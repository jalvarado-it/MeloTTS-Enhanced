# flake8: noqa: E402

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import logging
import gc

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from melo.download_utils import load_pretrain_model

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Enable memory-efficient attention
torch.backends.cuda.enable_mem_efficient_sdp(True)

global_step = 0


class MemoryManager:
    """Utility class for memory management during training"""
    
    @staticmethod
    def cleanup():
        """Force garbage collection and clear CUDA cache"""
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0
    
    @staticmethod
    def log_memory(logger, step, prefix=""):
        """Log current memory usage"""
        if logger:
            memory_gb = MemoryManager.get_memory_usage()
            logger.info(f"{prefix}Memory usage at step {step}: {memory_gb:.2f}GB")


def create_optimized_dataloader(dataset, hps, rank, n_gpus, is_train=True):
    """Create memory-optimized dataloader"""
    if is_train:
        sampler = DistributedBucketSampler(
            dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        
        return DataLoader(
            dataset,
            num_workers=4,  # Reduced from 16
            shuffle=False,
            pin_memory=False,  # Saves GPU memory
            collate_fn=TextAudioSpeakerCollate(),
            batch_sampler=sampler,
            persistent_workers=True,
            prefetch_factor=2,  # Reduced from 4
        )
    else:
        return DataLoader(
            dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=TextAudioSpeakerCollate(),
        )


def load_models_sequentially(hps, rank):
    """Load models one at a time to reduce memory peaks"""
    
    # Configure model parameters for memory efficiency
    if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas:
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    # Load generator
    print("Loading Generator...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)
    
    # Enable gradient checkpointing for generator
    if hasattr(net_g, 'enable_gradient_checkpointing'):
        net_g.enable_gradient_checkpointing()
    
    MemoryManager.cleanup()
    
    # Load discriminator
    print("Loading Discriminator...")
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    MemoryManager.cleanup()
    
    # Load duration discriminator if needed
    net_dur_disc = None
    if ("use_duration_discriminator" in hps.model.keys() 
        and hps.model.use_duration_discriminator):
        print("Loading Duration Discriminator...")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
        MemoryManager.cleanup()
    
    return net_g, net_d, net_dur_disc


def memory_efficient_forward(net_g, x, x_lengths, spec, spec_lengths, 
                           speakers, tone, language, bert, ja_bert, use_checkpoint=True):
    """Memory-efficient forward pass with optional gradient checkpointing"""
    if use_checkpoint and net_g.training:
        return checkpoint(
            net_g,
            x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, ja_bert,
            use_reentrant=False
        )
    else:
        return net_g(x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, ja_bert)


def run():
    hps = utils.get_hparams()
    
    # Memory optimization: Reduce batch size and segment size
    hps.train.batch_size = min(hps.train.batch_size, 4)  # Cap at 4 for <8GB VRAM
    hps.train.segment_size = min(hps.train.segment_size, 8192)  # Reduce segment size
    
    # Add gradient accumulation steps if not present
    if not hasattr(hps.train, 'gradient_accumulation_steps'):
        hps.train.gradient_accumulation_steps = 4
    
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=local_rank,
    )
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step
    
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:
        logger = None
        writer = None
        writer_eval = None
    
    # Create datasets
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_loader = create_optimized_dataloader(train_dataset, hps, rank, n_gpus, True)
    
    eval_loader = None
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = create_optimized_dataloader(eval_dataset, hps, rank, n_gpus, False)
    
    # Load models sequentially to reduce memory peaks
    net_g, net_d, net_dur_disc = load_models_sequentially(hps, rank)
    
    # Create optimizers
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_dur_disc = None
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    
    # Wrap models with DDP - need find_unused_parameters=True for gradient accumulation
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
    
    # Load pretrained models
    pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
    hps.pretrain_G = hps.pretrain_G or pretrain_G
    hps.pretrain_D = hps.pretrain_D or pretrain_D
    hps.pretrain_dur = hps.pretrain_dur or pretrain_dur

    if hps.pretrain_G:
        utils.load_checkpoint(hps.pretrain_G, net_g, None, skip_optimizer=True)
        MemoryManager.cleanup()
    if hps.pretrain_D:
        utils.load_checkpoint(hps.pretrain_D, net_d, None, skip_optimizer=True)
        MemoryManager.cleanup()
    if net_dur_disc is not None and hps.pretrain_dur:
        utils.load_checkpoint(hps.pretrain_dur, net_dur_disc, None, skip_optimizer=True)
        MemoryManager.cleanup()
    
    # Load checkpoints
    epoch_str = 1
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc, optim_dur_disc,
                skip_optimizer=getattr(hps.train, 'skip_optimizer', True),
            )
        _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g, optim_g,
            skip_optimizer=getattr(hps.train, 'skip_optimizer', True),
        )
        _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d, optim_d,
            skip_optimizer=getattr(hps.train, 'skip_optimizer', True),
        )
        
        # Set initial learning rates
        if not optim_g.param_groups[0].get("initial_lr"):
            optim_g.param_groups[0]["initial_lr"] = g_resume_lr
        if not optim_d.param_groups[0].get("initial_lr"):
            optim_d.param_groups[0]["initial_lr"] = d_resume_lr
        if net_dur_disc is not None and not optim_dur_disc.param_groups[0].get("initial_lr"):
            optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")
        epoch_str = 1
        global_step = 0
    
    # Create schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_dur_disc = None
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    
    scaler = GradScaler(enabled=hps.train.fp16_run)
    
    # Initial memory cleanup
    MemoryManager.cleanup()
    if logger:
        MemoryManager.log_memory(logger, 0, "Initial ")

    # Training loop
    for epoch in range(epoch_str, hps.train.epochs + 1):
        try:
            if rank == 0:
                train_and_evaluate(
                    rank, epoch, hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler, [train_loader, eval_loader],
                    logger, [writer, writer_eval],
                )
            else:
                train_and_evaluate(
                    rank, epoch, hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler, [train_loader, None],
                    None, None,
                )
        except Exception as e:
            print(f"Training error: {e}")
            MemoryManager.cleanup()
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
        if scheduler_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    
    if writers is not None:
        writer, writer_eval = writers
    else:
        writer = writer_eval = None

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    
    # Gradient accumulation counter
    accumulation_steps = getattr(hps.train, 'gradient_accumulation_steps', 1)
    
    # Initialize optimizers for the epoch
    optim_g.zero_grad()
    optim_d.zero_grad()
    if optim_dur_disc is not None:
        optim_dur_disc.zero_grad()
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        (x, x_lengths, spec, spec_lengths, y, y_lengths, 
         speakers, tone, language, bert, ja_bert) = batch_data
        
        # Update MAS noise scale
        if hasattr(net_g.module, 'use_noise_scaled_mas') and net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial - 
                net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        
        # Move tensors to GPU efficiently
        device_kwargs = {'device': rank, 'non_blocking': True}
        x, x_lengths = x.cuda(**device_kwargs), x_lengths.cuda(**device_kwargs)
        spec, spec_lengths = spec.cuda(**device_kwargs), spec_lengths.cuda(**device_kwargs)
        y, y_lengths = y.cuda(**device_kwargs), y_lengths.cuda(**device_kwargs)
        speakers = speakers.cuda(**device_kwargs)
        tone = tone.cuda(**device_kwargs)
        language = language.cuda(**device_kwargs)
        bert = bert.cuda(**device_kwargs)
        ja_bert = ja_bert.cuda(**device_kwargs)

        # Forward pass with gradient checkpointing
        with autocast(enabled=hps.train.fp16_run):
            generator_outputs = memory_efficient_forward(
                net_g, x, x_lengths, spec, spec_lengths, 
                speakers, tone, language, bert, ja_bert
            )
            
            (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
             (z, z_p, m_p, logs_p, m_q, logs_q),
             (hidden_x, logw, logw_)) = generator_outputs
            
            # Compute mel spectrograms
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            
            # Clear intermediate tensors
            del generator_outputs
            
            # Discriminator forward pass
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc / accumulation_steps
            
            # Duration discriminator (if enabled)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                        y_dur_hat_r, y_dur_hat_g
                    )
                    loss_dur_disc_all = loss_dur_disc / accumulation_steps
                
                # Duration discriminator backward (accumulate gradients)
                scaler.scale(loss_dur_disc_all).backward()
                
                # Step only when accumulation is complete
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optim_dur_disc)
                    commons.clip_grad_value_(net_dur_disc.parameters(), None)
                    scaler.step(optim_dur_disc)
                    optim_dur_disc.zero_grad()
        
        # Main discriminator backward (accumulate gradients) 
        scaler.scale(loss_disc_all).backward()
        
        # Step only when accumulation is complete
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            optim_d.zero_grad()
        
        # Generator forward pass
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_dur + loss_kl) / accumulation_steps
                
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen / accumulation_steps
        
        # Generator backward (accumulate gradients)
        scaler.scale(loss_gen_all).backward()
        
        # Step only when accumulation is complete
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
            optim_g.zero_grad()
        
        # Clear tensors to free memory
        del y_hat, y_mel, y_hat_mel, mel, y, x, spec
        del y_d_hat_r, y_d_hat_g, fmap_r, fmap_g
        if net_dur_disc is not None:
            del y_dur_hat_r, y_dur_hat_g, hidden_x, logw, logw_
        
        # Periodic memory cleanup
        if batch_idx % 10 == 0:
            MemoryManager.cleanup()
        
        # Logging and evaluation (only after gradient steps)
        if rank == 0 and (batch_idx + 1) % accumulation_steps == 0:
            actual_global_step = global_step // accumulation_steps
            
            if actual_global_step % (hps.train.log_interval * 2) == 0:  # Less frequent logging
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc * accumulation_steps, loss_gen * accumulation_steps, 
                         loss_fm, loss_mel, loss_dur, loss_kl]
                
                logger.info(f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]")
                logger.info([x.item() for x in losses] + [actual_global_step, lr])
                
                # Reduced tensorboard logging
                scalar_dict = {
                    "loss/g/total": loss_gen_all * accumulation_steps,
                    "loss/d/total": loss_disc_all * accumulation_steps,
                    "learning_rate": lr,
                }
                
                utils.summarize(writer=writer, global_step=actual_global_step, scalars=scalar_dict)
                MemoryManager.log_memory(logger, actual_global_step, "Training ")
            
            if actual_global_step % (hps.train.eval_interval * 2) == 0 and actual_global_step > 0:  # Less frequent evaluation
                evaluate(hps, net_g, eval_loader, writer_eval)
                
                # Save checkpoints
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{actual_global_step}.pth"),
                )
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"D_{actual_global_step}.pth"),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch,
                        os.path.join(hps.model_dir, f"DUR_{actual_global_step}.pth"),
                    )
                
                # Clean old checkpoints
                keep_ckpts = getattr(hps.train, "keep_ckpts", 3)  # Reduced from 5
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
        
        global_step += 1

    if rank == 0:
        progress_percent = (epoch / hps.train.epochs) * 100
        remaining_epochs = hps.train.epochs - epoch
        epoch_time = 2.0  # Approximate seconds per epoch based on logs
        estimated_remaining_time = remaining_epochs * epoch_time
        
        hours = int(estimated_remaining_time // 3600)
        minutes = int((estimated_remaining_time % 3600) // 60)
        
        # ANSI color codes
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        # Create progress bar visualization
        bar_length = 40
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        progress_msg = f"====> Epoch: {epoch}/{hps.train.epochs} ({progress_percent:.1f}% complete)"
        time_msg = f"Estimated time remaining: {hours}h {minutes}m ({remaining_epochs} epochs)"
        
        # Clear screen and move cursor to top (ANSI escape sequences)
        print('\033[2J\033[H', end='')  # Clear screen and move to top
        
        # Colorful console output
        print(f"{BOLD}{GREEN}====> Epoch: {CYAN}{epoch}{END}{BOLD}{GREEN}/{CYAN}{hps.train.epochs} {YELLOW}({progress_percent:.1f}% complete){END}")
        print(f"{BLUE}[{bar}]{END} {YELLOW}{progress_percent:.1f}%{END}")
        print(f"{BOLD}{GREEN}⏱  Estimated time remaining: {YELLOW}{hours}h {minutes}m {GREEN}({remaining_epochs} epochs){END}")
        print()  # Add blank line for spacing
        
        # Plain text for log
        logger.info(progress_msg)
        logger.info(time_msg)
    
    MemoryManager.cleanup()


def evaluate(hps, generator, eval_loader, writer_eval):
    """Memory-efficient evaluation"""
    generator.eval()
    print("Evaluating ...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_loader):
            if batch_idx >= 3:  # Limit evaluation samples to save memory
                break
                
            (x, x_lengths, spec, spec_lengths, y, y_lengths,
             speakers, tone, language, bert, ja_bert) = batch_data
            
            # Move to GPU
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            
            # Generate with SDP
            y_hat, attn, mask, *_ = generator.module.infer(
                x, x_lengths, speakers, tone, language, bert, ja_bert,
                y=spec, max_len=1000, sdp_ratio=1.0,
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
            
            # Compute mel spectrograms
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            
            # Log only essential data
            image_dict = {
                f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].cpu().numpy()
                ),
                f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                    mel[0].cpu().numpy()
                ),
            }
            
            audio_dict = {
                f"gen/audio_{batch_idx}": y_hat[0, :, :y_hat_lengths[0]],
                f"gt/audio_{batch_idx}": y[0, :, :y_lengths[0]],
            }
            
            utils.summarize(
                writer=writer_eval,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sampling_rate=hps.data.sampling_rate,
            )
            
            # Clear memory after each sample
            del y_hat, mel, y_hat_mel, x, spec, y
            MemoryManager.cleanup()

    generator.train()
    print('Evaluation done')


if __name__ == "__main__":
    run()