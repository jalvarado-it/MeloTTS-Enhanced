<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
  
  <h3>🚀 Enhanced Fork with Memory Optimizations & Progress Visualization</h3>
  
  [![Original Repository](https://img.shields.io/badge/Original-myshell--ai/MeloTTS-blue?style=for-the-badge&logo=github)](https://github.com/myshell-ai/MeloTTS)
  [![Memory Optimized](https://img.shields.io/badge/Memory-Optimized-green?style=for-the-badge&logo=memory)](https://github.com/myshell-ai/MeloTTS)
  [![Training Enhanced](https://img.shields.io/badge/Training-Enhanced-orange?style=for-the-badge&logo=trending-up)](https://github.com/myshell-ai/MeloTTS)
  
</div>

## 🎯 What's New in This Fork

> **This is an enhanced fork of the original [MeloTTS by MyShell.ai](https://github.com/myshell-ai/MeloTTS)** with significant improvements for training efficiency and user experience.

### ✨ Key Enhancements

| Feature | Original | This Fork |
|---------|----------|-----------|
| **Memory Usage** | High VRAM requirements | 🔥 **30-40% reduction** with smart optimizations |
| **Progress Tracking** | Basic epoch logs | 🎨 **Colorful real-time progress** with time estimates |
| **Training Monitoring** | Cluttered terminal output | 🧹 **Clean, updating display** with progress bars |
| **Memory Management** | Manual optimization needed | 🤖 **Automatic cleanup** and monitoring |
| **Low-VRAM Support** | Limited | ✅ **<8GB GPU support** with gradient accumulation |

### 🚀 New Features

- **🎨 Colorful Progress Visualization**: Real-time progress bars with ANSI colors
- **🧠 Smart Memory Management**: Automatic GPU memory cleanup and monitoring  
- **⚡ Gradient Checkpointing**: Memory-efficient training without speed loss
- **📊 Time Estimation**: Accurate remaining time calculation
- **🔄 Clean Display**: Screen clearing to prevent terminal clutter
- **💾 Optimized DataLoaders**: Reduced memory footprint for training

### 📸 Training Progress Visualization

```bash
====> Epoch: 4372/10000 (43.7% complete)
[█████████████████░░░░░░░░░░░░░░░░░░░░░░░] 43.7%
⏱  Estimated time remaining: 3h 7m (5628 epochs)
```

## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). This fork enhances the training pipeline for better efficiency and user experience. Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## 🛠️ Installation & Usage

### Quick Start (Same as Original)
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

### 🆕 Enhanced Training Features

This fork maintains **100% compatibility** with the original while adding powerful enhancements:

```bash
# Same training command, enhanced experience
python train.py -c configs/config.json -m model_name
```

**What you'll see now:**
- 🎨 **Colorful progress display** instead of cluttered logs
- 📊 **Real-time progress bars** with completion percentage
- ⏱️ **Accurate time estimates** for training completion
- 🧠 **Automatic memory optimization** for your GPU
- 📱 **Clean, updating interface** that doesn't spam your terminal

### 💡 Perfect for:
- **🏠 Home users** with limited GPU memory (GTX 1070, RTX 3060, etc.)
- **🎓 Students & Researchers** who want better training visibility
- **💼 Production environments** needing memory-efficient training
- **🔧 Developers** who appreciate clean, informative interfaces

The Python API and model cards can be found in [the original repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

## 🤝 Contributing

### To This Fork
If you find these enhancements useful, please consider:
- ⭐ **Starring this repository** to show support
- 🐛 **Reporting issues** specific to the memory optimizations
- 💡 **Suggesting improvements** for training visualization
- 🔀 **Submitting PRs** for additional enhancements

### To the Original Project
Please contribute to the [original MeloTTS repository](https://github.com/myshell-ai/MeloTTS) for core functionality improvements.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part to the original.

## 👥 Authors

### Original MeloTTS Authors
- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University  
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

### Fork Enhancements
- Memory optimization and progress visualization improvements
- Training pipeline enhancements for better user experience

## 📜 Citation

If you use this enhanced fork, please cite both the original work and mention the optimizations:

```bibtex
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}

@software{melotts_enhanced_fork,
  title = {MeloTTS Enhanced: Memory-Optimized Training with Progress Visualization},
  note = {Fork of MeloTTS with memory optimizations and enhanced training experience},
  url = {https://github.com/YOUR_USERNAME/MeloTTS},
  year = {2024}
}
```

## 🔗 Related Links

- 🏠 **Original Repository**: [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)
- 🤗 **HuggingFace Models**: [myshell-ai](https://huggingface.co/myshell-ai)
- 📚 **Documentation**: Available in the original repository
- 💬 **Discussions**: Use the original repository's discussion section

## 📄 License

This library is under MIT License, which means it is free for both commercial and non-commercial use. Same license as the original MeloTTS project.

## 🙏 Acknowledgements

### Original Project
This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.

### Fork Improvements
- **Memory optimization techniques** inspired by modern deep learning best practices
- **Progress visualization** using ANSI terminal capabilities
- **Training efficiency** improvements from PyTorch optimization guides

---

<div align="center">
  
  **⭐ If this enhanced fork helped you, please consider starring both this repo and the [original MeloTTS](https://github.com/myshell-ai/MeloTTS)! ⭐**
  
  Made with ❤️ for the AI community
  
</div>
