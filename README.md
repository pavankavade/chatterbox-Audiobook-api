# 🎧 Chatterbox Audiobook Generator

**Transform your text into high-quality audiobooks with advanced TTS models and voice cloning capabilities.**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
./install-audiobook.bat
```

### 2. Launch the Application
```bash
./launch_audiobook.bat
```

The web interface will open automatically in your browser at `http://localhost:7860`

---

## ✨ Features

### 📚 **Audiobook Creation**
- **Single Voice**: Generate entire audiobooks with one consistent voice
- **Multi-Voice**: Create dynamic audiobooks with multiple characters
- **Custom Voices**: Clone voices from audio samples for personalized narration

### 🎵 **Audio Processing**
- **Smart Cleanup**: Remove unwanted silence and audio artifacts
- **Preview System**: Test settings before applying to entire projects
- **Batch Processing**: Process multiple projects efficiently
- **Quality Control**: Advanced audio optimization tools

### 🎭 **Voice Management**
- **Voice Library**: Organize and manage your voice collection
- **Voice Cloning**: Create custom voices from audio samples
- **Character Assignment**: Map specific voices to story characters

### 📖 **Text Processing**
- **Chapter Support**: Automatic chapter detection and organization
- **Multi-Voice Parsing**: Parse character dialogue automatically
- **Text Validation**: Ensure proper formatting before generation

---

## 📁 Project Structure

```
📦 Your Audiobook Projects
├── 🎤 speakers/           # Voice library and samples
├── 📚 audiobook_projects/ # Generated audiobooks
├── 🔧 src/audiobook/      # Core processing modules
└── 📄 Generated files...  # Audio chunks and final outputs
```

---

## 🎯 Workflow

1. **📝 Prepare Text**: Format your story with proper chapter breaks
2. **🎤 Select Voices**: Choose or clone voices for your characters  
3. **⚙️ Configure Settings**: Adjust quality, speed, and processing options
4. **🎧 Generate Audio**: Create your audiobook with advanced TTS
5. **🧹 Clean & Optimize**: Use smart cleanup tools for perfect audio
6. **📦 Export**: Get your finished audiobook ready for distribution

---

## 🛠️ Technical Requirements

- **Python 3.8+**
- **CUDA GPU** (recommended for faster processing)
- **8GB+ RAM** (16GB recommended for large projects)
- **Modern web browser** for the interface

---

## 📋 Supported Formats

### Input
- **Text**: `.txt`, `.md`, formatted stories and scripts
- **Audio Samples**: `.wav`, `.mp3`, `.flac` for voice cloning

### Output
- **Audio**: High-quality `.wav` files
- **Projects**: Organized folder structure with chapters
- **Exports**: Ready-to-use audiobook files

---

## 🆘 Support

- **Features Guide**: See `AUDIOBOOK_FEATURES.md` for detailed capabilities
- **Development Notes**: Check `development/` folder for technical details
- **Issues**: Report problems via GitHub issues

---

## 📄 License

This project is licensed under the terms specified in `LICENSE`.

---

**🎉 Ready to create amazing audiobooks? Run `./launch_audiobook.bat` and start generating!** 