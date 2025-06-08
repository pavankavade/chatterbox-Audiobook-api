# 🎧 Chatterbox Audiobook Generator

**This is a work in progress. You can consider this a pre-launch repo at the moment, but if you find bugs, please put them in the issues area. Thank you.**
**Transform your text into high-quality audiobooks with advanced TTS models, voice cloning, and professional volume normalization.**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
./install-audiobook.bat
```

### 2. Launch the Application
```bash
./launch_audiobook.bat
```

### 3. CUDA Issue Fix (If Needed)
If you encounter CUDA assertion errors during generation, install the patched version:
```bash
# Activate your virtual environment first
venv\Scripts\activate.bat

# Install the CUDA-fixed version
pip install --force-reinstall --no-cache-dir "chatterbox-tts @ git+https://github.com/fakerybakery/better-chatterbox@fix-cuda-issue"
```

The web interface will open automatically in your browser at `http://localhost:7860`

---

## ✨ Features

### 📚 **Audiobook Creation**
- **Single Voice**: Generate entire audiobooks with one consistent voice
- **Multi-Voice**: Create dynamic audiobooks with multiple characters
- **Custom Voices**: Clone voices from audio samples for personalized narration
- **Professional Volume Normalization**: Ensure consistent audio levels across all voices

### 🎵 **Audio Processing**
- **Smart Cleanup**: Remove unwanted silence and audio artifacts
- **Volume Normalization**: Professional-grade volume balancing for all voices
- **Real-time Audio Analysis**: Live volume level monitoring and feedback
- **Preview System**: Test settings before applying to entire projects
- **Batch Processing**: Process multiple projects efficiently
- **Quality Control**: Advanced audio optimization tools

### 🎭 **Voice Management**
- **Voice Library**: Organize and manage your voice collection
- **Voice Cloning**: Create custom voices from audio samples
- **Volume Settings**: Configure target volume levels for each voice
- **Professional Presets**: Industry-standard volume levels (audiobook, podcast, broadcast)
- **Character Assignment**: Map specific voices to story characters

### 📊 **Volume Normalization System** ⭐ *NEW*
- **Professional Standards**: Audiobook (-18 dB), Podcast (-16 dB), Broadcast (-23 dB) presets
- **Consistent Character Voices**: All characters maintain the same volume level
- **Real-time Analysis**: Color-coded volume status with RMS and peak level display
- **Retroactive Normalization**: Apply volume settings to existing voice projects
- **Multi-Voice Support**: Batch normalize all voices in multi-character audiobooks
- **Soft Limiting**: Intelligent audio limiting to prevent distortion

### 📖 **Text Processing**
- **Chapter Support**: Automatic chapter detection and organization
- **Multi-Voice Parsing**: Parse character dialogue automatically
- **Text Validation**: Ensure proper formatting before generation

---

## 🎚️ Volume Normalization Guide

### **Individual Voice Setup**
1. Go to **Voice Library** tab
2. Upload your voice sample and configure settings
3. Set target volume level (default: -18 dB for audiobooks)
4. Choose from professional presets or use custom levels
5. Save voice profile with volume settings

### **Multi-Voice Projects**
1. Navigate to **Multi-Voice Audiobook Creation** tab
2. Enable volume normalization for all voices
3. Set target level for consistent character voices
4. All characters will be automatically normalized during generation

### **Professional Standards**
- **📖 Audiobook Standard**: -18 dB RMS (recommended for most audiobooks)
- **🎙️ Podcast Standard**: -16 dB RMS (for podcast-style content)
- **🔇 Quiet/Comfortable**: -20 dB RMS (for quiet listening environments)
- **🔊 Loud/Energetic**: -14 dB RMS (for dynamic, energetic content)
- **📺 Broadcast Standard**: -23 dB RMS (for broadcast television standards)

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
3. **🎚️ Configure Volume**: Set professional volume levels and normalization
4. **⚙️ Configure Settings**: Adjust quality, speed, and processing options
5. **🎧 Generate Audio**: Create your audiobook with advanced TTS
6. **🧹 Clean & Optimize**: Use smart cleanup tools for perfect audio
7. **📦 Export**: Get your finished audiobook ready for distribution

---

## 🛠️ Technical Requirements

- **Python 3.8+**
- **CUDA GPU** (recommended for faster processing)
- **8GB+ RAM** (16GB recommended for large projects)
- **Modern web browser** for the interface

### 🔧 **CUDA Support**
- CUDA compatibility issues have been resolved with updated dependencies
- GPU acceleration is now stable for extended generation sessions
- Fallback to CPU processing available if CUDA issues occur
- **If you encounter CUDA assertion errors**: Use the patched version from the installation instructions above
- The fix addresses PyTorch indexing issues that could cause crashes during audio generation

---

## ⚠️ Known Issues

### **Multi-Voice Generation**
- Short sentences or sections may occasionally cause issues during multi-voice generation
- This is a limitation of the underlying TTS models rather than the implementation
- **Workaround**: Use longer, more detailed sentences for better stability
- Single-voice generation is not affected by this issue

---

## 📋 Supported Formats

### Input
- **Text**: `.txt`, `.md`, formatted stories and scripts
- **Audio Samples**: `.wav`, `.mp3`, `.flac` for voice cloning

### Output
- **Audio**: High-quality `.wav` files with professional volume levels
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

**🎉 Ready to create amazing audiobooks with professional volume levels? Run `./launch_audiobook.bat` and start generating!** 
