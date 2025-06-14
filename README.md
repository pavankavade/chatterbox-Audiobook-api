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
- **📋 Text Queuing System** ⭐ *NEW*: Upload books in any size chapters and generate continuously
- **🔄 Chunk-Based Processing** ⭐ *NEW*: Improved reliability for longer text generations

### 🎵 **Audio Processing**
- **Smart Cleanup**: Remove unwanted silence and audio artifacts
- **Volume Normalization**: Professional-grade volume balancing for all voices
- **Real-time Audio Analysis**: Live volume level monitoring and feedback
- **Preview System**: Test settings before applying to entire projects
- **Batch Processing**: Process multiple projects efficiently
- **Quality Control**: Advanced audio optimization tools
- **🎯 Enhanced Audio Quality** ⭐ *NEW*: Improved P-top and minimum P parameters for better voice generation

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
- **📋 Queue Management** ⭐ *NEW*: Batch process multiple text files sequentially

---

## 🆕 Recent Improvements

### 🎯 **Audio Quality Enhancements**
We've significantly improved audio generation quality by optimizing the underlying TTS parameters:

- **Enhanced P-top and Minimum P Settings**: Fine-tuned probability parameters for more natural speech patterns
- **Reduced Audio Artifacts**: Better handling of pronunciation and intonation
- **Improved Voice Consistency**: More stable voice characteristics across long generations
- **Better Pronunciation**: Enhanced handling of complex words and names

**📝 Note for Existing Users**: 
- Older voice profiles will continue to work as before
- To take advantage of the new audio quality improvements, consider re-creating voice profiles
- Existing projects remain fully compatible

### 📋 **Text Queuing System**
Perfect for processing large books or multiple chapters:

- **Batch Upload**: Upload multiple text files of any size
- **Sequential Processing**: Automatically processes files one after another
- **Progress Tracking**: Monitor generation progress across all queued items
- **Flexible Chapter Sizes**: No restrictions on individual file length
- **Unattended Generation**: Set up large projects and let them run automatically

### 🔄 **Chunk-Based TTS System**
Enhanced the core text-to-speech engine for better reliability:

- **Background Chunking**: Automatically splits long texts into optimal chunks
- **Memory Management**: Better handling of large text inputs
- **Error Recovery**: Improved resilience during long generation sessions
- **Consistent Quality**: Maintains voice quality across chunk boundaries
- **Progress Feedback**: Real-time updates on generation progress

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

### **Text Queuing Workflow** ⭐ *NEW*
1. Go to **Production Studio** tab
2. Select "Batch Processing" mode
3. Upload multiple text files (chapters, sections, etc.)
4. Choose your voice and settings
5. Start batch processing - files will generate sequentially
6. Monitor progress and download completed audiobooks

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

### 📋 **Batch Processing Workflow** ⭐ *NEW*
1. **📚 Organize Chapters**: Split your book into individual text files
2. **📋 Queue Setup**: Upload all files to the batch processing system
3. **🎤 Voice Selection**: Choose voice and configure settings once
4. **🔄 Automated Generation**: Let the system process all files sequentially
5. **📊 Monitor Progress**: Track completion status in real-time
6. **📦 Collect Results**: Download all generated audiobook chapters

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

## ⚠️ Known Issues & Compatibility

### **Multi-Voice Generation**
- Short sentences or sections may occasionally cause issues during multi-voice generation
- This is a limitation of the underlying TTS models rather than the implementation
- **Workaround**: Use longer, more detailed sentences for better stability
- Single-voice generation is not affected by this issue

### **Voice Profile Compatibility**
- **Existing Voices**: All older voice profiles remain fully functional
- **New Features**: To benefit from improved audio quality, consider re-creating voice profiles
- **Project Compatibility**: Existing audiobook projects work without modification
- **Regeneration**: Individual chunks can be regenerated with improved quality settings

### **Batch Processing Considerations**
- Large batch jobs may take significant time depending on text length and hardware
- Monitor system resources during extended batch processing sessions
- Consider processing very large books in smaller batches for better control

---

## 📋 Supported Formats

### Input
- **Text**: `.txt`, `.md`, formatted stories and scripts
- **Audio Samples**: `.wav`, `.mp3`, `.flac` for voice cloning
- **Batch Files**: Multiple text files for queue processing

### Output
- **Audio**: High-quality `.wav` files with professional volume levels
- **Projects**: Organized folder structure with chapters
- **Exports**: Ready-to-use audiobook files
- **Batch Results**: Multiple completed audiobooks from queue processing

---

## 🆘 Support

- **Features Guide**: See `AUDIOBOOK_FEATURES.md` for detailed capabilities
- **Development Notes**: Check `development/` folder for technical details
- **Issues**: Report problems via GitHub issues

---

## 📄 License

This project is licensed under the terms specified in `LICENSE`.

---

**🎉 Ready to create amazing audiobooks with professional volume levels and enhanced audio quality? Run `./launch_audiobook.bat` and start generating!** 
