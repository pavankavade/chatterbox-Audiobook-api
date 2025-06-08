# Raw Voices Feature - Separate Processed and Unprocessed Audio

## ✅ IMPLEMENTED: Raw Voices Section

Successfully implemented a separate section for raw audio files alongside processed voice profiles, making it clear which voices are ready to use and which need configuration.

## Feature Overview

### 🎯 **Purpose**
- **Separate processed voice profiles** (with settings) from raw audio files
- **Keep main app clean** by only showing processed voices in dropdown selections
- **Easy conversion workflow** from raw audio to configured voice profiles
- **Clear visual distinction** between ready-to-use voices and raw materials

### 🎛️ **Interface Design**

#### **Processed Voice Profiles Section** (Existing)
- **🎯 Select Voice**: Shows only fully configured voice profiles with settings
- **Load/Refresh/Delete**: Standard voice profile management
- **Used throughout app**: These voices appear in all TTS, Single Voice, and Multi-Voice dropdowns

#### **Raw Audio Files Section** (New)
- **🎵 Raw Audio Files**: Shows unprocessed audio files without voice profiles
- **Smart Detection**: Automatically excludes files that already have voice profiles
- **Easy Loading**: One-click to load raw audio into voice configuration
- **Auto-naming**: Suggests voice profile names based on filenames

---

## How It Works

### **Smart File Detection**
```python
def _get_raw_audio_files(self, voice_library_path: str):
    # 1. Scan for audio files (.wav, .mp3, .flac, .ogg, .m4a)
    # 2. Get list of existing voice profiles
    # 3. Return only audio files WITHOUT voice profiles
    # 4. Support both direct files and subfolder structures
```

### **File Organization Support**
- **Direct Files**: `my_voice.wav` → Shows as "my_voice.wav"
- **Subfolder Structure**: `character_name/audio.wav` → Shows as "character_name/audio.wav"
- **Mixed Libraries**: Handles both organized and flat folder structures

### **Automatic Exclusion Logic**
- If `hero_voice.wav` has a voice profile named "hero_voice", it won't appear in raw voices
- Only shows truly unprocessed audio files
- Updates automatically when voice profiles are created or deleted

---

## User Workflow

### **Converting Raw Audio to Voice Profile**

1. **Browse Raw Voices**: Look at "🎵 Raw Audio Files" section
2. **Select Audio**: Choose an unprocessed audio file from dropdown
3. **Load Raw Audio**: Click "📁 Load Raw Audio" button
4. **Auto-population**:
   - Audio file loads into the voice configuration
   - Voice name auto-suggested from filename
   - Ready to configure settings and save
5. **Configure & Save**: Set exaggeration, temperature, etc. and save voice profile
6. **Automatic Update**: Raw audio disappears from raw list, appears in processed voices

### **Managing Voice Libraries**

#### **Clean Separation**
- **Main App Dropdowns**: Only show processed voices with settings
- **Voice Library Raw Section**: Shows unprocessed audio needing configuration
- **Status Updates**: Path changes show count of both processed and raw voices

#### **Batch Processing**
- See all unprocessed audio at a glance
- Process multiple raw voices into profiles systematically
- Organize voice library efficiently

---

## Technical Implementation

### **File Scanning Logic**
```python
# Supported audio formats
audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Check against existing voice profiles
processed_voices = get_voice_choices(voice_library_path)

# Find raw files (audio without voice profiles)
raw_files = [file for file in audio_files 
             if filename_without_extension not in processed_voices]
```

### **Auto-naming Algorithm**
```python
# For "character_voices/hero.wav" → suggests "character_voices"
# For "narrator_deep.wav" → suggests "narrator_deep"
# Cleans special characters for valid voice profile names
```

### **Interface Integration**
- **Path Updates**: Both sections refresh when voice library path changes
- **Real-time Updates**: Raw voices update when voice profiles are created/deleted
- **Status Messages**: Show counts of both processed and raw voices

---

## User Interface

### **Voice Library Tab Layout**
```
🎯 Select Voice
┌─────────────────────────────────────┐
│ Saved Voice Profiles                │
│ ▼ narrator_deep                     │
└─────────────────────────────────────┘
[📥 Load Voice] [🔄 Refresh] [🗑️ Delete]

🎵 Raw Audio Files
💡 Raw voices are audio files without voice profile settings.
Select a raw audio file below to create a voice profile.

┌─────────────────────────────────────┐
│ Available Audio Files               │
│ ▼ character_voices/hero.wav         │
└─────────────────────────────────────┘
[📁 Load Raw Audio] [🔄 Refresh Raw]
```

### **Status Display Updates**
```
✅ Voice library updated: C:\MyVoices
Found 5 voice profiles, 12 raw audio files
```

## Benefits

### ✅ **Clean Organization**
- **Processed voices only** in main app dropdowns
- **Clear distinction** between ready and unready voices
- **Efficient workflow** for voice profile creation

### ✅ **Smart Management**
- **Automatic detection** of unprocessed audio
- **Prevents duplication** by excluding processed files
- **Easy conversion** from raw to configured

### ✅ **Professional Workflow**
- **Batch processing** capability for large voice libraries
- **Status tracking** shows progress through voice collection
- **Organized development** from raw materials to finished profiles

### ✅ **User Experience**
- **No confusion** about which voices are ready to use
- **Streamlined interface** with logical separation
- **Quick conversion** process with auto-suggested names

## Current Status: ✅ FULLY FUNCTIONAL

The Raw Voices feature is now completely implemented and functional:
- ✅ **Smart File Detection**: Automatically finds unprocessed audio files
- ✅ **Clean Separation**: Processed vs raw voices clearly distinguished  
- ✅ **Easy Conversion**: One-click loading of raw audio into voice configuration
- ✅ **Auto-naming**: Intelligent voice name suggestions from filenames
- ✅ **Real-time Updates**: Interface refreshes automatically when files change
- ✅ **Professional Workflow**: Efficient voice library management

Users now have a clear, organized approach to managing their voice libraries with distinct areas for finished voice profiles and raw audio materials awaiting configuration! 