import random
import numpy as np
import torch
import gradio as gr
import json
import os
import shutil
import re
import wave
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU mode for multi-voice to avoid CUDA indexing errors
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice processing

# Default voice library path
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"

def load_config():
    """Load configuration including voice library path"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('voice_library_path', DEFAULT_VOICE_LIBRARY)
        except:
            return DEFAULT_VOICE_LIBRARY
    return DEFAULT_VOICE_LIBRARY

def save_config(voice_library_path):
    """Save configuration including voice library path"""
    config = {
        'voice_library_path': voice_library_path,
        'last_updated': str(Path().resolve())  # timestamp
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return f"✅ Configuration saved - Voice library path: {voice_library_path}"
    except Exception as e:
        return f"❌ Error saving configuration: {str(e)}"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model

def load_model_cpu():
    """Load model specifically for CPU processing"""
    model = ChatterboxTTS.from_pretrained("cpu")
    return model

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            clear_gpu_memory()
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return wav, "GPU"
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or 
                "CUDA" in str(e) or 
                "out of memory" in str(e).lower()):
                
                print(f"⚠️ CUDA error detected, falling back to CPU: {str(e)[:100]}...")
                # Fall through to CPU mode
            else:
                raise e
    
    # CPU fallback or primary CPU mode
    try:
        # Load CPU model if needed
        cpu_model = ChatterboxTTS.from_pretrained("cpu")
        wav = cpu_model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return True

def chunk_text_by_sentences(text, max_words=50):
    """
    Split text into chunks, breaking at sentence boundaries after reaching max_words
    """
    # Split text into sentences using regex to handle multiple punctuation marks
    sentences = re.split(r'([.!?]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        # Add punctuation if it exists
        if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, start new chunk
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """
    Save audio chunks as numbered WAV files
    """
    if not project_name.strip():
        project_name = "untitled_audiobook"
    
    # Sanitize project name
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_project_name = safe_project_name.replace(' ', '_')
    
    # Create output directory
    project_dir = os.path.join(output_dir, safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    saved_files = []
    
    for i, audio_chunk in enumerate(audio_chunks, 1):
        filename = f"{safe_project_name}_{i:03d}.wav"
        filepath = os.path.join(project_dir, filename)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert float32 to int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        saved_files.append(filepath)
    
    return saved_files, project_dir

def ensure_voice_library_exists(voice_library_path):
    """Ensure the voice library directory exists"""
    Path(voice_library_path).mkdir(parents=True, exist_ok=True)
    return voice_library_path

def get_voice_profiles(voice_library_path):
    """Get list of saved voice profiles"""
    if not os.path.exists(voice_library_path):
        return []
    
    profiles = []
    for item in os.listdir(voice_library_path):
        profile_path = os.path.join(voice_library_path, item)
        if os.path.isdir(profile_path):
            config_file = os.path.join(profile_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    profiles.append({
                        'name': item,
                        'display_name': config.get('display_name', item),
                        'description': config.get('description', ''),
                        'config': config
                    })
                except:
                    continue
    return profiles

def get_voice_choices(voice_library_path):
    """Get voice choices for dropdown with display names"""
    profiles = get_voice_profiles(voice_library_path)
    choices = [("Manual Input (Upload Audio)", None)]  # Default option
    for profile in profiles:
        display_text = f"🎭 {profile['display_name']} ({profile['name']})"
        choices.append((display_text, profile['name']))
    return choices

def get_audiobook_voice_choices(voice_library_path):
    """Get voice choices for audiobook creation (no manual input option)"""
    profiles = get_voice_profiles(voice_library_path)
    choices = []
    if not profiles:
        choices.append(("No voices available - Create voices first", None))
    else:
        for profile in profiles:
            display_text = f"🎭 {profile['display_name']} ({profile['name']})"
            choices.append((display_text, profile['name']))
    return choices

def load_text_file(file_path):
    """Load text from uploaded file"""
    if file_path is None:
        return "No file uploaded", "❌ Please upload a text file"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic validation
        if not content.strip():
            return "", "❌ File is empty"
        
        word_count = len(content.split())
        char_count = len(content)
        
        status = f"✅ File loaded successfully!\n📄 {word_count:,} words | {char_count:,} characters"
        
        return content, status
        
    except UnicodeDecodeError:
        try:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            word_count = len(content.split())
            char_count = len(content)
            status = f"✅ File loaded (latin-1 encoding)!\n📄 {word_count:,} words | {char_count:,} characters"
            return content, status
        except Exception as e:
            return "", f"❌ Error reading file: {str(e)}"
    except Exception as e:
        return "", f"❌ Error loading file: {str(e)}"

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate inputs for audiobook creation"""
    issues = []
    
    if not text_content or not text_content.strip():
        issues.append("📝 Text content is required")
    
    if not selected_voice:
        issues.append("🎭 Voice selection is required")
    
    if not project_name or not project_name.strip():
        issues.append("📁 Project name is required")
    
    if text_content and len(text_content.strip()) < 10:
        issues.append("📏 Text is too short (minimum 10 characters)")
    
    if issues:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Please fix these issues:\n" + "\n".join(f"• {issue}" for issue in issues), 
            gr.Audio(visible=False)
        )
    
    word_count = len(text_content.split())
    chunks = chunk_text_by_sentences(text_content)
    chunk_count = len(chunks)
    
    return (
        gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=True),
        f"✅ Ready for audiobook creation!\n📊 {word_count:,} words → {chunk_count} chunks\n📁 Project: {project_name.strip()}", 
        gr.Audio(visible=True)
    )

def get_voice_config(voice_library_path, voice_name):
    """Get voice configuration for audiobook generation"""
    if not voice_name:
        return None
    
    # Sanitize voice name - remove special characters that might cause issues
    safe_voice_name = voice_name.replace("_-_", "_").replace("__", "_")
    safe_voice_name = "".join(c for c in safe_voice_name if c.isalnum() or c in ('_', '-')).strip('_-')
    
    # Try original name first, then sanitized name
    for name_to_try in [voice_name, safe_voice_name]:
        profile_dir = os.path.join(voice_library_path, name_to_try)
        config_file = os.path.join(profile_dir, "config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                audio_file = None
                if config.get('audio_file'):
                    audio_path = os.path.join(profile_dir, config['audio_file'])
                    if os.path.exists(audio_path):
                        audio_file = audio_path
                
                return {
                    'audio_file': audio_file,
                    'exaggeration': config.get('exaggeration', 0.5),
                    'cfg_weight': config.get('cfg_weight', 0.5),
                    'temperature': config.get('temperature', 0.8),
                    'display_name': config.get('display_name', name_to_try)
                }
            except Exception as e:
                print(f"⚠️ Error reading config for voice '{name_to_try}': {str(e)}")
                continue
    
    return None

def clear_gpu_memory():
    """Clear GPU memory cache to prevent CUDA errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return f"GPU Memory - Allocated: {allocated//1024//1024}MB, Cached: {cached//1024//1024}MB"
    return "CUDA not available"

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """
    Adaptive text chunking that reduces chunk size if CUDA errors occur
    """
    if reduce_on_error:
        # Start with smaller chunks for multi-voice to reduce memory pressure
        max_words = min(max_words, 35)
    
    return chunk_text_by_sentences(text, max_words)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """Generate audio with retry logic for CUDA errors"""
    for retry in range(max_retries):
        try:
            # Clear memory before generation
            if retry > 0:
                clear_gpu_memory()
            
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return wav
            
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or 
                "CUDA" in str(e) or 
                "out of memory" in str(e).lower()):
                
                if retry < max_retries - 1:
                    print(f"⚠️ GPU error, retry {retry + 1}/{max_retries}: {str(e)[:100]}...")
                    clear_gpu_memory()
                    continue
                else:
                    raise RuntimeError(f"Failed after {max_retries} retries: {str(e)}")
            else:
                raise e
    
    raise RuntimeError("Generation failed after all retries")

def create_audiobook(model, text_content, voice_library_path, selected_voice, project_name):
    """Create audiobook from text using selected voice with smart chunking and improved error handling"""
    if not text_content or not selected_voice or not project_name:
        return None, "❌ Missing required fields"
    
    # Get voice configuration
    voice_config = get_voice_config(voice_library_path, selected_voice)
    if not voice_config:
        return None, f"❌ Could not load voice configuration for '{selected_voice}'"
    
    if not voice_config['audio_file']:
        return None, f"❌ No audio file found for voice '{voice_config['display_name']}'"
    
    try:
        # Chunk the text intelligently
        chunks = chunk_text_by_sentences(text_content)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return None, "❌ No text chunks to process"
        
        # Initialize model if needed
        if model is None:
            model = ChatterboxTTS.from_pretrained(DEVICE)
        
        audio_chunks = []
        status_updates = []
        
        # Clear memory before starting
        clear_gpu_memory()
        
        for i, chunk in enumerate(chunks, 1):
            try:
                # Update status
                chunk_words = len(chunk.split())
                status_msg = f"🎵 Processing chunk {i}/{total_chunks}\n🎭 Voice: {voice_config['display_name']}\n📝 Chunk {i}: {chunk_words} words\n📊 Progress: {i}/{total_chunks} chunks"
                status_updates.append(status_msg)
                
                # Generate audio for this chunk with retry logic
                wav = generate_with_retry(
                    model,
                    chunk,
                    voice_config['audio_file'],
                    voice_config['exaggeration'],
                    voice_config['temperature'],
                    voice_config['cfg_weight']
                )
                
                # Move to CPU immediately and clear GPU memory
                audio_chunks.append(wav.squeeze(0).cpu().numpy())
                del wav
                clear_gpu_memory()
                
            except Exception as chunk_error:
                return None, f"❌ Error processing chunk {i}: {str(chunk_error)}"
        
        # Save all chunks as numbered files
        saved_files, project_dir = save_audio_chunks(audio_chunks, model.sr, project_name)
        
        # Save project metadata for regeneration purposes
        voice_info = {
            'voice_name': selected_voice,
            'display_name': voice_config['display_name'],
            'audio_file': voice_config['audio_file'],
            'exaggeration': voice_config['exaggeration'],
            'cfg_weight': voice_config['cfg_weight'],
            'temperature': voice_config['temperature']
        }
        
        save_project_metadata(
            project_dir=project_dir,
            project_name=project_name,
            text_content=text_content,
            voice_info=voice_info,
            chunks=chunks,
            project_type="single_voice"
        )
        
        # Combine all audio for preview (just concatenate)
        combined_audio = np.concatenate(audio_chunks)
        
        total_words = len(text_content.split())
        duration_minutes = len(combined_audio) // model.sr // 60
        
        success_msg = f"✅ Audiobook created successfully!\n🎭 Voice: {voice_config['display_name']}\n📊 {total_words:,} words in {total_chunks} chunks\n⏱️ Duration: ~{duration_minutes} minutes\n📁 Saved to: {project_dir}\n🎵 Files: {len(saved_files)} audio chunks\n💾 Metadata saved for regeneration"
        
        return (model.sr, combined_audio), success_msg
        
    except Exception as e:
        clear_gpu_memory()
        error_msg = f"❌ Error creating audiobook: {str(e)}"
        return None, error_msg

def load_voice_for_tts(voice_library_path, voice_name):
    """Load a voice profile for TTS tab - returns settings for sliders"""
    if not voice_name:
        # Return to manual input mode
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), "📝 Manual input mode - upload your own audio file below"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), f"❌ Voice profile '{voice_name}' not found"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        audio_file = None
        if config.get('audio_file'):
            audio_path = os.path.join(profile_dir, config['audio_file'])
            if os.path.exists(audio_path):
                audio_file = audio_path
        
        # Hide manual audio upload when using saved voice
        audio_component = gr.Audio(visible=False) if audio_file else gr.Audio(visible=True)
        
        status_msg = f"✅ Using voice: {config.get('display_name', voice_name)}"
        if config.get('description'):
            status_msg += f" - {config['description']}"
        
        return (
            audio_file,
            config.get('exaggeration', 0.5),
            config.get('cfg_weight', 0.5),
            config.get('temperature', 0.8),
            audio_component,
            status_msg
        )
    except Exception as e:
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), f"❌ Error loading voice profile: {str(e)}"

def save_voice_profile(voice_library_path, voice_name, display_name, description, audio_file, exaggeration, cfg_weight, temperature):
    """Save a voice profile with its settings"""
    if not voice_name:
        return "❌ Error: Voice name cannot be empty"
    
    # Sanitize voice name for folder
    safe_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    if not safe_name:
        return "❌ Error: Invalid voice name"
    
    ensure_voice_library_exists(voice_library_path)
    
    profile_dir = os.path.join(voice_library_path, safe_name)
    os.makedirs(profile_dir, exist_ok=True)
    
    # Copy audio file if provided
    audio_path = None
    if audio_file:
        audio_ext = os.path.splitext(audio_file)[1]
        audio_path = os.path.join(profile_dir, f"reference{audio_ext}")
        shutil.copy2(audio_file, audio_path)
        # Store relative path
        audio_path = f"reference{audio_ext}"
    
    # Save configuration
    config = {
        "display_name": display_name or voice_name,
        "description": description or "",
        "audio_file": audio_path,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "created_date": str(Path().resolve())  # Add timestamp if needed
    }
    
    config_file = os.path.join(profile_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return f"✅ Voice profile '{display_name or voice_name}' saved successfully!"

def load_voice_profile(voice_library_path, voice_name):
    """Load a voice profile and return its settings"""
    if not voice_name:
        return None, 0.5, 0.5, 0.8, "No voice selected"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 0.5, 0.5, 0.8, f"❌ Voice profile '{voice_name}' not found"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        audio_file = None
        if config.get('audio_file'):
            audio_path = os.path.join(profile_dir, config['audio_file'])
            if os.path.exists(audio_path):
                audio_file = audio_path
        
        return (
            audio_file,
            config.get('exaggeration', 0.5),
            config.get('cfg_weight', 0.5),
            config.get('temperature', 0.8),
            f"✅ Loaded voice profile: {config.get('display_name', voice_name)}"
        )
    except Exception as e:
        return None, 0.5, 0.5, 0.8, f"❌ Error loading voice profile: {str(e)}"

def delete_voice_profile(voice_library_path, voice_name):
    """Delete a voice profile"""
    if not voice_name:
        return "❌ No voice selected", []
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    if os.path.exists(profile_dir):
        try:
            shutil.rmtree(profile_dir)
            return f"✅ Voice profile '{voice_name}' deleted successfully!", get_voice_profiles(voice_library_path)
        except Exception as e:
            return f"❌ Error deleting voice profile: {str(e)}", get_voice_profiles(voice_library_path)
    else:
        return f"❌ Voice profile '{voice_name}' not found", get_voice_profiles(voice_library_path)

def refresh_voice_list(voice_library_path):
    """Refresh the voice profile list"""
    profiles = get_voice_profiles(voice_library_path)
    choices = [p['name'] for p in profiles]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def refresh_voice_choices(voice_library_path):
    """Refresh voice choices for TTS dropdown"""
    choices = get_voice_choices(voice_library_path)
    return gr.Dropdown(choices=choices, value=None)

def refresh_audiobook_voice_choices(voice_library_path):
    """Refresh voice choices for audiobook creation"""
    choices = get_audiobook_voice_choices(voice_library_path)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices and choices[0][1] else None)

def update_voice_library_path(new_path):
    """Update the voice library path and save to config"""
    if not new_path.strip():
        return DEFAULT_VOICE_LIBRARY, "❌ Path cannot be empty, using default", refresh_voice_list(DEFAULT_VOICE_LIBRARY), refresh_voice_choices(DEFAULT_VOICE_LIBRARY), refresh_audiobook_voice_choices(DEFAULT_VOICE_LIBRARY)
    
    # Ensure the directory exists
    ensure_voice_library_exists(new_path)
    
    # Save to config
    save_msg = save_config(new_path)
    
    # Return updated components
    return (
        new_path,  # Update the state
        save_msg,  # Status message
        refresh_voice_list(new_path),  # Updated voice dropdown
        refresh_voice_choices(new_path),  # Updated TTS choices
        refresh_audiobook_voice_choices(new_path)  # Updated audiobook choices
    )

def parse_multi_voice_text(text):
    """
    Parse text with voice tags like [voice_name] and return segments with associated voices
    Automatically removes character names from spoken text when they match the voice tag
    Returns: [(voice_name, text_segment), ...]
    """
    import re
    
    # Split text by voice tags but keep the tags
    pattern = r'(\[([^\]]+)\])'
    parts = re.split(pattern, text)
    
    segments = []
    current_voice = None
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        if not part:
            i += 1
            continue
            
        # Check if this is a voice tag
        if part.startswith('[') and part.endswith(']'):
            # This is a voice tag
            current_voice = part[1:-1]  # Remove brackets
            i += 1
        else:
            # This is text content
            if part and current_voice:
                # Clean the text by removing character name if it matches the voice tag
                cleaned_text = clean_character_name_from_text(part, current_voice)
                # Only add non-empty segments after cleaning
                if cleaned_text.strip():
                    segments.append((current_voice, cleaned_text))
                else:
                    print(f"[DEBUG] Skipping empty segment after cleaning for voice '{current_voice}'")
            elif part:
                # Text without voice tag - use default
                segments.append((None, part))
            i += 1
    
    return segments

def clean_character_name_from_text(text, voice_name):
    """
    Remove character name from the beginning of text if it matches the voice name
    Handles various formats like 'P1', 'P1:', 'P1 -', etc.
    """
    text = text.strip()
    
    # If the entire text is just the voice name (with possible punctuation), return empty
    if text.lower().replace(':', '').replace('.', '').replace('-', '').strip() == voice_name.lower():
        print(f"[DEBUG] Text is just the voice name '{voice_name}', returning empty")
        return ""
    
    # Create variations of the voice name to check for
    voice_variations = [
        voice_name,                    # af_sarah
        voice_name.upper(),            # AF_SARAH  
        voice_name.lower(),            # af_sarah
        voice_name.capitalize(),       # Af_sarah
    ]
    
    # Also add variations without underscores for more flexible matching
    for voice_var in voice_variations[:]:
        if '_' in voice_var:
            voice_variations.append(voice_var.replace('_', ' '))  # af sarah
            voice_variations.append(voice_var.replace('_', ''))   # afsarah
    
    for voice_var in voice_variations:
        # Check for various patterns:
        # "af_sarah text..." -> "text..."
        # "af_sarah: text..." -> "text..."
        # "af_sarah - text..." -> "text..."
        # "af_sarah. text..." -> "text..."
        patterns = [
            rf'^{re.escape(voice_var)}\s+',      # "af_sarah "
            rf'^{re.escape(voice_var)}:\s*',     # "af_sarah:" or "af_sarah: "
            rf'^{re.escape(voice_var)}\.\s*',    # "af_sarah." or "af_sarah. "
            rf'^{re.escape(voice_var)}\s*-\s*',  # "af_sarah -" or "af_sarah-"
            rf'^{re.escape(voice_var)}\s*\|\s*', # "af_sarah |" or "af_sarah|"
            rf'^{re.escape(voice_var)}\s*\.\.\.', # "af_sarah..."
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                # Remove the matched pattern and return the remaining text
                cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                print(f"[DEBUG] Cleaned text for voice '{voice_name}': '{text[:50]}...' -> '{cleaned[:50] if cleaned else '(empty)'}'")
                return cleaned
    
    # If no character name pattern found, return original text
    return text

def chunk_multi_voice_segments(segments, max_words=50):
    """
    Take voice segments and chunk them appropriately while preserving voice assignments
    Returns: [(voice_name, chunk_text), ...]
    """
    final_chunks = []
    
    for voice_name, text in segments:
        # Chunk this segment using the same sentence boundary logic
        text_chunks = chunk_text_by_sentences(text, max_words)
        
        # Add voice assignment to each chunk
        for chunk in text_chunks:
            final_chunks.append((voice_name, chunk))
    
    return final_chunks

def validate_multi_voice_text(text_content, voice_library_path):
    """
    Validate multi-voice text and check if all referenced voices exist
    Returns: (is_valid, message, voice_counts)
    """
    if not text_content or not text_content.strip():
        return False, "❌ Text content is required", {}
    
    # Parse the text to find voice references
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return False, "❌ No valid voice segments found", {}
    
    # Count voice usage and check availability
    voice_counts = {}
    missing_voices = []
    available_voices = [p['name'] for p in get_voice_profiles(voice_library_path)]
    
    for voice_name, text_segment in segments:
        if voice_name is None:
            voice_name = "No Voice Tag"
        
        if voice_name not in voice_counts:
            voice_counts[voice_name] = 0
        voice_counts[voice_name] += len(text_segment.split())
        
        # Check if voice exists (skip None/default)
        if voice_name != "No Voice Tag" and voice_name not in available_voices:
            if voice_name not in missing_voices:
                missing_voices.append(voice_name)
    
    if missing_voices:
        return False, f"❌ Missing voices: {', '.join(missing_voices)}", voice_counts
    
    if "No Voice Tag" in voice_counts:
        return False, "❌ Found text without voice tags. All text must be assigned to a voice using [voice_name]", voice_counts
    
    return True, "✅ All voices found and text properly tagged", voice_counts

def validate_multi_audiobook_input(text_content, voice_library_path, project_name):
    """Validate inputs for multi-voice audiobook creation"""
    issues = []
    
    if not project_name or not project_name.strip():
        issues.append("📁 Project name is required")
    
    if text_content and len(text_content.strip()) < 10:
        issues.append("📏 Text is too short (minimum 10 characters)")
    
    # Validate voice parsing
    is_valid, voice_message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
    
    if not is_valid:
        issues.append(voice_message)
    
    if issues:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Please fix these issues:\n" + "\n".join(f"• {issue}" for issue in issues),
            "",
            gr.Audio(visible=False)
        )
    
    # Show voice breakdown
    voice_breakdown = "\n".join([f"🎭 {voice}: {words} words" for voice, words in voice_counts.items()])
    chunks = chunk_multi_voice_segments(parse_multi_voice_text(text_content))
    total_words = sum(voice_counts.values())
    
    return (
        gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"✅ Ready for multi-voice audiobook creation!\n📊 {total_words:,} total words → {len(chunks)} chunks\n📁 Project: {project_name.strip()}\n\n{voice_breakdown}",
        voice_breakdown,
        gr.Audio(visible=True)
    )

def create_multi_voice_audiobook(model, text_content, voice_library_path, project_name):
    """Create multi-voice audiobook from tagged text"""
    if not text_content or not project_name:
        return None, "❌ Missing required fields"
    
    try:
        # Parse and validate the text
        is_valid, message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
        if not is_valid:
            return None, f"❌ Text validation failed: {message}"
        
        # Get voice segments and chunk them
        segments = parse_multi_voice_text(text_content)
        chunks = chunk_multi_voice_segments(segments, max_words=50)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return None, "❌ No text chunks to process"
        
        # Initialize model if needed
        if model is None:
            model = ChatterboxTTS.from_pretrained(DEVICE)
        
        audio_chunks = []
        chunk_info = []  # For saving metadata
        
        for i, (voice_name, chunk_text) in enumerate(chunks, 1):
            # Get voice configuration
            voice_config = get_voice_config(voice_library_path, voice_name)
            if not voice_config:
                return None, f"❌ Could not load voice configuration for '{voice_name}'"
            
            if not voice_config['audio_file']:
                return None, f"❌ No audio file found for voice '{voice_config['display_name']}'"
            
            # Update status (this would be shown in real implementation)
            chunk_words = len(chunk_text.split())
            status_msg = f"🎵 Processing chunk {i}/{total_chunks}\n🎭 Voice: {voice_config['display_name']} ({voice_name})\n📝 Chunk {i}: {chunk_words} words\n📊 Progress: {i}/{total_chunks} chunks"
            
            # Generate audio for this chunk
            wav = model.generate(
                chunk_text,
                audio_prompt_path=voice_config['audio_file'],
                exaggeration=voice_config['exaggeration'],
                temperature=voice_config['temperature'],
                cfg_weight=voice_config['cfg_weight'],
            )
            
            audio_chunks.append(wav.squeeze(0).numpy())
            chunk_info.append({
                'chunk_num': i,
                'voice_name': voice_name,
                'character_name': voice_name,
                'voice_display': voice_config['display_name'],
                'text': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                'word_count': chunk_words
            })
        
        # Save all chunks with voice info in filenames
        saved_files, project_dir = save_audio_chunks(audio_chunks, model.sr, project_name)
        
        # Combine all audio for preview
        combined_audio = np.concatenate(audio_chunks)
        
        total_words = sum([info['word_count'] for info in chunk_info])
        duration_minutes = len(combined_audio) // model.sr // 60
        
        # Create assignment summary
        assignment_summary = "\n".join([f"🎭 [{char}] → {voice_counts[char]}" for char in voice_counts.keys()])
        
        success_msg = f"✅ Multi-voice audiobook created successfully!\n📊 {total_words:,} words in {total_chunks} chunks\n🎭 Characters: {len(voice_counts)}\n⏱️ Duration: ~{duration_minutes} minutes\n📁 Saved to: {project_dir}\n🎵 Files: {len(saved_files)} audio chunks\n\nVoice Assignments:\n{assignment_summary}"
        
        return (model.sr, combined_audio), success_msg
        
    except Exception as e:
        error_msg = f"❌ Error creating multi-voice audiobook: {str(e)}"
        return None, error_msg

def analyze_multi_voice_text(text_content, voice_library_path):
    """
    Analyze multi-voice text and return character breakdown with voice assignment interface
    """
    if not text_content or not text_content.strip():
        return "", {}, gr.Group(visible=False), "❌ No text to analyze"
    
    # Parse the text to find voice references
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return "", {}, gr.Group(visible=False), "❌ No voice tags found in text"
    
    # Count voice usage
    voice_counts = {}
    for voice_name, text_segment in segments:
        if voice_name is None:
            voice_name = "No Voice Tag"
        
        if voice_name not in voice_counts:
            voice_counts[voice_name] = 0
        voice_counts[voice_name] += len(text_segment.split())
    
    # Create voice breakdown display
    if "No Voice Tag" in voice_counts:
        breakdown_text = "❌ Found text without voice tags:\n"
        breakdown_text += f"• No Voice Tag: {voice_counts['No Voice Tag']} words\n"
        breakdown_text += "\nAll text must be assigned to a voice using [voice_name] tags!"
        return breakdown_text, voice_counts, gr.Group(visible=False), "❌ Text contains untagged content"
    
    breakdown_text = "✅ Voice tags found:\n"
    for voice, words in voice_counts.items():
        breakdown_text += f"🎭 [{voice}]: {words} words\n"
    
    return breakdown_text, voice_counts, gr.Group(visible=True), "✅ Analysis complete - assign voices below"

def create_assignment_interface_with_dropdowns(voice_counts, voice_library_path):
    """
    Create actual Gradio dropdown components for each character
    Returns the components and character names for proper handling
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return [], [], "<div class='voice-status'>❌ No valid characters found</div>"
    
    # Get available voices
    available_voices = get_voice_profiles(voice_library_path)
    
    if not available_voices:
        return [], [], "<div class='voice-status'>❌ No voices available in library. Create voices first!</div>"
    
    # Create voice choices for dropdowns
    voice_choices = [("Select a voice...", None)]
    for voice in available_voices:
        display_text = f"🎭 {voice['display_name']} ({voice['name']})"
        voice_choices.append((display_text, voice['name']))
    
    # Create components for each character
    dropdown_components = []
    character_names = []
    
    for character_name, word_count in voice_counts.items():
        if character_name != "No Voice Tag":
            dropdown = gr.Dropdown(
                choices=voice_choices,
                label=f"Voice for [{character_name}] ({word_count} words)",
                value=None,
                interactive=True,
                info=f"Select which voice to use for character '{character_name}'"
            )
            dropdown_components.append(dropdown)
            character_names.append(character_name)
    
    # Create info display
    info_html = f"<div class='voice-status'>✅ Found {len(character_names)} characters. Select voices for each character using the dropdowns below.</div>"
    
    return dropdown_components, character_names, info_html

def validate_dropdown_assignments(text_content, voice_library_path, project_name, voice_counts, character_names, *dropdown_values):
    """
    Validate voice assignments from dropdown values
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Invalid text or voice tags",
            {},
            gr.Audio(visible=False)
        )
    
    if not project_name or not project_name.strip():
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Project name is required",
            {},
            gr.Audio(visible=False)
        )
    
    if len(dropdown_values) != len(character_names):
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"❌ Assignment mismatch: {len(character_names)} characters, {len(dropdown_values)} dropdown values",
            {},
            gr.Audio(visible=False)
        )
    
    # Create voice assignments mapping from dropdown values
    voice_assignments = {}
    missing_assignments = []
    
    for i, character in enumerate(character_names):
        assigned_voice = dropdown_values[i] if i < len(dropdown_values) else None
        if not assigned_voice:
            missing_assignments.append(character)
        else:
            voice_assignments[character] = assigned_voice
    
    if missing_assignments:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"❌ Please assign voices for: {', '.join(missing_assignments)}",
            voice_assignments,
            gr.Audio(visible=False)
        )
    
    # All assignments valid
    total_words = sum(voice_counts.values())
    assignment_summary = "\n".join([f"🎭 [{char}] → {voice_assignments[char]}" for char in character_names])
    
    return (
        gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"✅ All characters assigned!\n📊 {total_words:,} words total\n📁 Project: {project_name.strip()}\n\nAssignments:\n{assignment_summary}",
        voice_assignments,
        gr.Audio(visible=True)
    )

def get_model_device_str(model_obj):
    """Safely get the device string ("cuda" or "cpu") from a model object."""
    if not model_obj or not hasattr(model_obj, 'device'):
        # print("⚠️ Model object is None or has no device attribute.")
        return None 
    
    device_attr = model_obj.device
    if isinstance(device_attr, torch.device):
        return device_attr.type
    elif isinstance(device_attr, str):
        if device_attr in ["cuda", "cpu"]:
            return device_attr
        else:
            print(f"⚠️ Unexpected string for model.device: {device_attr}")
            return None 
    else:
        print(f"⚠️ Unexpected type for model.device: {type(device_attr)}")
        return None

def _filter_problematic_short_chunks(chunks, voice_assignments):
    """Helper to filter out very short chunks that likely represent only character tags."""
    if not chunks:
        return []

    filtered_chunks = []
    # Extract just the keys from voice_assignments, which are the character tags like 'af_sarah', 'af_aoede'
    # Ensure keys are strings and lowercased for consistent matching.
    known_char_tags = [str(tag).lower().strip() for tag in voice_assignments.keys()]
    original_chunk_count = len(chunks)

    for chunk_idx, chunk_info in enumerate(chunks):
        # Handle tuple format: (voice_name, text)
        if isinstance(chunk_info, tuple) and len(chunk_info) == 2:
            voice_name, text = chunk_info
            if not isinstance(text, str):
                print(f"⚠️ Skipping chunk with non-string text at index {chunk_idx}: {chunk_info}")
                filtered_chunks.append(chunk_info)
                continue
                
            text_to_check = text.strip().lower()
            is_problematic_tag_chunk = False
            
            # Check if text is just the voice name or character tag (with possible punctuation)
            # This handles cases like "af_sarah", "af_sarah.", "af_sarah...", etc.
            cleaned_for_check = text_to_check.replace('_', '').replace('-', '').replace('.', '').replace(':', '').strip()
            
            # Check against known character tags
            for tag in known_char_tags:
                tag_cleaned = tag.replace('_', '').replace('-', '').strip()
                if cleaned_for_check == tag_cleaned:
                    is_problematic_tag_chunk = True
                    break
            
            # Also check if it's very short and matches a tag pattern
            if not is_problematic_tag_chunk and 1 <= len(text_to_check) <= 20:
                # More robust check for tag-like patterns
                core_text_segment = text_to_check
                # Strip common endings
                for ending in ["...", "..", ".", ":", "-", "_"]:
                    if core_text_segment.endswith(ending):
                        core_text_segment = core_text_segment[:-len(ending)]
                
                # Check if what remains is a known character tag
                if core_text_segment in known_char_tags:
                    is_problematic_tag_chunk = True
            
            if is_problematic_tag_chunk:
                print(f"⚠️ Filtering out suspected tag-only chunk {chunk_idx+1}/{original_chunk_count} for voice '{voice_name}': '{text}'")
            else:
                filtered_chunks.append(chunk_info)
        else:
            # Handle unexpected format
            print(f"⚠️ Unexpected chunk format at index {chunk_idx}: {chunk_info}")
            filtered_chunks.append(chunk_info)
            
    if len(filtered_chunks) < original_chunk_count:
        print(f"ℹ️ Filtered {original_chunk_count - len(filtered_chunks)} problematic short chunk(s) out of {original_chunk_count}.")
    
    return filtered_chunks

def create_multi_voice_audiobook_with_assignments(model, text_content, voice_library_path, project_name, voice_assignments):
    """Create multi-voice audiobook using the voice assignments mapping - Smart GPU/CPU hybrid"""
    print(f"\n[DEBUG] === create_multi_voice_audiobook_with_assignments CALLED ===")
    print(f"[DEBUG] Received project_name: {project_name}")
    print(f"[DEBUG] Received voice_assignments: {voice_assignments}")
    print(f"[DEBUG] Initial raw text_content (first 500 chars):\n{text_content[:500]}\n---------------------")

    if not text_content or not project_name or not voice_assignments:
        error_msg = "❌ Missing required fields or voice assignments. Ensure text is entered, project name is set, and voices are assigned after analyzing text."
        print(f"[DEBUG] Validation failed: {error_msg}")
        return None, None, error_msg, None # Ensure four values are returned

    try:
        # Parse the text and map voices
        segments = parse_multi_voice_text(text_content)
        mapped_segments = []
        for character_name, text_segment in segments:
            if character_name in voice_assignments:
                actual_voice = voice_assignments[character_name]
                mapped_segments.append((actual_voice, text_segment))
            else:
                return None, f"❌ No voice assignment found for character '{character_name}'"
        
        initial_max_words = 30 if DEVICE == "cuda" else 40 # Smaller initial chunks for GPU
        print(f"ℹ️ Using initial max_words for chunking: {initial_max_words}")
        chunks = chunk_multi_voice_segments(mapped_segments, max_words=initial_max_words)
        
        # Filter out problematic short chunks AFTER initial chunking
        chunks = _filter_problematic_short_chunks(chunks, voice_assignments)
        
        total_chunks = len(chunks)
        print(f"[DEBUG] After filtering, generated {total_chunks} chunks. First 3 chunks (if any):")
        for i, (voice_name, chunk_text_debug) in enumerate(chunks[:3]):
            print(f"  [DEBUG] Chunk {i+1} (voice: {voice_name}): '{chunk_text_debug[:100]}...' ")
        print(f"---------------------")

        if not chunks:
            return None, "❌ No text chunks to process"
        
        use_cpu = False
        processing_model = model 
        device_name = "GPU" if DEVICE == "cuda" else "CPU"
        
        # Ensure initial model is correct type
        current_model_device_str = get_model_device_str(processing_model)
        if DEVICE == "cuda" and current_model_device_str != 'cuda':
            print(f"Correcting initial model to GPU (was {current_model_device_str}).")
            if processing_model and hasattr(processing_model, 'to'): del processing_model
            torch.cuda.empty_cache()
            processing_model = ChatterboxTTS.from_pretrained(DEVICE)
        elif DEVICE == "cpu" and current_model_device_str != 'cpu':
            print(f"Correcting initial model to CPU (was {current_model_device_str}).")
            if processing_model and hasattr(processing_model, 'to'): del processing_model
            torch.cuda.empty_cache()
            processing_model = ChatterboxTTS.from_pretrained("cpu")

        audio_chunks = []
        chunk_info = []
        global_cuda_errors_count = 0
        max_global_cuda_errors = 2

        for i, (voice_name, chunk_text) in enumerate(chunks, 1):
            try:
                voice_config = get_voice_config(voice_library_path, voice_name)
                if not voice_config: return None, f"❌ Could not load voice config for '{voice_name}'"
                if not voice_config['audio_file']: return None, f"❌ No audio file for voice '{voice_config['display_name']}'"
                if not os.path.exists(voice_config['audio_file']): return None, f"❌ Audio file not found: {voice_config['audio_file']}"

                if "_-_" in voice_name or len(voice_name) > 50:
                    print(f"⚠️ Warning: Voice name '{voice_name}' may cause issues")
                
                # device_name here reflects the *intended* device for the outer loop (GPU if available, else CPU)
                # or the globally switched CPU mode.
                # current_attempt_device will be set more specifically within the attempt loop.

                generation_success = False
                wav = None

                for attempt in range(2): 
                    current_attempt_device = "" # To be set based on logic below

                    try:
                        model_actual_device_str = get_model_device_str(processing_model)

                        if use_cpu: # Global flag for CPU for the rest of the audiobook
                            if model_actual_device_str != 'cpu':
                                print(f"🔄 Chunk {i}: Global CPU mode. Ensuring CPU model (was {model_actual_device_str}).")
                                if processing_model and hasattr(processing_model, 'to'): del processing_model
                                torch.cuda.empty_cache()
                                processing_model = ChatterboxTTS.from_pretrained("cpu")
                            device_name = "CPU" # Update primary intended device
                            current_attempt_device = "CPU"
                        elif device_name == "GPU": # Intending GPU for this chunk (not globally on CPU yet)
                            if model_actual_device_str != 'cuda': # Model is not on GPU (or is None)
                                print(f"🔄 Chunk {i}, Attempt {attempt+1}: Reloading GPU model (was {model_actual_device_str}).")
                                if processing_model and hasattr(processing_model, 'to'): del processing_model
                                torch.cuda.empty_cache()
                                processing_model = ChatterboxTTS.from_pretrained(DEVICE) # DEVICE is "cuda"
                            current_attempt_device = "GPU"
                        else: # device_name is "CPU" from initial setup
                             if model_actual_device_str != 'cpu':
                                print(f"🔄 Chunk {i}, Attempt {attempt+1}: Ensuring CPU model (was {model_actual_device_str}).")
                                if processing_model and hasattr(processing_model, 'to'): del processing_model
                                torch.cuda.empty_cache()
                                processing_model = ChatterboxTTS.from_pretrained("cpu")
                             current_attempt_device = "CPU"
                        
                        print(f"🎙️ Chunk {i}, Attempt {attempt+1} on {current_attempt_device} for voice '{voice_name}'")
                        if current_attempt_device == "GPU":
                            torch.cuda.empty_cache()

                        wav = processing_model.generate(
                            chunk_text, audio_prompt_path=voice_config['audio_file'],
                            exaggeration=voice_config['exaggeration'], temperature=voice_config['temperature'],
                            cfg_weight=voice_config['cfg_weight'])
                        generation_success = True
                        print(f"✅ Chunk {i}, Attempt {attempt+1} SUCCEEDED on {current_attempt_device}")
                        break 
                        
                    except RuntimeError as gen_error:
                        error_str = str(gen_error).lower()
                        print(f"⚠️ Chunk {i}, Attempt {attempt+1} FAILED on {current_attempt_device}: {error_str[:250]}...") # Increased log length
                        if attempt == 0 and current_attempt_device == "GPU": # Log text on first GPU fail
                            print(f"Problematic text for chunk {i} (GPU Attempt 1 with {voice_name}): <<< {chunk_text[:300]}... >>>")
                        
                        torch.cuda.empty_cache()

                        is_cuda_related_error = "cuda" in error_str or "device-side assert" in error_str or "srcindex < srcselectdimsize" in error_str
                        
                        if is_cuda_related_error and current_attempt_device == "GPU":
                            global_cuda_errors_count += 1
                            if global_cuda_errors_count >= max_global_cuda_errors and not use_cpu:
                                print(f"🚫 Max global GPU errors ({global_cuda_errors_count}). Switching to CPU for subsequent chunks.")
                                use_cpu = True 
                                device_name = "CPU" # Reflect global switch for future chunks' device_name logic

                            if attempt == 0: 
                                print(f"🛠️ Chunk {i}: GPU Attempt 1 failed. Explicitly deleting and preparing to reload GPU model for Attempt 2.")
                                if processing_model and hasattr(processing_model, 'to'):
                                    try:
                                        # Attempt to move to CPU before deleting if it's a nn.Module
                                        if isinstance(processing_model, torch.nn.Module):
                                            processing_model.to('cpu')
                                    except Exception as move_err:
                                        print(f"  (Note: Error moving model to CPU before del: {move_err})")
                                    del processing_model
                                processing_model = None # Mark for reload
                                torch.cuda.synchronize() # Ensure all GPU operations are done
                                torch.cuda.empty_cache() # Clear cache thoroughly
                                continue 
                        
                        if attempt == 1:
                            print(f"❌ Chunk {i} failed after 2 attempts on intended device ({current_attempt_device}).")
                            break 
                
                if not generation_success:
                    print(f"📉 Chunk {i} failed initial attempts. Trying final CPU fallback.")
                    current_model_device_str = get_model_device_str(processing_model)
                    if current_model_device_str != 'cpu':
                        print(f"🔄 Chunk {i}: Switching to CPU for final attempt (was {current_model_device_str}). Syncing CUDA first...")
                        if torch.cuda.is_available(): # Synchronize before switching if CUDA was involved
                            torch.cuda.synchronize()
                        if processing_model and hasattr(processing_model, 'to'): del processing_model
                        torch.cuda.empty_cache() # Clear cache again after sync
                        processing_model = ChatterboxTTS.from_pretrained("cpu")
                    
                    try:
                        print(f"🎙️ Chunk {i}, Final attempt on CPU for voice '{voice_name}'")
                        wav = processing_model.generate(
                            chunk_text, audio_prompt_path=voice_config['audio_file'],
                            exaggeration=voice_config['exaggeration'], temperature=voice_config['temperature'],
                            cfg_weight=voice_config['cfg_weight'])
                        generation_success = True
                        print(f"✅ Chunk {i} SUCCEEDED on final CPU attempt.")
                        device_name = "CPU" # Model is now CPU
                    except Exception as cpu_final_error:
                        print(f"❌❌ Chunk {i} FAILED ALL ATTEMPTS, including final CPU. Voice: '{voice_name}'. Error: {str(cpu_final_error)}")
                        print(f"Problematic text for chunk {i}: <<< {chunk_text[:200]}... >>>")
                        return None, f"❌ Chunk {i} FAILED ALL ATTEMPTS (voice: {voice_name}), including final CPU: {str(cpu_final_error)}"

                if not generation_success:
                    return None, f"❌ Critical error: Chunk {i} status unclear post-attempts."

                # Store audio
                # Note: device_name reflects the device of the successfully used model for this chunk if successful
                # or the model state after potential switches.
                # If generation_success is true, wav is from processing_model.
                # If it used GPU and succeeded, wav is on GPU.
                # If it used CPU and succeeded, wav is on CPU.
                
                model_used_for_wav = get_model_device_str(processing_model)
                if model_used_for_wav == 'cuda' and not use_cpu : # if not globally on CPU
                     audio_chunks.append(wav.squeeze(0).cpu().numpy())
                else: # wav is already on CPU or should be treated as such
                     audio_chunks.append(wav.squeeze(0).numpy())

                character_for_voice = None
                for char_key, assigned_voice_val in voice_assignments.items(): # Renamed for clarity
                    if assigned_voice_val == voice_name:
                        character_for_voice = char_key
                        break
                chunk_info.append({
                    'chunk_num': i, 'voice_name': voice_name, 'character_name': character_for_voice or voice_name,
                    'voice_display': voice_config['display_name'], 
                    'text': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    'word_count': len(chunk_text.split())
                })
                
                del wav
                if get_model_device_str(processing_model) == 'cuda' and not use_cpu:
                    torch.cuda.empty_cache()
                
            except Exception as chunk_error_outer:
                return None, f"❌ Outer error processing chunk {i} (voice: {voice_name}): {str(chunk_error_outer)}"
        
        # Save all chunks with character and voice info
        saved_files = []
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        project_dir = os.path.join("audiobook_projects", safe_project_name)
        os.makedirs(project_dir, exist_ok=True)

        for idx, (audio_chunk_data, info_data) in enumerate(zip(audio_chunks, chunk_info), 1):
            character_name_file = info_data['character_name'].replace(' ', '_') if info_data['character_name'] else info_data['voice_name']
            filename = f"{safe_project_name}_{idx:03d}_{character_name_file}.wav"
            filepath = os.path.join(project_dir, filename)
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(processing_model.sr)
                audio_int16 = (audio_chunk_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            saved_files.append(filepath)

        metadata_file = os.path.join(project_dir, "project_info.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                'project_name': project_name, 'total_chunks': total_chunks,
                'final_processing_mode': 'CPU' if use_cpu else ('GPU' if DEVICE == 'cuda' else 'CPU'), # More accurate final mode
                'voice_assignments': voice_assignments, 'characters': list(voice_assignments.keys()),
                'chunks': chunk_info
            }, f, indent=2)
        
        # Save standardized project metadata for regeneration compatibility
        multi_voice_info = {}
        for voice_name in set(voice_assignments.values()):
            voice_config = get_voice_config(voice_library_path, voice_name)
            if voice_config:
                multi_voice_info[voice_name] = {
                    'voice_name': voice_name,
                    'display_name': voice_config['display_name'],
                    'audio_file': voice_config['audio_file'],
                    'exaggeration': voice_config['exaggeration'],
                    'cfg_weight': voice_config['cfg_weight'],
                    'temperature': voice_config['temperature']
                }
        
        # Convert chunks back to text list for metadata
        chunks_text = [chunk_text for _, chunk_text in chunks]
        
        save_project_metadata(
            project_dir=project_dir,
            project_name=project_name,
            text_content=text_content,
            voice_info=multi_voice_info,
            chunks=chunks_text,
            project_type="multi_voice"
        )
        
        combined_audio = np.concatenate(audio_chunks)
        total_words = sum([cinfo['word_count'] for cinfo in chunk_info])
        duration_minutes = len(combined_audio) // processing_model.sr // 60
        
        overall_mode_msg = 'CPU' if use_cpu else ('GPU' if DEVICE == 'cuda' else 'CPU')
        fallback_info = ""
        if global_cuda_errors_count > 0:
            fallback_info = f" with {global_cuda_errors_count} GPU error(s) leading to CPU use for some/all chunks" \
                           if use_cpu else f" with {global_cuda_errors_count} GPU error(s) handled (some chunks may have used CPU)"

        success_msg = (f"✅ Multi-voice audiobook created successfully! (Overall mode: {overall_mode_msg}{fallback_info})\n"
                       f"📊 {total_words:,} words in {total_chunks} chunks\n"
                       f"🎭 Characters: {len(voice_assignments)}\n"
                       f"⏱️ Duration: ~{duration_minutes} minutes\n"
                       f"📁 Saved to: {project_dir}\n"
                       f"🎵 Files: {len(saved_files)} audio chunks")
        
        assignment_summary = "\n".join([f"🎭 [{char}] → {assigned_voice}" for char, assigned_voice in voice_assignments.items()])
        success_msg += f"\n\nVoice Assignments:\n{assignment_summary}"
        
        return (processing_model.sr, combined_audio), success_msg
        
    except Exception as e:
        # Log the full traceback for debugging next time
        import traceback
        print(f"❌ CRITICAL ERROR in create_multi_voice_audiobook_with_assignments: {str(e)}")
        traceback.print_exc()
        error_msg = f"❌ Error creating multi-voice audiobook: {str(e)}"
        return None, error_msg

def handle_multi_voice_analysis(text_content, voice_library_path):
    """
    Analyze multi-voice text and populate character dropdowns
    Returns updated dropdown components
    """
    if not text_content or not text_content.strip():
        # Reset all dropdowns to hidden
        empty_dropdown = gr.Dropdown(choices=[("No character found", None)], visible=False, interactive=False)
        return (
            "<div class='voice-status'>❌ No text to analyze</div>",
            {},
            [],
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            gr.Button("🔍 Validate Voice Assignments", interactive=False),
            "❌ Add text first"
        )
    
    # Parse the text to find voice references
    breakdown_text, voice_counts, group_visibility, status = analyze_multi_voice_text(text_content, voice_library_path)
    
    if not voice_counts or "No Voice Tag" in voice_counts:
        # Reset all dropdowns to hidden
        empty_dropdown = gr.Dropdown(choices=[("No character found", None)], visible=False, interactive=False)
        return (
            breakdown_text,
            voice_counts,
            [],
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            gr.Button("🔍 Validate Voice Assignments", interactive=False),
            status
        )
    
    # Get available voices for dropdown choices
    available_voices = get_voice_profiles(voice_library_path)
    if not available_voices:
        empty_dropdown = gr.Dropdown(choices=[("No voices available", None)], visible=False, interactive=False)
        return (
            "<div class='voice-status'>❌ No voices available in library. Create voices first!</div>",
            voice_counts,
            [],
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            gr.Button("🔍 Validate Voice Assignments", interactive=False),
            "❌ No voices in library"
        )
    
    # Create voice choices for dropdowns
    voice_choices = [("Select a voice...", None)]
    for voice in available_voices:
        display_text = f"🎭 {voice['display_name']} ({voice['name']})"
        voice_choices.append((display_text, voice['name']))
    
    # Get character names (excluding "No Voice Tag")
    character_names = [name for name in voice_counts.keys() if name != "No Voice Tag"]
    
    # Create dropdown components for up to 6 characters
    dropdown_components = []
    for i in range(6):
        if i < len(character_names):
            character_name = character_names[i]
            word_count = voice_counts[character_name]
            dropdown = gr.Dropdown(
                choices=voice_choices,
                label=f"Voice for [{character_name}] ({word_count} words)",
                visible=True,
                interactive=True,
                info=f"Select which voice to use for character '{character_name}'"
            )
        else:
            dropdown = gr.Dropdown(
                choices=[("No character found", None)],
                label=f"Character {i+1}",
                visible=False,
                interactive=False
            )
        dropdown_components.append(dropdown)
    
    # Create summary message
    total_words = sum(voice_counts.values())
    summary_msg = f"✅ Found {len(character_names)} characters with {total_words:,} total words\n" + breakdown_text
    
    return (
        summary_msg,
        voice_counts,
        character_names,
        dropdown_components[0], dropdown_components[1], dropdown_components[2],
        dropdown_components[3], dropdown_components[4], dropdown_components[5],
        gr.Button("🔍 Validate Voice Assignments", interactive=True),
        "✅ Analysis complete - assign voices above"
    )

def validate_dropdown_voice_assignments(text_content, voice_library_path, project_name, voice_counts, character_names, 
                                       char1_voice, char2_voice, char3_voice, char4_voice, char5_voice, char6_voice):
    """
    Validate voice assignments from character dropdowns
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Invalid text or voice tags",
            {},
            gr.Audio(visible=False)
        )
    
    if not project_name or not project_name.strip():
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Project name is required",
            {},
            gr.Audio(visible=False)
        )
    
    if not character_names:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "❌ No characters found in text",
            {},
            gr.Audio(visible=False)
        )
    
    # Collect dropdown values
    dropdown_values = [char1_voice, char2_voice, char3_voice, char4_voice, char5_voice, char6_voice]
    
    # Create voice assignments mapping
    voice_assignments = {}
    missing_assignments = []
    
    for i, character_name in enumerate(character_names):
        if i < len(dropdown_values):
            assigned_voice = dropdown_values[i]
            if not assigned_voice:
                missing_assignments.append(character_name)
            else:
                voice_assignments[character_name] = assigned_voice
        else:
            missing_assignments.append(character_name)
    
    if missing_assignments:
        return (
            gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"❌ Please assign voices for: {', '.join(missing_assignments)}",
            voice_assignments,
            gr.Audio(visible=False)
        )
    
    # All assignments valid
    total_words = sum(voice_counts.values())
    assignment_summary = "\n".join([f"🎭 [{char}] → {voice_assignments[char]}" for char in character_names])
    
    return (
        gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"✅ All characters assigned!\n📊 {total_words:,} words total\n📁 Project: {project_name.strip()}\n\nAssignments:\n{assignment_summary}",
        voice_assignments,
        gr.Audio(visible=True)
    )

# Custom CSS for better styling - Fixed to preserve existing UI while targeting white backgrounds
css = """
.voice-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: #f9f9f9;
}

.tab-nav {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px;
    border-radius: 8px 8px 0 0;
}

.voice-library-header {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}

.voice-status {
    background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
    color: white;
    border-radius: 6px;
    padding: 12px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.config-status {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border-radius: 6px;
    padding: 10px;
    margin: 5px 0;
    font-size: 0.9em;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.audiobook-header {
    background: linear-gradient(90deg, #8b5cf6 0%, #06b6d4 100%);
    color: white;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}

.file-status {
    background: linear-gradient(135deg, #b45309 0%, #92400e 100%);
    color: white;
    border-radius: 6px;
    padding: 12px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.audiobook-status {
    background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%);
    color: white;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

/* Target specific instruction boxes that had white backgrounds */
.instruction-box {
    background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
    color: white !important;
    border-left: 4px solid #3b82f6 !important;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}
"""

# Load the saved voice library path
SAVED_VOICE_LIBRARY_PATH = load_config()

# Project metadata and regeneration functionality
def save_project_metadata(project_dir: str, project_name: str, text_content: str, 
                         voice_info: dict, chunks: list, project_type: str = "single_voice") -> None:
    """Save project metadata for regeneration purposes"""
    metadata = {
        "project_name": project_name,
        "project_type": project_type,  # "single_voice" or "multi_voice"
        "creation_date": str(time.time()),
        "text_content": text_content,
        "chunks": chunks,
        "voice_info": voice_info,
        "sample_rate": 24000,  # Default sample rate for ChatterboxTTS
        "version": "1.0"
    }
    
    metadata_file = os.path.join(project_dir, "project_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Warning: Could not save project metadata: {str(e)}")

def load_project_metadata(project_dir: str) -> dict:
    """Load project metadata from directory"""
    metadata_file = os.path.join(project_dir, "project_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load project metadata: {str(e)}")
    return None

def get_existing_projects(output_dir: str = "audiobook_projects") -> list:
    """Get list of existing projects with their metadata"""
    projects = []
    
    if not os.path.exists(output_dir):
        return projects
    
    for project_name in os.listdir(output_dir):
        project_path = os.path.join(output_dir, project_name)
        if os.path.isdir(project_path):
            # Get only the actual chunk files (not complete, backup, or temp files)
            all_audio_files = [f for f in os.listdir(project_path) if f.endswith('.wav')]
            
            # Filter to only count actual chunk files
            chunk_files = []
            for wav_file in all_audio_files:
                # Skip complete files, backup files, and temp files
                if (wav_file.endswith('_complete.wav') or 
                    '_backup_' in wav_file or 
                    'temp_regenerated_' in wav_file):
                    continue
                
                # Check if it matches the chunk pattern: projectname_XXX.wav or projectname_XXX_character.wav
                import re
                # Pattern for single voice: projectname_001.wav
                pattern1 = rf'^{re.escape(project_name)}_(\d{{3}})\.wav$'
                # Pattern for multi-voice: projectname_001_character.wav  
                pattern2 = rf'^{re.escape(project_name)}_(\d{{3}})_.+\.wav$'
                
                if re.match(pattern1, wav_file) or re.match(pattern2, wav_file):
                    chunk_files.append(wav_file)
            
            # Try to load metadata
            metadata = load_project_metadata(project_path)
            
            project_info = {
                "name": project_name,
                "path": project_path,
                "audio_files": chunk_files,  # Only actual chunk files
                "audio_count": len(chunk_files),
                "has_metadata": metadata is not None,
                "metadata": metadata
            }
            
            # If no metadata, try to infer some info
            if not metadata and chunk_files:
                project_info["creation_date"] = os.path.getctime(project_path)
                project_info["estimated_type"] = "unknown"
            
            projects.append(project_info)
    
    # Sort by creation date (newest first) - handle mixed types safely
    def get_sort_key(project):
        if project.get("metadata"):
            creation_date = project["metadata"].get("creation_date", 0)
            # Convert string timestamps to float for sorting
            if isinstance(creation_date, str):
                try:
                    return float(creation_date)
                except (ValueError, TypeError):
                    return 0.0
            return float(creation_date) if creation_date else 0.0
        else:
            return float(project.get("creation_date", 0))
    
    projects.sort(key=get_sort_key, reverse=True)
    
    return projects

def force_refresh_all_project_dropdowns():
    """Force refresh all project dropdowns to ensure new projects appear"""
    try:
        # Clear any potential caches and get fresh project list
        projects = get_existing_projects()
        choices = get_project_choices()
        # Return the same choices for all three dropdowns that might need updating
        return (
            gr.Dropdown(choices=choices, value=None),
            gr.Dropdown(choices=choices, value=None), 
            gr.Dropdown(choices=choices, value=None)
        )
    except Exception as e:
        print(f"Error refreshing project dropdowns: {str(e)}")
        error_choices = [("Error loading projects", None)]
        return (
            gr.Dropdown(choices=error_choices, value=None),
            gr.Dropdown(choices=error_choices, value=None),
            gr.Dropdown(choices=error_choices, value=None)
        )

def force_refresh_single_project_dropdown():
    """Force refresh a single project dropdown"""
    try:
        choices = get_project_choices()
        # Return a new dropdown with updated choices and no selected value
        return gr.Dropdown(choices=choices, value=None)
    except Exception as e:
        print(f"Error refreshing project dropdown: {str(e)}")
        error_choices = [("Error loading projects", None)]
        return gr.Dropdown(choices=error_choices, value=None)

def get_project_choices() -> list:
    """Get project choices for dropdown - always fresh data"""
    try:
        projects = get_existing_projects()  # This should always get fresh data
        if not projects:
            return [("No projects found", None)]
        
        choices = []
        for project in projects:
            metadata = project.get("metadata")
            if metadata:
                project_type = metadata.get('project_type', 'unknown')
                display_name = f"📁 {project['name']} ({project_type}) - {project['audio_count']} files"
            else:
                display_name = f"📁 {project['name']} (no metadata) - {project['audio_count']} files"
            choices.append((display_name, project['name']))
        
        return choices
        
    except Exception as e:
        print(f"Error getting project choices: {str(e)}")
        return [("Error loading projects", None)]

def load_project_for_regeneration(project_name: str) -> tuple:
    """Load a project for regeneration"""
    if not project_name:
        return "", "", "", None, "No project selected"
    
    projects = get_existing_projects()
    project = next((p for p in projects if p['name'] == project_name), None)
    
    if not project:
        return "", "", "", None, f"❌ Project '{project_name}' not found"
    
    metadata = project.get('metadata')
    if not metadata:
        # Legacy project without metadata
        audio_files = project['audio_files']
        if audio_files:
            # Load first audio file for waveform
            first_audio = os.path.join(project['path'], audio_files[0])
            return ("", 
                    "⚠️ Legacy project - no original text available", 
                    "⚠️ Voice information not available",
                    first_audio,
                    f"⚠️ Legacy project loaded. Found {len(audio_files)} audio files but no metadata.")
        else:
            return "", "", "", None, f"❌ No audio files found in project '{project_name}'"
    
    # Project with metadata
    text_content = metadata.get('text_content', '')
    voice_info = metadata.get('voice_info', {})
    
    # Format voice info display
    if metadata.get('project_type') == 'multi_voice':
        voice_display = "🎭 Multi-voice project:\n"
        for voice_name, info in voice_info.items():
            voice_display += f"  • {voice_name}: {info.get('display_name', voice_name)}\n"
    else:
        voice_display = f"🎤 Single voice: {voice_info.get('display_name', 'Unknown')}"
    
    # Load first audio file for waveform
    audio_files = project['audio_files']
    first_audio = os.path.join(project['path'], audio_files[0]) if audio_files else None
    
    creation_date = metadata.get('creation_date', '')
    if creation_date:
        try:
            import datetime
            date_obj = datetime.datetime.fromtimestamp(float(creation_date))
            date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = creation_date
    else:
        date_str = "Unknown"
    
    status_msg = f"✅ Project loaded successfully!\n📅 Created: {date_str}\n🎵 Audio files: {len(audio_files)}\n📝 Text length: {len(text_content)} characters"
    
    return text_content, voice_display, project_name, first_audio, status_msg

def regenerate_project_sample(model, project_name: str, voice_library_path: str, sample_text: str = None) -> tuple:
    """Regenerate a sample from an existing project"""
    if not project_name:
        return None, "❌ No project selected"
    
    projects = get_existing_projects()
    project = next((p for p in projects if p['name'] == project_name), None)
    
    if not project:
        return None, f"❌ Project '{project_name}' not found"
    
    metadata = project.get('metadata')
    if not metadata:
        return None, "❌ Cannot regenerate - project has no metadata (legacy project)"
    
    # Use provided sample text or take first chunk from original
    if sample_text and sample_text.strip():
        text_to_regenerate = sample_text.strip()
    else:
        chunks = metadata.get('chunks', [])
        if not chunks:
            original_text = metadata.get('text_content', '')
            if original_text:
                chunks = chunk_text_by_sentences(original_text, max_words=50)
                text_to_regenerate = chunks[0] if chunks else original_text[:200]
            else:
                return None, "❌ No text content available for regeneration"
        else:
            text_to_regenerate = chunks[0]
    
    # Get voice information
    voice_info = metadata.get('voice_info', {})
    project_type = metadata.get('project_type', 'single_voice')
    
    try:
        if project_type == 'single_voice':
            # Single voice regeneration
            voice_config = voice_info
            if not voice_config or not voice_config.get('audio_file'):
                return None, "❌ Voice configuration not available"
            
            # Generate audio
            wav = generate_with_retry(
                model,
                text_to_regenerate,
                voice_config['audio_file'],
                voice_config.get('exaggeration', 0.5),
                voice_config.get('temperature', 0.8),
                voice_config.get('cfg_weight', 0.5)
            )
            
            audio_output = wav.squeeze(0).cpu().numpy()
            status_msg = f"✅ Sample regenerated successfully!\n🎭 Voice: {voice_config.get('display_name', 'Unknown')}\n📝 Text: {text_to_regenerate[:100]}..."
            
            return (model.sr, audio_output), status_msg
            
        else:
            # Multi-voice regeneration - use first voice
            first_voice = list(voice_info.keys())[0] if voice_info else None
            if not first_voice:
                return None, "❌ No voice information available for multi-voice project"
            
            voice_config = voice_info[first_voice]
            if not voice_config or not voice_config.get('audio_file'):
                return None, f"❌ Voice configuration not available for '{first_voice}'"
            
            wav = generate_with_retry(
                model,
                text_to_regenerate,
                voice_config['audio_file'],
                voice_config.get('exaggeration', 0.5),
                voice_config.get('temperature', 0.8),
                voice_config.get('cfg_weight', 0.5)
            )
            
            audio_output = wav.squeeze(0).cpu().numpy()
            status_msg = f"✅ Sample regenerated successfully!\n🎭 Voice: {voice_config.get('display_name', first_voice)}\n📝 Text: {text_to_regenerate[:100]}..."
            
            return (model.sr, audio_output), status_msg
            
    except Exception as e:
        clear_gpu_memory()
        return None, f"❌ Error regenerating sample: {str(e)}"

def get_project_chunks(project_name: str) -> list:
    """Get all chunks from a project with audio files and text"""
    if not project_name:
        return []
    
    projects = get_existing_projects()
    project = next((p for p in projects if p['name'] == project_name), None)
    
    if not project:
        return []
    
    project_path = project['path']
    
    # Get only the actual chunk files (not complete, backup, or temp files)
    all_wav_files = [f for f in os.listdir(project_path) if f.endswith('.wav')]
    
    # Filter to only get numbered chunk files in format: projectname_001.wav, projectname_002.wav etc.
    chunk_files = []
    for wav_file in all_wav_files:
        # Skip complete files, backup files, and temp files
        if (wav_file.endswith('_complete.wav') or 
            '_backup_' in wav_file or 
            'temp_regenerated_' in wav_file):
            continue
        
        # Check if it matches the pattern: projectname_XXX.wav
        import re
        pattern = rf'^{re.escape(project_name)}_(\d{{3}})\.wav$'
        if re.match(pattern, wav_file):
            chunk_files.append(wav_file)
    
    # Sort by chunk number
    chunk_files.sort()
    
    chunks = []
    metadata = project.get('metadata')
    
    if metadata and metadata.get('chunks'):
        # Project with metadata - get original text chunks
        original_chunks = metadata.get('chunks', [])
        project_type = metadata.get('project_type', 'single_voice')
        voice_info = metadata.get('voice_info', {})
        
        # For multi-voice, also load the project_info.json to get voice assignments
        voice_assignments = {}
        if project_type == 'multi_voice':
            project_info_file = os.path.join(project_path, "project_info.json")
            if os.path.exists(project_info_file):
                try:
                    with open(project_info_file, 'r') as f:
                        project_info = json.load(f)
                        voice_assignments = project_info.get('voice_assignments', {})
                except Exception as e:
                    print(f"⚠️ Warning: Could not load voice assignments: {str(e)}")
        
        for i, audio_file in enumerate(chunk_files):
            chunk_info = {
                'chunk_num': i + 1,
                'audio_file': os.path.join(project_path, audio_file),
                'audio_filename': audio_file,
                'text': original_chunks[i] if i < len(original_chunks) else "Text not available",
                'has_metadata': True,
                'project_type': project_type,
                'voice_info': voice_info
            }
            
            # For multi-voice, try to extract character and find assigned voice
            if project_type == 'multi_voice':
                # Filename format: project_001_character.wav
                parts = audio_file.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    character_name = '_'.join(parts[2:])  # Everything after project_XXX_
                    chunk_info['character'] = character_name
                    
                    # Look up the actual voice assigned to this character
                    assigned_voice = voice_assignments.get(character_name, character_name)
                    chunk_info['assigned_voice'] = assigned_voice
                    
                    # Get the voice config for the assigned voice
                    chunk_info['voice_config'] = voice_info.get(assigned_voice, {})
                    
                else:
                    chunk_info['character'] = 'unknown'
                    chunk_info['assigned_voice'] = 'unknown'
                    chunk_info['voice_config'] = {}
            
            chunks.append(chunk_info)
    
    else:
        # Legacy project without metadata
        for i, audio_file in enumerate(chunk_files):
            chunk_info = {
                'chunk_num': i + 1,
                'audio_file': os.path.join(project_path, audio_file),
                'audio_filename': audio_file,
                'text': "Legacy project - original text not available",
                'has_metadata': False,
                'project_type': 'unknown',
                'voice_info': {}
            }
            chunks.append(chunk_info)
    
    return chunks

def regenerate_single_chunk(model, project_name: str, chunk_num: int, voice_library_path: str, custom_text: str = None) -> tuple:
    """Regenerate a single chunk from a project"""
    chunks = get_project_chunks(project_name)
    
    if not chunks or chunk_num < 1 or chunk_num > len(chunks):
        return None, f"❌ Invalid chunk number {chunk_num}"
    
    chunk = chunks[chunk_num - 1]  # Convert to 0-based index
    
    if not chunk['has_metadata']:
        return None, "❌ Cannot regenerate - legacy project has no voice metadata"
    
    # Use custom text or original text
    text_to_regenerate = custom_text.strip() if custom_text and custom_text.strip() else chunk['text']
    
    if not text_to_regenerate:
        return None, "❌ No text available for regeneration"
    
    try:
        project_type = chunk['project_type']
        
        if project_type == 'single_voice':
            # Single voice project
            voice_config = chunk['voice_info']
            if not voice_config or not voice_config.get('audio_file'):
                return None, "❌ Voice configuration not available"
            
            wav = generate_with_retry(
                model,
                text_to_regenerate,
                voice_config['audio_file'],
                voice_config.get('exaggeration', 0.5),
                voice_config.get('temperature', 0.8),
                voice_config.get('cfg_weight', 0.5)
            )
            
            voice_display = voice_config.get('display_name', 'Unknown')
            
        elif project_type == 'multi_voice':
            # Multi-voice project - use the voice config from the chunk
            voice_config = chunk.get('voice_config', {})
            character_name = chunk.get('character', 'unknown')
            assigned_voice = chunk.get('assigned_voice', 'unknown')
            
            if not voice_config:
                return None, f"❌ Voice configuration not found for character '{character_name}' (assigned voice: '{assigned_voice}')"
            
            if not voice_config.get('audio_file'):
                return None, f"❌ Audio file not found for character '{character_name}' (assigned voice: '{assigned_voice}')"
            
            # Check if audio file actually exists
            audio_file_path = voice_config.get('audio_file')
            if not os.path.exists(audio_file_path):
                return None, f"❌ Audio file does not exist: {audio_file_path}"
            
            wav = generate_with_retry(
                model,
                text_to_regenerate,
                voice_config['audio_file'],
                voice_config.get('exaggeration', 0.5),
                voice_config.get('temperature', 0.8),
                voice_config.get('cfg_weight', 0.5)
            )
            
            voice_display = f"{voice_config.get('display_name', assigned_voice)} (Character: {character_name})"
            
        else:
            return None, f"❌ Unknown project type: {project_type}"
        
        # Save regenerated audio to a temporary file
        audio_output = wav.squeeze(0).cpu().numpy()
        
        # Create temporary file path
        project_dir = os.path.dirname(chunk['audio_file'])
        temp_filename = f"temp_regenerated_chunk_{chunk_num}_{int(time.time())}.wav"
        temp_file_path = os.path.join(project_dir, temp_filename)
        
        # Save as WAV file
        with wave.open(temp_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(model.sr)
            # Convert float32 to int16
            audio_int16 = (audio_output * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        status_msg = f"✅ Chunk {chunk_num} regenerated successfully!\n🎭 Voice: {voice_display}\n📝 Text: {text_to_regenerate[:100]}{'...' if len(text_to_regenerate) > 100 else ''}\n💾 Temp file: {temp_filename}"
        
        # Return the temp file path instead of the audio tuple
        return temp_file_path, status_msg
        
    except Exception as e:
        clear_gpu_memory()
        return None, f"❌ Error regenerating chunk {chunk_num}: {str(e)}"

def load_project_chunks_for_interface(project_name: str) -> tuple:
    """Load project chunks and return data for interface components"""
    if not project_name:
        # Hide all chunk interfaces
        empty_returns = []
        for i in range(50):
            empty_returns.extend([
                gr.Group(visible=False),  # group
                None,  # audio
                "",  # text
                "<div class='voice-status'>No chunk loaded</div>",  # voice_info
                gr.Button(f"🎵 Regenerate Chunk {i+1}", interactive=False),  # button
                gr.Audio(visible=False),  # regenerated_audio
                "<div class='voice-status'>No chunk</div>"  # status
            ])
        
        return (
            "<div class='voice-status'>📝 Select a project first</div>",  # project_info_summary
            [],  # current_project_chunks
            project_name,  # current_project_name
            "<div class='audiobook-status'>📁 No project loaded</div>",  # project_status
            *empty_returns
        )
    
    chunks = get_project_chunks(project_name)
    
    if not chunks:
        # Hide all chunk interfaces
        empty_returns = []
        for i in range(50):
            empty_returns.extend([
                gr.Group(visible=False),
                None,
                "",
                "<div class='voice-status'>No chunk found</div>",
                gr.Button(f"🎵 Regenerate Chunk {i+1}", interactive=False),
                gr.Audio(visible=False),
                "<div class='voice-status'>No chunk</div>"
            ])
        
        return (
            f"<div class='voice-status'>❌ No chunks found in project '{project_name}'</div>",
            [],
            project_name,
            f"❌ No audio files found in project '{project_name}'",
            *empty_returns
        )
    
    # Create project summary
    project_info = f"""
    <div class='audiobook-status'>
        📁 <strong>Project:</strong> {project_name}<br/>
        🎵 <strong>Total Chunks:</strong> {len(chunks)}<br/>
        📝 <strong>Type:</strong> {chunks[0]['project_type'].replace('_', ' ').title()}<br/>
        ✅ <strong>Metadata:</strong> {'Available' if chunks[0]['has_metadata'] else 'Legacy Project'}
    </div>
    """
    
    status_msg = f"✅ Loaded {len(chunks)} chunks from project '{project_name}'"
    
    # Prepare interface updates
    interface_updates = []
    
    for i in range(50):
        if i < len(chunks):
            chunk = chunks[i]
            
            # Voice info display
            if chunk['project_type'] == 'multi_voice':
                character_name = chunk.get('character', 'unknown')
                assigned_voice = chunk.get('assigned_voice', 'unknown')
                voice_config = chunk.get('voice_config', {})
                voice_display_name = voice_config.get('display_name', assigned_voice)
                
                voice_info_html = f"<div class='voice-status'>🎭 Character: {character_name}<br/>🎤 Voice: {voice_display_name}</div>"
            elif chunk['project_type'] == 'single_voice':
                voice_name = chunk['voice_info'].get('display_name', 'Unknown') if chunk.get('voice_info') else 'Unknown'
                voice_info_html = f"<div class='voice-status'>🎤 Voice: {voice_name}</div>"
            else:
                voice_info_html = "<div class='voice-status'>⚠️ Legacy project - limited info</div>"
            
            # Status message
            chunk_status = f"<div class='voice-status'>📄 Chunk {chunk['chunk_num']} ready to regenerate</div>"
            
            interface_updates.extend([
                gr.Group(visible=True),  # group
                chunk['audio_file'],  # audio
                chunk['text'],  # text
                voice_info_html,  # voice_info
                gr.Button(f"🎵 Regenerate Chunk {chunk['chunk_num']}", interactive=chunk['has_metadata']),  # button
                gr.Audio(visible=False),  # regenerated_audio
                chunk_status  # status
            ])
        else:
            # Hide unused interfaces
            interface_updates.extend([
                gr.Group(visible=False),
                None,
                "",
                "<div class='voice-status'>No chunk</div>",
                gr.Button(f"🎵 Regenerate Chunk {i+1}", interactive=False),
                gr.Audio(visible=False),
                "<div class='voice-status'>No chunk</div>"
            ])
    
    return (
        project_info,  # project_info_summary
        chunks,  # current_project_chunks
        project_name,  # current_project_name
        status_msg,  # project_status
        gr.Button("📥 Download Full Project Audio", variant="primary", size="lg", interactive=bool(chunks)),  # download_project_btn
        f"<div class='voice-status'>✅ Ready to download complete project audio</div>" if chunks else "<div class='voice-status'>📁 Load a project first to enable download</div>",  # download_status
        *interface_updates
    )

def combine_project_audio_chunks(project_name: str, output_format: str = "wav") -> tuple:
    """Combine all audio chunks from a project into a single downloadable file"""
    if not project_name:
        return None, "❌ No project selected"
    
    chunks = get_project_chunks(project_name)
    
    if not chunks:
        return None, f"❌ No audio chunks found in project '{project_name}'"
    
    try:
        combined_audio = []
        sample_rate = 24000  # Default sample rate
        
        # Sort chunks by chunk number to ensure correct order
        chunks_sorted = sorted(chunks, key=lambda x: x['chunk_num'])
        
        # Load and combine all audio files in order
        for chunk in chunks_sorted:
            audio_file = chunk['audio_file']
            filename = os.path.basename(audio_file)
            
            # Double-check: Skip complete files, backup files, temp files
            if (filename.endswith('_complete.wav') or 
                '_backup_' in filename or 
                'temp_regenerated_' in filename):
                print(f"⚠️ Skipping non-chunk file: {filename}")
                continue
            
            # Only include actual numbered chunk files
            import re
            pattern = rf'^{re.escape(project_name)}_(\d{{3}})\.wav$'
            if not re.match(pattern, filename):
                print(f"⚠️ Skipping non-standard file: {filename}")
                continue
            
            if os.path.exists(audio_file):
                # Read WAV file
                try:
                    with wave.open(audio_file, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                        # Convert bytes to numpy array
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                        combined_audio.append(audio_data)
                        print(f"✅ Added chunk {chunk['chunk_num']}: {filename} ({len(audio_data)} samples)")
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {str(e)}")
            else:
                print(f"⚠️ Warning: Audio file not found: {audio_file}")
        
        if not combined_audio:
            return None, f"❌ No valid audio files found in project '{project_name}'"
        
        # Concatenate all audio
        full_audio = np.concatenate(combined_audio)
        
        # Create output filename
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
        output_filename = f"{safe_project_name}_complete.{output_format}"
        output_path = os.path.join("audiobook_projects", project_name, output_filename)
        
        # Remove existing complete file to avoid confusion
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"🗑️ Removed existing complete file: {output_filename}")
            except Exception as e:
                print(f"⚠️ Could not remove existing complete file: {str(e)}")
        
        # Save as WAV (Gradio will handle the download)
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setnchannels(1)  # Mono
            output_wav.setsampwidth(2)  # 16-bit
            output_wav.setframerate(sample_rate)
            # Convert back to int16
            audio_int16 = (full_audio * 32767).astype(np.int16)
            output_wav.writeframes(audio_int16.tobytes())
        
        # Calculate duration
        duration_seconds = len(full_audio) / sample_rate
        duration_minutes = int(duration_seconds // 60)
        duration_secs = int(duration_seconds % 60)
        
        success_msg = f"✅ Combined {len(combined_audio)} chunks successfully!\n🎵 Total duration: {duration_minutes}:{duration_secs:02d}\n📁 File: {output_filename}\n🔄 Fresh combination of current chunk files"
        
        return output_path, success_msg
        
    except Exception as e:
        return None, f"❌ Error combining audio: {str(e)}"

def load_previous_project_audio(project_name: str) -> tuple:
    """Load a previous project's combined audio for download in creation tabs"""
    if not project_name:
        return None, None, "📁 Select a project to load its audio"
    
    # Check if combined file already exists
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
    combined_file = os.path.join("audiobook_projects", project_name, f"{safe_project_name}_complete.wav")
    
    if os.path.exists(combined_file):
        # File already exists, load it
        return combined_file, combined_file, f"✅ Loaded existing combined audio for '{project_name}'"
    else:
        # Create combined file
        audio_path, status = combine_project_audio_chunks(project_name)
        return audio_path, audio_path, status

def save_trimmed_audio(audio_data, original_file_path: str, chunk_num: int) -> tuple:
    """Save trimmed audio data to replace the original file"""
    if not audio_data or not original_file_path:
        return "❌ No audio data to save", None
    
    print(f"[DEBUG] save_trimmed_audio called for chunk {chunk_num}")
    print(f"[DEBUG] audio_data type: {type(audio_data)}")
    print(f"[DEBUG] original_file_path: {original_file_path}")
    
    try:
        # Get project directory and create backup
        project_dir = os.path.dirname(original_file_path)
        backup_file = original_file_path.replace('.wav', f'_backup_original_{int(time.time())}.wav')
        
        # Backup original file
        if os.path.exists(original_file_path):
            shutil.copy2(original_file_path, backup_file)
            print(f"[DEBUG] Created backup: {os.path.basename(backup_file)}")
        
        # Handle different types of audio data from Gradio
        audio_saved = False
        
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            # Tuple format: (sample_rate, audio_array)
            sample_rate, audio_array = audio_data
            print(f"[DEBUG] Tuple format - sample_rate: {sample_rate}, audio_array shape: {getattr(audio_array, 'shape', 'unknown')}")
            
            # Ensure audio_array is numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            # Handle multi-dimensional arrays
            if len(audio_array.shape) > 1:
                # If stereo, take first channel
                audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()
            
            # Save trimmed audio as WAV file
            with wave.open(original_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to int16 if needed
                if audio_array.dtype != np.int16:
                    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                        # Ensure values are in range [-1, 1] before converting
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_array.astype(np.int16)
                else:
                    audio_int16 = audio_array
                
                wav_file.writeframes(audio_int16.tobytes())
            
            audio_saved = True
            print(f"[DEBUG] Saved audio from tuple format: {len(audio_int16)} samples")
            
        elif isinstance(audio_data, str):
            # File path - copy the trimmed file over
            print(f"[DEBUG] String format (file path): {audio_data}")
            if os.path.exists(audio_data):
                shutil.copy2(audio_data, original_file_path)
                audio_saved = True
                print(f"[DEBUG] Copied file from: {audio_data}")
            else:
                print(f"[DEBUG] File not found: {audio_data}")
                return f"❌ Trimmed audio file not found: {audio_data}", None
                
        elif hasattr(audio_data, 'name'):  # Gradio file object
            # Handle Gradio uploaded file
            print(f"[DEBUG] Gradio file object: {audio_data.name}")
            if os.path.exists(audio_data.name):
                shutil.copy2(audio_data.name, original_file_path)
                audio_saved = True
                print(f"[DEBUG] Copied from Gradio file: {audio_data.name}")
            else:
                return f"❌ Gradio file not found: {audio_data.name}", None
                
        else:
            print(f"[DEBUG] Unexpected audio data format: {type(audio_data)}")
            # Try to handle as raw audio data
            try:
                if hasattr(audio_data, '__iter__'):
                    audio_array = np.array(audio_data)
                    sample_rate = 24000  # Default sample rate
                    
                    with wave.open(original_file_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        
                        if audio_array.dtype != np.int16:
                            if np.max(np.abs(audio_array)) <= 1.0:
                                audio_int16 = (audio_array * 32767).astype(np.int16)
                            else:
                                audio_int16 = audio_array.astype(np.int16)
                        else:
                            audio_int16 = audio_array
                        
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    audio_saved = True
                    print(f"[DEBUG] Saved as raw audio data: {len(audio_int16)} samples")
                else:
                    return f"❌ Cannot process audio data type: {type(audio_data)}", None
            except Exception as e:
                print(f"[DEBUG] Failed to process as raw audio: {str(e)}")
                return f"❌ Cannot process audio data: {str(e)}", None
        
        if audio_saved:
            status_msg = f"✅ Chunk {chunk_num} trimmed and saved!\n💾 Original backed up as: {os.path.basename(backup_file)}\n🎵 Audio file updated successfully"
            print(f"[DEBUG] Successfully saved trimmed audio for chunk {chunk_num}")
            return status_msg, original_file_path
        else:
            return f"❌ Failed to save trimmed audio for chunk {chunk_num}", None
            
    except Exception as e:
        print(f"[DEBUG] Exception in save_trimmed_audio: {str(e)}")
        return f"❌ Error saving trimmed audio for chunk {chunk_num}: {str(e)}", None

def accept_regenerated_chunk(project_name: str, chunk_num: int, regenerated_audio_path: str) -> tuple:
    """Accept the regenerated chunk by replacing the original audio file and deleting the temp file"""
    if not project_name or not regenerated_audio_path:
        return "❌ No regenerated audio to accept", None
    
    try:
        chunks = get_project_chunks(project_name)
        if chunk_num < 1 or chunk_num > len(chunks):
            return f"❌ Invalid chunk number {chunk_num}", None
        
        chunk = chunks[chunk_num - 1]
        original_audio_file = chunk['audio_file']
        
        # Check if temp file exists
        if not os.path.exists(regenerated_audio_path):
            return f"❌ Regenerated audio file not found: {regenerated_audio_path}", None
        
        # Backup original file (optional, with timestamp)
        backup_file = original_audio_file.replace('.wav', f'_backup_{int(time.time())}.wav')
        if os.path.exists(original_audio_file):
            shutil.copy2(original_audio_file, backup_file)
        
        # Replace original with regenerated
        shutil.move(regenerated_audio_path, original_audio_file)
        
        # Clean up any other temp files for this chunk (in case there are multiple)
        project_dir = os.path.dirname(original_audio_file)
        temp_files = []
        try:
            for file in os.listdir(project_dir):
                if file.startswith(f"temp_regenerated_chunk_{chunk_num}_") and file.endswith('.wav'):
                    temp_path = os.path.join(project_dir, file)
                    try:
                        os.remove(temp_path)
                        temp_files.append(file)
                        print(f"🗑️ Cleaned up temp file: {file}")
                    except:
                        pass  # Ignore errors when cleaning up
        except Exception as e:
            print(f"⚠️ Warning during temp file cleanup: {str(e)}")
        
        status_msg = f"✅ Chunk {chunk_num} regeneration accepted!\n💾 Original backed up as: {os.path.basename(backup_file)}\n🗑️ Cleaned up {len(temp_files)} temporary file(s)"
        
        # Return both status message and the path to the NEW audio file (for interface update)
        return status_msg, original_audio_file
        
    except Exception as e:
        return f"❌ Error accepting chunk {chunk_num}: {str(e)}", None

def decline_regenerated_chunk(chunk_num: int, regenerated_audio_path: str = None) -> tuple:
    """Decline the regenerated chunk and clean up the temporary file"""
    
    # Handle the case where regenerated_audio_path might be a tuple (from Gradio Audio component)
    # or a string (file path)
    actual_file_path = None
    
    if regenerated_audio_path:
        if isinstance(regenerated_audio_path, tuple):
            # Gradio Audio component returns (sample_rate, audio_data) or just audio data
            # In our case, we should have the file path stored differently
            # For now, we can't get the file path from the tuple, so skip cleanup
            print(f"⚠️ Warning: Received tuple instead of file path for chunk {chunk_num} decline")
            actual_file_path = None
        elif isinstance(regenerated_audio_path, str):
            # This is the expected case - a file path string
            actual_file_path = regenerated_audio_path
        else:
            print(f"⚠️ Warning: Unexpected type for regenerated_audio_path: {type(regenerated_audio_path)}")
            actual_file_path = None
    
    # Clean up temporary file if we have a valid file path
    if actual_file_path and os.path.exists(actual_file_path):
        try:
            os.remove(actual_file_path)
            print(f"🗑️ Cleaned up declined regeneration: {os.path.basename(actual_file_path)}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp file: {str(e)}")
    
    return (
        gr.Audio(visible=False),  # Hide regenerated audio
        gr.Row(visible=False),    # Hide accept/decline buttons
        f"❌ Chunk {chunk_num} regeneration declined. Keeping original audio."
    )

def force_complete_project_refresh():
    """Force a complete refresh of project data, clearing any potential caches"""
    try:
        # Force reload of projects from filesystem
        import importlib
        import sys
        
        # Clear any module-level caches
        if hasattr(sys.modules[__name__], '_project_cache'):
            delattr(sys.modules[__name__], '_project_cache')
        
        # Get fresh project list
        projects = get_existing_projects()
        choices = get_project_choices()
        
        print(f"🔄 Complete refresh: Found {len(projects)} projects")
        for project in projects[:5]:  # Show first 5 projects
            print(f"  - {project['name']} ({project.get('audio_count', 0)} files)")
        
        return gr.Dropdown(choices=choices, value=None)
        
    except Exception as e:
        print(f"Error in complete refresh: {str(e)}")
        error_choices = [("Error loading projects", None)]
        return gr.Dropdown(choices=error_choices, value=None)

def cleanup_project_temp_files(project_name: str) -> str:
    """Clean up any temporary files in a project directory"""
    if not project_name:
        return "❌ No project name provided"
    
    try:
        project_dir = os.path.join("audiobook_projects", project_name)
        if not os.path.exists(project_dir):
            return f"❌ Project directory not found: {project_dir}"
        
        temp_files_removed = 0
        temp_patterns = ['temp_regenerated_', '_backup_original_']
        
        for file in os.listdir(project_dir):
            if any(pattern in file for pattern in temp_patterns) and file.endswith('.wav'):
                file_path = os.path.join(project_dir, file)
                try:
                    os.remove(file_path)
                    temp_files_removed += 1
                    print(f"🗑️ Removed temp file: {file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {file}: {str(e)}")
        
        if temp_files_removed > 0:
            return f"✅ Cleaned up {temp_files_removed} temporary file(s) from project '{project_name}'"
        else:
            return f"✅ No temporary files found in project '{project_name}'"
            
    except Exception as e:
        return f"❌ Error cleaning up temp files: {str(e)}"

def handle_audio_trimming(audio_data) -> tuple:
    """Handle audio trimming from Gradio audio component
    
    When users select a portion of audio in Gradio's waveform, we need to extract 
    that specific segment. This function attempts to work with Gradio's trimming data.
    """
    if not audio_data:
        return None, "❌ No audio data provided"
    
    print(f"[DEBUG] handle_audio_trimming called with data type: {type(audio_data)}")
    
    try:
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            # Standard format: (sample_rate, audio_array)
            sample_rate, audio_array = audio_data
            
            # Check if this is the full audio or a trimmed segment
            if hasattr(audio_array, 'shape'):
                print(f"[DEBUG] Audio shape: {audio_array.shape}, sample_rate: {sample_rate}")
                # For now, return the audio as-is since Gradio trimming is complex
                return audio_data, f"✅ Audio loaded - {len(audio_array)} samples at {sample_rate}Hz"
            else:
                return None, "❌ Invalid audio array format"
        else:
            return None, "❌ Invalid audio data format"
            
    except Exception as e:
        print(f"[DEBUG] Error in handle_audio_trimming: {str(e)}")
        return None, f"❌ Error processing audio: {str(e)}"

def extract_audio_segment(audio_data, start_time: float = None, end_time: float = None) -> tuple:
    """Extract a specific time segment from audio data
    
    Args:
        audio_data: Tuple of (sample_rate, audio_array)
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)
    """
    if not audio_data or not isinstance(audio_data, tuple) or len(audio_data) != 2:
        return None, "❌ Invalid audio data"
    
    try:
        sample_rate, audio_array = audio_data
        
        if not hasattr(audio_array, 'shape'):
            return None, "❌ Invalid audio array"
        
        # Handle multi-dimensional arrays
        if len(audio_array.shape) > 1:
            # Take first channel if stereo
            audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()
        
        total_samples = len(audio_array)
        total_duration = total_samples / sample_rate
        
        # Calculate sample indices
        start_sample = 0 if start_time is None else int(start_time * sample_rate)
        end_sample = total_samples if end_time is None else int(end_time * sample_rate)
        
        # Ensure valid bounds
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        
        # Extract segment
        trimmed_audio = audio_array[start_sample:end_sample]
        
        trimmed_duration = len(trimmed_audio) / sample_rate
        
        status_msg = f"✅ Extracted segment: {trimmed_duration:.2f}s (from {start_time or 0:.2f}s to {end_time or total_duration:.2f}s)"
        
        return (sample_rate, trimmed_audio), status_msg
        
    except Exception as e:
        return None, f"❌ Error extracting segment: {str(e)}"

def save_visual_trim_to_file(audio_data, original_file_path: str, chunk_num: int) -> tuple:
    """Save visually trimmed audio from Gradio audio component to file
    
    This attempts to work with whatever audio data Gradio provides from the visual trimming.
    If it's the full audio, it saves the full audio. If it's trimmed, it saves the trimmed portion.
    """
    if not audio_data or not original_file_path:
        return "❌ No audio data to save", None
    
    print(f"[DEBUG] save_visual_trim_to_file called for chunk {chunk_num}")
    print(f"[DEBUG] audio_data type: {type(audio_data)}")
    
    try:
        # Get project directory and create backup
        project_dir = os.path.dirname(original_file_path)
        backup_file = original_file_path.replace('.wav', f'_backup_visual_trim_{int(time.time())}.wav')
        
        # Backup original file
        if os.path.exists(original_file_path):
            shutil.copy2(original_file_path, backup_file)
            print(f"[DEBUG] Created backup: {os.path.basename(backup_file)}")
        
        # Handle Gradio audio data - it should be in (sample_rate, audio_array) format
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            print(f"[DEBUG] Audio format - sample_rate: {sample_rate}, array shape: {getattr(audio_array, 'shape', 'unknown')}")
            
            # Ensure audio_array is numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            # Handle multi-dimensional arrays
            if len(audio_array.shape) > 1:
                # If stereo, take first channel
                audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()
            
            # Save the audio as WAV file (whatever Gradio gave us - trimmed or full)
            with wave.open(original_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to int16 if needed
                if audio_array.dtype != np.int16:
                    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                        # Ensure values are in range [-1, 1] before converting
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_array.astype(np.int16)
                else:
                    audio_int16 = audio_array
                
                wav_file.writeframes(audio_int16.tobytes())
            
            duration_seconds = len(audio_int16) / sample_rate
            status_msg = f"✅ Chunk {chunk_num} audio saved! Duration: {duration_seconds:.2f}s\n💾 Original backed up as: {os.path.basename(backup_file)}\n🎵 Visual trimming applied (if any)"
            print(f"[DEBUG] Successfully saved audio for chunk {chunk_num}: {len(audio_int16)} samples")
            return status_msg, original_file_path
        else:
            return f"❌ Invalid audio format for chunk {chunk_num}: expected (sample_rate, array) tuple", None
            
    except Exception as e:
        print(f"[DEBUG] Exception in save_visual_trim_to_file: {str(e)}")
        return f"❌ Error saving audio for chunk {chunk_num}: {str(e)}", None

def auto_save_visual_trims_and_download(project_name: str) -> tuple:
    """Enhanced download that attempts to save any pending visual trims and then downloads"""
    if not project_name:
        return None, "❌ No project selected"
    
    # Standard download functionality
    download_result = combine_project_audio_chunks(project_name)
    
    if download_result[0]:  # If download was successful
        success_msg = download_result[1] + "\n\n🎵 Note: If you made visual trims but didn't save them, use the 'Save Trimmed Chunk' buttons first, then refresh download"
        return download_result[0], success_msg
    else:
        return download_result

with gr.Blocks(css=css, title="Chatterbox TTS - Audiobook Edition") as demo:
    model_state = gr.State(None)
    voice_library_path_state = gr.State(SAVED_VOICE_LIBRARY_PATH)
    
    gr.HTML("""
    <div class="voice-library-header">
        <h1>🎧 Chatterbox TTS - Audiobook Edition</h1>
        <p>Professional voice cloning for audiobook creation</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # Enhanced TTS Tab with Voice Selection
        with gr.TabItem("🎤 Text-to-Speech", id="tts"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(
                        value="Welcome to Chatterbox TTS Audiobook Edition. This tool will help you create amazing audiobooks with consistent character voices.",
                        label="Text to synthesize",
                        lines=3
                    )
                    
                    # Voice Selection Section
                    with gr.Group():
                        gr.HTML("<h4>🎭 Voice Selection</h4>")
                        tts_voice_selector = gr.Dropdown(
                            choices=get_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                            label="Choose Voice",
                            value=None,
                            info="Select a saved voice profile or use manual input"
                        )
                        
                        # Voice status display
                        tts_voice_status = gr.HTML(
                            "<div class='voice-status'>📝 Manual input mode - upload your own audio file below</div>"
                        )
                    
                    # Audio input (conditionally visible)
                    ref_wav = gr.Audio(
                        sources=["upload", "microphone"], 
                        type="filepath", 
                        label="Reference Audio File (Manual Input)", 
                        value=None,
                        visible=True
                    )
                    
                    with gr.Row():
                        exaggeration = gr.Slider(
                            0.25, 2, step=.05, 
                            label="Exaggeration (Neutral = 0.5)", 
                            value=.5
                        )
                        cfg_weight = gr.Slider(
                            0.2, 1, step=.05, 
                            label="CFG/Pace", 
                            value=0.5
                        )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

                    with gr.Row():
                        run_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                        refresh_voices_btn = gr.Button("🔄 Refresh Voices", size="sm")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio")
                    
                    gr.HTML("""
                    <div class="instruction-box">
                        <h4>💡 TTS Tips:</h4>
                        <ul>
                            <li><strong>Voice Selection:</strong> Choose a saved voice for consistent character voices</li>
                            <li><strong>Reference Audio:</strong> 10-30 seconds of clear speech works best</li>
                            <li><strong>Exaggeration:</strong> 0.3-0.7 for most voices, higher for dramatic effect</li>
                            <li><strong>CFG/Pace:</strong> Lower values = slower, more deliberate speech</li>
                            <li><strong>Temperature:</strong> Higher values = more variation, lower = more consistent</li>
                        </ul>
                    </div>
                    """)

        # Voice Library Tab
        with gr.TabItem("📚 Voice Library", id="voices"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🎭 Voice Management</h3>")
                    
                    # Voice Library Settings
                    with gr.Group():
                        gr.HTML("<h4>📁 Library Settings</h4>")
                        voice_library_path = gr.Textbox(
                            value=SAVED_VOICE_LIBRARY_PATH,
                            label="Voice Library Folder",
                            placeholder="Enter path to voice library folder",
                            info="This path will be remembered between sessions"
                        )
                        update_path_btn = gr.Button("💾 Save & Update Library Path", size="sm")
                        
                        # Configuration status
                        config_status = gr.HTML(
                            f"<div class='config-status'>📂 Current library: {SAVED_VOICE_LIBRARY_PATH}</div>"
                        )
                    
                    # Voice Selection
                    with gr.Group():
                        gr.HTML("<h4>🎯 Select Voice</h4>")
                        voice_dropdown = gr.Dropdown(
                            choices=[],
                            label="Saved Voice Profiles",
                            value=None
                        )
                        
                        with gr.Row():
                            load_voice_btn = gr.Button("📥 Load Voice", size="sm")
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            delete_voice_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                
                with gr.Column(scale=2):
                    # Voice Testing & Saving
                    gr.HTML("<h3>🎙️ Voice Testing & Configuration</h3>")
                    
                    with gr.Group():
                        gr.HTML("<h4>📝 Voice Details</h4>")
                        voice_name = gr.Textbox(label="Voice Name", placeholder="e.g., narrator_male_deep")
                        voice_display_name = gr.Textbox(label="Display Name", placeholder="e.g., Deep Male Narrator")
                        voice_description = gr.Textbox(
                            label="Description", 
                            placeholder="e.g., Deep, authoritative voice for main character",
                            lines=2
                        )
                    
                    with gr.Group():
                        gr.HTML("<h4>🎵 Voice Settings</h4>")
                        voice_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Reference Audio"
                        )
                        
                        with gr.Row():
                            voice_exaggeration = gr.Slider(
                                0.25, 2, step=.05,
                                label="Exaggeration",
                                value=0.5
                            )
                            voice_cfg = gr.Slider(
                                0.2, 1, step=.05,
                                label="CFG/Pace",
                                value=0.5
                            )
                            voice_temp = gr.Slider(
                                0.05, 5, step=.05,
                                label="Temperature",
                                value=0.8
                            )
                    
                    # Test Voice
                    with gr.Group():
                        gr.HTML("<h4>🧪 Test Voice</h4>")
                        test_text = gr.Textbox(
                            value="Hello, this is a test of the voice settings. How does this sound?",
                            label="Test Text",
                            lines=2
                        )
                        
                        with gr.Row():
                            test_voice_btn = gr.Button("🎵 Test Voice", variant="secondary")
                            save_voice_btn = gr.Button("💾 Save Voice Profile", variant="primary")
                        
                        test_audio_output = gr.Audio(label="Test Audio Output")
                        
                        # Status messages
                        voice_status = gr.HTML("<div class='voice-status'>Ready to test and save voices...</div>")

        # Enhanced Audiobook Creation Tab
        with gr.TabItem("📖 Audiobook Creation - Single Sample", id="audiobook_single"):
            gr.HTML("""
            <div class="audiobook-header">
                <h2>📖 Audiobook Creation Studio - Single Voice</h2>
                <p>Transform your text into professional audiobooks with one consistent voice</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Text Input Section
                    with gr.Group():
                        gr.HTML("<h3>📝 Text Content</h3>")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                audiobook_text = gr.Textbox(
                                    label="Audiobook Text",
                                    placeholder="Paste your text here or upload a file below...",
                                    lines=12,
                                    max_lines=20,
                                    info="Text will be split into chunks at sentence boundaries"
                                )
                            
                            with gr.Column(scale=1):
                                # File upload
                                text_file = gr.File(
                                    label="📄 Upload Text File",
                                    file_types=[".txt", ".md", ".rtf"],
                                    type="filepath"
                                )
                                
                                load_file_btn = gr.Button(
                                    "📂 Load File", 
                                    size="sm",
                                    variant="secondary"
                                )
                                
                                # File status
                                file_status = gr.HTML(
                                    "<div class='file-status'>📄 No file loaded</div>"
                                )
                
                with gr.Column(scale=1):
                    # Voice Selection & Project Settings
                    with gr.Group():
                        gr.HTML("<h3>🎭 Voice Configuration</h3>")
                        
                        audiobook_voice_selector = gr.Dropdown(
                            choices=get_audiobook_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                            label="Select Voice",
                            value=None,
                            info="Choose from your saved voice profiles"
                        )
                        
                        refresh_audiobook_voices_btn = gr.Button(
                            "🔄 Refresh Voices", 
                            size="sm"
                        )
                        
                        # Voice info display
                        audiobook_voice_info = gr.HTML(
                            "<div class='voice-status'>🎭 Select a voice to see details</div>"
                        )
                    
                    # Project Settings
                    with gr.Group():
                        gr.HTML("<h3>📁 Project Settings</h3>")
                        
                        project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="e.g., my_first_audiobook",
                            info="Used for naming output files (project_001.wav, project_002.wav, etc.)"
                        )
                        
                        # Previous Projects Section
                        with gr.Group():
                            gr.HTML("<h4>📚 Previous Projects</h4>")
                            
                            previous_project_dropdown = gr.Dropdown(
                                choices=get_project_choices(),
                                label="Load Previous Project Audio",
                                value=None,
                                info="Select a previous project to download its complete audio"
                            )
                            
                            with gr.Row():
                                load_previous_btn = gr.Button(
                                    "📂 Load Project Audio",
                                    size="sm",
                                    variant="secondary"
                                )
                                refresh_previous_btn = gr.Button(
                                    "🔄 Refresh",
                                    size="sm"
                                )
                            
                            # Previous project audio and download
                            previous_project_audio = gr.Audio(
                                label="Previous Project Audio",
                                visible=False
                            )
                            
                            previous_project_download = gr.File(
                                label="📁 Download Previous Project",
                                visible=False
                            )
                            
                            previous_project_status = gr.HTML(
                                "<div class='voice-status'>📁 Select a previous project to load its audio</div>"
                            )
            
            # Processing Section
            with gr.Group():
                gr.HTML("<h3>🚀 Audiobook Processing</h3>")
                
                with gr.Row():
                    validate_btn = gr.Button(
                        "🔍 Validate Input", 
                        variant="secondary",
                        size="lg"
                    )
                    
                    process_btn = gr.Button(
                        "🎵 Create Audiobook", 
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                # Status and progress
                audiobook_status = gr.HTML(
                    "<div class='audiobook-status'>📋 Ready to create audiobooks! Load text, select voice, and set project name.</div>"
                )
                
                # Preview/Output area
                audiobook_output = gr.Audio(
                    label="Generated Audiobook (Preview - Full files saved to project folder)",
                    visible=False
                )
            
            # Instructions
            gr.HTML("""
            <div class="instruction-box">
                <h4>📋 How to Create Single-Voice Audiobooks:</h4>
                <ol>
                    <li><strong>Add Text:</strong> Paste text or upload a .txt file</li>
                    <li><strong>Select Voice:</strong> Choose from your saved voice profiles</li>
                    <li><strong>Set Project Name:</strong> This will be used for output file naming</li>
                    <li><strong>Validate:</strong> Check that everything is ready</li>
                    <li><strong>Create:</strong> Generate your audiobook with smart chunking!</li>
                </ol>
                <p><strong>🎯 Smart Chunking:</strong> Text is automatically split at sentence boundaries after ~50 words for optimal processing.</p>
                <p><strong>📁 File Output:</strong> Individual chunks saved as project_001.wav, project_002.wav, etc.</p>
            </div>
            """)

        # NEW: Multi-Voice Audiobook Creation Tab
        with gr.TabItem("🎭 Audiobook Creation - Multi-Sample", id="audiobook_multi"):
            gr.HTML("""
            <div class="audiobook-header">
                <h2>🎭 Multi-Voice Audiobook Creation Studio</h2>
                <p>Create dynamic audiobooks with multiple character voices using voice tags</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Text Input Section with Voice Tags
                    with gr.Group():
                        gr.HTML("<h3>📝 Multi-Voice Text Content</h3>")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                multi_audiobook_text = gr.Textbox(
                                    label="Multi-Voice Audiobook Text",
                                    placeholder='Use voice tags like: [narrator] Once upon a time... [character1] "Hello!" said the princess. [narrator] She walked away...',
                                    lines=12,
                                    max_lines=20,
                                    info="Use [voice_name] tags to assign text to different voices"
                                )
                            
                            with gr.Column(scale=1):
                                # File upload
                                multi_text_file = gr.File(
                                    label="📄 Upload Text File",
                                    file_types=[".txt", ".md", ".rtf"],
                                    type="filepath"
                                )
                                
                                load_multi_file_btn = gr.Button(
                                    "📂 Load File", 
                                    size="sm",
                                    variant="secondary"
                                )
                                
                                # File status
                                multi_file_status = gr.HTML(
                                    "<div class='file-status'>📄 No file loaded</div>"
                                )
                
                with gr.Column(scale=1):
                    # Voice Analysis & Project Settings
                    with gr.Group():
                        gr.HTML("<h3>🔍 Text Analysis</h3>")
                        
                        analyze_text_btn = gr.Button(
                            "🔍 Analyze Text & Find Characters",
                            variant="secondary",
                            size="lg"
                        )
                        
                        # Voice breakdown display
                        voice_breakdown_display = gr.HTML(
                            "<div class='voice-status'>📝 Click 'Analyze Text' to find characters in your text</div>"
                        )
                        
                        refresh_multi_voices_btn = gr.Button(
                            "🔄 Refresh Available Voices", 
                            size="sm"
                        )
                    
                    # Voice Assignment Section
                    with gr.Group():
                        gr.HTML("<h3>🎭 Voice Assignments</h3>")
                        
                        # Character assignment dropdowns (max 6 common characters)
                        with gr.Column():
                            char1_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 1",
                                visible=False,
                                interactive=True
                            )
                            char2_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 2", 
                                visible=False,
                                interactive=True
                            )
                            char3_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 3",
                                visible=False,
                                interactive=True
                            )
                            char4_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 4",
                                visible=False,
                                interactive=True
                            )
                            char5_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 5",
                                visible=False,
                                interactive=True
                            )
                            char6_dropdown = gr.Dropdown(
                                choices=[("No character found", None)],
                                label="Character 6",
                                visible=False,
                                interactive=True
                            )
                    
                    # Project Settings
                    with gr.Group():
                        gr.HTML("<h3>📁 Project Settings</h3>")
                        
                        multi_project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="e.g., my_multi_voice_story",
                            info="Used for naming output files (project_001_character.wav, etc.)"
                        )
                        
                        # Previous Projects Section
                        with gr.Group():
                            gr.HTML("<h4>📚 Previous Projects</h4>")
                            
                            multi_previous_project_dropdown = gr.Dropdown(
                                choices=get_project_choices(),
                                label="Load Previous Project Audio",
                                value=None,
                                info="Select a previous project to download its complete audio"
                            )
                            
                            with gr.Row():
                                load_multi_previous_btn = gr.Button(
                                    "📂 Load Project Audio",
                                    size="sm",
                                    variant="secondary"
                                )
                                refresh_multi_previous_btn = gr.Button(
                                    "🔄 Refresh",
                                    size="sm"
                                )
                            
                            # Previous project audio and download
                            multi_previous_project_audio = gr.Audio(
                                label="Previous Project Audio",
                                visible=False
                            )
                            
                            multi_previous_project_download = gr.File(
                                label="📁 Download Previous Project",
                                visible=False
                            )
                            
                            multi_previous_project_status = gr.HTML(
                                "<div class='voice-status'>📁 Select a previous project to load its audio</div>"
                            )
            
            # Processing Section
            with gr.Group():
                gr.HTML("<h3>🚀 Multi-Voice Processing</h3>")
                
                with gr.Row():
                    validate_multi_btn = gr.Button(
                        "🔍 Validate Voice Assignments", 
                        variant="secondary",
                        size="lg",
                        interactive=False
                    )
                    
                    process_multi_btn = gr.Button(
                        "🎵 Create Multi-Voice Audiobook", 
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                # Status and progress
                multi_audiobook_status = gr.HTML(
                    "<div class='audiobook-status'>📋 Step 1: Analyze text to find characters<br/>📋 Step 2: Assign voices to each character<br/>📋 Step 3: Validate and create audiobook</div>"
                )
                
                # Preview/Output area
                multi_audiobook_output = gr.Audio(
                    label="Generated Multi-Voice Audiobook (Preview - Full files saved to project folder)",
                    visible=False
                )
            
            # Hidden state to store voice counts and assignments
            voice_counts_state = gr.State({})
            voice_assignments_state = gr.State({})
            character_names_state = gr.State([])
            
            # Instructions for Multi-Voice
            gr.HTML("""
            <div class="instruction-box">
                <h4>📋 How to Create Multi-Voice Audiobooks:</h4>
                <ol>
                    <li><strong>Add Voice Tags:</strong> Use [character_name] before text for that character</li>
                    <li><strong>Analyze Text:</strong> Click 'Analyze Text' to find all characters</li>
                    <li><strong>Assign Voices:</strong> Choose voices from your library for each character</li>
                    <li><strong>Set Project Name:</strong> Used for output file naming</li>
                    <li><strong>Validate & Create:</strong> Generate your multi-voice audiobook!</li>
                </ol>
                <h4>🎯 Voice Tag Format:</h4>
                <p><code>[narrator] The story begins here...</code></p>
                <p><code>[princess] "Hello there!" she said cheerfully.</code></p>
                <p><code>[narrator] The mysterious figure walked away.</code></p>
                <p><strong>📁 File Output:</strong> Files named with character: project_001_narrator.wav, project_002_princess.wav, etc.</p>
                <p><strong>🎭 New Workflow:</strong> Characters in [brackets] can be mapped to any voice in your library!</p>
                <p><strong>💡 Smart Processing:</strong> Tries GPU first for speed, automatically falls back to CPU if CUDA errors occur (your 3090 should handle most cases!).</p>
            </div>
            """)

        # NEW: Regenerate Sample Tab
        with gr.TabItem("🔄 Regenerate Sample", id="regenerate"):
            gr.HTML("""
            <div class="audiobook-header">
                <h2>🔄 Project Chunk Regeneration Studio</h2>
                <p>Load existing projects and regenerate individual chunks with original voice settings</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Project Selection
                    with gr.Group():
                        gr.HTML("<h3>📁 Project Selection</h3>")
                        
                        project_dropdown = gr.Dropdown(
                            choices=get_project_choices(),
                            label="Select Project",
                            value=None,
                            info="Choose from your existing audiobook projects"
                        )
                        
                        with gr.Row():
                            load_project_btn = gr.Button(
                                "📂 Load Project Chunks",
                                variant="secondary",
                                size="lg"
                            )
                            refresh_projects_btn = gr.Button(
                                "🔄 Refresh Projects",
                                size="sm"
                            )
                        
                        # Project status
                        project_status = gr.HTML(
                            "<div class='audiobook-status'>📁 Select a project to view all chunks</div>"
                        )
                
                with gr.Column(scale=2):
                    # Project Information Display
                    with gr.Group():
                        gr.HTML("<h3>📋 Project Overview</h3>")
                        
                        # Project info summary
                        project_info_summary = gr.HTML(
                            "<div class='voice-status'>📝 Load a project to see details</div>"
                        )
                        
                        # Chunks container - this will be populated dynamically
                        chunks_container = gr.HTML(
                            "<div class='audiobook-status'>📚 Project chunks will appear here after loading</div>"
                        )
                        
                        # Download Section
                        with gr.Group():
                            gr.HTML("<h4>💾 Download Complete Project</h4>")
                            
                            with gr.Row():
                                download_project_btn = gr.Button(
                                    "📥 Download Full Project Audio",
                                    variant="primary",
                                    size="lg",
                                    interactive=False
                                )
                                
                                refresh_download_btn = gr.Button(
                                    "🔄 Refresh Download",
                                    size="sm"
                                )
                                
                                cleanup_temp_btn = gr.Button(
                                    "🗑️ Clean Temp Files",
                                    size="sm",
                                    variant="secondary"
                                )
                            
                            # Download status and file
                            download_status = gr.HTML(
                                "<div class='voice-status'>📁 Load a project first to enable download</div>"
                            )
                            
                            download_file = gr.File(
                                label="📁 Download Complete Audio File",
                                visible=False
                            )
            
            # Dynamic chunk interface - created when project is loaded
            chunk_interfaces = []
            
            # Create interface for up to 50 chunks (should be enough for most projects)
            for i in range(50):
                with gr.Group(visible=False) as chunk_group:
                    with gr.Row():
                        with gr.Column(scale=1):
                            chunk_audio = gr.Audio(
                                label=f"Chunk {i+1} Audio",
                                interactive=True,  # Enable trimming
                                show_download_button=True,
                                show_share_button=False,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#01C6FF",
                                    waveform_progress_color="#0066B4", 
                                    trim_region_color="#FF6B6B",
                                    show_recording_waveform=True,
                                    skip_length=5,
                                    sample_rate=24000
                                )
                            )
                            
                            # Add trim/save button for original audio
                            save_original_trim_btn = gr.Button(
                                f"💾 Save Trimmed Chunk {i+1}",
                                variant="secondary",
                                size="sm",
                                visible=True
                            )
                        
                        with gr.Column(scale=2):
                            chunk_text = gr.Textbox(
                                label=f"Chunk {i+1} Text",
                                lines=3,
                                max_lines=6,
                                info="Edit this text and regenerate to create a new version"
                            )
                            
                            with gr.Row():
                                chunk_voice_info = gr.HTML(
                                    "<div class='voice-status'>Voice info</div>"
                                )
                                
                                regenerate_chunk_btn = gr.Button(
                                    f"🎵 Regenerate Chunk {i+1}",
                                    variant="primary",
                                    size="sm"
                                )
                            
                            regenerated_chunk_audio = gr.Audio(
                                label=f"Regenerated Chunk {i+1}",
                                visible=False,
                                interactive=True,  # Enable trimming
                                show_download_button=True,
                                show_share_button=False,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#FF6B6B",
                                    waveform_progress_color="#FF4444",
                                    trim_region_color="#FFB6C1",
                                    show_recording_waveform=True,
                                    skip_length=5,
                                    sample_rate=24000
                                )
                            )
                            
                            # Accept/Decline buttons for regenerated audio
                            with gr.Row(visible=False) as accept_decline_row:
                                accept_chunk_btn = gr.Button(
                                    "✅ Accept Regeneration",
                                    variant="primary",
                                    size="sm"
                                )
                                decline_chunk_btn = gr.Button(
                                    "❌ Decline Regeneration", 
                                    variant="stop",
                                    size="sm"
                                )
                                save_regen_trim_btn = gr.Button(
                                    "💾 Save Trimmed Regeneration",
                                    variant="secondary",
                                    size="sm"
                                )
                            
                            chunk_status = gr.HTML(
                                "<div class='voice-status'>Ready to regenerate</div>"
                            )
                
                chunk_interfaces.append({
                    'group': chunk_group,
                    'audio': chunk_audio,
                    'text': chunk_text,
                    'voice_info': chunk_voice_info,
                    'button': regenerate_chunk_btn,
                    'regenerated_audio': regenerated_chunk_audio,
                    'accept_decline_row': accept_decline_row,
                    'accept_btn': accept_chunk_btn,
                    'decline_btn': decline_chunk_btn,
                    'save_original_trim_btn': save_original_trim_btn,
                    'save_regen_trim_btn': save_regen_trim_btn,
                    'status': chunk_status,
                    'chunk_num': i + 1
                })
            
            # Hidden states for chunk management
            current_project_chunks = gr.State([])
            current_project_name = gr.State("")
            
            # Instructions for Regeneration
            gr.HTML("""
            <div class="instruction-box">
                <h4>📋 How to Use Chunk Regeneration & Audio Trimming:</h4>
                <ol>
                    <li><strong>Select Project:</strong> Choose from your existing audiobook projects</li>
                    <li><strong>Load Project:</strong> View all audio chunks with their original text</li>
                    <li><strong>Review & Trim:</strong> Listen to each chunk and trim if needed using the waveform controls</li>
                    <li><strong>Save Trimmed Audio:</strong> Click "💾 Save Trimmed Chunk" to save your trimmed version</li>
                    <li><strong>Edit & Regenerate:</strong> Modify text if needed and regenerate individual chunks</li>
                    <li><strong>Trim Regenerated:</strong> Use trim controls on regenerated audio and save with "💾 Save Trimmed Regeneration"</li>
                    <li><strong>Accept/Decline:</strong> Accept regenerated chunks or decline to keep originals</li>
                </ol>
                <h4>🎯 Audio Trimming Features:</h4>
                <ul>
                    <li><strong>🎵 Interactive Waveforms:</strong> Click and drag on the waveform to select audio segments</li>
                    <li><strong>✂️ Visual Trimming:</strong> Drag the trim handles to select the desired audio portion</li>
                    <li><strong>💾 Save Trimmed Audio:</strong> Click the save button to apply your trimming to the actual file</li>
                    <li><strong>🔄 Real-time Preview:</strong> Play the selected portion before saving</li>
                    <li><strong>📁 Auto-Backup:</strong> Original files are automatically backed up when trimming</li>
                    <li><strong>⚠️ Important:</strong> Visual trimming selection must be saved using the "Save Trimmed" button</li>
                </ul>
                <h4>🎯 Traditional Features:</h4>
                <ul>
                    <li><strong>📄 Individual Control:</strong> Regenerate only the chunks you need to fix</li>
                    <li><strong>🎭 Voice Preservation:</strong> Uses exact voice settings from original project</li>
                    <li><strong>🎵 Side-by-side Comparison:</strong> Compare original and regenerated audio</li>
                    <li><strong>✏️ Text Editing:</strong> Modify text before regenerating</li>
                    <li><strong>🚀 Efficient Workflow:</strong> Fix specific issues without regenerating entire projects</li>
                </ul>
                <p><strong>💡 Trimming Workflow:</strong></p>
                <ol>
                    <li>🎧 Load a project and play the audio chunk</li>
                    <li>✂️ Drag the trim region handles on the waveform to select your desired segment</li>
                    <li>▶️ Play the selection to verify it sounds correct</li>
                    <li>💾 Click "Save Trimmed Chunk" to apply the trimming to the actual file</li>
                    <li>🔄 The download will now include your trimmed version</li>
                </ol>
                <p><strong>⚠️ Note:</strong> Gradio's visual trimming is just for selection - you must click "Save Trimmed" to actually apply the changes to the downloadable file!</p>
                <p><strong>💡 Note:</strong> Only projects created with metadata support can be fully regenerated. Legacy projects will show limited information.</p>
            </div>
            """)

    # Load initial voice list and model
    demo.load(fn=load_model, inputs=[], outputs=model_state)
    demo.load(
        fn=lambda: refresh_voice_list(SAVED_VOICE_LIBRARY_PATH),
        inputs=[],
        outputs=voice_dropdown
    )
    demo.load(
        fn=lambda: refresh_voice_choices(SAVED_VOICE_LIBRARY_PATH),
        inputs=[],
        outputs=tts_voice_selector
    )
    demo.load(
        fn=lambda: refresh_audiobook_voice_choices(SAVED_VOICE_LIBRARY_PATH),
        inputs=[],
        outputs=audiobook_voice_selector
    )
    demo.load(
        fn=lambda: get_project_choices(),
        inputs=[],
        outputs=previous_project_dropdown
    )
    demo.load(
        fn=lambda: get_project_choices(),
        inputs=[],
        outputs=multi_previous_project_dropdown
    )

    # TTS Voice Selection
    tts_voice_selector.change(
        fn=lambda path, voice: load_voice_for_tts(path, voice),
        inputs=[voice_library_path_state, tts_voice_selector],
        outputs=[ref_wav, exaggeration, cfg_weight, temp, ref_wav, tts_voice_status]
    )

    # Refresh voices in TTS tab
    refresh_voices_btn.click(
        fn=lambda path: refresh_voice_choices(path),
        inputs=voice_library_path_state,
        outputs=tts_voice_selector
    )

    # TTS Generation
    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )

    # Voice Library Functions
    update_path_btn.click(
        fn=update_voice_library_path,
        inputs=voice_library_path,
        outputs=[voice_library_path_state, config_status, voice_dropdown, tts_voice_selector, audiobook_voice_selector]
    )

    refresh_btn.click(
        fn=lambda path: (refresh_voice_list(path), refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
        inputs=voice_library_path_state,
        outputs=[voice_dropdown, tts_voice_selector, audiobook_voice_selector]
    )

    load_voice_btn.click(
        fn=lambda path, name: load_voice_profile(path, name),
        inputs=[voice_library_path_state, voice_dropdown],
        outputs=[voice_audio, voice_exaggeration, voice_cfg, voice_temp, voice_status]
    )

    test_voice_btn.click(
        fn=lambda model, text, audio, exag, temp, cfg: generate(model, text, audio, exag, temp, 0, cfg),
        inputs=[model_state, test_text, voice_audio, voice_exaggeration, voice_temp, voice_cfg],
        outputs=test_audio_output
    )

    save_voice_btn.click(
        fn=lambda path, name, display, desc, audio, exag, cfg, temp: save_voice_profile(
            path, name, display, desc, audio, exag, cfg, temp
        ),
        inputs=[
            voice_library_path_state, voice_name, voice_display_name, voice_description,
            voice_audio, voice_exaggeration, voice_cfg, voice_temp
        ],
        outputs=voice_status
    ).then(
        fn=lambda path: (refresh_voice_list(path), refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
        inputs=voice_library_path_state,
        outputs=[voice_dropdown, tts_voice_selector, audiobook_voice_selector]
    )

    delete_voice_btn.click(
        fn=lambda path, name: delete_voice_profile(path, name),
        inputs=[voice_library_path_state, voice_dropdown],
        outputs=[voice_status, voice_dropdown]
    ).then(
        fn=lambda path: (refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
        inputs=voice_library_path_state,
        outputs=[tts_voice_selector, audiobook_voice_selector]
    )

    # NEW: Multi-Voice Audiobook Creation Functions
    
    # Multi-voice file loading
    load_multi_file_btn.click(
        fn=load_text_file,
        inputs=multi_text_file,
        outputs=[multi_audiobook_text, multi_file_status]
    )
    
    # Single-voice audiobook functions (restored)
    # File loading
    load_file_btn.click(
        fn=load_text_file,
        inputs=text_file,
        outputs=[audiobook_text, file_status]
    )
    
    # Voice selection for audiobook
    refresh_audiobook_voices_btn.click(
        fn=lambda path: refresh_audiobook_voice_choices(path),
        inputs=voice_library_path_state,
        outputs=audiobook_voice_selector
    )
    
    # Enhanced Validation with project name
    validate_btn.click(
        fn=validate_audiobook_input,
        inputs=[audiobook_text, audiobook_voice_selector, project_name],
        outputs=[process_btn, audiobook_status, audiobook_output]
    )
    
    # Enhanced Audiobook Creation with chunking and saving
    process_btn.click(
        fn=create_audiobook,
        inputs=[model_state, audiobook_text, voice_library_path_state, audiobook_voice_selector, project_name],
        outputs=[audiobook_output, audiobook_status]
    ).then(
        fn=force_refresh_all_project_dropdowns,
        inputs=[],
        outputs=[previous_project_dropdown, multi_previous_project_dropdown, project_dropdown]
    )
    
    # Text analysis to find characters and populate dropdowns
    analyze_text_btn.click(
        fn=handle_multi_voice_analysis,
        inputs=[multi_audiobook_text, voice_library_path_state],
        outputs=[voice_breakdown_display, voice_counts_state, character_names_state, 
                char1_dropdown, char2_dropdown, char3_dropdown, char4_dropdown, char5_dropdown, char6_dropdown,
                validate_multi_btn, multi_audiobook_status]
    )
    
    # Multi-voice validation using dropdown values
    validate_multi_btn.click(
        fn=validate_dropdown_voice_assignments,
        inputs=[multi_audiobook_text, voice_library_path_state, multi_project_name, voice_counts_state, character_names_state,
               char1_dropdown, char2_dropdown, char3_dropdown, char4_dropdown, char5_dropdown, char6_dropdown],
        outputs=[process_multi_btn, multi_audiobook_status, voice_assignments_state, multi_audiobook_output]
    )
    
    # Multi-voice audiobook creation (using voice assignments)
    process_multi_btn.click(
        fn=create_multi_voice_audiobook_with_assignments,
        inputs=[model_state, multi_audiobook_text, voice_library_path_state, multi_project_name, voice_assignments_state],
        outputs=[multi_audiobook_output, multi_audiobook_status]
    ).then(
        fn=force_refresh_all_project_dropdowns,
        inputs=[],
        outputs=[previous_project_dropdown, multi_previous_project_dropdown, project_dropdown]
    )
    
    # Refresh voices for multi-voice (updates dropdown choices)
    refresh_multi_voices_btn.click(
        fn=lambda path: f"<div class='voice-status'>🔄 Available voices refreshed from: {path}<br/>📚 Re-analyze your text to update character assignments</div>",
        inputs=voice_library_path_state,
        outputs=voice_breakdown_display
    )

    # NEW: Regenerate Sample Tab Functions
    
    # Load projects on tab initialization
    demo.load(
        fn=force_refresh_single_project_dropdown,
        inputs=[],
        outputs=project_dropdown
    )
    
    # Refresh projects dropdown
    refresh_projects_btn.click(
        fn=force_complete_project_refresh,
        inputs=[],
        outputs=project_dropdown
    )
    
    # Create output list for all chunk interface components
    chunk_outputs = []
    for i in range(50):
        chunk_outputs.extend([
            chunk_interfaces[i]['group'],
            chunk_interfaces[i]['audio'],
            chunk_interfaces[i]['text'],
            chunk_interfaces[i]['voice_info'],
            chunk_interfaces[i]['button'],
            chunk_interfaces[i]['regenerated_audio'],
            chunk_interfaces[i]['status']
        ])
    
    # Load project chunks
    load_project_btn.click(
        fn=load_project_chunks_for_interface,
        inputs=project_dropdown,
        outputs=[project_info_summary, current_project_chunks, current_project_name, project_status, download_project_btn, download_status] + chunk_outputs
    )
    
    # Add regeneration handlers for each chunk
    for i, chunk_interface in enumerate(chunk_interfaces):
        chunk_num = i + 1
        
        # Create state to store regenerated file path for this chunk
        chunk_regen_file_state = gr.State("")
        
        # Use closure to capture chunk_num properly
        def make_regenerate_handler(chunk_num):
            def regenerate_handler(model, project_name, voice_lib_path, custom_text):
                result = regenerate_single_chunk(model, project_name, chunk_num, voice_lib_path, custom_text)
                if result and len(result) == 2:
                    temp_file_path, status_msg = result
                    if temp_file_path and isinstance(temp_file_path, str):
                        # Return both the file path (for audio display) and store it for later use
                        return temp_file_path, status_msg, temp_file_path
                    else:
                        return None, status_msg, ""
                else:
                    return None, result[1] if result else "Error occurred", ""
            return regenerate_handler
        
        # Use closure for accept/decline handlers
        def make_accept_handler(chunk_num):
            def accept_handler(project_name, regen_file_path):
                if regen_file_path:
                    return accept_regenerated_chunk(project_name, chunk_num, regen_file_path)
                else:
                    return f"❌ No regenerated file to accept for chunk {chunk_num}", None
            return accept_handler
        
        def make_decline_handler(chunk_num):
            def decline_handler(regen_file_path):
                return decline_regenerated_chunk(chunk_num, regen_file_path)
            return decline_handler
        
        chunk_interface['button'].click(
            fn=make_regenerate_handler(chunk_num),
            inputs=[model_state, current_project_name, voice_library_path_state, chunk_interface['text']],
            outputs=[chunk_interface['regenerated_audio'], chunk_interface['status'], chunk_regen_file_state]
        ).then(
            fn=lambda audio: (gr.Audio(visible=bool(audio)), gr.Row(visible=bool(audio))),
            inputs=chunk_interface['regenerated_audio'],
            outputs=[chunk_interface['regenerated_audio'], chunk_interface['accept_decline_row']]
        )
        
        # Accept button handler
        chunk_interface['accept_btn'].click(
            fn=make_accept_handler(chunk_num),
            inputs=[current_project_name, chunk_regen_file_state],
            outputs=[chunk_interface['status'], chunk_interface['audio']]
        ).then(
            fn=lambda: (gr.Audio(visible=False), gr.Row(visible=False), ""),
            inputs=[],
            outputs=[chunk_interface['regenerated_audio'], chunk_interface['accept_decline_row'], chunk_regen_file_state]
        )
        
        # Decline button handler  
        chunk_interface['decline_btn'].click(
            fn=make_decline_handler(chunk_num),
            inputs=chunk_regen_file_state,
            outputs=[chunk_interface['regenerated_audio'], chunk_interface['accept_decline_row'], chunk_interface['status']]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=chunk_regen_file_state
        )
        
        # Save original trimmed audio handler
        def make_save_original_trim_handler(chunk_num):
            def save_original_trim(trimmed_audio_data):
                print(f"[DEBUG] save_original_trim called for chunk {chunk_num}")
                print(f"[DEBUG] trimmed_audio_data type: {type(trimmed_audio_data)}")
                
                if not current_project_chunks.value or chunk_num > len(current_project_chunks.value):
                    return f"❌ No project loaded or invalid chunk number {chunk_num}", None
                
                chunk_info = current_project_chunks.value[chunk_num - 1]
                original_file_path = chunk_info['audio_file']
                
                # Process the audio data with the new simplified function
                return save_visual_trim_to_file(trimmed_audio_data, original_file_path, chunk_num)
            return save_original_trim
        
        # Audio change handler to provide feedback about trimming
        def make_audio_change_handler(chunk_num):
            def audio_change_handler(audio_data):
                if audio_data:
                    return f"<div class='voice-status'>🎵 Chunk {chunk_num} audio ready - you can trim the waveform and save changes</div>"
                else:
                    return f"<div class='voice-status'>📄 Chunk {chunk_num} - no audio loaded</div>"
            return audio_change_handler
        
        chunk_interface['audio'].change(
            fn=make_audio_change_handler(chunk_num),
            inputs=chunk_interface['audio'],
            outputs=chunk_interface['status']
        )
        
        chunk_interface['save_original_trim_btn'].click(
            fn=make_save_original_trim_handler(chunk_num),
            inputs=chunk_interface['audio'],
            outputs=[chunk_interface['status'], chunk_interface['audio']]
        )
        
        # Save regenerated trimmed audio handler
        def make_save_regen_trim_handler(chunk_num):
            def save_regen_trim(trimmed_audio_data):
                if not current_project_chunks.value or chunk_num > len(current_project_chunks.value):
                    return f"❌ No project loaded or invalid chunk number {chunk_num}", None
                
                chunk_info = current_project_chunks.value[chunk_num - 1]
                original_file_path = chunk_info['audio_file']
                
                # Save the trimmed regenerated audio as the new original using simplified function
                return save_visual_trim_to_file(trimmed_audio_data, original_file_path, chunk_num)
            return save_regen_trim
        
        chunk_interface['save_regen_trim_btn'].click(
            fn=make_save_regen_trim_handler(chunk_num),
            inputs=chunk_interface['regenerated_audio'],
            outputs=[chunk_interface['status'], chunk_interface['audio']]
        ).then(
            fn=lambda: (gr.Audio(visible=False), gr.Row(visible=False), ""),
            inputs=[],
            outputs=[chunk_interface['regenerated_audio'], chunk_interface['accept_decline_row'], chunk_regen_file_state]
        )
    
        # Manual trimming handlers for this chunk
        def make_get_duration_handler(chunk_num):
            def get_duration_handler():
                if not current_project_chunks.value or chunk_num > len(current_project_chunks.value):
                    return 0, f"❌ No project loaded or invalid chunk number {chunk_num}"
                
                chunk_info = current_project_chunks.value[chunk_num - 1]
                audio_file = chunk_info['audio_file']
                
                try:
                    with wave.open(audio_file, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / sample_rate
                        
                        return duration, f"<div class='voice-status'>🎵 Chunk {chunk_num} duration: {duration:.2f} seconds</div>"
                except Exception as e:
                    return 0, f"<div class='voice-status'>❌ Error reading audio: {str(e)}</div>"
            return get_duration_handler
        
        def make_apply_manual_trim_handler(chunk_num):
            def apply_manual_trim(start_time, end_time):
                if not current_project_chunks.value or chunk_num > len(current_project_chunks.value):
                    return f"❌ No project loaded or invalid chunk number {chunk_num}", None
                
                chunk_info = current_project_chunks.value[chunk_num - 1]
                audio_file = chunk_info['audio_file']
                
                try:
                    # Load the audio file
                    with wave.open(audio_file, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    
                    # Apply manual trimming
                    audio_tuple = (sample_rate, audio_data)
                    end_time_actual = None if end_time <= 0 else end_time
                    trimmed_audio, status_msg = extract_audio_segment(audio_tuple, start_time, end_time_actual)
                    
                    if trimmed_audio:
                        # Save the trimmed audio
                        save_status, new_file_path = save_trimmed_audio(trimmed_audio, audio_file, chunk_num)
                        combined_status = f"{status_msg}\n{save_status}"
                        return combined_status, new_file_path
                    else:
                        return status_msg, None
                        
                except Exception as e:
                    return f"❌ Error applying manual trim to chunk {chunk_num}: {str(e)}", None
            return apply_manual_trim
        
        chunk_interface['get_duration_btn'].click(
            fn=make_get_duration_handler(chunk_num),
            inputs=[],
            outputs=[chunk_interface['trim_end'], chunk_interface['status']]
        )
        
        chunk_interface['apply_trim_btn'].click(
            fn=make_apply_manual_trim_handler(chunk_num),
            inputs=[chunk_interface['trim_start'], chunk_interface['trim_end']],
            outputs=[chunk_interface['status'], chunk_interface['audio']]
        )
    
    # Download full project audio
    download_project_btn.click(
        fn=combine_project_audio_chunks,
        inputs=current_project_name,
        outputs=[download_file, download_status]
    ).then(
        fn=lambda file_path: gr.File(visible=bool(file_path)),
        inputs=download_file,
        outputs=download_file
    )
    
    # Refresh download (regenerate combined file)
    refresh_download_btn.click(
        fn=combine_project_audio_chunks,
        inputs=current_project_name,
        outputs=[download_file, download_status]
    ).then(
        fn=lambda file_path: gr.File(visible=bool(file_path)),
        inputs=download_file,
        outputs=download_file
    )
    
    # Clean up temp files
    cleanup_temp_btn.click(
        fn=cleanup_project_temp_files,
        inputs=current_project_name,
        outputs=download_status
    )
    
    # Previous Projects - Single Voice Tab
    refresh_previous_btn.click(
        fn=force_complete_project_refresh,
        inputs=[],
        outputs=previous_project_dropdown
    )
    
    load_previous_btn.click(
        fn=load_previous_project_audio,
        inputs=previous_project_dropdown,
        outputs=[previous_project_audio, previous_project_download, previous_project_status]
    ).then(
        fn=lambda audio_path, download_path: (gr.Audio(visible=bool(audio_path)), gr.File(visible=bool(download_path))),
        inputs=[previous_project_audio, previous_project_download],
        outputs=[previous_project_audio, previous_project_download]
    )
    
    # Previous Projects - Multi-Voice Tab
    refresh_multi_previous_btn.click(
        fn=force_complete_project_refresh,
        inputs=[],
        outputs=multi_previous_project_dropdown
    )
    
    load_multi_previous_btn.click(
        fn=load_previous_project_audio,
        inputs=multi_previous_project_dropdown,
        outputs=[multi_previous_project_audio, multi_previous_project_download, multi_previous_project_status]
    ).then(
        fn=lambda audio_path, download_path: (gr.Audio(visible=bool(audio_path)), gr.File(visible=bool(download_path))),
        inputs=[multi_previous_project_audio, multi_previous_project_download],
        outputs=[multi_previous_project_audio, multi_previous_project_download]
    )

    demo.load(
        fn=force_refresh_single_project_dropdown,
        inputs=[],
        outputs=previous_project_dropdown
    )
    demo.load(
        fn=force_refresh_single_project_dropdown,
        inputs=[],
        outputs=multi_previous_project_dropdown
    )
    demo.load(
        fn=force_refresh_single_project_dropdown,
        inputs=[],
        outputs=project_dropdown
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True) 