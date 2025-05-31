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
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU mode for multi-voice to avoid CUDA indexing errors
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice processing

# Default voice library path
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"
MAX_CHUNKS_FOR_INTERFACE = 100 # Increased from 50 to 100, will add pagination later
MAX_CHUNKS_FOR_AUTO_SAVE = 100 # Match the interface limit for now

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
        return f"âœ… Configuration saved - Voice library path: {voice_library_path}"
    except Exception as e:
        return f"âŒ Error saving configuration: {str(e)}"

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
                
                print(f"âš ï¸ CUDA error detected, falling back to CPU: {str(e)[:100]}...")
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
        display_text = f"ðŸŽ­ {profile['display_name']} ({profile['name']})"
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
            display_text = f"ðŸŽ­ {profile['display_name']} ({profile['name']})"
            choices.append((display_text, profile['name']))
    return choices

def load_text_file(file_path):
    """Load text from uploaded file"""
    if file_path is None:
        return "No file uploaded", "âŒ Please upload a text file"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic validation
        if not content.strip():
            return "", "âŒ File is empty"
        
        word_count = len(content.split())
        char_count = len(content)
        
        status = f"âœ… File loaded successfully!\nðŸ“„ {word_count:,} words | {char_count:,} characters"
        
        return content, status
        
    except UnicodeDecodeError:
        try:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            word_count = len(content.split())
            char_count = len(content)
            status = f"âœ… File loaded (latin-1 encoding)!\nðŸ“„ {word_count:,} words | {char_count:,} characters"
            return content, status
        except Exception as e:
            return "", f"âŒ Error reading file: {str(e)}"
    except Exception as e:
        return "", f"âŒ Error loading file: {str(e)}"

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate inputs for audiobook creation"""
    issues = []
    
    if not text_content or not text_content.strip():
        issues.append("ðŸ“ Text content is required")
    
    if not selected_voice:
        issues.append("ðŸŽ­ Voice selection is required")
    
    if not project_name or not project_name.strip():
        issues.append("ðŸ“ Project name is required")
    
    if text_content and len(text_content.strip()) < 10:
        issues.append("ðŸ“ Text is too short (minimum 10 characters)")
    
    if issues:
        return (
            gr.Button("ðŸŽµ Create Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Please fix these issues:\n" + "\n".join(f"â€¢ {issue}" for issue in issues), 
            gr.Audio(visible=False)
        )
    
    word_count = len(text_content.split())
    chunks = chunk_text_by_sentences(text_content)
    chunk_count = len(chunks)
    
    return (
        gr.Button("ðŸŽµ Create Audiobook", variant="primary", size="lg", interactive=True),
        f"âœ… Ready for audiobook creation!\nðŸ“Š {word_count:,} words â†’ {chunk_count} chunks\nðŸ“ Project: {project_name.strip()}", 
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
                print(f"âš ï¸ Error reading config for voice '{name_to_try}': {str(e)}")
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
                    print(f"âš ï¸ GPU error, retry {retry + 1}/{max_retries}: {str(e)[:100]}...")
                    clear_gpu_memory()
                    continue
                else:
                    raise RuntimeError(f"Failed after {max_retries} retries: {str(e)}")
            else:
                raise e
    
    raise RuntimeError("Generation failed after all retries")

def create_audiobook(
    model,
    text_content: str,
    voice_library_path: str,
    selected_voice: str,
    project_name: str,
    resume: bool = False,
    autosave_interval: int = 10
) -> tuple:
    """
    Create audiobook from text using selected voice with smart chunking, autosave every N chunks, and resume support.
    Args:
        model: TTS model
        text_content: Full text
        voice_library_path: Path to voice library
        selected_voice: Voice name
        project_name: Project name
        resume: If True, resume from last saved chunk
        autosave_interval: Chunks per autosave (default 10)
    Returns:
        (sample_rate, combined_audio), status_message
    """
    import numpy as np
    import os
    import json
    import wave
    from typing import List

    if not text_content or not selected_voice or not project_name:
        return None, "âŒ Missing required fields"

    # Get voice configuration
    voice_config = get_voice_config(voice_library_path, selected_voice)
    if not voice_config:
        return None, f"âŒ Could not load voice configuration for '{selected_voice}'"
    if not voice_config['audio_file']:
        return None, f"âŒ No audio file found for voice '{voice_config['display_name']}'"

    # Prepare chunking
    chunks = chunk_text_by_sentences(text_content)
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None, "âŒ No text chunks to process"

    # Project directory
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Resume logic: find already completed chunk files
    completed_chunks = set()
    chunk_filenames = [f"{safe_project_name}_{i+1:03d}.wav" for i in range(total_chunks)]
    for idx, fname in enumerate(chunk_filenames):
        if os.path.exists(os.path.join(project_dir, fname)):
            completed_chunks.add(idx)

    # If resuming, only process missing chunks
    start_idx = 0
    if resume and completed_chunks:
        # Find first missing chunk
        for i in range(total_chunks):
            if i not in completed_chunks:
                start_idx = i
                break
        else:
            return None, "âœ… All chunks already completed. Nothing to resume."
    else:
        start_idx = 0

    # Initialize model if needed
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    audio_chunks: List[np.ndarray] = []
    status_updates = []
    clear_gpu_memory()

    # For resume, load already completed audio
    for i in range(start_idx):
        fname = os.path.join(project_dir, chunk_filenames[i])
        with wave.open(fname, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            audio_chunks.append(audio_data)

    # Process missing chunks
    for i in range(start_idx, total_chunks):
        if i in completed_chunks:
            continue  # Already done
        chunk = chunks[i]
        try:
            chunk_words = len(chunk.split())
            status_msg = f"ðŸŽµ Processing chunk {i+1}/{total_chunks}\nðŸŽ­ Voice: {voice_config['display_name']}\nðŸ“ Chunk {i+1}: {chunk_words} words\nðŸ“Š Progress: {i+1}/{total_chunks} chunks"
            status_updates.append(status_msg)
            wav = generate_with_retry(
                model,
                chunk,
                voice_config['audio_file'],
                voice_config['exaggeration'],
                voice_config['temperature'],
                voice_config['cfg_weight']
            )
            audio_np = wav.squeeze(0).cpu().numpy()
            audio_chunks.append(audio_np)
            # Save this chunk immediately
            fname = os.path.join(project_dir, chunk_filenames[i])
            with wave.open(fname, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(model.sr)
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            del wav
            clear_gpu_memory()
        except Exception as chunk_error:
            return None, f"âŒ Error processing chunk {i+1}: {str(chunk_error)}"
        # Autosave every N chunks
        if (i + 1) % autosave_interval == 0 or (i + 1) == total_chunks:
            # Save project metadata
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
    success_msg = f"âœ… Audiobook created successfully!\nðŸŽ­ Voice: {voice_config['display_name']}\nðŸ“Š {total_words:,} words in {total_chunks} chunks\nâ±ï¸ Duration: ~{duration_minutes} minutes\nðŸ“ Saved to: {project_dir}\nðŸŽµ Files: {len(audio_chunks)} audio chunks\nðŸ’¾ Metadata saved for regeneration"
    return (model.sr, combined_audio), success_msg

def load_voice_for_tts(voice_library_path, voice_name):
    """Load a voice profile for TTS tab - returns settings for sliders"""
    if not voice_name:
        # Return to manual input mode
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), "ðŸ“ Manual input mode - upload your own audio file below"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), f"âŒ Voice profile '{voice_name}' not found"
    
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
        
        status_msg = f"âœ… Using voice: {config.get('display_name', voice_name)}"
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
        return None, 0.5, 0.5, 0.8, gr.Audio(visible=True), f"âŒ Error loading voice profile: {str(e)}"

def save_voice_profile(voice_library_path, voice_name, display_name, description, audio_file, exaggeration, cfg_weight, temperature):
    """Save a voice profile with its settings"""
    if not voice_name:
        return "âŒ Error: Voice name cannot be empty"
    
    # Sanitize voice name for folder
    safe_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    if not safe_name:
        return "âŒ Error: Invalid voice name"
    
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
    
    return f"âœ… Voice profile '{display_name or voice_name}' saved successfully!"

def load_voice_profile(voice_library_path, voice_name):
    """Load a voice profile and return its settings"""
    if not voice_name:
        return None, 0.5, 0.5, 0.8, "No voice selected"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 0.5, 0.5, 0.8, f"âŒ Voice profile '{voice_name}' not found"
    
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
            f"âœ… Loaded voice profile: {config.get('display_name', voice_name)}"
        )
    except Exception as e:
        return None, 0.5, 0.5, 0.8, f"âŒ Error loading voice profile: {str(e)}"

def delete_voice_profile(voice_library_path, voice_name):
    """Delete a voice profile"""
    if not voice_name:
        return "âŒ No voice selected", []
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    if os.path.exists(profile_dir):
        try:
            shutil.rmtree(profile_dir)
            return f"âœ… Voice profile '{voice_name}' deleted successfully!", get_voice_profiles(voice_library_path)
        except Exception as e:
            return f"âŒ Error deleting voice profile: {str(e)}", get_voice_profiles(voice_library_path)
    else:
        return f"âŒ Voice profile '{voice_name}' not found", get_voice_profiles(voice_library_path)

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
        return DEFAULT_VOICE_LIBRARY, "âŒ Path cannot be empty, using default", refresh_voice_list(DEFAULT_VOICE_LIBRARY), refresh_voice_choices(DEFAULT_VOICE_LIBRARY), refresh_audiobook_voice_choices(DEFAULT_VOICE_LIBRARY)
    
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
        return False, "âŒ Text content is required", {}
    
    # Parse the text to find voice references
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return False, "âŒ No valid voice segments found", {}
    
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
        return False, f"âŒ Missing voices: {', '.join(missing_voices)}", voice_counts
    
    if "No Voice Tag" in voice_counts:
        return False, "âŒ Found text without voice tags. All text must be assigned to a voice using [voice_name]", voice_counts
    
    return True, "âœ… All voices found and text properly tagged", voice_counts

def validate_multi_audiobook_input(text_content, voice_library_path, project_name):
    """Validate inputs for multi-voice audiobook creation"""
    issues = []
    
    if not project_name or not project_name.strip():
        issues.append("ðŸ“ Project name is required")
    
    if text_content and len(text_content.strip()) < 10:
        issues.append("ðŸ“ Text is too short (minimum 10 characters)")
    
    # Validate voice parsing
    is_valid, voice_message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
    
    if not is_valid:
        issues.append(voice_message)
    
    if issues:
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Please fix these issues:\n" + "\n".join(f"â€¢ {issue}" for issue in issues),
            "",
            gr.Audio(visible=False)
        )
    
    # Show voice breakdown
    voice_breakdown = "\n".join([f"ðŸŽ­ {voice}: {words} words" for voice, words in voice_counts.items()])
    chunks = chunk_multi_voice_segments(parse_multi_voice_text(text_content))
    total_words = sum(voice_counts.values())
    
    return (
        gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"âœ… Ready for multi-voice audiobook creation!\nðŸ“Š {total_words:,} total words â†’ {len(chunks)} chunks\nðŸ“ Project: {project_name.strip()}\n\n{voice_breakdown}",
        voice_breakdown,
        gr.Audio(visible=True)
    )

def create_multi_voice_audiobook(model, text_content, voice_library_path, project_name):
    """Create multi-voice audiobook from tagged text"""
    if not text_content or not project_name:
        return None, "âŒ Missing required fields"
    
    try:
        # Parse and validate the text
        is_valid, message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
        if not is_valid:
            return None, f"âŒ Text validation failed: {message}"
        
        # Get voice segments and chunk them
        segments = parse_multi_voice_text(text_content)
        chunks = chunk_multi_voice_segments(segments, max_words=50)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return None, "âŒ No text chunks to process"
        
        # Initialize model if needed
        if model is None:
            model = ChatterboxTTS.from_pretrained(DEVICE)
        
        audio_chunks = []
        chunk_info = []  # For saving metadata
        
        for i, (voice_name, chunk_text) in enumerate(chunks, 1):
            # Get voice configuration
            voice_config = get_voice_config(voice_library_path, voice_name)
            if not voice_config:
                return None, f"âŒ Could not load voice configuration for '{voice_name}'"
            
            if not voice_config['audio_file']:
                return None, f"âŒ No audio file found for voice '{voice_config['display_name']}'"
            
            # Update status (this would be shown in real implementation)
            chunk_words = len(chunk_text.split())
            status_msg = f"ðŸŽµ Processing chunk {i}/{total_chunks}\nðŸŽ­ Voice: {voice_config['display_name']} ({voice_name})\nðŸ“ Chunk {i}: {chunk_words} words\nðŸ“Š Progress: {i}/{total_chunks} chunks"
            
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
        assignment_summary = "\n".join([f"ðŸŽ­ [{char}] â†’ {voice_counts[char]}" for char in voice_counts.keys()])
        
        success_msg = f"âœ… Multi-voice audiobook created successfully!\nðŸ“Š {total_words:,} words in {total_chunks} chunks\nðŸŽ­ Characters: {len(voice_counts)}\nâ±ï¸ Duration: ~{duration_minutes} minutes\nðŸ“ Saved to: {project_dir}\nðŸŽµ Files: {len(saved_files)} audio chunks\n\nVoice Assignments:\n{assignment_summary}"
        
        return (model.sr, combined_audio), success_msg
        
    except Exception as e:
        error_msg = f"âŒ Error creating multi-voice audiobook: {str(e)}"
        return None, error_msg

def analyze_multi_voice_text(text_content, voice_library_path):
    """
    Analyze multi-voice text and return character breakdown with voice assignment interface
    """
    if not text_content or not text_content.strip():
        return "", {}, gr.Group(visible=False), "âŒ No text to analyze"
    
    # Parse the text to find voice references
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return "", {}, gr.Group(visible=False), "âŒ No voice tags found in text"
    
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
        breakdown_text = "âŒ Found text without voice tags:\n"
        breakdown_text += f"â€¢ No Voice Tag: {voice_counts['No Voice Tag']} words\n"
        breakdown_text += "\nAll text must be assigned to a voice using [voice_name] tags!"
        return breakdown_text, voice_counts, gr.Group(visible=False), "âŒ Text contains untagged content"
    
    breakdown_text = "âœ… Voice tags found:\n"
    for voice, words in voice_counts.items():
        breakdown_text += f"ðŸŽ­ [{voice}]: {words} words\n"
    
    return breakdown_text, voice_counts, gr.Group(visible=True), "âœ… Analysis complete - assign voices below"

def create_assignment_interface_with_dropdowns(voice_counts, voice_library_path):
    """
    Create actual Gradio dropdown components for each character
    Returns the components and character names for proper handling
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return [], [], "<div class='voice-status'>âŒ No valid characters found</div>"
    
    # Get available voices
    available_voices = get_voice_profiles(voice_library_path)
    
    if not available_voices:
        return [], [], "<div class='voice-status'>âŒ No voices available in library. Create voices first!</div>"
    
    # Create voice choices for dropdowns
    voice_choices = [("Select a voice...", None)]
    for voice in available_voices:
        display_text = f"ðŸŽ­ {voice['display_name']} ({voice['name']})"
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
    info_html = f"<div class='voice-status'>âœ… Found {len(character_names)} characters. Select voices for each character using the dropdowns below.</div>"
    
    return dropdown_components, character_names, info_html

def validate_dropdown_assignments(text_content, voice_library_path, project_name, voice_counts, character_names, *dropdown_values):
    """
    Validate voice assignments from dropdown values
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Invalid text or voice tags",
            {},
            gr.Audio(visible=False)
        )
    
    if not project_name or not project_name.strip():
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Project name is required",
            {},
            gr.Audio(visible=False)
        )
    
    if len(dropdown_values) != len(character_names):
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"âŒ Assignment mismatch: {len(character_names)} characters, {len(dropdown_values)} dropdown values",
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
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"âŒ Please assign voices for: {', '.join(missing_assignments)}",
            voice_assignments,
            gr.Audio(visible=False)
        )
    
    # All assignments valid
    total_words = sum(voice_counts.values())
    assignment_summary = "\n".join([f"ðŸŽ­ [{char}] â†’ {voice_assignments[char]}" for char in character_names])
    
    return (
        gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"âœ… All characters assigned!\nðŸ“Š {total_words:,} words total\nðŸ“ Project: {project_name.strip()}\n\nAssignments:\n{assignment_summary}",
        voice_assignments,
        gr.Audio(visible=True)
    )

def get_model_device_str(model_obj):
    """Safely get the device string ("cuda" or "cpu") from a model object."""
    if not model_obj or not hasattr(model_obj, 'device'):
        # print("âš ï¸ Model object is None or has no device attribute.")
        return None 
    
    device_attr = model_obj.device
    if isinstance(device_attr, torch.device):
        return device_attr.type
    elif isinstance(device_attr, str):
        if device_attr in ["cuda", "cpu"]:
            return device_attr
        else:
            print(f"âš ï¸ Unexpected string for model.device: {device_attr}")
            return None 
    else:
        print(f"âš ï¸ Unexpected type for model.device: {type(device_attr)}")
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
                print(f"âš ï¸ Skipping chunk with non-string text at index {chunk_idx}: {chunk_info}")
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
                print(f"âš ï¸ Filtering out suspected tag-only chunk {chunk_idx+1}/{original_chunk_count} for voice '{voice_name}': '{text}'")
            else:
                filtered_chunks.append(chunk_info)
        else:
            # Handle unexpected format
            print(f"âš ï¸ Unexpected chunk format at index {chunk_idx}: {chunk_info}")
            filtered_chunks.append(chunk_info)
            
    if len(filtered_chunks) < original_chunk_count:
        print(f"â„¹ï¸ Filtered {original_chunk_count - len(filtered_chunks)} problematic short chunk(s) out of {original_chunk_count}.")
    
    return filtered_chunks

def create_multi_voice_audiobook_with_assignments(
    model,
    text_content: str,
    voice_library_path: str,
    project_name: str,
    voice_assignments: dict,
    resume: bool = False,
    autosave_interval: int = 10
) -> tuple:
    """
    Create multi-voice audiobook using the voice assignments mapping, autosave every N chunks, and resume support.
    Args:
        model: TTS model
        text_content: Full text
        voice_library_path: Path to voice library
        project_name: Project name
        voice_assignments: Character to voice mapping
        resume: If True, resume from last saved chunk
        autosave_interval: Chunks per autosave (default 10)
    Returns:
        (sample_rate, combined_audio), status_message
    """
    import numpy as np
    import os
    import json
    import wave
    from typing import List

    if not text_content or not project_name or not voice_assignments:
        error_msg = "âŒ Missing required fields or voice assignments. Ensure text is entered, project name is set, and voices are assigned after analyzing text."
        return None, None, error_msg, None

    # Parse the text and map voices
    segments = parse_multi_voice_text(text_content)
    mapped_segments = []
    for character_name, text_segment in segments:
        if character_name in voice_assignments:
            actual_voice = voice_assignments[character_name]
            mapped_segments.append((actual_voice, text_segment))
        else:
            return None, None, f"âŒ No voice assignment found for character '{character_name}'", None

    initial_max_words = 30 if DEVICE == "cuda" else 40
    chunks = chunk_multi_voice_segments(mapped_segments, max_words=initial_max_words)
    chunks = _filter_problematic_short_chunks(chunks, voice_assignments)
    total_chunks = len(chunks)
    if not chunks:
        return None, None, "âŒ No text chunks to process", None

    # Project directory
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Resume logic: find already completed chunk files
    completed_chunks = set()
    chunk_filenames = []
    chunk_info = []
    for i, (voice_name, chunk_text) in enumerate(chunks):
        character_name = None
        for char_key, assigned_voice_val in voice_assignments.items():
            if assigned_voice_val == voice_name:
                character_name = char_key
                break
        character_name_file = character_name.replace(' ', '_') if character_name else voice_name
        filename = f"{safe_project_name}_{i+1:03d}_{character_name_file}.wav"
        chunk_filenames.append(filename)
        if os.path.exists(os.path.join(project_dir, filename)):
            completed_chunks.add(i)
        chunk_info.append({
            'chunk_num': i+1, 'voice_name': voice_name, 'character_name': character_name or voice_name,
            'voice_display': voice_name, 'text': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
            'word_count': len(chunk_text.split())
        })

    # If resuming, only process missing chunks
    start_idx = 0
    if resume and completed_chunks:
        for i in range(total_chunks):
            if i not in completed_chunks:
                start_idx = i
                break
        else:
            return None, None, "âœ… All chunks already completed. Nothing to resume.", None
    else:
        start_idx = 0

    # Initialize model if needed
    processing_model = model
    if processing_model is None:
        processing_model = ChatterboxTTS.from_pretrained(DEVICE)

    audio_chunks: List[np.ndarray] = []
    # For resume, load already completed audio
    for i in range(start_idx):
        fname = os.path.join(project_dir, chunk_filenames[i])
        with wave.open(fname, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            audio_chunks.append(audio_data)

    # Process missing chunks
    for i in range(start_idx, total_chunks):
        if i in completed_chunks:
            continue
        voice_name, chunk_text = chunks[i]
        try:
            voice_config = get_voice_config(voice_library_path, voice_name)
            if not voice_config:
                return None, None, f"âŒ Could not load voice config for '{voice_name}'", None
            if not voice_config['audio_file']:
                return None, None, f"âŒ No audio file for voice '{voice_config['display_name']}'", None
            if not os.path.exists(voice_config['audio_file']):
                return None, None, f"âŒ Audio file not found: {voice_config['audio_file']}", None
            wav = processing_model.generate(
                chunk_text, audio_prompt_path=voice_config['audio_file'],
                exaggeration=voice_config['exaggeration'], temperature=voice_config['temperature'],
                cfg_weight=voice_config['cfg_weight'])
            audio_np = wav.squeeze(0).cpu().numpy()
            audio_chunks.append(audio_np)
            # Save this chunk immediately
            fname = os.path.join(project_dir, chunk_filenames[i])
            with wave.open(fname, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(processing_model.sr)
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            del wav
            if get_model_device_str(processing_model) == 'cuda':
                torch.cuda.empty_cache()
        except Exception as chunk_error_outer:
            return None, None, f"âŒ Outer error processing chunk {i+1} (voice: {voice_name}): {str(chunk_error_outer)}", None
        # Autosave every N chunks
        if (i + 1) % autosave_interval == 0 or (i + 1) == total_chunks:
            # Save project metadata
            metadata_file = os.path.join(project_dir, "project_info.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'project_name': project_name, 'total_chunks': total_chunks,
                    'final_processing_mode': 'CPU' if DEVICE == 'cpu' else 'GPU',
                    'voice_assignments': voice_assignments, 'characters': list(voice_assignments.keys()),
                    'chunks': chunk_info
                }, f, indent=2)
    # Combine all audio for preview (just concatenate)
    combined_audio = np.concatenate(audio_chunks)
    total_words = sum(len(chunk[1].split()) for chunk in chunks)
    duration_minutes = len(combined_audio) // processing_model.sr // 60
    assignment_summary = "\n".join([f"ðŸŽ­ [{char}] â†’ {assigned_voice}" for char, assigned_voice in voice_assignments.items()])
    success_msg = (f"âœ… Multi-voice audiobook created successfully!\n"
                   f"ðŸ“Š {total_words:,} words in {total_chunks} chunks\n"
                   f"ðŸŽ­ Characters: {len(voice_assignments)}\n"
                   f"â±ï¸ Duration: ~{duration_minutes} minutes\n"
                   f"ðŸ“ Saved to: {project_dir}\n"
                   f"ðŸŽµ Files: {len(audio_chunks)} audio chunks\n"
                   f"\nVoice Assignments:\n{assignment_summary}")
    return (processing_model.sr, combined_audio), None, success_msg, None

def handle_multi_voice_analysis(text_content, voice_library_path):
    """
    Analyze multi-voice text and populate character dropdowns
    Returns updated dropdown components
    """
    if not text_content or not text_content.strip():
        # Reset all dropdowns to hidden
        empty_dropdown = gr.Dropdown(choices=[("No character found", None)], visible=False, interactive=False)
        return (
            "<div class='voice-status'>âŒ No text to analyze</div>",
            {},
            [],
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            gr.Button("ðŸ” Validate Voice Assignments", interactive=False),
            "âŒ Add text first"
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
            gr.Button("ðŸ” Validate Voice Assignments", interactive=False),
            status
        )
    
    # Get available voices for dropdown choices
    available_voices = get_voice_profiles(voice_library_path)
    if not available_voices:
        empty_dropdown = gr.Dropdown(choices=[("No voices available", None)], visible=False, interactive=False)
        return (
            "<div class='voice-status'>âŒ No voices available in library. Create voices first!</div>",
            voice_counts,
            [],
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            gr.Button("ðŸ” Validate Voice Assignments", interactive=False),
            "âŒ No voices in library"
        )
    
    # Create voice choices for dropdowns
    voice_choices = [("Select a voice...", None)]
    for voice in available_voices:
        display_text = f"ðŸŽ­ {voice['display_name']} ({voice['name']})"
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
    summary_msg = f"âœ… Found {len(character_names)} characters with {total_words:,} total words\n" + breakdown_text
    
    return (
        summary_msg,
        voice_counts,
        character_names,
        dropdown_components[0], dropdown_components[1], dropdown_components[2],
        dropdown_components[3], dropdown_components[4], dropdown_components[5],
        gr.Button("ðŸ” Validate Voice Assignments", interactive=True),
        "âœ… Analysis complete - assign voices above"
    )

def validate_dropdown_voice_assignments(text_content, voice_library_path, project_name, voice_counts, character_names, 
                                       char1_voice, char2_voice, char3_voice, char4_voice, char5_voice, char6_voice):
    """
    Validate voice assignments from character dropdowns
    """
    if not voice_counts or "No Voice Tag" in voice_counts:
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Invalid text or voice tags",
            {},
            gr.Audio(visible=False)
        )
    
    if not project_name or not project_name.strip():
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ Project name is required",
            {},
            gr.Audio(visible=False)
        )
    
    if not character_names:
        return (
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            "âŒ No characters found in text",
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
            gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
            f"âŒ Please assign voices for: {', '.join(missing_assignments)}",
            voice_assignments,
            gr.Audio(visible=False)
        )
    
    # All assignments valid
    total_words = sum(voice_counts.values())
    assignment_summary = "\n".join([f"ðŸŽ­ [{char}] â†’ {voice_assignments[char]}" for char in character_names])
    
    return (
        gr.Button("ðŸŽµ Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
        f"âœ… All characters assigned!\nðŸ“Š {total_words:,} words total\nðŸ“ Project: {project_name.strip()}\n\nAssignments:\n{assignment_summary}",
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
        print(f"âš ï¸ Warning: Could not save project metadata: {str(e)}")

def load_project_metadata(project_dir: str) -> dict:
    """Load project metadata from directory"""
    metadata_file = os.path.join(project_dir, "project_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"âš ï¸ Warning: Could not load project metadata: {str(e)}")
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
        # Return the same choices for the two remaining dropdowns
        return (
            gr.Dropdown(choices=choices, value=None),
            gr.Dropdown(choices=choices, value=None)
        )
    except Exception as e:
        print(f"Error refreshing project dropdowns: {str(e)}")
        error_choices = [("Error loading projects", None)]
        return (
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
                display_name = f"ðŸ“ {project['name']} ({project_type}) - {project['audio_count']} files"
            else:
                display_name = f"ðŸ“ {project['name']} (no metadata) - {project['audio_count']} files"
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
        return "", "", "", None, f"âŒ Project '{project_name}' not found"
    
    metadata = project.get('metadata')
    if not metadata:
        # Legacy project without metadata
        audio_files = project['audio_files']
        if audio_files:
            # Load first audio file for waveform
            first_audio = os.path.join(project['path'], audio_files[0])
            return ("", 
                    "âš ï¸ Legacy project - no original text available", 
                    "âš ï¸ Voice information not available",
                    first_audio,
                    f"âš ï¸ Legacy project loaded. Found {len(audio_files)} audio files but no metadata.")
        else:
            return "", "", "", None, f"âŒ No audio files found in project '{project_name}'"
    
    # Project with metadata
    text_content = metadata.get('text_content', '')
    voice_info = metadata.get('voice_info', {})
    
    # Format voice info display
    if metadata.get('project_type') == 'multi_voice':
        voice_display = "ðŸŽ­ Multi-voice project:\n"
        for voice_name, info in voice_info.items():
            voice_display += f"  â€¢ {voice_name}: {info.get('display_name', voice_name)}\n"
    else:
        voice_display = f"ðŸŽ¤ Single voice: {voice_info.get('display_name', 'Unknown')}"
    
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
    
    status_msg = f"âœ… Project loaded successfully!\nðŸ“… Created: {date_str}\nðŸŽµ Audio files: {len(audio_files)}\nðŸ“ Text length: {len(text_content)} characters"
    
    return text_content, voice_display, project_name, first_audio, status_msg

def create_continuous_playback_audio(project_name: str) -> tuple:
    """Create a single continuous audio file from all project chunks for Listen & Edit mode"""
    if not project_name:
        return None, "âŒ No project selected"
    
    chunks = get_project_chunks(project_name)
    if not chunks:
        return None, f"âŒ No audio chunks found in project '{project_name}'"
    
    try:
        combined_audio = []
        sample_rate = 24000  # Default sample rate
        chunk_timings = []  # Store start/end times for each chunk
        current_time = 0.0
        
        # Sort chunks by chunk number to ensure correct order
        def extract_chunk_number(chunk_info):
            return chunk_info.get('chunk_num', 0)
        
        chunks_sorted = sorted(chunks, key=extract_chunk_number)
        
        # Load and combine all audio files in order
        for chunk in chunks_sorted:
            audio_file = chunk['audio_file']
            
            if os.path.exists(audio_file):
                try:
                    with wave.open(audio_file, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                        
                        # Record timing info for this chunk
                        chunk_duration = len(audio_data) / sample_rate
                        chunk_timings.append({
                            'chunk_num': chunk['chunk_num'],
                            'start_time': current_time,
                            'end_time': current_time + chunk_duration,
                            'text': chunk.get('text', ''),
                            'audio_file': audio_file
                        })
                        
                        combined_audio.append(audio_data)
                        current_time += chunk_duration
                        
                except Exception as e:
                    print(f"âš ï¸ Error reading chunk {chunk['chunk_num']}: {str(e)}")
            else:
                print(f"âš ï¸ Warning: Audio file not found: {audio_file}")
        
        if not combined_audio:
            return None, f"âŒ No valid audio files found in project '{project_name}'"
        
        # Concatenate all audio
        full_audio = np.concatenate(combined_audio)
        
        # Create temporary combined file
        temp_filename = f"temp_continuous_{project_name}_{int(time.time())}.wav"
        temp_file_path = os.path.join("audiobook_projects", project_name, temp_filename)
        
        # Save as WAV file
        with wave.open(temp_file_path, 'wb') as output_wav:
            output_wav.setnchannels(1)  # Mono
            output_wav.setsampwidth(2)  # 16-bit
            output_wav.setframerate(sample_rate)
            audio_int16 = (full_audio * 32767).astype(np.int16)
            output_wav.writeframes(audio_int16.tobytes())
        
        # Calculate total duration
        total_duration = len(full_audio) / sample_rate
        duration_minutes = int(total_duration // 60)
        duration_seconds = int(total_duration % 60)
        
        success_msg = f"âœ… Continuous audio created: {duration_minutes}:{duration_seconds:02d} ({len(chunks_sorted)} chunks)"
        
        # Return audio file path and timing data
        return (temp_file_path, chunk_timings), success_msg
        
    except Exception as e:
        return None, f"âŒ Error creating continuous audio: {str(e)}"

def get_current_chunk_from_time(chunk_timings: list, current_time: float) -> dict:
    """Get the current chunk information based on playback time"""
    if not chunk_timings or current_time is None:
        return {}
    
    for chunk_timing in chunk_timings:
        if chunk_timing['start_time'] <= current_time < chunk_timing['end_time']:
            return chunk_timing
    
    # If we're past the end, return the last chunk
    if chunk_timings and current_time >= chunk_timings[-1]['end_time']:
        return chunk_timings[-1]
    
    # If we're before the start, return the first chunk
    if chunk_timings and current_time < chunk_timings[0]['start_time']:
        return chunk_timings[0]
    
    return {}

def regenerate_chunk_and_update_continuous(model, project_name: str, chunk_num: int, voice_library_path: str, 
                                         custom_text: str = None) -> tuple:
    """Regenerate a chunk and update the continuous audio file"""
    # First regenerate the chunk
    result = regenerate_single_chunk(model, project_name, chunk_num, voice_library_path, custom_text)
    
    if result[0] is None:  # Error occurred
        return None, result[1], None
    
    temp_file_path, status_msg = result
    
    # Accept the regenerated chunk immediately (auto-accept for continuous mode)
    chunks = get_project_chunks(project_name)
    accept_result = accept_regenerated_chunk(project_name, chunk_num, temp_file_path, chunks)
    
    if "âœ…" not in accept_result[0]:  # Error in acceptance
        return None, f"âŒ Regeneration succeeded but failed to update: {accept_result[0]}", None
    
    # Recreate the continuous audio with the updated chunk
    continuous_result = create_continuous_playback_audio(project_name)
    
    if continuous_result[0] is None:  # Error creating continuous audio
        return None, f"âœ… Chunk regenerated but failed to update continuous audio: {continuous_result[1]}", None
    
    continuous_data, continuous_msg = continuous_result
    
    return continuous_data, f"âœ… Chunk {chunk_num} regenerated and continuous audio updated!", status_msg

def cleanup_temp_continuous_files(project_name: str) -> None:
    """Clean up temporary continuous audio files"""
    if not project_name:
        return
    
    project_path = os.path.join("audiobook_projects", project_name)
    if not os.path.exists(project_path):
        return
    
    try:
        for file in os.listdir(project_path):
            if file.startswith("temp_continuous_") and file.endswith('.wav'):
                file_path = os.path.join(project_path, file)
                try:
                    os.remove(file_path)
                    print(f"ðŸ—‘ï¸ Cleaned up: {file}")
                except Exception as e:
                    print(f"âš ï¸ Could not remove {file}: {str(e)}")
    except Exception as e:
        print(f"âš ï¸ Error cleaning temp files: {str(e)}")

def regenerate_project_sample(model, project_name: str, voice_library_path: str, sample_text: str = None) -> tuple:
    """Regenerate a sample from an existing project"""
    if not project_name:
        return None, "âŒ No project selected"
    
    projects = get_existing_projects()
    project = next((p for p in projects if p['name'] == project_name), None)
    
    if not project:
        return None, f"âŒ Project '{project_name}' not found"
    
    metadata = project.get('metadata')
    if not metadata:
        return None, "âŒ Cannot regenerate - project has no metadata (legacy project)"
    
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
                return None, "âŒ No text content available for regeneration"
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
                return None, "âŒ Voice configuration not available"
            
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
            status_msg = f"âœ… Sample regenerated successfully!\nðŸŽ­ Voice: {voice_config.get('display_name', 'Unknown')}\nðŸ“ Text: {text_to_regenerate[:100]}..."
            
            return (model.sr, audio_output), status_msg
            
        else:
            # Multi-voice regeneration - use first voice
            first_voice = list(voice_info.keys())[0] if voice_info else None
            if not first_voice:
                return None, "âŒ No voice information available for multi-voice project"
            
            voice_config = voice_info[first_voice]
            if not voice_config or not voice_config.get('audio_file'):
                return None, f"âŒ Voice configuration not available for '{first_voice}'"
            
            wav = generate_with_retry(
                model,
                text_to_regenerate,
                voice_config['audio_file'],
                voice_config.get('exaggeration', 0.5),
                voice_config.get('temperature', 0.8),
                voice_config.get('cfg_weight', 0.5)
            )
            
            audio_output = wav.squeeze(0).cpu().numpy()
            status_msg = f"âœ… Sample regenerated successfully!\nðŸŽ­ Voice: {voice_config.get('display_name', first_voice)}\nðŸ“ Text: {text_to_regenerate[:100]}..."
            
            return (model.sr, audio_output), status_msg
            
    except Exception as e:
        clear_gpu_memory()
        return None, f"âŒ Error regenerating sample: {str(e)}"

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
    
    # Sort by chunk number (numerically, not lexicographically)
    def extract_chunk_num_from_filename(filename: str) -> int:
        import re
        match = re.search(r'_(\d{3})\.wav$', filename)
        if not match:
            match = re.search(r'_(\d+)\.wav$', filename)
        if match:
            return int(match.group(1))
        return 0
    chunk_files = sorted(chunk_files, key=extract_chunk_num_from_filename)
    
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
                    print(f"âš ï¸ Warning: Could not load voice assignments: {str(e)}")
        
        for i, audio_file in enumerate(chunk_files):
            # Extract the actual chunk number from the filename instead of using the enumerate index
            actual_chunk_num = extract_chunk_num_from_filename(audio_file)
            
            chunk_info = {
                'chunk_num': actual_chunk_num,  # Use actual chunk number from filename
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
            # Extract the actual chunk number from the filename instead of using the enumerate index
            actual_chunk_num = extract_chunk_num_from_filename(audio_file)
            
            chunk_info = {
                'chunk_num': actual_chunk_num,  # Use actual chunk number from filename
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
        return None, f"âŒ Invalid chunk number {chunk_num}"
    
    chunk = chunks[chunk_num - 1]  # Convert to 0-based index
    
    if not chunk['has_metadata']:
        return None, "âŒ Cannot regenerate - legacy project has no voice metadata"
    
    # Use custom text or original text
    text_to_regenerate = custom_text.strip() if custom_text and custom_text.strip() else chunk['text']
    
    if not text_to_regenerate:
        return None, "âŒ No text available for regeneration"
    
    try:
        project_type = chunk['project_type']
        
        if project_type == 'single_voice':
            # Single voice project
            voice_config = chunk['voice_info']
            if not voice_config or not voice_config.get('audio_file'):
                return None, "âŒ Voice configuration not available"
            
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
                return None, f"âŒ Voice configuration not found for character '{character_name}' (assigned voice: '{assigned_voice}')"
            
            if not voice_config.get('audio_file'):
                return None, f"âŒ Audio file not found for character '{character_name}' (assigned voice: '{assigned_voice}')"
            
            # Check if audio file actually exists
            audio_file_path = voice_config.get('audio_file')
            if not os.path.exists(audio_file_path):
                return None, f"âŒ Audio file does not exist: {audio_file_path}"
            
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
            return None, f"âŒ Unknown project type: {project_type}"
        
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
        
        status_msg = f"âœ… Chunk {chunk_num} regenerated successfully!\nðŸŽ­ Voice: {voice_display}\nðŸ“ Text: {text_to_regenerate[:100]}{'...' if len(text_to_regenerate) > 100 else ''}\nðŸ’¾ Temp file: {temp_filename}"
        
        # Return the temp file path instead of the audio tuple
        return temp_file_path, status_msg
        
    except Exception as e:
        clear_gpu_memory()
        return None, f"âŒ Error regenerating chunk {chunk_num}: {str(e)}"

def load_project_chunks_for_interface(project_name: str, page_num: int = 1, chunks_per_page: int = 50) -> tuple:
    """Load project chunks and return data for interface components with pagination support"""
    if not project_name:
        # Hide all chunk interfaces
        empty_returns = []
        for i in range(MAX_CHUNKS_FOR_INTERFACE):
            empty_returns.extend([
                gr.Group(visible=False),  # group
                None,  # audio
                "",  # text
                "<div class='voice-status'>No chunk loaded</div>",  # voice_info
                gr.Button(f"ðŸŽµ Regenerate Chunk {i+1}", interactive=False),  # button
                gr.Audio(visible=False),  # regenerated_audio
                "<div class='voice-status'>No chunk</div>"  # status
            ])
        
        return (
            "<div class='voice-status'>ðŸ“ Select a project first</div>",  # project_info_summary
            [],  # current_project_chunks (all chunks, not just displayed)
            project_name,  # current_project_name
            "<div class='audiobook-status'>ðŸ“ No project loaded</div>",  # project_status
            gr.Button("ðŸ“¥ Download Full Project Audio", variant="primary", size="lg", interactive=False),  # download_project_btn
            "<div class='voice-status'>ðŸ“ Load a project first to enable download</div>",  # download_status
            1,  # current_page_state
            1,  # total_pages_state
            gr.Button("â¬…ï¸ Previous Page", size="sm", interactive=False),  # prev_page_btn
            gr.Button("âž¡ï¸ Next Page", size="sm", interactive=False),  # next_page_btn
            "<div class='voice-status'>ðŸ“„ No project loaded</div>",  # page_info
            *empty_returns
        )
    
    all_chunks = get_project_chunks(project_name)
    
    if not all_chunks:
        # Hide all chunk interfaces
        empty_returns = []
        for i in range(MAX_CHUNKS_FOR_INTERFACE):
            empty_returns.extend([
                gr.Group(visible=False),
                None,
                "",
                "<div class='voice-status'>No chunk found</div>",
                gr.Button(f"ðŸŽµ Regenerate Chunk {i+1}", interactive=False),
                gr.Audio(visible=False),
                "<div class='voice-status'>No chunk</div>"
            ])
        
        return (
            f"<div class='voice-status'>âŒ No chunks found in project '{project_name}'</div>",
            [],
            project_name,
            f"âŒ No audio files found in project '{project_name}'",
            gr.Button("ðŸ“¥ Download Full Project Audio", variant="primary", size="lg", interactive=False),
            f"âŒ No audio files found in project '{project_name}'",
            1,  # current_page_state
            1,  # total_pages_state
            gr.Button("â¬…ï¸ Previous Page", size="sm", interactive=False),  # prev_page_btn
            gr.Button("âž¡ï¸ Next Page", size="sm", interactive=False),  # next_page_btn
            f"âŒ No chunks found in project '{project_name}'",  # page_info
            *empty_returns
        )
    
    # Calculate pagination
    total_chunks = len(all_chunks)
    total_pages = max(1, (total_chunks + chunks_per_page - 1) // chunks_per_page)  # Ceiling division
    page_num = max(1, min(page_num, total_pages))  # Clamp page number
    
    start_idx = (page_num - 1) * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, total_chunks)
    chunks_for_current_page = all_chunks[start_idx:end_idx]
    
    # Create project summary
    project_info = f"""
    <div class='audiobook-status'>
        ðŸ“ <strong>Project:</strong> {project_name}<br/>
        ðŸŽµ <strong>Total Chunks:</strong> {total_chunks}<br/>
        ðŸ“„ <strong>Showing:</strong> {len(chunks_for_current_page)} chunks (Page {page_num} of {total_pages})<br/>
        ðŸ“ <strong>Type:</strong> {all_chunks[0]['project_type'].replace('_', ' ').title()}<br/>
        âœ… <strong>Metadata:</strong> {'Available' if all_chunks[0]['has_metadata'] else 'Legacy Project'}
    </div>
    """
    
    status_msg = f"âœ… Loaded page {page_num} of {total_pages} ({len(chunks_for_current_page)} chunks shown, {total_chunks} total) from project '{project_name}'"
    
    # Page info
    page_info_html = f"<div class='voice-status'>ðŸ“„ Page {page_num} of {total_pages} | Chunks {start_idx + 1}-{end_idx} of {total_chunks}</div>"
    
    # Navigation buttons
    prev_btn = gr.Button("â¬…ï¸ Previous Page", size="sm", interactive=(page_num > 1))
    next_btn = gr.Button("âž¡ï¸ Next Page", size="sm", interactive=(page_num < total_pages))
    
    # Prepare interface updates
    interface_updates = []
    
    for i in range(MAX_CHUNKS_FOR_INTERFACE):
        if i < len(chunks_for_current_page):
            chunk = chunks_for_current_page[i]
            
            # Voice info display
            if chunk['project_type'] == 'multi_voice':
                character_name = chunk.get('character', 'unknown')
                assigned_voice = chunk.get('assigned_voice', 'unknown')
                voice_config = chunk.get('voice_config', {})
                voice_display_name = voice_config.get('display_name', assigned_voice)
                
                voice_info_html = f"<div class='voice-status'>ðŸŽ­ Character: {character_name}<br/>ðŸŽ¤ Voice: {voice_display_name}</div>"
            elif chunk['project_type'] == 'single_voice':
                voice_name = chunk['voice_info'].get('display_name', 'Unknown') if chunk.get('voice_info') else 'Unknown'
                voice_info_html = f"<div class='voice-status'>ðŸŽ¤ Voice: {voice_name}</div>"
            else:
                voice_info_html = "<div class='voice-status'>âš ï¸ Legacy project - limited info</div>"
            
            # Status message
            chunk_status = f"<div class='voice-status'>ðŸ“„ Chunk {chunk['chunk_num']} ready to regenerate</div>"
            
            interface_updates.extend([
                gr.Group(visible=True),  # group
                chunk['audio_file'],  # audio
                chunk['text'],  # text
                voice_info_html,  # voice_info
                gr.Button(f"ðŸŽµ Regenerate Chunk {chunk['chunk_num']}", interactive=chunk['has_metadata']),  # button
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
                gr.Button(f"ðŸŽµ Regenerate Chunk {i+1}", interactive=False),
                gr.Audio(visible=False),
                "<div class='voice-status'>No chunk</div>"
            ])
    
    return (
        project_info,  # project_info_summary
        all_chunks,  # current_project_chunks (ALL chunks, not just displayed)
        project_name,  # current_project_name
        status_msg,  # project_status
        gr.Button("ðŸ“¥ Download Full Project Audio", variant="primary", size="lg", interactive=bool(all_chunks)),  # download_project_btn
        f"<div class='voice-status'>âœ… Ready to download complete project audio ({total_chunks} chunks)</div>" if all_chunks else "<div class='voice-status'>ðŸ“ Load a project first to enable download</div>",  # download_status
        page_num,  # current_page_state
        total_pages,  # total_pages_state
        prev_btn,  # prev_page_btn
        next_btn,  # next_page_btn
        page_info_html,  # page_info
        *interface_updates
    )

def combine_project_audio_chunks(project_name: str, output_format: str = "wav") -> tuple:
    """Combine all audio chunks from a project into a single downloadable file"""
    if not project_name:
        return None, "âŒ No project selected"
    
    chunks = get_project_chunks(project_name)
    
    if not chunks:
        return None, f"âŒ No audio chunks found in project '{project_name}'"
    
    try:
        combined_audio = []
        sample_rate = 24000  # Default sample rate
        total_samples_processed = 0
        
        # Sort chunks by chunk number to ensure correct order (not alphabetical)
        def extract_chunk_number(chunk_info):
            """Extract chunk number from chunk info for proper numerical sorting"""
            try:
                # First try to get chunk_num directly from the chunk info
                chunk_num = chunk_info.get('chunk_num')
                if chunk_num is not None:
                    return int(chunk_num)  # Ensure it's an integer
            except (ValueError, TypeError):
                pass
            
            # Fallback: try to extract from filename
            try:
                filename = chunk_info.get('audio_filename', '') or chunk_info.get('audio_file', '')
                if filename:
                    import re
                    # Look for patterns like "_123.wav" or "_chunk_123.wav"
                    match = re.search(r'_(\d+)\.wav$', filename)
                    if match:
                        return int(match.group(1))
                    
                    # Try other patterns like "projectname_123.wav"
                    match = re.search(r'(\d+)\.wav$', filename)
                    if match:
                        return int(match.group(1))
            except (ValueError, TypeError, AttributeError):
                pass
            
            # Last resort: return 0 (should sort first)
            print(f"[WARNING] Could not extract chunk number from: {chunk_info}")
            return 0
        
        chunks_sorted = sorted(chunks, key=extract_chunk_number)
        
        print(f"[INFO] Combining {len(chunks_sorted)} chunks for project '{project_name}'")
        chunk_numbers = [extract_chunk_number(c) for c in chunks_sorted[:5]]
        print(f"[DEBUG] First few chunks: {chunk_numbers}")
        chunk_numbers = [extract_chunk_number(c) for c in chunks_sorted[-5:]]
        print(f"[DEBUG] Last few chunks: {chunk_numbers}")
        
        # Process chunks in batches to manage memory better
        batch_size = 50
        for batch_start in range(0, len(chunks_sorted), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks_sorted))
            batch_chunks = chunks_sorted[batch_start:batch_end]
            
            print(f"[INFO] Processing batch {batch_start//batch_size + 1}/{(len(chunks_sorted) + batch_size - 1)//batch_size} (chunks {batch_start+1}-{batch_end})")
            
            for chunk_info in batch_chunks:
                chunk_path = chunk_info.get('audio_file')  # Use 'audio_file' instead of 'audio_path'
                chunk_num = extract_chunk_number(chunk_info)
                
                if not chunk_path or not os.path.exists(chunk_path):
                    print(f"âš ï¸ Warning: Chunk {chunk_num} file not found: {chunk_path}")
                    continue
                
                try:
                    with wave.open(chunk_path, 'rb') as wav_file:
                        chunk_sample_rate = wav_file.getframerate()
                        chunk_frames = wav_file.getnframes()
                        chunk_audio_data = wav_file.readframes(chunk_frames)
                        
                        # Convert to numpy array (16-bit to float32 for better precision)
                        chunk_audio_array = np.frombuffer(chunk_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if sample_rate != chunk_sample_rate:
                            print(f"âš ï¸ Warning: Sample rate mismatch in chunk {chunk_num}: {chunk_sample_rate} vs {sample_rate}")
                            sample_rate = chunk_sample_rate  # Use the chunk's sample rate
                        
                        combined_audio.append(chunk_audio_array)
                        total_samples_processed += len(chunk_audio_array)
                        
                        if chunk_num <= 5 or chunk_num % 100 == 0 or chunk_num > len(chunks_sorted) - 5:
                            print(f"âœ… Added chunk {chunk_num}: {os.path.basename(chunk_path)} ({len(chunk_audio_array)} samples)")
                        
                except Exception as e:
                    print(f"âŒ Error reading chunk {chunk_num} ({chunk_path}): {e}")
                    continue
        
        if not combined_audio:
            return None, "âŒ No valid audio chunks found to combine"
        
        print(f"[INFO] Concatenating {len(combined_audio)} chunks...")
        print(f"[INFO] Total samples to process: {total_samples_processed}")
        
        # Concatenate all audio using numpy for efficiency
        final_audio = np.concatenate(combined_audio, axis=0)
        
        print(f"[INFO] Final audio array shape: {final_audio.shape}")
        print(f"[INFO] Final audio duration: {len(final_audio) / sample_rate / 60:.2f} minutes")
        
        # Convert back to int16 for WAV format
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
        
        # Create output filename
        output_filename = f"{project_name}_complete.{output_format}"
        output_path = os.path.join("audiobook_projects", project_name, output_filename)
        
        # Save the combined audio file with proper WAV encoding
        print(f"[INFO] Saving combined audio to: {output_path}")
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(final_audio_int16.tobytes())
        
        # Verify the saved file
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            # Check the saved file duration
            with wave.open(output_path, 'rb') as verify_wav:
                saved_frames = verify_wav.getnframes()
                saved_rate = verify_wav.getframerate()
                saved_duration_minutes = saved_frames / saved_rate / 60
            
            print(f"[INFO] Saved file size: {file_size_mb:.2f} MB")
            print(f"[INFO] Saved file duration: {saved_duration_minutes:.2f} minutes")
            
            if saved_duration_minutes < (len(final_audio) / sample_rate / 60 * 0.95):  # Allow 5% tolerance
                print(f"âš ï¸ WARNING: Saved file duration ({saved_duration_minutes:.2f} min) is significantly shorter than expected ({len(final_audio) / sample_rate / 60:.2f} min)")
        
        # Calculate total duration
        total_duration_seconds = len(final_audio) / sample_rate
        duration_hours = int(total_duration_seconds // 3600)
        duration_minutes = int((total_duration_seconds % 3600) // 60)
        
        success_message = (
            f"âœ… Combined {len(chunks_sorted)} chunks successfully! "
            f"ðŸŽµ Total duration: {duration_hours}:{duration_minutes:02d} "
            f"ðŸ“ File: {output_filename} "
            f"ðŸ”„ Fresh combination of current chunk files"
        )
        
        return output_path, success_message
        
    except Exception as e:
        error_msg = f"âŒ Error combining audio chunks: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return None, error_msg

def load_previous_project_audio(project_name: str) -> tuple:
    """Load a previous project's combined audio for download in creation tabs"""
    if not project_name:
        return None, None, "ðŸ“ Select a project to load its audio"
    
    # Check if combined file already exists
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
    combined_file = os.path.join("audiobook_projects", project_name, f"{safe_project_name}_complete.wav")
    
    if os.path.exists(combined_file):
        # File already exists, load it
        return combined_file, combined_file, f"âœ… Loaded existing combined audio for '{project_name}'"
    else:
        # Create combined file
        audio_path, status = combine_project_audio_chunks(project_name)
        return audio_path, audio_path, status

def save_trimmed_audio(audio_data, original_file_path: str, chunk_num: int) -> tuple:
    """Save trimmed audio data to replace the original file"""
    if not audio_data or not original_file_path:
        return "âŒ No audio data to save", None
    
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
                return f"âŒ Trimmed audio file not found: {audio_data}", None
                
        elif hasattr(audio_data, 'name'):  # Gradio file object
            # Handle Gradio uploaded file
            print(f"[DEBUG] Gradio file object: {audio_data.name}")
            if os.path.exists(audio_data.name):
                shutil.copy2(audio_data.name, original_file_path)
                audio_saved = True
                print(f"[DEBUG] Copied from Gradio file: {audio_data.name}")
            else:
                return f"âŒ Gradio file not found: {audio_data.name}", None
                
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
                    return f"âŒ Cannot process audio data type: {type(audio_data)}", None
            except Exception as e:
                print(f"[DEBUG] Failed to process as raw audio: {str(e)}")
                return f"âŒ Cannot process audio data: {str(e)}", None
        
        if audio_saved:
            status_msg = f"âœ… Chunk {chunk_num} trimmed and saved!\nðŸ’¾ Original backed up as: {os.path.basename(backup_file)}\nðŸŽµ Audio file updated successfully"
            print(f"[DEBUG] Successfully saved trimmed audio for chunk {chunk_num}")
            return status_msg, original_file_path
        else:
            return f"âŒ Failed to save trimmed audio for chunk {chunk_num}", None
            
    except Exception as e:
        print(f"[DEBUG] Exception in save_trimmed_audio: {str(e)}")
        return f"âŒ Error saving trimmed audio for chunk {chunk_num}: {str(e)}", None

def accept_regenerated_chunk(project_name: str, actual_chunk_num_to_accept: int, regenerated_audio_path: str, current_project_chunks_list: list) -> tuple:
    """Accept the regenerated chunk by replacing the original audio file and deleting the temp file."""
    if not project_name or not regenerated_audio_path:
        return "âŒ No regenerated audio to accept", None
    
    try:
        # We already have the correct actual_chunk_num_to_accept and the full list of chunks
        if actual_chunk_num_to_accept < 1 or actual_chunk_num_to_accept > len(current_project_chunks_list):
            return f"âŒ Invalid actual chunk number {actual_chunk_num_to_accept}", None
        
        # Find the specific chunk_info using the actual_chunk_num_to_accept
        # This assumes current_project_chunks_list is sorted and chunk_num is 1-based and matches index+1
        # More robust: find it by matching 'chunk_num' field
        chunk_info_to_update = next((c for c in current_project_chunks_list if c['chunk_num'] == actual_chunk_num_to_accept), None)
        
        if not chunk_info_to_update:
            return f"âŒ Could not find info for actual chunk {actual_chunk_num_to_accept} in project data.", None
            
        original_audio_file = chunk_info_to_update['audio_file']
        
        # Check if temp file exists
        if not os.path.exists(regenerated_audio_path):
            return f"âŒ Regenerated audio file not found: {regenerated_audio_path}", None
        
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
                # Match temp_regenerated_chunk_ACTUALCHUNKNUM_timestamp.wav
                if file.startswith(f"temp_regenerated_chunk_{actual_chunk_num_to_accept}_") and file.endswith('.wav'):
                    temp_path = os.path.join(project_dir, file)
                    try:
                        os.remove(temp_path)
                        temp_files.append(file)
                        print(f"ðŸ—‘ï¸ Cleaned up temp file: {file}")
                    except:
                        pass  # Ignore errors when cleaning up
        except Exception as e:
            print(f"âš ï¸ Warning during temp file cleanup: {str(e)}")
        
        status_msg = f"âœ… Chunk {actual_chunk_num_to_accept} regeneration accepted!\nðŸ’¾ Original backed up as: {os.path.basename(backup_file)}\nðŸ—‘ï¸ Cleaned up {len(temp_files)} temporary file(s)"
        
        # Return both status message and the path to the NEW audio file (for interface update)
        return status_msg, original_audio_file
        
    except Exception as e:
        return f"âŒ Error accepting chunk {actual_chunk_num_to_accept}: {str(e)}", None

def decline_regenerated_chunk(actual_chunk_num_to_decline: int, regenerated_audio_path: str = None) -> tuple:
    """Decline the regenerated chunk and clean up the temporary file."""
    
    actual_file_path = None
    
    if regenerated_audio_path:
        if isinstance(regenerated_audio_path, tuple):
            print(f"âš ï¸ Warning: Received tuple instead of file path for chunk {actual_chunk_num_to_decline} decline")
            actual_file_path = None
        elif isinstance(regenerated_audio_path, str):
            actual_file_path = regenerated_audio_path
        else:
            print(f"âš ï¸ Warning: Unexpected type for regenerated_audio_path: {type(regenerated_audio_path)}")
            actual_file_path = None
    
    if actual_file_path and os.path.exists(actual_file_path):
        try:
            os.remove(actual_file_path)
            print(f"ðŸ—‘ï¸ Cleaned up declined regeneration for chunk {actual_chunk_num_to_decline}: {os.path.basename(actual_file_path)}")
        except Exception as e:
            print(f"âš ï¸ Warning: Could not clean up temp file for chunk {actual_chunk_num_to_decline}: {str(e)}")
    
    return (
        gr.Audio(visible=False),  # Hide regenerated audio
        gr.Row(visible=False),    # Hide accept/decline buttons
        f"âŒ Chunk {actual_chunk_num_to_decline} regeneration declined. Keeping original audio."
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
        
        print(f"ðŸ”„ Complete refresh: Found {len(projects)} projects")
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
        return "âŒ No project name provided"
    
    try:
        project_dir = os.path.join("audiobook_projects", project_name)
        if not os.path.exists(project_dir):
            return f"âŒ Project directory not found: {project_dir}"
        
        temp_files_removed = 0
        temp_patterns = ['temp_regenerated_', '_backup_original_']
        
        for file in os.listdir(project_dir):
            if any(pattern in file for pattern in temp_patterns) and file.endswith('.wav'):
                file_path = os.path.join(project_dir, file)
                try:
                    os.remove(file_path)
                    temp_files_removed += 1
                    print(f"ðŸ—‘ï¸ Removed temp file: {file}")
                except Exception as e:
                    print(f"âš ï¸ Could not remove {file}: {str(e)}")
        
        if temp_files_removed > 0:
            return f"âœ… Cleaned up {temp_files_removed} temporary file(s) from project '{project_name}'"
        else:
            return f"âœ… No temporary files found in project '{project_name}'"
            
    except Exception as e:
        return f"âŒ Error cleaning up temp files: {str(e)}"

def handle_audio_trimming(audio_data) -> tuple:
    """Handle audio trimming from Gradio audio component
    
    When users select a portion of audio in Gradio's waveform, we need to extract 
    that specific segment. This function attempts to work with Gradio's trimming data.
    """
    if not audio_data:
        return None, "âŒ No audio data provided"
    
    print(f"[DEBUG] handle_audio_trimming called with data type: {type(audio_data)}")
    
    try:
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            # Standard format: (sample_rate, audio_array)
            sample_rate, audio_array = audio_data
            
            # Check if this is the full audio or a trimmed segment
            if hasattr(audio_array, 'shape'):
                print(f"[DEBUG] Audio shape: {audio_array.shape}, sample_rate: {sample_rate}")
                # For now, return the audio as-is since Gradio trimming is complex
                return audio_data, f"âœ… Audio loaded - {len(audio_array)} samples at {sample_rate}Hz"
            else:
                return None, "âŒ Invalid audio array format"
        else:
            return None, "âŒ Invalid audio data format"
            
    except Exception as e:
        print(f"[DEBUG] Error in handle_audio_trimming: {str(e)}")
        return None, f"âŒ Error processing audio: {str(e)}"

def extract_audio_segment(audio_data, start_time: float = None, end_time: float = None) -> tuple:
    """Extract a specific time segment from audio data
    
    Args:
        audio_data: Tuple of (sample_rate, audio_array)
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)
    """
    if not audio_data or not isinstance(audio_data, tuple) or len(audio_data) != 2:
        return None, "âŒ Invalid audio data"
    
    try:
        sample_rate, audio_array = audio_data
        
        if not hasattr(audio_array, 'shape'):
            return None, "âŒ Invalid audio array"
        
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
        
        status_msg = f"âœ… Extracted segment: {trimmed_duration:.2f}s (from {start_time or 0:.2f}s to {end_time or total_duration:.2f}s)"
        
        return (sample_rate, trimmed_audio), status_msg
        
    except Exception as e:
        return None, f"âŒ Error extracting segment: {str(e)}"

def save_visual_trim_to_file(audio_data, original_file_path: str, chunk_num: int) -> tuple:
    """Save visually trimmed audio from Gradio audio component to file, directly overwriting the original chunk file."""
    import wave
    import numpy as np
    import os

    if not audio_data or not original_file_path:
        return "âŒ No audio data to save", None

    print(f"[DEBUG] Direct save_visual_trim_to_file called for chunk {chunk_num}")
    print(f"[DEBUG] Audio data type: {type(audio_data)}")
    print(f"[DEBUG] Original file path: {original_file_path}")

    try:
        if not os.path.exists(os.path.dirname(original_file_path)):
            return f"âŒ Error: Directory for original file does not exist: {os.path.dirname(original_file_path)}", None

        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()

            print(f"[DEBUG] Saving chunk {chunk_num} - Sample rate: {sample_rate}, Trimmed array length: {len(audio_array)}")

            with wave.open(original_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                if audio_array.dtype != np.int16:
                    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_array.astype(np.int16)
                else:
                    audio_int16 = audio_array
                wav_file.writeframes(audio_int16.tobytes())
            
            duration_seconds = len(audio_int16) / sample_rate
            status_msg = f"âœ… Chunk {chunk_num} trimmed & directly saved! New duration: {duration_seconds:.2f}s. Original overwritten."
            print(f"[INFO] Chunk {chunk_num} saved to {original_file_path}, duration {duration_seconds:.2f}s.")
            return status_msg, original_file_path
        else:
            print(f"[ERROR] Invalid audio format for chunk {chunk_num}: expected (sample_rate, array) tuple, got {type(audio_data)}")
            return f"âŒ Invalid audio format for chunk {chunk_num}: expected (sample_rate, array) tuple", None
    except Exception as e:
        print(f"[ERROR] Exception in save_visual_trim_to_file for chunk {chunk_num}: {str(e)}")
        return f"âŒ Error saving audio for chunk {chunk_num}: {str(e)}", None

def auto_save_visual_trims_and_download(project_name: str) -> tuple:
    """Enhanced download that attempts to save any pending visual trims and then downloads"""
    if not project_name:
        return None, "âŒ No project selected"
    
    # Standard download functionality
    download_result = combine_project_audio_chunks(project_name)
    
    if download_result[0]:  # If download was successful
        success_msg = download_result[1] + "\n\nðŸŽµ Note: If you made visual trims but didn't save them, use the 'Save Trimmed Chunk' buttons first, then refresh download"
        return download_result[0], success_msg
    else:
        return download_result

def save_all_pending_trims_and_combine(project_name: str, loaded_chunks_data: list, *all_audio_component_values) -> str:
    """
    Automatically saves visual trims from displayed audio components for the current project,
    then creates split downloadable files.
    """
    if not project_name:
        return "âŒ No project selected for download."
    if not loaded_chunks_data:
        return "âŒ No chunks loaded for the project to save or combine."

    print(f"[INFO] Auto-saving trims for project '{project_name}' before creating split files.")
    auto_save_reports = []

    num_loaded_chunks = len(loaded_chunks_data)
    num_audio_components_passed = len(all_audio_component_values)
    
    # Only process chunks that have corresponding audio players in the interface
    max_chunks_to_process = min(num_loaded_chunks, num_audio_components_passed, MAX_CHUNKS_FOR_INTERFACE)
    
    print(f"[INFO] Project has {num_loaded_chunks} total chunks, processing first {max_chunks_to_process} for auto-save.")

    for i in range(max_chunks_to_process):
        chunk_info = loaded_chunks_data[i]
        chunk_num = chunk_info['chunk_num']
        original_file_path = chunk_info['audio_file']

        current_audio_data_from_player = all_audio_component_values[i]
        if current_audio_data_from_player:  # If there's audio in the player (e.g., (sample_rate, data))
            print(f"[DEBUG] Auto-saving trim for chunk {chunk_num} (Audio data type: {type(current_audio_data_from_player)})")
            status_msg, _ = save_visual_trim_to_file(current_audio_data_from_player, original_file_path, chunk_num)
            auto_save_reports.append(f"Chunk {chunk_num}: {status_msg.splitlines()[0]}") # Take first line of status
        else:
            auto_save_reports.append(f"Chunk {chunk_num}: No audio data in player; skipping auto-save.")

    # After attempting to save all trims from displayed chunks, create split files instead of one massive file
    print(f"[INFO] Creating split MP3 files for project '{project_name}' after auto-save attempts.")
    split_result = combine_project_audio_chunks_split(project_name)
    
    final_status_message = split_result
    if auto_save_reports:
        auto_save_summary = f"Auto-saved trims for {max_chunks_to_process} displayed chunks out of {num_loaded_chunks} total chunks."
        final_status_message = f"--- Auto-Save Report ---\n{auto_save_summary}\n" + "\n".join(auto_save_reports[:10])  # Show first 10 reports
        if len(auto_save_reports) > 10:
            final_status_message += f"\n... and {len(auto_save_reports) - 10} more auto-saves."
        final_status_message += f"\n\n{split_result}"
        
    return final_status_message

def combine_project_audio_chunks_split(project_name: str, chunks_per_file: int = 50, output_format: str = "mp3") -> str:
    """Create multiple smaller downloadable MP3 files from project chunks"""
    if not project_name:
        return "âŒ No project selected"
    
    chunks = get_project_chunks(project_name)
    
    if not chunks:
        return f"âŒ No audio chunks found in project '{project_name}'"
    
    try:
        # Check if pydub is available for MP3 export
        try:
            from pydub import AudioSegment
            mp3_available = True
        except ImportError:
            mp3_available = False
            output_format = "wav"  # Fallback to WAV
            print("[WARNING] pydub not available, using WAV format instead of MP3")
        
        sample_rate = 24000  # Default sample rate
        
        # Sort chunks by chunk number to ensure correct order
        def extract_chunk_number(chunk_info):
            """Extract chunk number from chunk info for proper numerical sorting"""
            try:
                # First try to get chunk_num directly from the chunk info
                chunk_num = chunk_info.get('chunk_num')
                if chunk_num is not None:
                    return int(chunk_num)  # Ensure it's an integer
            except (ValueError, TypeError):
                pass
            
            # Fallback: try to extract from filename
            try:
                filename = chunk_info.get('audio_filename', '') or chunk_info.get('audio_file', '')
                if filename:
                    import re
                    # Look for patterns like "_123.wav" or "_chunk_123.wav"
                    match = re.search(r'_(\d+)\.wav$', filename)
                    if match:
                        return int(match.group(1))
                    
                    # Try other patterns like "projectname_123.wav"
                    match = re.search(r'(\d+)\.wav$', filename)
                    if match:
                        return int(match.group(1))
            except (ValueError, TypeError, AttributeError):
                pass
            
            # Last resort: return 0 (should sort first)
            print(f"[WARNING] Could not extract chunk number from: {chunk_info}")
            return 0
        
        chunks_sorted = sorted(chunks, key=extract_chunk_number)
        
        # Debug: Show first and last few chunk numbers to verify sorting
        if len(chunks_sorted) > 0:
            first_few = [extract_chunk_number(c) for c in chunks_sorted[:5]]
            last_few = [extract_chunk_number(c) for c in chunks_sorted[-5:]]
            print(f"[DEBUG] First 5 chunk numbers after sorting: {first_few}")
            print(f"[DEBUG] Last 5 chunk numbers after sorting: {last_few}")
            
            # NEW: Also show the actual filenames to verify they match the chunk numbers
            first_few_files = [os.path.basename(c.get('audio_file', 'unknown')) for c in chunks_sorted[:5]]
            last_few_files = [os.path.basename(c.get('audio_file', 'unknown')) for c in chunks_sorted[-5:]]
            print(f"[DEBUG] First 5 filenames after sorting: {first_few_files}")
            print(f"[DEBUG] Last 5 filenames after sorting: {last_few_files}")
        
        print(f"[INFO] Creating {len(chunks_sorted)} chunks into multiple {output_format.upper()} files ({chunks_per_file} chunks per file)")
        
        created_files = []
        total_duration_seconds = 0
        
        # Process chunks in groups
        for file_index in range(0, len(chunks_sorted), chunks_per_file):
            file_end = min(file_index + chunks_per_file, len(chunks_sorted))
            file_chunks = chunks_sorted[file_index:file_end]
            
            file_number = (file_index // chunks_per_file) + 1
            
            # Use actual chunk numbers from the files, not array indices
            chunk_start = extract_chunk_number(file_chunks[0]) if file_chunks else file_index + 1
            chunk_end = extract_chunk_number(file_chunks[-1]) if file_chunks else file_end
            
            print(f"[INFO] Creating file {file_number}: chunks {chunk_start}-{chunk_end}")
            
            # Debug: Show which files will be processed for this part
            if len(file_chunks) > 0:
                first_files = [os.path.basename(c.get('audio_file', 'unknown')) for c in file_chunks[:3]]
                last_files = [os.path.basename(c.get('audio_file', 'unknown')) for c in file_chunks[-3:]]
                print(f"[DEBUG] Part {file_number} - First 3 files: {first_files}")
                print(f"[DEBUG] Part {file_number} - Last 3 files: {last_files}")
            
            combined_audio = []
            
            for chunk_info in file_chunks:
                chunk_path = chunk_info.get('audio_file')
                chunk_num = extract_chunk_number(chunk_info)
                
                if not chunk_path or not os.path.exists(chunk_path):
                    print(f"âš ï¸ Warning: Chunk {chunk_num} file not found: {chunk_path}")
                    continue
                
                try:
                    with wave.open(chunk_path, 'rb') as wav_file:
                        chunk_sample_rate = wav_file.getframerate()
                        chunk_frames = wav_file.getnframes()
                        chunk_audio_data = wav_file.readframes(chunk_frames)
                        
                        # Convert to numpy array
                        chunk_audio_array = np.frombuffer(chunk_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if sample_rate != chunk_sample_rate:
                            sample_rate = chunk_sample_rate
                        
                        combined_audio.append(chunk_audio_array)
                        
                except Exception as e:
                    print(f"âŒ Error reading chunk {chunk_num} ({chunk_path}): {e}")
                    continue
            
            if not combined_audio:
                print(f"âš ï¸ No valid chunks found for file {file_number}")
                continue
            
            # Concatenate audio for this file
            file_audio = np.concatenate(combined_audio, axis=0)
            file_duration_seconds = len(file_audio) / sample_rate
            total_duration_seconds += file_duration_seconds
            
            # Convert back to int16 for audio processing
            file_audio_int16 = (file_audio * 32767).astype(np.int16)
            
            # Create output filename
            output_filename = f"{project_name}_part{file_number:02d}_chunks{chunk_start:03d}-{chunk_end:03d}.{output_format}"
            output_path = os.path.join("audiobook_projects", project_name, output_filename)
            
            if mp3_available and output_format == "mp3":
                # Use pydub to create MP3 with good compression
                audio_segment = AudioSegment(
                    file_audio_int16.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
                # Export as MP3 with good quality settings
                audio_segment.export(output_path, format="mp3", bitrate="128k")
            else:
                # Save as WAV file
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(file_audio_int16.tobytes())
            
            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                file_duration_minutes = file_duration_seconds / 60
                
                created_files.append({
                    'filename': output_filename,
                    'chunks': f"{chunk_start}-{chunk_end}",
                    'duration_minutes': file_duration_minutes,
                    'size_mb': file_size_mb
                })
                
                print(f"âœ… Created {output_filename}: {file_duration_minutes:.2f} minutes, {file_size_mb:.2f} MB")
        
        if not created_files:
            return "âŒ No files were created"
        
        # Calculate total statistics
        total_duration_minutes = total_duration_seconds / 60
        total_duration_hours = int(total_duration_minutes // 60)
        remaining_minutes = int(total_duration_minutes % 60)
        total_size_mb = sum(f['size_mb'] for f in created_files)
        
        # Create a summary of all created files
        file_list = "\n".join([
            f"ðŸ“ {f['filename']} - Chunks {f['chunks']} - {f['duration_minutes']:.1f} min - {f['size_mb']:.1f} MB"
            for f in created_files
        ])
        
        format_display = output_format.upper()
        size_comparison = f"ðŸ“¦ Total size: {total_size_mb:.1f} MB ({format_display} format" + (f" - ~70% smaller than WAV!" if output_format == "mp3" else "") + ")"
        
        success_message = (
            f"âœ… Created {len(created_files)} downloadable {format_display} files from {len(chunks_sorted)} chunks!\n"
            f"ðŸŽµ Total duration: {total_duration_hours}h {remaining_minutes}m\n"
            f"{size_comparison}\n\n"
            f"ðŸ“ **Files are saved in your project folder:**\n"
            f"ðŸ“‚ Navigate to: audiobook_projects/{project_name}/\n\n"
            f"ðŸ“‹ Files created:\n{file_list}\n\n"
            f"ðŸ’¡ **Tip:** Browse to your project folder to download individual {format_display} files!"
        )
        
        return success_message
        
    except Exception as e:
        error_msg = f"âŒ Error creating split audio files: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg

with gr.Blocks(css=css, title="Chatterbox TTS - Audiobook Edition") as demo:
    model_state = gr.State(None)
    voice_library_path_state = gr.State(SAVED_VOICE_LIBRARY_PATH)
    
    gr.HTML("""
    <div class="voice-library-header">
        <h1>ðŸŽ§ Chatterbox TTS - Audiobook Edition</h1>
        <p>Professional voice cloning for audiobook creation</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # Enhanced TTS Tab with Voice Selection
        with gr.TabItem("ðŸŽ¤ Text-to-Speech", id="tts"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(
                        value="Welcome to Chatterbox TTS Audiobook Edition. This tool will help you create amazing audiobooks with consistent character voices.",
                        label="Text to synthesize",
                        lines=3
                    )
                    
                    # Voice Selection Section
                    with gr.Group():
                        gr.HTML("<h4>ðŸŽ­ Voice Selection</h4>")
                        tts_voice_selector = gr.Dropdown(
                            choices=get_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                            label="Choose Voice",
                            value=None,
                            info="Select a saved voice profile or use manual input"
                        )
                        
                        # Voice status display
                        tts_voice_status = gr.HTML(
                            "<div class='voice-status'>ðŸ“ Manual input mode - upload your own audio file below</div>"
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

                    with gr.Accordion("âš™ï¸ Advanced Options", open=False):
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

                    with gr.Row():
                        run_btn = gr.Button("ðŸŽµ Generate Speech", variant="primary", size="lg")
                        refresh_voices_btn = gr.Button("ðŸ”„ Refresh Voices", size="sm")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio")
                    
                    gr.HTML("""
                    <div class="instruction-box">
                        <h4>ðŸ’¡ TTS Tips:</h4>
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
        with gr.TabItem("ðŸ“š Voice Library", id="voices"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>ðŸŽ­ Voice Management</h3>")
                    
                    # Voice Library Settings
                    with gr.Group():
                        gr.HTML("<h4>ðŸ“ Library Settings</h4>")
                        voice_library_path = gr.Textbox(
                            value=SAVED_VOICE_LIBRARY_PATH,
                            label="Voice Library Folder",
                            placeholder="Enter path to voice library folder",
                            info="This path will be remembered between sessions"
                        )
                        update_path_btn = gr.Button("ðŸ’¾ Save & Update Library Path", size="sm")
                        
                        # Configuration status
                        config_status = gr.HTML(
                            f"<div class='config-status'>ðŸ“‚ Current library: {SAVED_VOICE_LIBRARY_PATH}</div>"
                        )
                    
                    # Voice Selection
                    with gr.Group():
                        gr.HTML("<h4>ðŸŽ¯ Select Voice</h4>")
                        voice_dropdown = gr.Dropdown(
                            choices=[],
                            label="Saved Voice Profiles",
                            value=None
                        )
                        
                        with gr.Row():
                            load_voice_btn = gr.Button("ðŸ“¥ Load Voice", size="sm")
                            refresh_btn = gr.Button("ðŸ”„ Refresh", size="sm")
                            delete_voice_btn = gr.Button("ðŸ—‘ï¸ Delete", size="sm", variant="stop")
                
                with gr.Column(scale=2):
                    # Voice Testing & Saving
                    gr.HTML("<h3>ðŸŽ™ï¸ Voice Testing & Configuration</h3>")
                    
                    with gr.Group():
                        gr.HTML("<h4>ðŸ“ Voice Details</h4>")
                        voice_name = gr.Textbox(label="Voice Name", placeholder="e.g., narrator_male_deep")
                        voice_display_name = gr.Textbox(label="Display Name", placeholder="e.g., Deep Male Narrator")
                        voice_description = gr.Textbox(
                            label="Description", 
                            placeholder="e.g., Deep, authoritative voice for main character",
                            lines=2
                        )
                    
                    with gr.Group():
                        gr.HTML("<h4>ðŸŽµ Voice Settings</h4>")
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
                        gr.HTML("<h4>ðŸ§ª Test Voice</h4>")
                        test_text = gr.Textbox(
                            value="Hello, this is a test of the voice settings. How does this sound?",
                            label="Test Text",
                            lines=2
                        )
                        
                        with gr.Row():
                            test_voice_btn = gr.Button("ðŸŽµ Test Voice", variant="secondary")
                            save_voice_btn = gr.Button("ðŸ’¾ Save Voice Profile", variant="primary")
                        
                        test_audio_output = gr.Audio(label="Test Audio Output")
                        
                        # Status messages
                        voice_status = gr.HTML("<div class='voice-status'>Ready to test and save voices...</div>")

        # Enhanced Audiobook Creation Tab
        with gr.TabItem("ðŸ“– Audiobook Creation - Single Sample", id="audiobook_single"):
            gr.HTML("""
            <div class="audiobook-header">
                <h2>ðŸ“– Audiobook Creation Studio - Single Voice</h2>
                <p>Transform your text into professional audiobooks with one consistent voice</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Text Input Section
                    with gr.Group():
                        gr.HTML("<h3>ðŸ“ Text Content</h3>")
                        
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
                                    label="ðŸ“„ Upload Text File",
                                    file_types=[".txt", ".md", ".rtf"],
                                    type="filepath"
                                )
                                
                                load_file_btn = gr.Button(
                                    "ðŸ“‚ Load File", 
                                    size="sm",
                                    variant="secondary"
                                )
                                
                                # File status
                                file_status = gr.HTML(
                                    "<div class='file-status'>ðŸ“„ No file loaded</div>"
                                )
                    # NEW: Project Management Section
                    with gr.Group():
                        gr.HTML("<h3>ðŸ“ Project Management</h3>")
                        single_project_dropdown = gr.Dropdown(
                            choices=get_project_choices(),
                            label="Select Existing Project",
                            value=None,
                            info="Load or resume an existing project"
                        )
                        with gr.Row():
                            load_project_btn = gr.Button("ðŸ“‚ Load Project", size="sm", variant="secondary")
                            resume_project_btn = gr.Button("â–¶ï¸ Resume Project", size="sm", variant="primary")
                        single_project_progress = gr.HTML("<div class='voice-status'>No project loaded</div>")
                
                with gr.Column(scale=1):
                    # Voice Selection & Project Settings
                    with gr.Group():
                        gr.HTML("<h3>ðŸŽ­ Voice Configuration</h3>")
                        
                        audiobook_voice_selector = gr.Dropdown(
                            choices=get_audiobook_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                            label="Select Voice",
                            value=None,
                            info="Choose from your saved voice profiles"
                        )
                        
                        refresh_audiobook_voices_btn = gr.Button(
                            "ðŸ”„ Refresh Voices", 
                            size="sm"
                        )
                        
                        # Voice info display
                        audiobook_voice_info = gr.HTML(
                            "<div class='voice-status'>ðŸŽ­ Select a voice to see details</div>"
                        )
                    
                    # Project Settings
                    with gr.Group():
                        gr.HTML("<h3>ðŸ“ Project Settings</h3>")
                        
                        project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="e.g., my_first_audiobook",
                            info="Used for naming output files (project_001.wav, project_002.wav, etc.)"
                        )
                        
                        # Previous Projects Section
                        with gr.Group():
                            gr.HTML("<h4>ðŸ“š Previous Projects</h4>")
                            
                            previous_project_dropdown = gr.Dropdown(
                                choices=get_project_choices(),
                                label="Load Previous Project Audio",
                                value=None,
                                info="Select a previous project to download its complete audio"
                            )
                            
                            with gr.Row():
                                load_previous_btn = gr.Button(
                                    "ðŸ“‚ Load Project Audio",
                                    size="sm",
                                    variant="secondary"
                                )
                                refresh_previous_btn = gr.Button(
                                    "ðŸ”„ Refresh",
                                    size="sm"
                                )
                            
                            # Previous project audio and download
                            previous_project_audio = gr.Audio(
                                label="Previous Project Audio",
                                visible=False
                            )
                            
                            previous_project_download = gr.File(
                                label="ðŸ“ Download Previous Project",
                                visible=False
                            )
                            
                            previous_project_status = gr.HTML(
                                "<div class='voice-status'>ðŸ“ Select a previous project to load its audio</div>"
                            )
            
            # Processing Section
            with gr.Group():
                gr.HTML("<h3>ðŸš€ Audiobook Processing</h3>")
                
                with gr.Row():
                    validate_btn = gr.Button(
                        "ðŸ” Validate Input", 
                        variant="secondary",
                        size="lg"
                    )
                    
                    process_btn = gr.Button(
                        "ðŸŽµ Create Audiobook", 
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                # Status and progress
                audiobook_status = gr.HTML(
                    "<div class='audiobook-status'>ðŸ“‹ Ready to create audiobooks! Load text, select voice, and set project name.</div>"
                )
                
                # Preview/Output area
                audiobook_output = gr.Audio(
                    label="Generated Audiobook (Preview - Full files saved to project folder)",
                    visible=False
                )
            
            # Instructions
            gr.HTML("""
            <div class="instruction-box">
                <h4>ðŸ“‹ How to Create Single-Voice Audiobooks:</h4>
                <ol>
                    <li><strong>Add Text:</strong> Paste text or upload a .txt file</li>
                    <li><strong>Select Voice:</strong> Choose from your saved voice profiles</li>
                    <li><strong>Set Project Name:</strong> This will be used for output file naming</li>
                    <li><strong>Validate:</strong> Check that everything is ready</li>
                    <li><strong>Create:</strong> Generate your audiobook with smart chunking!</li>
                </ol>
                <p><strong>ðŸŽ¯ Smart Chunking:</strong> Text is automatically split at sentence boundaries after ~50 words for optimal processing.</p>
                <p><strong>ðŸ“ File Output:</strong> Individual chunks saved as project_001.wav, project_002.wav, etc.</p>
            </div>
            """)

        # NEW: Multi-Voice Audiobook Creation Tab
        with gr.TabItem("ðŸŽ­ Audiobook Creation - Multi-Sample", id="audiobook_multi"):
            gr.HTML("""
            <div class="audiobook-header">
                <h2>ðŸŽ­ Multi-Voice Audiobook Creation Studio</h2>
                <p>Create dynamic audiobooks with multiple character voices using voice tags</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Text Input Section with Voice Tags
                    with gr.Group():
                        gr.HTML("<h3>ðŸ“ Multi-Voice Text Content</h3>")
                        
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
                                    label="ðŸ“„ Upload Text File",
                                    file_types=[".txt", ".md", ".rtf"],
                                    type="filepath"
                                )
                                
                                load_multi_file_btn = gr.Button(
                                    "ðŸ“‚ Load File", 
                                    size="sm",
                                    variant="secondary"
                                )
                                
                                # File status
                                multi_file_status = gr.HTML(
                                    "<div class='file-status'>ðŸ“„ No file loaded</div>"
                                )
                    # NEW: Project Management Section
                    with gr.Group():
                        gr.HTML("<h3>ðŸ“ Project Management</h3>")
                        multi_project_dropdown = gr.Dropdown(
                            choices=get_project_choices(),
                            label="Select Existing Project",
                            value=None,
                            info="Load or resume an existing project"
                        )
                        with gr.Row():
                            load_multi_project_btn = gr.Button("ðŸ“‚ Load Project", size="sm", variant="secondary")
                            resume_multi_project_btn = gr.Button("â–¶ï¸ Resume Project", size="sm", variant="primary")
                        multi_project_progress = gr.HTML("<div class='voice-status'>No project loaded</div>")
                
                with gr.Column(scale=1):
                    # Voice Analysis & Project Settings
                    with gr.Group():
                        gr.HTML("<h3>ðŸ” Text Analysis</h3>")
                        
                        analyze_text_btn = gr.Button(
                            "ðŸ” Analyze Text & Find Characters",
                            variant="secondary",
                            size="lg"
                        )
                        
                        # Voice breakdown display
                        voice_breakdown_display = gr.HTML(
                            "<div class='voice-status'>ðŸ“ Click 'Analyze Text' to find characters in your text</div>"
                        )
                        
                        refresh_multi_voices_btn = gr.Button(
                            "ðŸ”„ Refresh Available Voices", 
                            size="sm"
                        )
                    
                    # Voice Assignment Section
                    with gr.Group():
                        gr.HTML("<h3>ðŸŽ­ Voice Assignments</h3>")
                        
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
                        gr.HTML("<h3>ðŸ“ Project Settings</h3>")
                        
                        multi_project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="e.g., my_multi_voice_story",
                            info="Used for naming output files (project_001_character.wav, etc.)"
                        )
                        
                        # Previous Projects Section
                        with gr.Group():
                            gr.HTML("<h4>ðŸ“š Previous Projects</h4>")
                            
                            multi_previous_project_dropdown = gr.Dropdown(
                                choices=get_project_choices(),
                                label="Load Previous Project Audio",
                                value=None,
                                info="Select a previous project to download its complete audio"
                            )
                            
                            with gr.Row():
                                load_multi_previous_btn = gr.Button(
                                    "ðŸ“‚ Load Project Audio",
                                    size="sm",
                                    variant="secondary"
                                )
                                refresh_multi_previous_btn = gr.Button(
                                    "ðŸ”„ Refresh",
                                    size="sm"
                                )
                            
                            # Previous project audio and download
                            multi_previous_project_audio = gr.Audio(
                                label="Previous Project Audio",
                                visible=False
                            )
                            
                            multi_previous_project_download = gr.File(
                                label="ðŸ“ Download Previous Project",
                                visible=False
                            )
                            
                            multi_previous_project_status = gr.HTML(
                                "<div class='voice-status'>ðŸ“ Select a previous project to load its audio</div>"
                            )
            
            # Processing Section
            with gr.Group():
                gr.HTML("<h3>ðŸš€ Multi-Voice Processing</h3>")
                
                with gr.Row():
                    validate_multi_btn = gr.Button(
                        "ðŸ” Validate Voice Assignments", 
                        variant="secondary",
                        size="lg",
                        interactive=False
                    )
                    
                    process_multi_btn = gr.Button(
                        "ðŸŽµ Create Multi-Voice Audiobook", 
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                # Status and progress
                multi_audiobook_status = gr.HTML(
                    "<div class='audiobook-status'>ðŸ“‹ Step 1: Analyze text to find characters<br/>ðŸ“‹ Step 2: Assign voices to each character<br/>ðŸ“‹ Step 3: Validate and create audiobook</div>"
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
                <h4>ðŸ“‹ How to Create Multi-Voice Audiobooks:</h4>
                <ol>
                    <li><strong>Add Voice Tags:</strong> Use [character_name] before text for that character</li>
                    <li><strong>Analyze Text:</strong> Click 'Analyze Text' to find all characters</li>
                    <li><strong>Assign Voices:</strong> Choose voices from your library for each character</li>
                    <li><strong>Set Project Name:</strong> Used for output file naming</li>
                    <li><strong>Validate & Create:</strong> Generate your multi-voice audiobook!</li>
                </ol>
                <h4>ðŸŽ¯ Voice Tag Format:</h4>
                <p><code>[narrator] The story begins here...</code></p>
                <p><code>[princess] "Hello there!" she said cheerfully.</code></p>
                <p><code>[narrator] The mysterious figure walked away.</code></p>
                <p><strong>ðŸ“ File Output:</strong> Files named with character: project_001_narrator.wav, project_002_princess.wav, etc.</p>
                <p><strong>ðŸŽ­ New Workflow:</strong> Characters in [brackets] can be mapped to any voice in your library!</p>
                <p><strong>ðŸ’¡ Smart Processing:</strong> Tries GPU first for speed, automatically falls back to CPU if CUDA errors occur (your 3090 should handle most cases!).</p>
            </div>
            """)

        # NEW: Regenerate Sample Tab with Sub-tabs
