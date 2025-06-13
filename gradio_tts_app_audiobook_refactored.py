"""
Gradio TTS Audiobook Application - REFACTORED FOR STATELESS GENERATION

This module provides a comprehensive web-based interface for creating audiobooks using
Text-to-Speech (TTS) technology. It supports both single-voice and multi-voice audiobook
generation with features like voice profile management, project management, audio editing,
and real-time playback.

This version has been refactored to use a stateless generation approach to eliminate
random audio artifacts caused by state bleed between chunks.

Key Features:
- Single and multi-voice audiobook generation
- Voice profile management with custom settings
- Project-based organization with save/resume functionality
- Real-time audio editing and trimming
- Continuous playback with chunk tracking
- Audio quality analysis and normalization
- GPU/CPU processing with automatic fallback

Dependencies:
- ChatterboxTTS: Core TTS engine for audio generation
- Gradio: Web interface framework
- PyTorch: ML framework for TTS model execution
- Torchaudio: Audio processing utilities

Author: Generated for ChatterBox Audiobook Project
"""

# Standard library imports for core functionality
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
import torchaudio
import tempfile
import time
from typing import List
import warnings
warnings.filterwarnings("ignore")

# Try importing the TTS module with graceful fallback
try:
    from src.chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChatterboxTTS not available - {e}")
    CHATTERBOX_AVAILABLE = False

# --- Core Application State ---
# This dictionary will cache the voice conditioning data to avoid re-computation
VOICE_CONDS_CACHE = {}

# Device configuration for TTS processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force CPU mode for multi-voice to avoid CUDA indexing errors
MULTI_VOICE_DEVICE = "cpu"

# Application configuration constants
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"
MAX_CHUNKS_FOR_INTERFACE = 100
MAX_CHUNKS_FOR_AUTO_SAVE = 100

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('voice_library_path', DEFAULT_VOICE_LIBRARY)
        except:
            return DEFAULT_VOICE_LIBRARY
    return DEFAULT_VOICE_LIBRARY

def save_config(voice_library_path):
    config = {
        'voice_library_path': voice_library_path,
        'last_updated': str(Path().resolve())
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
    model = ChatterboxTTS.from_pretrained("cpu")
    return model

def get_or_create_conds(model, audio_prompt_path, exaggeration, force_recreate=False):
    """
    Get voice conditioning from cache or create and cache it.
    This is the core of the stateless fix, ensuring each voice has its own,
    isolated conditioning data.
    """
    cache_key = (os.path.abspath(audio_prompt_path), exaggeration, model.device)
    if cache_key in VOICE_CONDS_CACHE and not force_recreate:
        # print(f"CACHE HIT for {os.path.basename(audio_prompt_path)}")
        return VOICE_CONDS_CACHE[cache_key]
    
    # print(f"CACHE MISS for {os.path.basename(audio_prompt_path)}. Creating new conds.")
    conds = model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
    VOICE_CONDS_CACHE[cache_key] = conds
    return conds

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    """
    Stateless generate wrapper.
    """
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
    if seed_num != 0:
        set_seed(int(seed_num))

    conds = get_or_create_conds(model, audio_prompt_path, exaggeration)
    
    wav = model.generate(
        text,
        conds=conds,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    if DEVICE == "cuda":
        try:
            clear_gpu_memory()
            conds = get_or_create_conds(model, audio_prompt_path, exaggeration)
            wav = model.generate(
                text,
                conds=conds,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return wav, "GPU"
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower()):
                print(f"⚠️ CUDA error, falling back to CPU: {str(e)[:100]}...")
            else:
                raise e
    
    try:
        cpu_model = ChatterboxTTS.from_pretrained("cpu")
        conds = get_or_create_conds(cpu_model, audio_prompt_path, exaggeration)
        wav = cpu_model.generate(
            text,
            conds=conds,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def generate_with_retry(model, text, conds, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    Stateless generate_with_retry.
    """
    for retry in range(max_retries):
        try:
            if retry > 0:
                clear_gpu_memory()
            
            wav = model.generate(
                text,
                conds=conds,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return wav
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower()):
                if retry < max_retries - 1:
                    print(f"⚠️ GPU error, retry {retry + 1}/{max_retries}: {str(e)[:100]}...")
                    clear_gpu_memory()
                    continue
                else:
                    raise RuntimeError(f"Failed after {max_retries} retries: {str(e)}")
            else:
                raise e
    raise RuntimeError("Generation failed after all retries")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_voice_config(voice_library_path, voice_name):
    if not voice_name: return None
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f: config = json.load(f)
            audio_file = os.path.join(profile_dir, config['audio_file']) if config.get('audio_file') and os.path.exists(os.path.join(profile_dir, config['audio_file'])) else None
            return {'audio_file': os.path.abspath(audio_file) if audio_file else None, 'exaggeration': config.get('exaggeration', 0.5), 'cfg_weight': config.get('cfg_weight', 0.5), 'temperature': config.get('temperature', 0.8), 'display_name': config.get('display_name', voice_name)}
        except Exception as e:
            print(f"⚠️ Error reading config for voice '{voice_name}': {e}")
    return None

def chunk_text_by_sentences(text, max_words=80):
    sentences = re.split(r'([.!?]+\s*)', text)
    chunks, current_chunk, current_word_count = [], "", 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence: i += 1; continue
        if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]): sentence += sentences[i + 1]; i += 2
        else: i += 1
        sentence_words = len(sentence.split())
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip(): chunks.append(current_chunk.strip())
            current_chunk, current_word_count = sentence, sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    if current_chunk.strip(): chunks.append(current_chunk.strip())
    return chunks

def create_audiobook(
    model,
    text_content: str,
    voice_library_path: str,
    selected_voice: str,
    project_name: str,
    resume: bool = False,
    autosave_interval: int = 10
) -> tuple:
    # ... (initial setup code is the same) ...
    # === PHASE 1-5 ... ===
    if not text_content or not selected_voice or not project_name:
        return None, "❌ Missing required fields"

    voice_config = get_voice_config(voice_library_path, selected_voice)
    if not voice_config:
        return None, f"❌ Could not load voice configuration for '{selected_voice}'"
    if not voice_config['audio_file']:
        return None, f"❌ No audio file found for voice '{voice_config['display_name']}'"

    chunks = chunk_text_by_sentences(text_content)
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None, "❌ No text chunks to process"

    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # ... (resume logic is the same) ...

    # === PHASE 6: MODEL INITIALIZATION AND CONDITIONING ===
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
        
    # ** NEW: Prepare voice conditioning ONCE for the entire project **
    try:
        conds = get_or_create_conds(model, voice_config['audio_file'], voice_config['exaggeration'])
    except Exception as e:
        return None, f"❌ Failed to prepare voice conditioning: {e}"

    # === PHASE 7 & 8 ... ===
    # ...

    # === PHASE 9: MAIN GENERATION LOOP ===
    for i in range(total_chunks):
        chunk = chunks[i]
        try:
            # ... (status reporting) ...
            
            # ** CHANGED: Pass the `conds` object to the retry function **
            wav = generate_with_retry(
                model,
                chunk,
                conds,  # Pass the prepared conds
                voice_config['exaggeration'],
                voice_config['temperature'],
                voice_config['cfg_weight']
            )
            audio_np = wav.squeeze(0).cpu().numpy()
            
            # ... (The rest of the loop for saving files, normalization, etc. remains the same) ...
        except Exception as chunk_error:
            return None, f"❌ Error processing chunk {i+1}: {str(chunk_error)}"
            
    # ... (Final combination and success message) ...
    return (model.sr, combined_audio), success_msg

def create_multi_voice_audiobook_with_assignments(
    model,
    text_content: str,
    voice_library_path: str,
    project_name: str,
    voice_assignments: dict,
    resume: bool = False,
    autosave_interval: int = 10
) -> tuple:
    # ... (setup is the same) ...

    # === NEW: Prepare all voice conditionings at the beginning ===
    voice_conds_map = {}
    for char_name, voice_name in voice_assignments.items():
        if voice_name not in voice_conds_map:
            try:
                voice_config = get_voice_config(voice_library_path, voice_name)
                if not voice_config or not voice_config['audio_file']:
                    return None, None, f"❌ Invalid config for voice '{voice_name}'", None
                
                conds = get_or_create_conds(model, voice_config['audio_file'], voice_config['exaggeration'])
                voice_conds_map[voice_name] = conds
            except Exception as e:
                return None, None, f"❌ Failed to prepare voice '{voice_name}': {e}", None

    # ... (resume logic is the same) ...

    # Process missing chunks
    for i in range(total_chunks):
        # ...
        voice_name, chunk_text = chunks[i]
        try:
            voice_config = get_voice_config(voice_library_path, voice_name)
            if not voice_config:
                return None, None, f"❌ Could not load voice config for '{voice_name}'", None
            
            # ** CHANGED: Get pre-made conds from our map **
            conds_for_chunk = voice_conds_map.get(voice_name)
            if not conds_for_chunk:
                 return None, None, f"❌ Internal error: Missing prepared conds for voice '{voice_name}'", None

            wav = generate_with_retry(
                model,
                chunk_text,
                conds=conds_for_chunk,
                exaggeration=voice_config['exaggeration'],
                temperature=voice_config['temperature'],
                cfg_weight=voice_config['cfg_weight']
            )
            audio_np = wav.squeeze(0).cpu().numpy()
            
            # ... (rest of the loop) ...
        except Exception as chunk_error_outer:
            # ...
    # ... (Final combination and success message) ...
    return (model.sr, combined_audio), None, success_msg, None

# ... (All other functions, especially UI handlers, need to be updated) ...

# Example of updating a UI handler
def regenerate_single_chunk(model, project_name: str, chunk_num: int, voice_library_path: str, custom_text: str = None) -> tuple:
    # ...
    try:
        # ... (loading project info) ...
        voice_config = get_voice_config(voice_library_path, voice_name)
        # ...
        
        # ** CHANGED: Prepare conds before generating **
        conds = get_or_create_conds(model, voice_config['audio_file'], voice_config['exaggeration'])
        
        wav = generate_with_retry(
            model,
            text_to_generate,
            conds=conds,
            exaggeration=voice_config.get('exaggeration', 0.5),
            temperature=voice_config.get('temperature', 0.8),
            cfg_weight=voice_config.get('cfg_weight', 0.5)
        )
        # ... (rest of function) ...
    except Exception as e:
        # ...
# This is a sample of the changes required. I will apply them to all necessary functions.
# The full refactored code will be in the new file.
# Note: I have omitted the full code for brevity, but the logic above will be applied throughout.
# The full file will be written in the next step.
pass # Placeholder to make this a valid code block 