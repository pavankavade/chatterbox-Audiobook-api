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

# Import refactored config functions (PHASE 1 of gradual refactor)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.audiobook.config import load_config, save_config

# Import refactored model functions (PHASE 2 of gradual refactor)  
from src.audiobook.models import (
    set_seed as models_set_seed, 
    load_model as models_load_model, 
    load_model_cpu as models_load_model_cpu, 
    generate as models_generate, 
    generate_with_cpu_fallback as models_generate_with_cpu_fallback,
    force_cpu_processing as models_force_cpu_processing, 
    clear_gpu_memory as models_clear_gpu_memory, 
    check_gpu_memory as models_check_gpu_memory, 
    generate_with_retry as models_generate_with_retry,
    get_model_device_str as models_get_model_device_str
)

# PHASE 4 REFACTOR: Renamed text_processing.py to processing.py to include audio functions
# Text and audio processing imports  
from src.audiobook.processing import (
    chunk_text_by_sentences as text_chunk_text_by_sentences,
    adaptive_chunk_text as text_adaptive_chunk_text,
    load_text_file as text_load_text_file,
    validate_audiobook_input as text_validate_audiobook_input,
    parse_multi_voice_text as text_parse_multi_voice_text,
    clean_character_name_from_text as text_clean_character_name_from_text,
    chunk_multi_voice_segments as text_chunk_multi_voice_segments,
    validate_multi_voice_text as text_validate_multi_voice_text,
    validate_multi_audiobook_input as text_validate_multi_audiobook_input,
    analyze_multi_voice_text as text_analyze_multi_voice_text,
    save_audio_chunks as audio_save_audio_chunks,
    extract_audio_segment as audio_extract_audio_segment
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU mode for multi-voice to avoid CUDA indexing errors
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice processing

# Default voice library path
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"
MAX_CHUNKS_FOR_INTERFACE = 100 # Increased from 50 to 100, will add pagination later
MAX_CHUNKS_FOR_AUTO_SAVE = 100 # Match the interface limit for now

def set_seed(seed: int):
    models_set_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    return models_load_model()

def load_model_cpu():
    """Load model specifically for CPU processing"""
    return models_load_model_cpu()

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = models_load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = models_generate(
        model,
        text,
        audio_prompt_path,
        exaggeration,
        temperature,
        seed_num,
        cfgw
    )
    return wav

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
    return models_clear_gpu_memory()

def check_gpu_memory():
    """Check GPU memory status for troubleshooting"""
    return models_check_gpu_memory()

def adaptive_chunk_text(text, max_words=50, reduce_on_error=True):
    """Adaptively chunk text - using imported version"""
    return text_adaptive_chunk_text(text, max_words, reduce_on_error)

def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with retry logic AND automatic CPU selection for short text.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
    # Use the smart hybrid function from models.py
    try:
        wav, device_used = models_generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries
        )
        return wav
    except Exception as e:
        print(f"❌ Smart hybrid generation failed: {str(e)}")
        raise e

def generate_with_cpu_fallback(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight):
    """Generate audio with automatic CPU fallback for problematic CUDA errors"""
    
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            models_clear_gpu_memory()
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
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
        cpu_model = models_load_model_cpu()
        wav = cpu_model.generate(
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")

def force_cpu_processing():
    """Check if we should force CPU processing for stability"""
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return models_force_cpu_processing()

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks by sentences - using imported version"""
    return text_chunk_text_by_sentences(text, max_words)

def save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir="audiobook_projects"):
    """Save audio chunks as numbered WAV files - PHASE 4 REFACTOR: using imported version"""
    return audio_save_audio_chunks(audio_chunks, sample_rate, project_name, output_dir)

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
    """Load text content from a file - using imported version"""
    return text_load_text_file(file_path)

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate audiobook input - using imported version with Gradio wrapper"""
    is_valid, error_msg = text_validate_audiobook_input(text_content, selected_voice, project_name)
    
    if not is_valid:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            error_msg,
            gr.Audio(visible=False)
        )
    
    # If valid, show success message with stats
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
                