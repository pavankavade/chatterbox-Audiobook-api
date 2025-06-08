"""
TTS Models Management Module

This module handles all TTS model-related functionality including:
- Model initialization and loading
- TTS engine management
- CPU/GPU model switching
- Model caching and optimization
- ChatterboxTTS integration
"""

# Import key functions to make them available at package level
from .tts_model import (
    set_seed,
    load_model,
    load_model_cpu,
    clear_gpu_memory,
    check_gpu_memory,
    generate,
    generate_with_cpu_fallback,
    generate_with_retry,
    generate_for_gradio,
    force_cpu_processing,
    get_model_device_str,
    DEVICE,
    MULTI_VOICE_DEVICE
)

__all__ = [
    'set_seed',
    'load_model',
    'load_model_cpu',
    'clear_gpu_memory',
    'check_gpu_memory',
    'generate',
    'generate_with_cpu_fallback',
    'generate_with_retry',
    'generate_for_gradio',
    'force_cpu_processing',
    'get_model_device_str',
    'DEVICE',
    'MULTI_VOICE_DEVICE'
] 