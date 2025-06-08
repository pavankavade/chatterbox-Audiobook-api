"""Model management and TTS operations for the audiobook generation system."""

import sys
import os
from typing import Any, Tuple, Optional

# CRITICAL: Ensure we're using the virtual environment's PyTorch
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'venv')
venv_site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)
    print(f"🔧 Forcing virtual environment PyTorch: {venv_site_packages}")

import torch
import random
import numpy as np

# Add parent directory to path for ChatterboxTTS import
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
    print("✅ ChatterboxTTS engine loaded successfully")
except ImportError as e:
    print(f"⚠️  Warning: ChatterboxTTS not available - {e}")
    CHATTERBOX_AVAILABLE = False
    
    # Create dummy class for when ChatterboxTTS is not available
    class ChatterboxTTS:
        pass


# Global device setting - will be imported from main file
import torch

def get_optimal_device() -> str:
    """Get the optimal device for TTS processing with detailed detection."""
    print(f"🔧 Device Detection:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"   GPU Device: {device_name}")
            print(f"   GPU Memory: {memory_gb} GB")
            print(f"   Selected TTS Device: cuda")
            return "cuda"
        except Exception as e:
            print(f"   GPU Detection Error: {e}")
            print(f"   Selected TTS Device: cpu (fallback)")
            return "cpu"
    else:
        print(f"   Selected TTS Device: cpu")
        return "cpu"

# Lazy device detection - will be set when first model is loaded
DEVICE = None
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice processing


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible generation.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model() -> Optional[ChatterboxTTS]:
    """Load TTS model for the default device.
    
    Returns:
        ChatterboxTTS: Loaded TTS model or None if not available
    """
    if not CHATTERBOX_AVAILABLE:
        print("❌ Cannot load model: ChatterboxTTS not available")
        return None
        
    global DEVICE
    if DEVICE is None:
        DEVICE = get_optimal_device()
    
    print(f"🚀 Loading ChatterboxTTS model on {DEVICE}...")
    model = ChatterboxTTS.from_pretrained(device=DEVICE)  # Explicitly pass device parameter
    print(f"✅ ChatterboxTTS model loaded successfully on {DEVICE}")
    return model


def load_model_cpu() -> ChatterboxTTS:
    """Load model specifically for CPU processing.
    
    Returns:
        ChatterboxTTS: CPU-loaded TTS model
    """
    model = ChatterboxTTS.from_pretrained("cpu")
    return model


def clear_gpu_memory() -> None:
    """Clear GPU memory cache to prevent CUDA errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_memory() -> str:
    """Check current GPU memory usage.
    
    Returns:
        str: GPU memory status information
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return f"GPU Memory - Allocated: {allocated//1024//1024}MB, Cached: {cached//1024//1024}MB"
    return "CUDA not available"


def generate(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    seed_num: int, 
    cfgw: float
) -> Tuple[int, np.ndarray]:
    """Generate audio from text using the TTS model.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter for generation
        temperature: Temperature for generation randomness
        seed_num: Random seed (0 for random)
        cfgw: CFG weight parameter
        
    Returns:
        tuple: (sample_rate, audio_array)
    """
    if model is None:
        global DEVICE
        if DEVICE is None:
            DEVICE = get_optimal_device()
        model = ChatterboxTTS.from_pretrained(device=DEVICE)

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


def generate_with_cpu_fallback(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    cfg_weight: float
) -> Tuple[Any, str]:
    """Generate audio with automatic CPU fallback for problematic CUDA errors.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter
        temperature: Temperature parameter
        cfg_weight: CFG weight parameter
        
    Returns:
        tuple: (audio_wav, device_used)
    """
    # Ensure device is detected
    global DEVICE
    if DEVICE is None:
        DEVICE = get_optimal_device()
    
    # First try with the provided model (should be GPU if available)
    if DEVICE == "cuda" and model is not None:
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
        # Load CPU model for fallback
        cpu_model = ChatterboxTTS.from_pretrained(device="cpu")
            
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


def generate_with_retry(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    cfg_weight: float, 
    max_retries: int = 3
) -> Tuple[Any, str]:
    """Generate audio with retry mechanism for robustness.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter
        temperature: Temperature parameter
        cfg_weight: CFG weight parameter
        max_retries: Maximum number of retry attempts
        
    Returns:
        tuple: (audio_wav, device_used)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return generate_with_cpu_fallback(
                model, text, audio_prompt_path, exaggeration, temperature, cfg_weight
            )
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                clear_gpu_memory()
    
    raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")


def force_cpu_processing() -> bool:
    """Check if we should force CPU processing for stability.
    
    Returns:
        bool: True if CPU processing should be forced
    """
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return True


def get_model_device_str(model_obj: Optional[ChatterboxTTS]) -> str:
    """Get the device string for a model object.
    
    Args:
        model_obj: TTS model instance
        
    Returns:
        str: Device information string
    """
    if model_obj is None:
        return "No model loaded"
    
    try:
        # Try to access model device info
        if hasattr(model_obj, 'device'):
            return f"Model device: {model_obj.device}"
        elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'device'):
            return f"Model device: {model_obj.model.device}"
        else:
            return "Device info unavailable"
    except Exception as e:
        return f"Error getting device info: {str(e)}"


def get_device_info() -> str:
    """Get comprehensive device information for the UI.
    
    Returns:
        str: Device information string
    """
    global DEVICE
    if DEVICE is None:
        DEVICE = get_optimal_device()
    
    return f"Device: {DEVICE}, CUDA Available: {torch.cuda.is_available()}, ChatterboxTTS Available: {CHATTERBOX_AVAILABLE}"


def generate_for_gradio(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    seed_num: int, 
    cfg_weight: float
) -> Optional[Tuple[int, np.ndarray]]:
    """Generate audio for Gradio interface with proper error handling.
    
    This function wraps the TTS generation with proper error handling and
    returns results in the format expected by Gradio audio components.
    
    Args:
        model: TTS model instance (None to auto-load)
        text: Text to convert to speech
        audio_prompt_path: Path to voice reference audio
        exaggeration: Voice exaggeration parameter
        temperature: Generation randomness
        seed_num: Random seed (0 for random)
        cfg_weight: Classifier-free guidance weight
        
    Returns:
        tuple: (sample_rate, audio_array) or None on failure
    """
    try:
        if not CHATTERBOX_AVAILABLE:
            print("❌ ChatterboxTTS not available")
            return None
            
        # Auto-load model if needed
        if model is None:
            model = load_model()
        
        # Set seed if specified
        if seed_num != 0:
            set_seed(int(seed_num))
        
        # Generate with retry mechanism
        wav, device_used = generate_with_retry(
            model, text, audio_prompt_path, exaggeration, temperature, cfg_weight
        )
        
        print(f"✅ Audio generated successfully using {device_used}")
        return (model.sr, wav.squeeze(0).numpy())
        
    except Exception as e:
        print(f"❌ TTS Generation failed: {str(e)}")
        return None 