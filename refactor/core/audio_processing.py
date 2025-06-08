"""
# ==============================================================================
# CORE AUDIO PROCESSING MODULE
# ==============================================================================
# 
# This module provides fundamental audio processing capabilities for the
# Chatterbox Audiobook Studio refactored system. It handles basic audio
# operations, format conversions, and quality management.
# 
# **Key Features:**
# - **Audio Format Handling**: Support for multiple audio formats (WAV, MP3)
# - **Sample Rate Management**: Consistent sample rate handling (24kHz)
# - **Quality Preservation**: Lossless audio processing where possible
# - **Memory Efficiency**: Optimized processing for large audiobook files
# - **Original Compatibility**: Maintains exact audio characteristics from original system
"""

import numpy as np
import torch
import torchaudio
import warnings
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# AUDIO CONSTANTS AND CONFIGURATION
# ==============================================================================

# Standard sample rate for ChatterboxTTS (from original system)
DEFAULT_SAMPLE_RATE = 24000

# Audio format configurations
SUPPORTED_FORMATS = ['wav', 'mp3', 'flac']
DEFAULT_FORMAT = 'wav'

# Quality settings
DEFAULT_BIT_DEPTH = 16
HIGH_QUALITY_BIT_DEPTH = 24

# ==============================================================================
# CORE AUDIO PROCESSING FUNCTIONS
# ==============================================================================

def normalize_audio_array(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio array to prevent clipping while preserving dynamics.
    
    Args:
        audio_array (np.ndarray): Input audio array
        
    Returns:
        np.ndarray: Normalized audio array
        
    **Normalization Strategy:**
    - **Peak Normalization**: Prevents clipping by scaling to maximum safe level
    - **Dynamic Preservation**: Maintains original dynamic range characteristics
    - **Quality Preservation**: Uses high-precision calculations
    """
    if len(audio_array) == 0:
        return audio_array
    
    # Find peak amplitude
    peak = np.max(np.abs(audio_array))
    
    if peak == 0:
        return audio_array
    
    # Normalize to 95% of maximum to prevent clipping
    normalized = audio_array * (0.95 / peak)
    
    return normalized.astype(np.float32)

def resample_audio(
    audio_array: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Resample audio to target sample rate using high-quality resampling.
    
    Args:
        audio_array (np.ndarray): Input audio array
        original_sample_rate (int): Original sample rate
        target_sample_rate (int): Target sample rate
        
    Returns:
        np.ndarray: Resampled audio array
        
    **Resampling Features:**
    - **High Quality**: Uses torchaudio's high-quality resampling
    - **Anti-Aliasing**: Prevents aliasing artifacts
    - **Precision**: Maintains audio quality during conversion
    """
    if original_sample_rate == target_sample_rate:
        return audio_array
    
    try:
        # Convert to tensor for torchaudio processing
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Add channel dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample using torchaudio
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate
        )
        resampled_tensor = resampler(audio_tensor)
        
        # Convert back to numpy and remove channel dimension if added
        resampled_array = resampled_tensor.squeeze().numpy()
        
        return resampled_array.astype(np.float32)
        
    except Exception as e:
        print(f"⚠️  Resampling failed, using linear interpolation: {e}")
        
        # Fallback to simple linear interpolation
        duration = len(audio_array) / original_sample_rate
        target_length = int(duration * target_sample_rate)
        
        indices = np.linspace(0, len(audio_array) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio_array)), audio_array)
        
        return resampled.astype(np.float32)

def load_audio_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load audio file with automatic format detection and conversion.
    
    Args:
        file_path (Union[str, Path]): Path to audio file
        
    Returns:
        Tuple[np.ndarray, int]: (audio_array, sample_rate)
        
    **Loading Features:**
    - **Format Auto-Detection**: Automatically detects audio format
    - **Quality Preservation**: Loads at original quality
    - **Error Handling**: Graceful handling of corrupted files
    - **Memory Efficiency**: Efficient loading for large files
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load using torchaudio
        audio_tensor, sample_rate = torchaudio.load(str(file_path))
        
        # Convert to numpy and handle multiple channels
        audio_array = audio_tensor.numpy()
        
        # Convert stereo to mono by averaging channels
        if audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)
        else:
            audio_array = audio_array[0]
        
        return audio_array.astype(np.float32), sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def save_audio_file(
    audio_array: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    format: str = DEFAULT_FORMAT,
    bit_depth: int = DEFAULT_BIT_DEPTH
) -> bool:
    """
    Save audio array to file with specified format and quality.
    
    Args:
        audio_array (np.ndarray): Audio data to save
        file_path (Union[str, Path]): Output file path
        sample_rate (int): Sample rate for output
        format (str): Audio format ('wav', 'mp3', 'flac')
        bit_depth (int): Bit depth for output (16 or 24)
        
    Returns:
        bool: True if saved successfully
        
    **Saving Features:**
    - **Format Support**: Multiple output formats
    - **Quality Control**: Configurable bit depth and sample rate
    - **Path Management**: Automatic directory creation
    - **Error Recovery**: Comprehensive error handling
    """
    file_path = Path(file_path)
    
    try:
        # Create output directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure audio is normalized and in correct format
        audio_array = normalize_audio_array(audio_array)
        
        # Convert to tensor for torchaudio
        if audio_array.ndim == 1:
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio_array).float()
        
        # Convert bit depth
        if bit_depth == 16:
            audio_tensor = (audio_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
        elif bit_depth == 24:
            audio_tensor = (audio_tensor * 8388607).clamp(-8388608, 8388607).to(torch.int32)
        
        # Save using torchaudio
        torchaudio.save(
            str(file_path),
            audio_tensor.float(),  # torchaudio.save expects float
            sample_rate,
            format=format
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save audio file {file_path}: {e}")
        return False

def combine_audio_arrays(
    audio_arrays: list,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    pause_duration: float = 0.0
) -> np.ndarray:
    """
    Combine multiple audio arrays into a single continuous array.
    
    Args:
        audio_arrays (list): List of audio arrays to combine
        sample_rate (int): Sample rate for pause calculation
        pause_duration (float): Pause duration between segments in seconds
        
    Returns:
        np.ndarray: Combined audio array
        
    **Combination Features:**
    - **Seamless Joining**: Smooth concatenation of audio segments
    - **Configurable Pauses**: Optional silence between segments
    - **Quality Preservation**: Maintains audio quality during combination
    - **Memory Efficiency**: Optimized for large numbers of segments
    """
    if not audio_arrays:
        return np.array([], dtype=np.float32)
    
    if len(audio_arrays) == 1:
        return audio_arrays[0].astype(np.float32)
    
    # Calculate pause samples
    pause_samples = int(pause_duration * sample_rate)
    pause_array = np.zeros(pause_samples, dtype=np.float32)
    
    # Combine arrays with pauses
    combined_segments = []
    
    for i, audio_array in enumerate(audio_arrays):
        # Add audio segment
        combined_segments.append(audio_array.astype(np.float32))
        
        # Add pause (except after last segment)
        if i < len(audio_arrays) - 1 and pause_samples > 0:
            combined_segments.append(pause_array)
    
    # Concatenate all segments
    combined_audio = np.concatenate(combined_segments)
    
    return combined_audio

def get_audio_duration(audio_array: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the duration of an audio array in seconds.
    
    Args:
        audio_array (np.ndarray): Audio array
        sample_rate (int): Sample rate
        
    Returns:
        float: Duration in seconds
    """
    return len(audio_array) / sample_rate

def validate_audio_array(audio_array: np.ndarray) -> Tuple[bool, str]:
    """
    Validate an audio array for common issues.
    
    Args:
        audio_array (np.ndarray): Audio array to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(audio_array, np.ndarray):
        return False, "Audio data must be a numpy array"
    
    if audio_array.size == 0:
        return False, "Audio array is empty"
    
    if audio_array.ndim > 2:
        return False, "Audio array has too many dimensions (max 2 for stereo)"
    
    if not np.issubdtype(audio_array.dtype, np.floating):
        if not np.issubdtype(audio_array.dtype, np.integer):
            return False, "Audio array must be numeric (float or int)"
    
    # Check for NaN or infinite values
    if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
        return False, "Audio array contains NaN or infinite values"
    
    # Check for extreme values that might indicate corruption
    if np.max(np.abs(audio_array)) > 100:  # Assuming normalized float audio
        return False, "Audio array contains suspiciously large values"
    
    return True, "Audio array is valid"

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Core Audio Processing module loaded")
print(f"🎵 Default sample rate: {DEFAULT_SAMPLE_RATE} Hz")
print(f"📁 Supported formats: {', '.join(SUPPORTED_FORMATS)}") 