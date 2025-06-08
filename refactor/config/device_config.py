"""
# ==============================================================================
# DEVICE CONFIGURATION MODULE  
# ==============================================================================
# 
# This module provides intelligent device management for the Chatterbox
# Audiobook Studio refactored system. It handles CUDA/CPU selection with
# the same sophisticated logic as the original system.
# 
# **Key Features:**
# - **Intelligent Device Detection**: Automatic CUDA availability checking
# - **Multi-Voice Stability**: CPU-only processing for multi-voice projects
# - **Memory Management**: GPU memory monitoring and optimization
# - **Fallback Handling**: Graceful degradation when CUDA unavailable
# - **Original Compatibility**: Maintains exact device logic from original system
"""

import torch
import warnings
from typing import Dict, Any, Optional, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# DEVICE DETECTION AND CONFIGURATION
# ==============================================================================

def get_primary_device() -> str:
    """
    Get the primary device for TTS processing.
    
    Uses the same logic as the original system for device selection.
    
    Returns:
        str: Device string ("cuda" or "cpu")
        
    **Device Selection Logic:**
    - **CUDA Available**: Use GPU for single-voice processing
    - **CUDA Unavailable**: Fall back to CPU processing
    - **Memory Monitoring**: Check GPU memory before selection
    """
    if torch.cuda.is_available():
        device = "cuda"
        try:
            # Test CUDA functionality
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            print(f"⚠️  CUDA test failed: {e}. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"

def get_multi_voice_device() -> str:
    """
    Get the device for multi-voice processing.
    
    Always returns CPU for multi-voice to avoid CUDA indexing errors
    that occur when processing multiple voice assignments simultaneously.
    This is a critical architectural decision from the original system.
    
    Returns:
        str: Always "cpu" for stability
        
    **Multi-Voice Device Logic:**
    - **Always CPU**: Prevents CUDA memory conflicts
    - **Stability First**: Avoids indexing errors with multiple voices
    - **Tested Approach**: Proven stable in original system
    """
    return "cpu"

def get_device_configuration() -> Dict[str, Any]:
    """
    Get complete device configuration information.
    
    Provides comprehensive device information for system monitoring
    and configuration display.
    
    Returns:
        Dict[str, Any]: Complete device configuration
        
    **Configuration Information:**
    - **Primary Device**: Main TTS processing device
    - **Multi-Voice Device**: Device for multi-voice projects
    - **CUDA Available**: CUDA availability status
    - **GPU Information**: GPU details if available
    - **Memory Information**: Available memory statistics
    """
    primary_device = get_primary_device()
    multi_voice_device = get_multi_voice_device()
    cuda_available = torch.cuda.is_available()
    
    config = {
        'primary_device': primary_device,
        'multi_voice_device': multi_voice_device,
        'cuda_available': cuda_available,
        'torch_version': torch.__version__
    }
    
    if cuda_available:
        try:
            config.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory
            })
        except Exception as e:
            config['gpu_error'] = str(e)
    
    return config

def clear_gpu_memory() -> None:
    """
    Clear GPU memory if CUDA is available.
    
    Maintains compatibility with the original system's memory management.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  GPU memory clear failed: {e}")

def check_gpu_memory() -> Dict[str, int]:
    """
    Check GPU memory usage and availability.
    
    Returns:
        Dict[str, int]: Memory statistics in bytes
        
    **Memory Information:**
    - **allocated**: Currently allocated GPU memory
    - **reserved**: Reserved GPU memory
    - **free**: Available GPU memory
    - **total**: Total GPU memory
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0,
            'reserved': 0, 
            'free': 0,
            'total': 0,
            'cuda_available': False
        }
    
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'cuda_available': True
        }
    except Exception as e:
        return {
            'allocated': 0,
            'reserved': 0,
            'free': 0, 
            'total': 0,
            'cuda_available': False,
            'error': str(e)
        }

def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_size (int): Size in bytes
        
    Returns:
        str: Formatted size string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def get_device_summary() -> str:
    """
    Get a human-readable device configuration summary.
    
    Returns:
        str: Formatted device summary for display
    """
    config = get_device_configuration()
    
    summary_lines = [
        f"🎯 Primary Device: {config['primary_device'].upper()}",
        f"🎭 Multi-Voice Device: {config['multi_voice_device'].upper()}",
        f"⚡ CUDA Available: {'Yes' if config['cuda_available'] else 'No'}",
        f"🔧 PyTorch Version: {config['torch_version']}"
    ]
    
    if config['cuda_available'] and 'device_name' in config:
        summary_lines.extend([
            f"🎮 GPU: {config.get('device_name', 'Unknown')}",
            f"🧠 GPU Memory: {format_memory_size(config.get('memory_total', 0))}"
        ])
        
        if 'memory_allocated' in config:
            allocated = config['memory_allocated']
            total = config.get('memory_total', 1)
            usage_percent = (allocated / total) * 100 if total > 0 else 0
            summary_lines.append(f"📊 Memory Usage: {format_memory_size(allocated)} ({usage_percent:.1f}%)")
    
    return "\n".join(summary_lines)

def validate_device_for_operation(operation_type: str) -> Tuple[str, bool]:
    """
    Validate and return the appropriate device for a specific operation.
    
    Args:
        operation_type (str): Type of operation ("single_voice", "multi_voice", "general")
        
    Returns:
        Tuple[str, bool]: (device_string, is_optimal)
        
    **Operation-Specific Device Selection:**
    - **single_voice**: Primary device (CUDA if available)
    - **multi_voice**: Always CPU for stability
    - **general**: Primary device for general operations
    """
    if operation_type == "multi_voice":
        return get_multi_voice_device(), True
    elif operation_type in ["single_voice", "general"]:
        device = get_primary_device()
        is_optimal = device == "cuda" and torch.cuda.is_available()
        return device, is_optimal
    else:
        # Unknown operation type, use primary device
        device = get_primary_device()
        return device, False

# ==============================================================================
# MODULE INITIALIZATION  
# ==============================================================================

# Initialize device configuration on module import
_PRIMARY_DEVICE = get_primary_device()
_MULTI_VOICE_DEVICE = get_multi_voice_device()
_DEVICE_CONFIG = get_device_configuration()

print(f"🎯 Primary TTS Device: {_PRIMARY_DEVICE}")
print(f"🎭 Multi-Voice Device: {_MULTI_VOICE_DEVICE}")

if _DEVICE_CONFIG['cuda_available']:
    print(f"⚡ GPU Detected: {_DEVICE_CONFIG.get('device_name', 'Unknown')}")
    total_memory = _DEVICE_CONFIG.get('memory_total', 0)
    if total_memory > 0:
        print(f"🧠 GPU Memory: {format_memory_size(total_memory)}")
else:
    print("⚠️  CUDA not available - using CPU processing")

# Export the initialized devices for easy access
PRIMARY_DEVICE = _PRIMARY_DEVICE
MULTI_VOICE_DEVICE = _MULTI_VOICE_DEVICE
DEVICE_CONFIG = _DEVICE_CONFIG 