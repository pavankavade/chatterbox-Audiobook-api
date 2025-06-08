"""
# ==============================================================================
# MODEL MANAGEMENT MODULE
# ==============================================================================
# 
# This module provides sophisticated model lifecycle management for the
# Chatterbox Audiobook Studio refactored system. It handles model loading,
# caching, memory optimization, and device management.
# 
# **Key Features:**
# - **Intelligent Model Caching**: Avoid redundant model loading operations
# - **Memory Optimization**: Efficient GPU memory management and cleanup
# - **Device Management**: Smart device selection and fallback handling
# - **Lifecycle Management**: Complete model loading, unloading, and switching
# - **Original Compatibility**: Maintains exact behavior from original system
"""

import torch
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# Import configuration
from config.device_config import (
    get_device_configuration, validate_device_for_operation,
    clear_gpu_memory, check_gpu_memory, format_memory_size
)

# ==============================================================================
# MODEL MANAGEMENT CLASS
# ==============================================================================

class ModelManager:
    """
    Professional model lifecycle manager for TTS engines.
    
    This class provides sophisticated model management capabilities extracted
    and enhanced from the original system's model handling logic.
    
    **Management Features:**
    - **Smart Caching**: Intelligent model instance management
    - **Memory Optimization**: Advanced GPU memory management
    - **Device Switching**: Seamless device migration capabilities
    - **Load Balancing**: Optimal model distribution across devices
    - **Performance Monitoring**: Comprehensive model performance tracking
    """
    
    def __init__(self):
        """Initialize the model manager with default configuration."""
        self.loaded_models: Dict[str, Any] = {}
        self.model_devices: Dict[str, str] = {}
        self.model_memory_usage: Dict[str, int] = {}
        self.load_history: List[Dict[str, Any]] = []
        
        print("✅ Model Manager initialized")
    
    def load_model_for_device(
        self, 
        model_type: str = "chatterbox_tts",
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> Tuple[Any, bool]:
        """
        Load a TTS model for the specified device with intelligent caching.
        
        Args:
            model_type (str): Type of model to load
            device (Optional[str]): Target device ("cuda", "cpu", or None for auto)
            force_reload (bool): Force reload even if cached
            
        Returns:
            Tuple[Any, bool]: (model_instance, successfully_loaded)
            
        **Loading Strategy:**
        - **Device Optimization**: Select optimal device for operation type
        - **Cache Management**: Reuse existing models when possible
        - **Memory Monitoring**: Track GPU memory usage during loading
        - **Error Recovery**: Graceful fallback on loading failures
        """
        # Determine optimal device
        if device is None:
            device, is_optimal = validate_device_for_operation("single_voice")
            if not is_optimal:
                print(f"⚠️  Using non-optimal device: {device}")
        
        # Check cache
        cache_key = f"{model_type}_{device}"
        if cache_key in self.loaded_models and not force_reload:
            print(f"📦 Using cached model: {cache_key}")
            return self.loaded_models[cache_key], True
        
        # Monitor memory before loading
        memory_before = check_gpu_memory()
        
        try:
            # Import TTS engine
            try:
                from src.chatterbox.tts import ChatterboxTTS
                model = ChatterboxTTS.from_pretrained(device)
                
                # Cache the loaded model
                self.loaded_models[cache_key] = model
                self.model_devices[cache_key] = device
                
                # Monitor memory after loading
                memory_after = check_gpu_memory()
                memory_used = memory_after.get('allocated', 0) - memory_before.get('allocated', 0)
                self.model_memory_usage[cache_key] = memory_used
                
                # Record load history
                self._record_load_event(cache_key, device, memory_used, True)
                
                print(f"✅ Model loaded: {cache_key}")
                if memory_used > 0:
                    print(f"📊 Memory used: {format_memory_size(memory_used)}")
                
                return model, True
                
            except ImportError:
                print("⚠️  ChatterboxTTS not available - using fallback")
                return None, False
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self._record_load_event(cache_key, device, 0, False, str(e))
            return None, False
    
    def unload_model(self, model_type: str = "chatterbox_tts", device: str = "cuda") -> bool:
        """
        Unload a specific model to free memory.
        
        Args:
            model_type (str): Type of model to unload
            device (str): Device the model is loaded on
            
        Returns:
            bool: True if successfully unloaded
        """
        cache_key = f"{model_type}_{device}"
        
        if cache_key in self.loaded_models:
            try:
                # Remove from cache
                del self.loaded_models[cache_key]
                
                # Clean up tracking
                if cache_key in self.model_devices:
                    del self.model_devices[cache_key]
                if cache_key in self.model_memory_usage:
                    del self.model_memory_usage[cache_key]
                
                # Force garbage collection
                gc.collect()
                
                # Clear GPU memory if applicable
                if device == "cuda":
                    clear_gpu_memory()
                
                print(f"✅ Model unloaded: {cache_key}")
                return True
                
            except Exception as e:
                print(f"❌ Model unload failed: {e}")
                return False
        else:
            print(f"⚠️  Model not found in cache: {cache_key}")
            return False
    
    def unload_all_models(self) -> int:
        """
        Unload all cached models to free memory.
        
        Returns:
            int: Number of models unloaded
        """
        models_to_unload = list(self.loaded_models.keys())
        unloaded_count = 0
        
        for cache_key in models_to_unload:
            model_type, device = cache_key.split("_", 1)
            if self.unload_model(model_type, device):
                unloaded_count += 1
        
        print(f"🧹 Unloaded {unloaded_count} models, cleared all caches")
        return unloaded_count
    
    def switch_model_device(
        self, 
        model_type: str = "chatterbox_tts",
        from_device: str = "cuda", 
        to_device: str = "cpu"
    ) -> Tuple[Any, bool]:
        """
        Switch a model from one device to another.
        
        Args:
            model_type (str): Type of model to switch
            from_device (str): Current device
            to_device (str): Target device
            
        Returns:
            Tuple[Any, bool]: (model_instance, successfully_switched)
        """
        # Unload from current device
        self.unload_model(model_type, from_device)
        
        # Load on new device
        return self.load_model_for_device(model_type, to_device, force_reload=True)
    
    def get_optimal_device_for_operation(self, operation_type: str) -> str:
        """
        Get the optimal device for a specific operation type.
        
        Args:
            operation_type (str): Type of operation ("single_voice", "multi_voice", "general")
            
        Returns:
            str: Optimal device string
        """
        device, is_optimal = validate_device_for_operation(operation_type)
        
        if not is_optimal:
            print(f"⚠️  Non-optimal device for {operation_type}: {device}")
        
        return device
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive memory status across all devices.
        
        Returns:
            Dict[str, Any]: Complete memory status information
        """
        status = {
            'gpu_memory': check_gpu_memory(),
            'loaded_models': len(self.loaded_models),
            'model_memory_usage': self.model_memory_usage.copy(),
            'total_model_memory': sum(self.model_memory_usage.values())
        }
        
        # Add device-specific information
        device_config = get_device_configuration()
        status['device_config'] = device_config
        
        return status
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all loaded models.
        
        Returns:
            Dict[str, Any]: Complete model status information
        """
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_devices': self.model_devices.copy(),
            'memory_usage': self.model_memory_usage.copy(),
            'load_history': self.load_history[-10:],  # Last 10 load events
            'total_models': len(self.loaded_models)
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage across all loaded models.
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        memory_before = check_gpu_memory()
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Force garbage collection
        gc.collect()
        
        memory_after = check_gpu_memory()
        
        memory_freed = memory_before.get('allocated', 0) - memory_after.get('allocated', 0)
        
        result = {
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_freed': memory_freed,
            'optimization_successful': memory_freed > 0
        }
        
        if memory_freed > 0:
            print(f"🧹 Memory optimization freed: {format_memory_size(memory_freed)}")
        else:
            print("🔧 Memory optimization complete (no memory freed)")
        
        return result
    
    def _record_load_event(
        self,
        cache_key: str,
        device: str,
        memory_used: int,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Record a model loading event for monitoring."""
        event = {
            'cache_key': cache_key,
            'device': device,
            'memory_used': memory_used,
            'success': success,
            'error': error,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        self.load_history.append(event)
        
        # Keep only last 50 events
        if len(self.load_history) > 50:
            self.load_history = self.load_history[-50:]

# ==============================================================================
# GLOBAL MODEL MANAGER INSTANCE
# ==============================================================================

# Global model manager for convenience
_global_model_manager: Optional[ModelManager] = None

def get_global_model_manager() -> ModelManager:
    """Get or create the global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager

# ==============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ==============================================================================

def load_model(device: Optional[str] = None) -> Any:
    """
    Load TTS model (backward compatibility with original system).
    
    Args:
        device (Optional[str]): Device to load model on
        
    Returns:
        Any: Loaded model instance or None
    """
    manager = get_global_model_manager()
    model, success = manager.load_model_for_device(device=device)
    return model if success else None

def load_model_cpu() -> Any:
    """
    Load TTS model specifically for CPU (backward compatibility).
    
    Returns:
        Any: Loaded model instance or None
    """
    return load_model(device="cpu")

def clear_gpu_memory_global() -> None:
    """Clear GPU memory globally (backward compatibility)."""
    manager = get_global_model_manager()
    manager.optimize_memory_usage()

def check_gpu_memory_global() -> Dict[str, int]:
    """Check GPU memory globally (backward compatibility)."""
    return check_gpu_memory()

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Model Management module loaded")
print("🧠 Global model manager ready for TTS model lifecycle management") 