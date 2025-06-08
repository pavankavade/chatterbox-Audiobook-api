"""
# ==============================================================================
# CONFIGURATION SETTINGS MODULE
# ==============================================================================
# 
# This module provides centralized configuration management for the Chatterbox
# Audiobook Studio refactored system. It maintains compatibility with the
# original configuration while providing a clean, modular interface.
# 
# **Key Features:**
# - **Centralized Configuration**: Single source of truth for all settings
# - **Original Compatibility**: Maintains exact compatibility with original system  
# - **Environment Support**: Development and production configurations
# - **Type Safety**: Proper type hints and validation
# - **Professional Standards**: PEP 8 compliance and comprehensive documentation
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# ==============================================================================
# SYSTEM CONFIGURATION CONSTANTS
# ==============================================================================

# Port configuration for parallel testing
REFACTORED_PORT = 7682  # Port for refactored system (original uses 7860)

# File and directory configuration (maintaining original compatibility)
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "audiobook_config.json"

# Interface limits and pagination settings (from original system)
MAX_CHUNKS_FOR_INTERFACE = 100
MAX_CHUNKS_FOR_AUTO_SAVE = 100

# Performance optimization constants
DEFAULT_MAX_WORDS_PER_CHUNK = 50
DEFAULT_AUTOSAVE_INTERVAL = 10

# Audio quality settings
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_TARGET_VOLUME_DB = -18.0  # ACX audiobook standard

# ==============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# ==============================================================================

def load_config() -> Dict[str, Any]:
    """
    Load application configuration from JSON file.
    
    Maintains exact compatibility with the original system while providing
    better error handling and type safety.
    
    Returns:
        Dict[str, Any]: Configuration dictionary with all settings
        
    **Compatibility Features:**
    - **Identical File Format**: Uses same JSON structure as original
    - **Graceful Fallback**: Handles missing or corrupted config files
    - **Default Values**: Provides sensible defaults for all settings
    """
    default_config = {
        'voice_library_path': DEFAULT_VOICE_LIBRARY,
        'max_chunks_interface': MAX_CHUNKS_FOR_INTERFACE,
        'max_chunks_autosave': MAX_CHUNKS_FOR_AUTO_SAVE,
        'default_target_volume': DEFAULT_TARGET_VOLUME_DB,
        'refactored_port': REFACTORED_PORT
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Merge with defaults to ensure all keys exist
            merged_config = default_config.copy()
            merged_config.update(config)
            return merged_config
            
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"⚠️  Config file error: {e}. Using defaults.")
            return default_config
    
    return default_config

def save_config(config: Dict[str, Any]) -> str:
    """
    Save application configuration to JSON file.
    
    Maintains compatibility with original system while providing enhanced
    error handling and validation.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        
    Returns:
        str: Success or error message for user feedback
        
    **Enhanced Features:**
    - **Input Validation**: Validates config structure before saving
    - **Atomic Writes**: Prevents corruption during save operation
    - **Detailed Feedback**: Provides clear success/error messages
    """
    try:
        # Validate required keys
        required_keys = ['voice_library_path']
        for key in required_keys:
            if key not in config:
                return f"❌ Missing required configuration key: {key}"
        
        # Add metadata
        config_with_metadata = config.copy()
        config_with_metadata.update({
            'last_updated': str(Path().resolve()),
            'refactored_system': True,
            'version': '1.0'
        })
        
        # Atomic write - write to temp file first
        temp_file = CONFIG_FILE + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(config_with_metadata, f, indent=2, ensure_ascii=False)
        
        # Move temp file to final location
        os.replace(temp_file, CONFIG_FILE)
        
        return f"✅ Configuration saved successfully - Voice library: {config['voice_library_path']}"
        
    except (PermissionError, OSError) as e:
        return f"❌ Error saving configuration: {str(e)}"

def get_system_config() -> Dict[str, Any]:
    """
    Get the complete system configuration.
    
    Provides a unified interface for accessing all system configuration
    settings with proper defaults and validation.
    
    Returns:
        Dict[str, Any]: Complete system configuration
        
    **System Configuration Categories:**
    - **File Paths**: Voice library, project directories
    - **Performance Limits**: Chunk limits, processing constraints  
    - **Audio Settings**: Sample rates, volume targets
    - **UI Settings**: Pagination, interface limits
    - **Network Settings**: Ports, connection parameters
    """
    return load_config()

def get_voice_library_path() -> str:
    """
    Get the configured voice library path.
    
    Returns:
        str: Path to the voice library directory
    """
    config = load_config()
    return config.get('voice_library_path', DEFAULT_VOICE_LIBRARY)

def update_voice_library_path(new_path: str) -> str:
    """
    Update the voice library path configuration.
    
    Args:
        new_path (str): New path to the voice library
        
    Returns:
        str: Success or error message
    """
    config = load_config()
    config['voice_library_path'] = new_path
    return save_config(config)

def get_performance_limits() -> Dict[str, int]:
    """
    Get performance-related configuration limits.
    
    Returns:
        Dict[str, int]: Performance limits configuration
    """
    config = load_config()
    return {
        'max_chunks_interface': config.get('max_chunks_interface', MAX_CHUNKS_FOR_INTERFACE),
        'max_chunks_autosave': config.get('max_chunks_autosave', MAX_CHUNKS_FOR_AUTO_SAVE),
        'max_words_per_chunk': config.get('max_words_per_chunk', DEFAULT_MAX_WORDS_PER_CHUNK),
        'autosave_interval': config.get('autosave_interval', DEFAULT_AUTOSAVE_INTERVAL)
    }

def get_audio_settings() -> Dict[str, float]:
    """
    Get audio-related configuration settings.
    
    Returns:
        Dict[str, float]: Audio configuration settings
    """
    config = load_config()
    return {
        'sample_rate': config.get('sample_rate', DEFAULT_SAMPLE_RATE),
        'target_volume_db': config.get('default_target_volume', DEFAULT_TARGET_VOLUME_DB)
    }

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    required_keys = ['voice_library_path']
    
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"
    
    # Validate voice library path
    voice_lib_path = config.get('voice_library_path')
    if not isinstance(voice_lib_path, str) or not voice_lib_path.strip():
        return False, "voice_library_path must be a non-empty string"
    
    # Validate numeric settings
    numeric_settings = {
        'max_chunks_interface': (1, 1000),
        'max_chunks_autosave': (1, 1000),
        'refactored_port': (1024, 65535)
    }
    
    for key, (min_val, max_val) in numeric_settings.items():
        if key in config:
            value = config[key]
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                return False, f"{key} must be between {min_val} and {max_val}"
    
    return True, "Configuration is valid"

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

# Load configuration on module import
_SYSTEM_CONFIG = load_config()

def get_loaded_config() -> Dict[str, Any]:
    """Get the configuration loaded at module import time."""
    return _SYSTEM_CONFIG.copy()

def get_refactored_config() -> Dict[str, Any]:
    """Get the refactored configuration instance (alias for compatibility)."""
    return get_loaded_config()

print(f"✅ Configuration module loaded - Voice library: {get_voice_library_path()}")
print(f"🌐 Refactored system port: {REFACTORED_PORT}")

def find_available_port(start_port: int = REFACTORED_PORT, max_attempts: int = 50) -> int:
    """
    Find an available port starting from the given port.
    
    Args:
        start_port (int): Starting port to check
        max_attempts (int): Maximum number of ports to try
        
    Returns:
        int: Available port number
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                print(f"✅ Found available port: {port}")
                return port
        except OSError:
            continue
    
    # If no port found, return the default
    print(f"⚠️  No available port found in range {start_port}-{start_port + max_attempts}, using default {start_port}")
    return start_port 