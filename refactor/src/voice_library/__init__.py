"""
Voice Library Management Module

This module handles all voice library functionality including:
- Voice profile creation and management
- Voice library configuration and paths
- Voice loading and validation
- Voice testing and preview
- Voice file organization
"""

from .voice_management import (
    ensure_voice_library_exists,
    get_voice_profiles,
    get_voice_choices,
    load_voice_profile,
    save_voice_profile,
    delete_voice_profile,
    test_voice_profile,
    get_voice_config,
    validate_voice_audio
)

__all__ = [
    'ensure_voice_library_exists',
    'get_voice_profiles',
    'get_voice_choices',
    'load_voice_profile',
    'save_voice_profile',
    'delete_voice_profile',
    'test_voice_profile',
    'get_voice_config',
    'validate_voice_audio'
] 