"""
Voice management module for Chatterbox Audiobook Studio.

This module provides comprehensive voice profile management and voice library
operations for the refactored audiobook studio system.
"""

from .voice_manager import *
from .voice_library import *
from .multi_voice_processor import *

__all__ = [
    'VoiceManager', 'VoiceLibrary', 'MultiVoiceProcessor',
    'get_voice_profiles', 'save_voice_profile', 'load_voice_profile'
] 