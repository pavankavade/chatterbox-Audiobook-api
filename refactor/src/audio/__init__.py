"""
Audio Management Module

This module handles all audio-related functionality including:
- Audio file I/O operations
- Audio processing and effects
- Quality analysis and validation
- Playback and streaming
- Audio format conversions
- Batch audio operations
"""

# Import file management functions
from .file_management import (
    save_audio_file,
    load_audio_file,
    get_audio_file_info,
    combine_audio_files,
    create_audio_manifest
)

__all__ = [
    # File management
    'save_audio_file',
    'load_audio_file',
    'get_audio_file_info',
    'combine_audio_files',
    'create_audio_manifest'
] 