"""
Audiobook generation package.

This module handles the complete audiobook generation pipeline:
- Text chunking
- TTS generation 
- Audio processing
- Project management
"""

from .generation import generate_single_voice_audiobook, AudiobookGenerator

__all__ = [
    'generate_single_voice_audiobook',
    'AudiobookGenerator'
] 