"""
Text Processing Module

This module handles all text processing functionality including:
- Text chunking and segmentation
- Multi-voice parsing and detection
- Text cleaning and normalization
- Chunk optimization
- Voice tag processing
"""

# Import chunking functions
from .chunking import (
    chunk_text_by_sentences,
    adaptive_chunk_text,
    load_text_file,
    validate_audiobook_input,
    save_audio_chunks,
    extract_audio_segment
)

# Import multi-voice functions
from .multi_voice import (
    parse_multi_voice_text,
    clean_character_name_from_text,
    chunk_multi_voice_segments,
    validate_multi_voice_text,
    validate_multi_audiobook_input,
    analyze_multi_voice_text
)

__all__ = [
    # Chunking functions
    'chunk_text_by_sentences',
    'adaptive_chunk_text',
    'load_text_file',
    'validate_audiobook_input',
    'save_audio_chunks',
    'extract_audio_segment',
    
    # Multi-voice functions
    'parse_multi_voice_text',
    'clean_character_name_from_text',
    'chunk_multi_voice_segments',
    'validate_multi_voice_text',
    'validate_multi_audiobook_input',
    'analyze_multi_voice_text'
] 