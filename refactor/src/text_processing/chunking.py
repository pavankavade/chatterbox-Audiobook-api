"""
Text processing utilities for audiobook generation.

Handles text chunking, validation, multi-voice parsing, and text cleanup.
"""

import re
import os
import wave
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any


def chunk_text_by_sentences(text: str, max_words: int = 50) -> List[str]:
    """Split text into chunks, breaking at sentence boundaries after reaching max_words.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    # Split text into sentences using regex to handle multiple punctuation marks
    sentences = re.split(r'([.!?]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        # Add punctuation if it exists
        if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, start new chunk
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def adaptive_chunk_text(text: str, max_words: int = 50, reduce_on_error: bool = True) -> List[str]:
    """Adaptively chunk text with error handling.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        reduce_on_error: Whether to reduce chunk size on errors
        
    Returns:
        List of text chunks
    """
    return chunk_text_by_sentences(text, max_words)


def load_text_file(file_path: str) -> Tuple[str, str]:
    """Load text content from a file with encoding detection.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        tuple: (text_content, status_message)
    """
    if not file_path:
        return "", "No file selected"
    
    try:
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for older files
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            return "", "File is empty"
        
        return content.strip(), f"✅ Loaded {len(content.split())} words from file"
    
    except FileNotFoundError:
        return "", "❌ File not found"
    except Exception as e:
        return "", f"❌ Error reading file: {str(e)}"


def validate_audiobook_input(text_content: str, selected_voice: str, project_name: str) -> Tuple[bool, str]:
    """Validate input for single-voice audiobook creation.
    
    Args:
        text_content: Text to validate
        selected_voice: Selected voice name
        project_name: Project name
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text_content or not text_content.strip():
        return False, "❌ Please provide text content or upload a text file"
    
    if not selected_voice:
        return False, "❌ Please select a voice"
    
    # Check if user selected the separator instead of a voice
    if selected_voice and "─────── Voice Pool ───────" in selected_voice:
        return False, "❌ Please select a voice, not the separator"
    
    if not project_name or not project_name.strip():
        return False, "❌ Please provide a project name"
    
    word_count = len(text_content.split())
    if word_count < 10:
        return False, "❌ Text content too short (minimum 10 words)"
    
    if word_count > 50000:
        return False, "❌ Text content too long (maximum 50,000 words for performance)"
    
    return True, "✅ Input validation passed - ready to generate audiobook!"


def _filter_problematic_short_chunks(chunks: List[str], voice_assignments: Dict[str, str]) -> List[str]:
    """Filter out problematic short chunks that might cause TTS issues.
    
    Args:
        chunks: List of text chunks
        voice_assignments: Character to voice mappings
        
    Returns:
        Filtered list of chunks
    """
    filtered_chunks = []
    min_length = 10  # Minimum character length
    
    for chunk in chunks:
        # Skip very short chunks
        if len(chunk.strip()) < min_length:
            continue
        
        # Skip chunks that are just punctuation or whitespace
        if not re.search(r'[a-zA-Z]', chunk):
            continue
        
        filtered_chunks.append(chunk)
    
    return filtered_chunks


def save_audio_chunks(audio_chunks: List[np.ndarray], sample_rate: int, project_name: str, output_dir: str = "audiobook_projects") -> Tuple[List[str], str]:
    """
    Save audio chunks as numbered WAV files
    
    Args:
        audio_chunks: List of audio numpy arrays
        sample_rate: Sample rate for audio files
        project_name: Name of the project
        output_dir: Directory to save project files
        
    Returns:
        tuple: (list of saved file paths, project directory path)
    """
    if not project_name.strip():
        project_name = "untitled_audiobook"
    
    # Sanitize project name
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_project_name = safe_project_name.replace(' ', '_')
    
    # Create output directory
    project_dir = os.path.join(output_dir, safe_project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    saved_files = []
    
    for i, audio_chunk in enumerate(audio_chunks, 1):
        filename = f"{safe_project_name}_{i:03d}.wav"
        filepath = os.path.join(project_dir, filename)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert float32 to int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        saved_files.append(filepath)
    
    return saved_files, project_dir


def extract_audio_segment(audio_data, start_time: float = None, end_time: float = None) -> tuple:
    """Extract a specific time segment from audio data
    
    Args:
        audio_data: Tuple of (sample_rate, audio_array)
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)
        
    Returns:
        tuple: (extracted_audio_data, status_message)
    """
    if not audio_data or not isinstance(audio_data, tuple) or len(audio_data) != 2:
        return None, "❌ Invalid audio data"
    
    try:
        sample_rate, audio_array = audio_data
        
        if not hasattr(audio_array, 'shape'):
            return None, "❌ Invalid audio array"
        
        # Handle multi-dimensional arrays
        if len(audio_array.shape) > 1:
            # Take first channel if stereo
            audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()
        
        total_samples = len(audio_array)
        total_duration = total_samples / sample_rate
        
        # Calculate sample indices
        start_sample = 0 if start_time is None else int(start_time * sample_rate)
        end_sample = total_samples if end_time is None else int(end_time * sample_rate)
        
        # Ensure valid bounds
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        
        # Extract segment
        trimmed_audio = audio_array[start_sample:end_sample]
        
        trimmed_duration = len(trimmed_audio) / sample_rate
        
        status_msg = f"✅ Extracted segment: {trimmed_duration:.2f}s (from {start_time or 0:.2f}s to {end_time or total_duration:.2f}s)"
        
        return (sample_rate, trimmed_audio), status_msg
        
    except Exception as e:
        return None, f"❌ Error extracting segment: {str(e)}" 