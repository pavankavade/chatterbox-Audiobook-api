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
    
    if not project_name or not project_name.strip():
        return False, "❌ Please provide a project name"
    
    word_count = len(text_content.split())
    if word_count < 10:
        return False, "❌ Text content too short (minimum 10 words)"
    
    if word_count > 50000:
        return False, "❌ Text content too long (maximum 50,000 words for performance)"
    
    return True, ""


def parse_multi_voice_text(text: str) -> List[Dict[str, str]]:
    """Parse text with multi-voice format markers.
    
    Expected format:
    [CHARACTER_NAME] dialogue text (no colon needed, tag and dialogue can be on the same line)
    
    Args:
        text: Input text with character markers
        
    Returns:
        List of segments with character and text, e.g.,
        [{'character': 'Character1', 'text': 'Dialogue for character 1.'}, ...]
    """
    segments = []
    
    # Regex to find [CharacterName] tags
    # It captures the character name and the text that follows until the next tag or end of string
    # Using re.split to capture text between tags and the tags themselves
    parts = re.split(r'(\[[^\]]+\])', text)
    
    current_character = None
    buffer = ""
    
    for part in parts:
        if not part:
            continue # Skip empty parts that can result from re.split
            
        part_stripped = part.strip()
        if re.match(r'^\[[^\]]+\]$', part_stripped): # It's a character tag
            if current_character and buffer.strip():
                segments.append({
                    'character': current_character,
                    'text': buffer.strip()
                })
            current_character = part_stripped[1:-1] # Remove brackets
            buffer = ""
        else: # It's text content
            if current_character is None and part_stripped: # Text before any character tag
                # Assign to a default "Narrator" if no character tag precedes it.
                # This can be adjusted based on desired behavior for untagged leading text.
                segments.append({
                    'character': "Narrator", # Or None, if untagged leading text should be handled differently
                    'text': part_stripped
                })
                buffer = "" # Clear buffer as this part is processed
            elif current_character:
                buffer += part # Append to current character's text buffer
            # If no current_character and it's not leading text, this part might be ignored
            # or could be appended to a default narrator if strict tagging isn't enforced.
            # Current logic: only appends if current_character is set.
                
    # Add any remaining text in the buffer for the last character
    if current_character and buffer.strip():
        segments.append({
            'character': current_character,
            'text': buffer.strip()
        })
    elif not current_character and buffer.strip() and not segments: # Only if it's the *only* content
        # If the entire text has no tags, assign it all to Narrator
        segments.append({
            'character': "Narrator",
            'text': buffer.strip()
        })
        
    # Filter out any segments where the text is empty after stripping
    final_segments = [seg for seg in segments if seg['text']]
    
    # Debug: Print parsed segments by the module
    # print("Parsed Segments by text_processing.py module:", final_segments)
    return final_segments


def clean_character_name_from_text(text: str, voice_name: str) -> str:
    """Clean character name markers from text.
       The new parse_multi_voice_text in this module should handle name/dialogue separation.
       This function primarily acts as a pass-through or for minor cleanup.
    
    Args:
        text: Text that may contain character markers
        voice_name: Voice name (largely ignored by this simplified version)
        
    Returns:
        Cleaned text
    """
    # The parsing logic should have already separated the character name.
    # This function just ensures the text is stripped.
    return text.strip()


def chunk_multi_voice_segments(segments: List[Dict[str, str]], max_words: int = 50) -> List[Dict[str, str]]:
    """Chunk multi-voice segments while preserving character assignments.
    
    Args:
        segments: List of character segments
        max_words: Maximum words per chunk
        
    Returns:
        List of chunked segments with character assignments
    """
    chunked_segments = []
    
    for segment in segments:
        character = segment['character']
        text = segment['text']
        
        # Chunk the text for this character
        text_chunks = chunk_text_by_sentences(text, max_words)
        
        # Create segment for each chunk
        for chunk in text_chunks:
            chunked_segments.append({
                'character': character,
                'text': chunk
            })
    
    return chunked_segments


def validate_multi_voice_text(text_content: str, voice_library_path: str) -> Tuple[bool, str, List[str]]:
    """Validate multi-voice text format and extract characters.
    
    Args:
        text_content: Text to validate
        voice_library_path: Path to voice library
        
    Returns:
        tuple: (is_valid, error_message, character_list)
    """
    if not text_content or not text_content.strip():
        return False, "❌ Please provide text content", []
    
    # Parse segments to extract characters
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return False, "❌ No valid character segments found. Use format: [CHARACTER_NAME]: dialogue", []
    
    # Extract unique characters
    characters = list(set(segment['character'] for segment in segments))
    
    if len(characters) < 2:
        return False, "❌ Multi-voice requires at least 2 different characters", characters
    
    if len(characters) > 6:
        return False, "❌ Too many characters (maximum 6 for performance)", characters
    
    # Check if we have enough text
    total_words = sum(len(segment['text'].split()) for segment in segments)
    if total_words < 20:
        return False, "❌ Not enough text content (minimum 20 words)", characters
    
    return True, "", characters


def validate_multi_audiobook_input(text_content: str, voice_library_path: str, project_name: str) -> Tuple[bool, str]:
    """Validate input for multi-voice audiobook creation.
    
    Args:
        text_content: Text to validate
        voice_library_path: Path to voice library
        project_name: Project name
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not project_name or not project_name.strip():
        return False, "❌ Please provide a project name"
    
    is_valid, error_msg, _ = validate_multi_voice_text(text_content, voice_library_path)
    return is_valid, error_msg


def analyze_multi_voice_text(text_content: str, voice_library_path: str) -> Tuple[bool, str, Dict[str, int]]:
    """Analyze multi-voice text and return character statistics.
    
    Args:
        text_content: Text to analyze
        voice_library_path: Path to voice library
        
    Returns:
        tuple: (is_valid, message, character_counts)
    """
    is_valid, error_msg, characters = validate_multi_voice_text(text_content, voice_library_path)
    
    if not is_valid:
        return False, error_msg, {}
    
    # Parse segments and count words per character
    segments = parse_multi_voice_text(text_content)
    character_counts = {}
    
    for segment in segments:
        character = segment['character']
        word_count = len(segment['text'].split())
        character_counts[character] = character_counts.get(character, 0) + word_count
    
    total_words = sum(character_counts.values())
    message = f"✅ Found {len(characters)} characters with {total_words} total words"
    
    return True, message, character_counts


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


# PHASE 4 REFACTOR: Adding audio processing functions to this module
# Originally from gradio_tts_app_audiobook.py save_audio_chunks() function

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


# PHASE 4 REFACTOR: Adding extract_audio_segment function from gradio_tts_app_audiobook.py
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