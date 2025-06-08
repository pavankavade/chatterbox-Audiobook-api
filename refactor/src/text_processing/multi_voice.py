"""Multi-voice text processing utilities for audiobook generation.

Handles multi-voice text parsing, character extraction, and validation.
"""

import re
from typing import List, Dict, Tuple
from .chunking import chunk_text_by_sentences


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