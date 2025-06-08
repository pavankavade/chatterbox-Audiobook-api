"""
# ==============================================================================
# MULTI-VOICE PROCESSOR MODULE
# ==============================================================================
# 
# This module provides sophisticated multi-voice processing capabilities for
# the Chatterbox Audiobook Studio refactored system. It handles voice assignment,
# character mapping, and coordinated multi-voice TTS generation.
# 
# **Key Features:**
# - **Character Voice Mapping**: Intelligent character-to-voice assignments
# - **Context-Aware Processing**: Understanding of dialogue and narrative structure
# - **Voice Coordination**: Seamless multi-character scene generation
# - **Performance Optimization**: Efficient processing of complex voice scenarios
# - **Original Compatibility**: Full compatibility with existing multi-voice workflows
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from pathlib import Path

# Import voice management
from .voice_manager import VoiceManager, VoiceProfile

# Import TTS engine
from core.tts_engine import RefactoredTTSEngine

# ==============================================================================
# VOICE ASSIGNMENT DATA STRUCTURES
# ==============================================================================

@dataclass
class VoiceAssignment:
    """
    Voice assignment configuration for a character or narrative element.
    """
    character_name: str
    voice_profile: str
    voice_weight: float = 1.0  # Relative importance for voice selection
    scene_context: str = ""    # Context where this assignment applies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'character_name': self.character_name,
            'voice_profile': self.voice_profile,
            'voice_weight': self.voice_weight,
            'scene_context': self.scene_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceAssignment':
        """Create from dictionary."""
        return cls(
            character_name=data.get('character_name', ''),
            voice_profile=data.get('voice_profile', ''),
            voice_weight=data.get('voice_weight', 1.0),
            scene_context=data.get('scene_context', '')
        )

@dataclass
class TextSegment:
    """
    A segment of text with associated voice and processing metadata.
    """
    text: str
    voice_profile: str
    character_name: str = ""
    segment_type: str = "dialogue"  # "dialogue", "narrative", "description"
    priority: int = 1               # Processing priority
    processing_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processing_options is None:
            self.processing_options = {}

# ==============================================================================
# MULTI-VOICE PROCESSOR CLASS
# ==============================================================================

class MultiVoiceProcessor:
    """
    Professional multi-voice processing system.
    
    This class provides comprehensive multi-voice capabilities including
    character voice mapping, intelligent text segmentation, and coordinated
    TTS generation for complex audiobook scenarios.
    
    **Processing Features:**
    - **Character Detection**: Automatic character identification in text
    - **Voice Assignment**: Intelligent voice-to-character mapping
    - **Context Analysis**: Understanding of narrative structure
    - **Batch Processing**: Efficient multi-segment generation
    - **Quality Consistency**: Maintaining voice characteristics across segments
    """
    
    def __init__(self, voice_manager: Optional[VoiceManager] = None):
        """
        Initialize the multi-voice processor.
        
        Args:
            voice_manager (Optional[VoiceManager]): Voice manager instance
        """
        self.voice_manager = voice_manager or VoiceManager()
        self.tts_engine = RefactoredTTSEngine()
        self.voice_assignments: Dict[str, VoiceAssignment] = {}
        self.default_voice_profile = ""
        self.narrator_voice_profile = ""
        
        print("✅ Multi-Voice Processor initialized")
    
    def set_voice_assignments(self, assignments: List[VoiceAssignment]) -> None:
        """
        Set character voice assignments.
        
        Args:
            assignments (List[VoiceAssignment]): List of voice assignments
        """
        self.voice_assignments.clear()
        
        for assignment in assignments:
            self.voice_assignments[assignment.character_name] = assignment
        
        print(f"📝 Set {len(assignments)} voice assignments")
    
    def add_voice_assignment(
        self,
        character_name: str,
        voice_profile: str,
        voice_weight: float = 1.0,
        scene_context: str = ""
    ) -> None:
        """
        Add a single voice assignment.
        
        Args:
            character_name (str): Name of the character
            voice_profile (str): Voice profile to assign
            voice_weight (float): Assignment weight
            scene_context (str): Context for assignment
        """
        assignment = VoiceAssignment(
            character_name=character_name,
            voice_profile=voice_profile,
            voice_weight=voice_weight,
            scene_context=scene_context
        )
        
        self.voice_assignments[character_name] = assignment
        print(f"✅ Added voice assignment: {character_name} → {voice_profile}")
    
    def set_default_voices(self, default_voice: str, narrator_voice: str = "") -> None:
        """
        Set default voice profiles for unassigned content.
        
        Args:
            default_voice (str): Default voice for unassigned dialogue
            narrator_voice (str): Narrator voice for narrative text
        """
        self.default_voice_profile = default_voice
        self.narrator_voice_profile = narrator_voice or default_voice
        
        print(f"🎭 Set default voices - Default: {default_voice}, Narrator: {self.narrator_voice_profile}")
    
    def detect_characters_in_text(self, text: str) -> Set[str]:
        """
        Detect potential character names in text using pattern matching.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Set[str]: Set of detected character names
        """
        characters = set()
        
        # Pattern for dialogue attribution (e.g., "John said", "Mary replied")
        dialogue_patterns = [
            r'"[^"]*",?\s*(\w+)\s+(?:said|replied|asked|whispered|shouted|exclaimed)',
            r'(\w+)\s+(?:said|replied|asked|whispered|shouted|exclaimed),?\s*"[^"]*"',
            r'"[^"]*"\s*-\s*(\w+)',
            r'(\w+):\s*"[^"]*"'
        ]
        
        for pattern in dialogue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                character_name = match.group(1).strip()
                if character_name and len(character_name) > 1:  # Filter out single letters
                    characters.add(character_name.title())
        
        return characters
    
    def segment_text_by_voice(self, text: str) -> List[TextSegment]:
        """
        Segment text into voice-specific chunks.
        
        Args:
            text (str): Input text to segment
            
        Returns:
            List[TextSegment]: List of segmented text with voice assignments
        """
        segments = []
        
        # Split text into sentences for processing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Analyze sentence for voice assignment
            voice_info = self._analyze_sentence_for_voice(sentence)
            
            segment = TextSegment(
                text=sentence.strip(),
                voice_profile=voice_info['voice_profile'],
                character_name=voice_info['character_name'],
                segment_type=voice_info['segment_type'],
                priority=voice_info['priority'],
                processing_options=voice_info['processing_options']
            )
            
            segments.append(segment)
        
        return segments
    
    def process_multi_voice_text(
        self,
        text: str,
        generation_options: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, Any]]:
        """
        Process text with multiple voices and generate audio segments.
        
        Args:
            text (str): Input text to process
            generation_options (Optional[Dict[str, Any]]): TTS generation options
            
        Returns:
            List[Tuple[int, Any]]: List of (sample_rate, audio_data) tuples
        """
        if generation_options is None:
            generation_options = {}
        
        # Segment text by voice
        segments = self.segment_text_by_voice(text)
        
        # Generate audio for each segment
        audio_segments = []
        
        for segment in segments:
            try:
                # Get voice profile path
                voice_profile = self._get_voice_profile_path(segment.voice_profile)
                
                if not voice_profile:
                    print(f"⚠️  Voice profile not found: {segment.voice_profile}, using default")
                    voice_profile = self._get_voice_profile_path(self.default_voice_profile)
                
                if not voice_profile:
                    print("❌ No valid voice profile available, skipping segment")
                    continue
                
                # Merge segment-specific options with global options
                segment_options = generation_options.copy()
                segment_options.update(segment.processing_options)
                
                # Generate audio
                sample_rate, audio_data = self.tts_engine.generate_speech(
                    text=segment.text,
                    audio_prompt_path=voice_profile,
                    exaggeration=segment_options.get('exaggeration', 1.0),
                    temperature=segment_options.get('temperature', 0.7),
                    cfg_weight=segment_options.get('cfg_weight', 3.0)
                )
                
                audio_segments.append((sample_rate, audio_data))
                
            except Exception as e:
                print(f"❌ Error generating audio for segment: {str(e)}")
                continue
        
        return audio_segments
    
    def generate_character_voice_map(self, text: str) -> Dict[str, str]:
        """
        Generate an automatic character-to-voice mapping for text.
        
        Args:
            text (str): Text to analyze for characters
            
        Returns:
            Dict[str, str]: Character to voice profile mapping
        """
        # Detect characters in text
        characters = self.detect_characters_in_text(text)
        
        # Get available voice profiles
        voice_profiles = self.voice_manager.get_voice_choices()
        
        # Create automatic mapping
        character_voice_map = {}
        
        for i, character in enumerate(sorted(characters)):
            if i < len(voice_profiles):
                character_voice_map[character] = voice_profiles[i]
            else:
                # Use default voice for overflow characters
                character_voice_map[character] = self.default_voice_profile
        
        return character_voice_map
    
    def optimize_voice_assignments(self) -> Dict[str, Any]:
        """
        Optimize voice assignments for better performance and quality.
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        results = {
            'total_assignments': len(self.voice_assignments),
            'optimizations_applied': [],
            'recommendations': []
        }
        
        # Check for duplicate voice assignments
        voice_usage = {}
        for char, assignment in self.voice_assignments.items():
            voice = assignment.voice_profile
            if voice not in voice_usage:
                voice_usage[voice] = []
            voice_usage[voice].append(char)
        
        # Identify overused voices
        for voice, characters in voice_usage.items():
            if len(characters) > 3:
                results['recommendations'].append(
                    f"Voice '{voice}' is assigned to {len(characters)} characters: {', '.join(characters)}. "
                    f"Consider using different voices for better character distinction."
                )
        
        # Check for missing voice profiles
        available_voices = set(self.voice_manager.get_voice_choices())
        
        for char, assignment in self.voice_assignments.items():
            if assignment.voice_profile not in available_voices:
                results['recommendations'].append(
                    f"Character '{char}' is assigned to missing voice '{assignment.voice_profile}'. "
                    f"This will fall back to the default voice."
                )
        
        return results
    
    def _analyze_sentence_for_voice(self, sentence: str) -> Dict[str, Any]:
        """
        Analyze a sentence to determine appropriate voice assignment.
        
        Args:
            sentence (str): Sentence to analyze
            
        Returns:
            Dict[str, Any]: Voice assignment information
        """
        # Default assignment
        voice_info = {
            'voice_profile': self.narrator_voice_profile,
            'character_name': 'Narrator',
            'segment_type': 'narrative',
            'priority': 1,
            'processing_options': {}
        }
        
        # Check for dialogue patterns
        dialogue_patterns = [
            (r'"[^"]*"', 'dialogue'),
            (r"'[^']*'", 'dialogue'),
            (r'"[^"]*"', 'dialogue')
        ]
        
        for pattern, segment_type in dialogue_patterns:
            if re.search(pattern, sentence):
                voice_info['segment_type'] = segment_type
                
                # Try to identify character
                character = self._extract_character_from_dialogue(sentence)
                if character and character in self.voice_assignments:
                    assignment = self.voice_assignments[character]
                    voice_info['voice_profile'] = assignment.voice_profile
                    voice_info['character_name'] = character
                else:
                    voice_info['voice_profile'] = self.default_voice_profile
                    voice_info['character_name'] = character or 'Unknown Speaker'
                
                break
        
        return voice_info
    
    def _extract_character_from_dialogue(self, sentence: str) -> Optional[str]:
        """
        Extract character name from dialogue sentence.
        
        Args:
            sentence (str): Sentence containing dialogue
            
        Returns:
            Optional[str]: Extracted character name or None
        """
        # Patterns for character identification
        patterns = [
            r'"[^"]*",?\s*(\w+)\s+(?:said|replied|asked|whispered|shouted|exclaimed)',
            r'(\w+)\s+(?:said|replied|asked|whispered|shouted|exclaimed)',
            r'"[^"]*"\s*-\s*(\w+)',
            r'(\w+):\s*"[^"]*"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                character_name = match.group(1).strip().title()
                if len(character_name) > 1:  # Filter out single letters
                    return character_name
        
        return None
    
    def _get_voice_profile_path(self, voice_profile_name: str) -> Optional[str]:
        """
        Get the full path to a voice profile's audio file.
        
        Args:
            voice_profile_name (str): Name of the voice profile
            
        Returns:
            Optional[str]: Path to voice audio file or None
        """
        if not voice_profile_name:
            return None
        
        try:
            profile, message = self.voice_manager.load_voice_profile(voice_profile_name)
            if profile and profile.audio_file:
                return profile.audio_file
        except Exception as e:
            print(f"⚠️  Error loading voice profile {voice_profile_name}: {e}")
        
        return None
    
    def get_processor_status(self) -> Dict[str, Any]:
        """
        Get comprehensive processor status information.
        
        Returns:
            Dict[str, Any]: Complete processor status
        """
        return {
            'voice_assignments': len(self.voice_assignments),
            'assigned_characters': list(self.voice_assignments.keys()),
            'default_voice': self.default_voice_profile,
            'narrator_voice': self.narrator_voice_profile,
            'available_voices': len(self.voice_manager.get_voice_choices()),
            'tts_engine_status': self.tts_engine.get_engine_status()
        }

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_voice_assignment(
    character_name: str,
    voice_profile: str,
    voice_weight: float = 1.0
) -> VoiceAssignment:
    """Create a voice assignment (convenience function)."""
    return VoiceAssignment(
        character_name=character_name,
        voice_profile=voice_profile,
        voice_weight=voice_weight
    )

def get_global_multi_voice_processor() -> MultiVoiceProcessor:
    """Get a global multi-voice processor instance."""
    return MultiVoiceProcessor()

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Multi-Voice Processor module loaded")
print("🎭 Multi-character voice processing ready for complex audiobook scenarios") 