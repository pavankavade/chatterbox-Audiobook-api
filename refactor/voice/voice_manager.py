"""
# ==============================================================================
# VOICE MANAGEMENT MODULE
# ==============================================================================
# 
# This module provides comprehensive voice profile management for the Chatterbox
# Audiobook Studio refactored system. It handles voice creation, loading, editing,
# and deletion with complete compatibility to the original system.
# 
# **Key Features:**
# - **Voice Profile CRUD**: Complete create, read, update, delete operations
# - **Original Compatibility**: Maintains exact JSON structure and behavior
# - **Audio Processing**: Voice file validation and processing
# - **Volume Normalization**: Optional voice-level audio normalization
# - **Professional Standards**: Type safety, validation, and comprehensive documentation
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Import audio processing
from core.audio_processing import load_audio_file, save_audio_file, validate_audio_array

# Import configuration
from config.settings import get_voice_library_path, DEFAULT_TARGET_VOLUME_DB

# ==============================================================================
# VOICE PROFILE STRUCTURE AND VALIDATION
# ==============================================================================

class VoiceProfile:
    """
    Professional voice profile data structure.
    
    This class encapsulates all voice profile information with validation
    and maintains exact compatibility with the original system's JSON format.
    """
    
    def __init__(
        self,
        voice_name: str,
        display_name: str = "",
        description: str = "",
        audio_file: str = "",
        exaggeration: float = 1.0,
        cfg_weight: float = 3.0,
        temperature: float = 0.7,
        enable_normalization: bool = False,
        target_level_db: float = DEFAULT_TARGET_VOLUME_DB
    ):
        """
        Initialize a voice profile.
        
        Args:
            voice_name (str): Unique voice identifier
            display_name (str): Human-readable voice name
            description (str): Voice description
            audio_file (str): Path to voice audio file
            exaggeration (float): Voice exaggeration level (0.0-2.0)
            cfg_weight (float): Classifier-free guidance weight
            temperature (float): Generation randomness (0.0-1.0)
            enable_normalization (bool): Enable volume normalization
            target_level_db (float): Target volume level in dB
        """
        self.voice_name = voice_name
        self.display_name = display_name or voice_name
        self.description = description
        self.audio_file = audio_file
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.enable_normalization = enable_normalization
        self.target_level_db = target_level_db
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voice profile to dictionary (original JSON format)."""
        return {
            'voice_name': self.voice_name,
            'display_name': self.display_name,
            'description': self.description,
            'audio_file': self.audio_file,
            'exaggeration': self.exaggeration,
            'cfg_weight': self.cfg_weight,
            'temperature': self.temperature,
            'enable_normalization': self.enable_normalization,
            'target_level_db': self.target_level_db
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create voice profile from dictionary (original JSON format)."""
        return cls(
            voice_name=data.get('voice_name', ''),
            display_name=data.get('display_name', ''),
            description=data.get('description', ''),
            audio_file=data.get('audio_file', ''),
            exaggeration=data.get('exaggeration', 1.0),
            cfg_weight=data.get('cfg_weight', 3.0),
            temperature=data.get('temperature', 0.7),
            enable_normalization=data.get('enable_normalization', False),
            target_level_db=data.get('target_level_db', DEFAULT_TARGET_VOLUME_DB)
        )
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate voice profile data.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not self.voice_name.strip():
            return False, "Voice name cannot be empty"
        
        if not self.display_name.strip():
            return False, "Display name cannot be empty"
        
        if self.audio_file and not os.path.exists(self.audio_file):
            return False, f"Audio file not found: {self.audio_file}"
        
        if not (0.0 <= self.exaggeration <= 2.0):
            return False, "Exaggeration must be between 0.0 and 2.0"
        
        if not (0.0 <= self.temperature <= 1.0):
            return False, "Temperature must be between 0.0 and 1.0"
        
        if not (0.0 <= self.cfg_weight <= 10.0):
            return False, "CFG weight must be between 0.0 and 10.0"
        
        if not (-60.0 <= self.target_level_db <= 0.0):
            return False, "Target level must be between -60.0 and 0.0 dB"
        
        return True, "Voice profile is valid"

# ==============================================================================
# VOICE MANAGER CLASS
# ==============================================================================

class VoiceManager:
    """
    Professional voice profile management system.
    
    This class provides comprehensive voice management functionality extracted
    and enhanced from the original system with improved error handling and
    professional architecture.
    
    **Management Features:**
    - **Profile CRUD**: Complete create, read, update, delete operations
    - **Library Management**: Voice library organization and maintenance
    - **Audio Processing**: Voice file validation and optional normalization
    - **Compatibility**: Maintains exact behavior from original system
    - **Validation**: Comprehensive data validation and error handling
    """
    
    def __init__(self, voice_library_path: Optional[str] = None):
        """
        Initialize the voice manager.
        
        Args:
            voice_library_path (Optional[str]): Path to voice library directory
        """
        self.voice_library_path = voice_library_path or get_voice_library_path()
        self.ensure_voice_library_exists()
        print(f"✅ Voice Manager initialized - Library: {self.voice_library_path}")
    
    def ensure_voice_library_exists(self) -> None:
        """Ensure the voice library directory exists."""
        library_path = Path(self.voice_library_path)
        library_path.mkdir(parents=True, exist_ok=True)
    
    def get_voice_profiles(self) -> List[Dict[str, str]]:
        """
        Get list of all voice profiles in the library.
        
        Maintains exact compatibility with original get_voice_profiles function.
        
        Returns:
            List[Dict[str, str]]: List of voice profile info dictionaries
            
        **Profile Info Structure:**
        - **name**: Voice profile name
        - **display_name**: Human-readable name
        - **description**: Voice description
        - **audio_file**: Path to voice audio file
        """
        profiles = []
        library_path = Path(self.voice_library_path)
        
        if not library_path.exists():
            return profiles
        
        try:
            for item in library_path.iterdir():
                if item.is_dir():
                    config_file = item / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            profiles.append({
                                'name': item.name,
                                'display_name': config.get('display_name', item.name),
                                'description': config.get('description', ''),
                                'audio_file': config.get('audio_file', '')
                            })
                        except Exception as e:
                            print(f"⚠️  Error reading voice config {config_file}: {e}")
            
            # Sort by display name for consistent ordering
            profiles.sort(key=lambda x: x['display_name'].lower())
            
        except Exception as e:
            print(f"❌ Error reading voice library: {e}")
        
        return profiles
    
    def save_voice_profile(
        self,
        voice_name: str,
        display_name: str,
        description: str,
        audio_file: Union[str, Path, Any],
        exaggeration: float = 1.0,
        cfg_weight: float = 3.0,
        temperature: float = 0.7,
        enable_normalization: bool = False,
        target_level_db: float = DEFAULT_TARGET_VOLUME_DB
    ) -> str:
        """
        Save a voice profile with complete original system compatibility.
        
        Extracted and enhanced from original save_voice_profile function.
        
        Args:
            voice_name (str): Unique voice identifier
            display_name (str): Human-readable voice name
            description (str): Voice description
            audio_file: Audio file (path, Gradio file object, or uploaded file)
            exaggeration (float): Voice exaggeration level
            cfg_weight (float): Classifier-free guidance weight
            temperature (float): Generation randomness
            enable_normalization (bool): Enable volume normalization
            target_level_db (float): Target volume level in dB
            
        Returns:
            str: Success or error message
            
        **Saving Process:**
        - **Profile Validation**: Comprehensive data validation
        - **Audio Processing**: File copying and optional normalization
        - **Directory Creation**: Automatic voice directory creation
        - **Config Generation**: JSON configuration file creation
        - **Error Recovery**: Detailed error handling and cleanup
        """
        try:
            # Validate inputs
            if not voice_name or not voice_name.strip():
                return "❌ Voice name cannot be empty"
            
            if not display_name or not display_name.strip():
                return "❌ Display name cannot be empty"
            
            # Create voice profile
            profile = VoiceProfile(
                voice_name=voice_name.strip(),
                display_name=display_name.strip(),
                description=description.strip(),
                audio_file="",  # Will be set after processing
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                enable_normalization=enable_normalization,
                target_level_db=target_level_db
            )
            
            # Validate profile
            is_valid, error_msg = profile.validate()
            if not is_valid:
                return f"❌ {error_msg}"
            
            # Create voice directory
            voice_dir = Path(self.voice_library_path) / voice_name
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            # Process audio file
            audio_result = self._process_voice_audio_file(
                audio_file, voice_dir, enable_normalization, target_level_db
            )
            
            if not audio_result.startswith("✅"):
                return audio_result  # Error message
            
            # Set audio file path in profile
            profile.audio_file = str(voice_dir / "voice.wav")
            
            # Save configuration
            config_file = voice_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
            
            return f"✅ Voice profile '{display_name}' saved successfully"
            
        except Exception as e:
            return f"❌ Error saving voice profile: {str(e)}"
    
    def load_voice_profile(self, voice_name: str) -> Tuple[Optional[VoiceProfile], str]:
        """
        Load a voice profile from the library.
        
        Args:
            voice_name (str): Name of voice profile to load
            
        Returns:
            Tuple[Optional[VoiceProfile], str]: (profile_object, status_message)
        """
        try:
            voice_dir = Path(self.voice_library_path) / voice_name
            config_file = voice_dir / "config.json"
            
            if not config_file.exists():
                return None, f"❌ Voice profile '{voice_name}' not found"
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            profile = VoiceProfile.from_dict(config_data)
            
            # Validate profile
            is_valid, error_msg = profile.validate()
            if not is_valid:
                return None, f"❌ Invalid voice profile: {error_msg}"
            
            return profile, f"✅ Voice profile '{voice_name}' loaded successfully"
            
        except Exception as e:
            return None, f"❌ Error loading voice profile: {str(e)}"
    
    def delete_voice_profile(self, voice_name: str) -> str:
        """
        Delete a voice profile from the library.
        
        Args:
            voice_name (str): Name of voice profile to delete
            
        Returns:
            str: Success or error message
        """
        try:
            voice_dir = Path(self.voice_library_path) / voice_name
            
            if not voice_dir.exists():
                return f"❌ Voice profile '{voice_name}' not found"
            
            # Remove the entire voice directory
            shutil.rmtree(voice_dir)
            
            return f"✅ Voice profile '{voice_name}' deleted successfully"
            
        except Exception as e:
            return f"❌ Error deleting voice profile: {str(e)}"
    
    def get_voice_choices(self) -> List[str]:
        """
        Get list of voice choices for UI dropdowns.
        
        Maintains compatibility with original get_voice_choices function.
        
        Returns:
            List[str]: List of voice profile names
        """
        profiles = self.get_voice_profiles()
        return [profile['name'] for profile in profiles]
    
    def get_audiobook_voice_choices(self) -> List[str]:
        """
        Get voice choices formatted for audiobook creation.
        
        Maintains compatibility with original get_audiobook_voice_choices function.
        
        Returns:
            List[str]: List of formatted voice choices
        """
        profiles = self.get_voice_profiles()
        choices = []
        
        for profile in profiles:
            display_name = profile.get('display_name', profile['name'])
            choice = f"{display_name} ({profile['name']})"
            choices.append(choice)
        
        return choices
    
    def refresh_voice_list(self) -> List[Dict[str, str]]:
        """
        Refresh and return the voice profile list.
        
        Returns:
            List[Dict[str, str]]: Updated list of voice profiles
        """
        return self.get_voice_profiles()
    
    def _process_voice_audio_file(
        self,
        audio_file: Union[str, Path, Any],
        voice_dir: Path,
        enable_normalization: bool,
        target_level_db: float
    ) -> str:
        """
        Process and save voice audio file.
        
        Args:
            audio_file: Input audio file (various formats)
            voice_dir (Path): Voice directory to save to
            enable_normalization (bool): Apply volume normalization
            target_level_db (float): Target volume level
            
        Returns:
            str: Success or error message
        """
        try:
            # Handle different audio file input types
            if hasattr(audio_file, 'name'):
                # Gradio file object
                input_path = Path(audio_file.name)
            elif isinstance(audio_file, (str, Path)):
                # File path
                input_path = Path(audio_file)
            else:
                return "❌ Unsupported audio file format"
            
            if not input_path.exists():
                return f"❌ Audio file not found: {input_path}"
            
            # Load audio data
            audio_array, sample_rate = load_audio_file(input_path)
            
            # Validate audio
            is_valid, error_msg = validate_audio_array(audio_array)
            if not is_valid:
                return f"❌ Invalid audio: {error_msg}"
            
            # Apply normalization if enabled
            if enable_normalization:
                # Simple volume normalization (more advanced normalization in audio module)
                from core.audio_processing import normalize_audio_array
                audio_array = normalize_audio_array(audio_array)
            
            # Save processed audio
            output_path = voice_dir / "voice.wav"
            success = save_audio_file(audio_array, output_path, sample_rate)
            
            if success:
                return f"✅ Voice audio saved: {output_path}"
            else:
                return "❌ Failed to save voice audio"
                
        except Exception as e:
            return f"❌ Error processing voice audio: {str(e)}"

# ==============================================================================
# GLOBAL VOICE MANAGER INSTANCE
# ==============================================================================

# Global voice manager for convenience
_global_voice_manager: Optional[VoiceManager] = None

def get_global_voice_manager() -> VoiceManager:
    """Get or create the global voice manager instance."""
    global _global_voice_manager
    if _global_voice_manager is None:
        _global_voice_manager = VoiceManager()
    return _global_voice_manager

# ==============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ==============================================================================

def get_voice_profiles(voice_library_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Get voice profiles (backward compatibility)."""
    if voice_library_path:
        manager = VoiceManager(voice_library_path)
    else:
        manager = get_global_voice_manager()
    return manager.get_voice_profiles()

def save_voice_profile(
    voice_library_path: str,
    voice_name: str,
    display_name: str,
    description: str,
    audio_file: Any,
    exaggeration: float = 1.0,
    cfg_weight: float = 3.0,
    temperature: float = 0.7,
    enable_normalization: bool = False,
    target_level_db: float = DEFAULT_TARGET_VOLUME_DB
) -> str:
    """Save voice profile (backward compatibility)."""
    manager = VoiceManager(voice_library_path)
    return manager.save_voice_profile(
        voice_name=voice_name,
        display_name=display_name,
        description=description,
        audio_file=audio_file,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        enable_normalization=enable_normalization,
        target_level_db=target_level_db
    )

def load_voice_profile(voice_library_path: str, voice_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load voice profile (backward compatibility)."""
    manager = VoiceManager(voice_library_path)
    profile, message = manager.load_voice_profile(voice_name)
    
    if profile:
        return profile.to_dict(), message
    else:
        return None, message

def delete_voice_profile(voice_library_path: str, voice_name: str) -> str:
    """Delete voice profile (backward compatibility)."""
    manager = VoiceManager(voice_library_path)
    return manager.delete_voice_profile(voice_name)

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Voice Management module loaded")
print(f"🎭 Voice profile management ready for library operations") 