"""
Voice Library Management System

This module provides comprehensive voice profile management functionality,
extracted and enhanced from the original monolithic application.

Features:
- Voice profile creation and configuration
- Voice library organization and validation
- Voice testing and preview capabilities
- Voice parameter management and optimization
- Integration with TTS models and audio processing
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import configuration and audio processing
try:
    from ..config.settings import config
    from ..models.tts_model import load_model, generate_for_gradio
except ImportError:
    from src.config.settings import config
    from src.models.tts_model import load_model, generate_for_gradio


def ensure_voice_library_exists(voice_library_path: str) -> str:
    """
    Ensure the voice library directory exists.
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        str: Confirmation message
    """
    try:
        Path(voice_library_path).mkdir(parents=True, exist_ok=True)
        return f"✅ Voice library ready at: {voice_library_path}"
    except Exception as e:
        return f"❌ Error creating voice library: {str(e)}"


def get_voice_profiles(voice_library_path: str) -> List[Dict]:
    """
    Get all voice profiles from the voice library.
    Includes both subfolder profiles and legacy JSON profiles.
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        List[Dict]: List of voice profile dictionaries
    """
    profiles = []
    
    try:
        voice_dir = Path(voice_library_path)
        if not voice_dir.exists():
            return profiles
        
        # 1. Get subfolder profiles (modern format)
        for profile_dir in voice_dir.iterdir():
            if profile_dir.is_dir():
                config_file = profile_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            profile_data = json.load(f)
                            profile_data['name'] = profile_dir.name
                            profile_data['path'] = str(profile_dir)
                            profile_data['profile_type'] = 'subfolder'
                            profiles.append(profile_data)
                    except Exception as e:
                        print(f"⚠️ Error loading profile {profile_dir.name}: {e}")
                        continue
                
                # Check for legacy voice folders (reference.wav + any .json file)
                else:
                    reference_wav = profile_dir / "reference.wav"
                    if reference_wav.exists():
                        # Look for any JSON file in the folder
                        json_files = list(profile_dir.glob("*.json"))
                        if json_files:
                            # Use the first JSON file found
                            json_file = json_files[0]
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    profile_data = json.load(f)
                                    profile_data['name'] = profile_dir.name
                                    profile_data['path'] = str(profile_dir)
                                    profile_data['audio_file'] = "reference.wav"
                                    profile_data['profile_type'] = 'legacy_folder'
                                    
                                    # Set display name if not present
                                    if 'display_name' not in profile_data:
                                        profile_data['display_name'] = profile_dir.name.replace('_', ' ').title()
                                    
                                    profiles.append(profile_data)
                                    print(f"✅ Loaded legacy voice folder: {profile_dir.name}")
                            except Exception as e:
                                print(f"⚠️ Error loading legacy folder {profile_dir.name}: {e}")
                                continue
                        else:
                            # Has reference.wav but no JSON - create basic profile
                            profile_data = {
                                'name': profile_dir.name,
                                'display_name': profile_dir.name.replace('_', ' ').title(),
                                'description': f"Legacy voice: {profile_dir.name}",
                                'audio_file': "reference.wav",
                                'exaggeration': 0.5,
                                'cfg_weight': 0.5,
                                'temperature': 0.8,
                                'path': str(profile_dir),
                                'profile_type': 'legacy_folder_no_config'
                            }
                            profiles.append(profile_data)
                            print(f"✅ Loaded legacy voice folder (no config): {profile_dir.name}")
        
        # 2. Get legacy JSON profiles (in main directory)
        for json_file in voice_dir.glob("*.json"):
            voice_name = json_file.stem
            wav_file = voice_dir / f"{voice_name}.wav"
            
            # Skip if this is already covered by a subfolder
            if (voice_dir / voice_name).is_dir():
                continue
                
            if wav_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                        profile_data['name'] = voice_name
                        profile_data['path'] = str(voice_dir)
                        profile_data['audio_file'] = f"{voice_name}.wav"
                        profile_data['profile_type'] = 'legacy_json'
                        
                        # Set display name if not present
                        if 'display_name' not in profile_data:
                            profile_data['display_name'] = voice_name.replace('_', ' ').title()
                        
                        profiles.append(profile_data)
                except Exception as e:
                    print(f"⚠️ Error loading legacy profile {voice_name}: {e}")
                    continue
        
        # Sort by display name
        profiles.sort(key=lambda x: x.get('display_name', x.get('name', '')))
        
    except Exception as e:
        print(f"❌ Error scanning voice library: {e}")
    
    return profiles


def get_legacy_voices(voice_library_path: str) -> List[str]:
    """
    Get voices from legacy format (just .wav files in main directory).
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        List[str]: List of legacy voice names
    """
    legacy_voices = []
    
    try:
        voice_dir = Path(voice_library_path)
        if not voice_dir.exists():
            return legacy_voices
        
        # Look for .wav files in the main directory
        for wav_file in voice_dir.glob("*.wav"):
            voice_name = wav_file.stem
            # Skip if this voice already has a modern directory
            voice_profile_dir = voice_dir / voice_name
            if not voice_profile_dir.exists():
                legacy_voices.append(voice_name)
        
        legacy_voices.sort()
        
    except Exception as e:
        print(f"❌ Error scanning legacy voices: {e}")
    
    return legacy_voices


def convert_legacy_voice_to_profile(voice_library_path: str, voice_name: str) -> Tuple[bool, str]:
    """
    Convert a legacy voice (.wav file) to new profile format.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Name of the legacy voice to convert
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        voice_dir = Path(voice_library_path)
        wav_file = voice_dir / f"{voice_name}.wav"
        json_file = voice_dir / f"{voice_name}.json"
        
        if not wav_file.exists():
            return False, f"Legacy voice file {wav_file} not found"
        
        # Create profile directory
        profile_dir = voice_dir / voice_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy audio file
        dest_audio = profile_dir / "reference_audio.wav"
        shutil.copy2(wav_file, dest_audio)
        
        # Load existing JSON config if available
        existing_config = {}
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            except:
                pass
        
        # Create profile config
        config_data = {
            'display_name': existing_config.get('display_name', voice_name.replace('_', ' ').title()),
            'description': existing_config.get('description', f"Legacy voice profile: {voice_name}"),
            'audio_file': "reference_audio.wav",
            'exaggeration': existing_config.get('exaggeration', 0.5),
            'cfg_weight': existing_config.get('cfg_weight', 0.5),
            'temperature': existing_config.get('temperature', 0.8),
            'enable_normalization': False,
            'target_level_db': -18.0,
            'created_date': str(wav_file.stat().st_mtime),
            'sample_rate': 24000,
            'legacy_converted': True
        }
        
        # Save profile config
        config_file = profile_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        return True, f"✅ Converted legacy voice {voice_name} to profile format"
        
    except Exception as e:
        return False, f"❌ Error converting legacy voice {voice_name}: {str(e)}"


def get_voice_choices(voice_library_path: str) -> List[str]:
    """
    Get a list of voice names for dropdown choices.
    Includes both modern profiles and legacy voices.
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        List[str]: List of voice names
    """
    all_voices = []
    
    # Get modern profile voices
    profiles = get_voice_profiles(voice_library_path)
    all_voices.extend([profile['name'] for profile in profiles])
    
    # Get legacy voices (not yet converted)
    legacy_voices = get_legacy_voices(voice_library_path)
    all_voices.extend(legacy_voices)
    
    # Remove duplicates and sort
    unique_voices = list(set(all_voices))
    unique_voices.sort()
    
    return unique_voices


def get_voice_choices_organized(voice_library_path: str) -> List[str]:
    """
    Get organized voice choices with tuned voices at top, then separator, then raw voices.
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        List[str]: Organized list of voice names with tuned voices first
    """
    # Get all voice profiles (subfolder + legacy JSON)
    profiles = get_voice_profiles(voice_library_path)
    tuned_voices = [profile['name'] for profile in profiles]
    tuned_voices.sort()
    
    # Get raw WAV voices (no config)
    voice_dir = Path(voice_library_path)
    raw_voices = []
    
    if voice_dir.exists():
        for wav_file in voice_dir.glob("*.wav"):
            voice_name = wav_file.stem
            
            # Skip if this voice already has a profile
            has_profile = False
            for profile in profiles:
                if profile['name'] == voice_name:
                    has_profile = True
                    break
            
            if not has_profile:
                raw_voices.append(voice_name)
    
    raw_voices.sort()
    
    # Create organized list
    organized_voices = []
    
    # Add tuned voices first (these have configs)
    if tuned_voices:
        organized_voices.extend(tuned_voices)
    
    # Add separator if we have both types
    if tuned_voices and raw_voices:
        organized_voices.append("─────── Voice Pool ───────")
    
    # Add raw voices
    if raw_voices:
        organized_voices.extend(raw_voices)
    
    return organized_voices


def load_voice_profile(voice_library_path: str, voice_name: str) -> Optional[Dict]:
    """
    Load a specific voice profile.
    Handles subfolder profiles, legacy JSON profiles, and raw WAV files.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Name of the voice profile to load
        
    Returns:
        Dict: Voice profile data or None if not found
    """
    try:
        voice_dir = Path(voice_library_path)
        
        # 1. Try subfolder format (config.json in subfolder)
        subfolder_config = voice_dir / voice_name / "config.json"
        if subfolder_config.exists():
            with open(subfolder_config, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                profile_data['name'] = voice_name
                profile_data['path'] = str(subfolder_config.parent)
                profile_data['profile_type'] = 'subfolder'
                
                # Ensure audio_file is just the filename, not full path
                if 'audio_file' not in profile_data:
                    profile_data['audio_file'] = 'reference.wav'
                
                return profile_data
        
        # 2. Try legacy folder format (reference.wav + any .json in subfolder)
        voice_subfolder = voice_dir / voice_name
        if voice_subfolder.is_dir():
            reference_wav = voice_subfolder / "reference.wav"
            if reference_wav.exists():
                # Look for any JSON file in the folder
                json_files = list(voice_subfolder.glob("*.json"))
                if json_files:
                    # Use the first JSON file found
                    json_file = json_files[0]
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            profile_data = json.load(f)
                            profile_data['name'] = voice_name
                            profile_data['path'] = str(voice_subfolder)
                            profile_data['audio_file'] = "reference.wav"
                            profile_data['profile_type'] = 'legacy_folder'
                            
                            # Set display name if not present
                            if 'display_name' not in profile_data:
                                profile_data['display_name'] = voice_name.replace('_', ' ').title()
                            
                            return profile_data
                    except Exception as e:
                        print(f"⚠️ Error loading legacy folder JSON: {e}")
                        # Fall through to create basic profile
                
                # Has reference.wav but no JSON - create basic profile
                profile_data = {
                    'name': voice_name,
                    'display_name': voice_name.replace('_', ' ').title(),
                    'description': f"Legacy voice: {voice_name}",
                    'audio_file': "reference.wav",
                    'exaggeration': 0.5,
                    'cfg_weight': 0.5,
                    'temperature': 0.8,
                    'path': str(voice_subfolder),
                    'profile_type': 'legacy_folder_no_config'
                }
                return profile_data
        
        # 3. Try legacy JSON format (JSON + WAV in main directory)
        json_file = voice_dir / f"{voice_name}.json"
        wav_file = voice_dir / f"{voice_name}.wav"
        
        if json_file.exists() and wav_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                profile_data['name'] = voice_name
                profile_data['path'] = str(voice_dir)
                profile_data['audio_file'] = f"{voice_name}.wav"
                profile_data['profile_type'] = 'legacy_json'
                
                # Set display name if not present
                if 'display_name' not in profile_data:
                    profile_data['display_name'] = voice_name.replace('_', ' ').title()
                
                return profile_data
        
        # 4. Try raw WAV file (no JSON config)
        if wav_file.exists():
            profile_data = {
                'name': voice_name,
                'display_name': voice_name.replace('_', ' ').title(),
                'description': f"Raw voice: {voice_name}",
                'audio_file': f"{voice_name}.wav",
                'exaggeration': 0.5,  # Default values
                'cfg_weight': 0.5,
                'temperature': 0.8,
                'profile_type': 'raw_wav',
                'legacy_format': True,
                'path': str(voice_dir)
            }
            
            return profile_data
        
        return None
            
    except Exception as e:
        print(f"❌ Error loading voice profile {voice_name}: {e}")
        return None


def save_voice_profile(
    voice_library_path: str,
    voice_name: str,
    display_name: str,
    description: str,
    audio_file_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    enable_normalization: bool = False,
    target_level_db: float = -18.0
) -> Tuple[bool, str]:
    """
    Save a voice profile to the voice library.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Internal name for the voice (no spaces)
        display_name (str): Display name for the voice
        description (str): Description of the voice
        audio_file_path (str): Path to the reference audio file
        exaggeration (float): Voice exaggeration parameter
        cfg_weight (float): CFG weight parameter
        temperature (float): Temperature parameter
        enable_normalization (bool): Whether to enable volume normalization
        target_level_db (float): Target volume level in dB
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        # Create voice directory
        voice_dir = Path(voice_library_path) / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy audio file to voice directory
        audio_filename = "reference.wav"
        dest_audio_path = voice_dir / audio_filename
        
        if audio_file_path:
            # Copy audio file to voice directory
            try:
                shutil.copy2(audio_file_path, dest_audio_path)
                
                # Note: Audio normalization is handled during preprocessing in the UI
                # No additional normalization needed here as it's already applied
                
            except Exception as e:
                return False, f"Failed to copy audio file: {str(e)}"
        else:
            return False, "No audio file provided"
        
        # Create voice configuration
        config_data = {
            'display_name': display_name,
            'description': description,
            'audio_file': audio_filename,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'temperature': temperature,
            'enable_normalization': enable_normalization,
            'target_level_db': target_level_db,
            'created_date': str(Path(audio_file_path).stat().st_mtime),
            'sample_rate': config.get_setting("sample_rate")
        }
        
        # Save configuration
        config_path = voice_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return True, f"✅ Voice profile '{display_name}' saved successfully"
        
    except Exception as e:
        return False, f"❌ Error saving voice profile: {str(e)}"


def delete_voice_profile(voice_library_path: str, voice_name: str) -> Tuple[bool, str]:
    """
    Delete a voice profile from the library.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Name of the voice profile to delete
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        voice_dir = Path(voice_library_path) / voice_name
        
        if not voice_dir.exists():
            return False, f"Voice profile '{voice_name}' not found"
        
        # Remove the entire voice directory
        shutil.rmtree(voice_dir)
        
        return True, f"✅ Voice profile '{voice_name}' deleted successfully"
        
    except Exception as e:
        return False, f"❌ Error deleting voice profile: {str(e)}"


def test_voice_profile(
    voice_library_path: str, 
    voice_name: str, 
    test_text: str = "Hello, this is a test of the voice profile."
) -> Optional[Tuple[int, np.ndarray]]:
    """
    Test a voice profile by generating sample audio.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Name of the voice profile to test
        test_text (str): Text to use for testing
        
    Returns:
        Optional[Tuple[int, np.ndarray]]: Audio data or None on failure
    """
    try:
        # Load voice profile
        profile = load_voice_profile(voice_library_path, voice_name)
        if not profile:
            print(f"❌ Voice profile '{voice_name}' not found")
            return None
        
        # Get audio file path
        audio_file_path = Path(profile['path']) / profile['audio_file']
        if not audio_file_path.exists():
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        # Load model and generate audio
        model = load_model()
        result = generate_for_gradio(
            model,
            test_text,
            str(audio_file_path),
            profile.get('exaggeration', 0.5),
            profile.get('temperature', 0.8),
            0,  # Random seed
            profile.get('cfg_weight', 0.5)
        )
        
        if result:
            print(f"✅ Voice test successful for '{voice_name}'")
            return result
        else:
            print(f"❌ Voice test failed for '{voice_name}'")
            return None
            
    except Exception as e:
        print(f"❌ Error testing voice profile: {e}")
        return None


def get_voice_config(voice_library_path: str, voice_name: str) -> Optional[Dict]:
    """
    Get the configuration for a specific voice.
    
    Args:
        voice_library_path (str): Path to voice library directory
        voice_name (str): Name of the voice profile
        
    Returns:
        Optional[Dict]: Voice configuration or None if not found
    """
    return load_voice_profile(voice_library_path, voice_name)


def validate_voice_audio(audio_file_path: str) -> Tuple[bool, str]:
    """
    Validate an audio file for use as a voice reference.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    try:
        if not audio_file_path or not os.path.exists(audio_file_path):
            return False, "Audio file not found"
        
        # Try to load the audio file
        audio_data, sample_rate = load_audio_file(audio_file_path)
        
        if audio_data is None:
            return False, "Unable to load audio file"
        
        # Check duration (should be between 5 seconds and 2 minutes)
        duration = len(audio_data) / sample_rate
        if duration < 5:
            return False, f"Audio too short ({duration:.1f}s). Minimum 5 seconds required."
        
        if duration > 120:
            return False, f"Audio too long ({duration:.1f}s). Maximum 2 minutes recommended."
        
        # Check if audio has content (not just silence)
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.01:  # Very quiet
            return False, "Audio appears to be silent or too quiet"
        
        return True, f"✅ Audio valid ({duration:.1f}s, {sample_rate}Hz)"
        
    except Exception as e:
        return False, f"Error validating audio: {str(e)}"


def get_voice_library_stats(voice_library_path: str) -> Dict[str, Any]:
    """
    Get statistics about the voice library.
    
    Args:
        voice_library_path (str): Path to voice library directory
        
    Returns:
        Dict[str, Any]: Statistics about the voice library
    """
    try:
        profiles = get_voice_profiles(voice_library_path)
        
        stats = {
            'total_voices': len(profiles),
            'library_path': voice_library_path,
            'library_exists': os.path.exists(voice_library_path)
        }
        
        if profiles:
            # Calculate some stats
            normalized_count = sum(1 for p in profiles if p.get('enable_normalization', False))
            stats['normalized_voices'] = normalized_count
            
            # Get average parameters
            avg_exaggeration = sum(p.get('exaggeration', 0.5) for p in profiles) / len(profiles)
            avg_temperature = sum(p.get('temperature', 0.8) for p in profiles) / len(profiles)
            
            stats['avg_exaggeration'] = avg_exaggeration
            stats['avg_temperature'] = avg_temperature
        
        return stats
        
    except Exception as e:
        return {
            'total_voices': 0,
            'library_path': voice_library_path,
            'library_exists': False,
            'error': str(e)
        } 