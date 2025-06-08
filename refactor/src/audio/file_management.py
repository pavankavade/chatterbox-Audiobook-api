"""Audio file management for audiobook generation.

Handles audio file operations, format conversion, and file I/O.
"""

import os
import wave
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import json
from scipy.io import wavfile
import warnings


def save_audio_file(
    audio_data: np.ndarray, 
    sample_rate: int, 
    file_path: str, 
    format: str = "wav"
) -> bool:
    """Save audio data to a file.
    
    Args:
        audio_data: Audio numpy array
        sample_rate: Sample rate for the audio
        file_path: Path where to save the file
        format: Audio format ('wav', 'mp3')
        
    Returns:
        bool: True if save was successful
    """
    try:
        if format.lower() == "wav":
            return _save_wav_file(audio_data, sample_rate, file_path)
        elif format.lower() == "mp3":
            return _save_mp3_file(audio_data, sample_rate, file_path)
        else:
            print(f"Unsupported audio format: {format}")
            return False
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")
        return False


def _save_wav_file(audio_data: np.ndarray, sample_rate: int, file_path: str) -> bool:
    """Save audio data as WAV file using scipy for robustness."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Ensure data is float32 between -1 and 1
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit integer for maximum compatibility
        int16_audio = (audio_data * 32767).astype(np.int16)

        wavfile.write(file_path, sample_rate, int16_audio)
        return True
    except Exception as e:
        print(f"Error saving WAV file with scipy: {e}")
        return False


def _save_mp3_file(audio_data: np.ndarray, sample_rate: int, file_path: str) -> bool:
    """Save audio data as MP3 file.
    
    Args:
        audio_data: Audio numpy array
        sample_rate: Sample rate for the audio
        file_path: Path where to save the MP3 file
        
    Returns:
        bool: True if save was successful
    """
    try:
        # Try to use pydub for MP3 conversion
        try:
            from pydub import AudioSegment
            import io
            
            # Convert numpy array to AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            )
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if there's a directory component
                os.makedirs(dir_path, exist_ok=True)
            
            # Export as MP3
            audio_segment.export(file_path, format="mp3")
            return True
            
        except ImportError:
            print("pydub not available, falling back to WAV format")
            wav_path = file_path.replace('.mp3', '.wav')
            return _save_wav_file(audio_data, sample_rate, wav_path)
            
    except Exception as e:
        print(f"Error saving MP3 file: {e}")
        return False


def load_audio_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load audio file and return audio data and sample rate.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate) or (None, None) if failed
    """
    try:
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            return None, None
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.wav':
            return _load_wav_file(file_path)
        elif file_ext in ['.mp3', '.flac', '.ogg']:
            return _load_audio_with_pydub(file_path)
        else:
            print(f"Unsupported audio format: {file_ext}")
            return None, None
            
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None


def _load_wav_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load WAV file using scipy for robustness against different bit depths."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            sample_rate, audio_data = wavfile.read(file_path)

        # Convert to float32 in [-1, 1] range if it's an integer type
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        elif np.issubdtype(audio_data.dtype, np.floating):
             # Audio is already float, just clip it to be safe
            audio_data = np.clip(audio_data, -1.0, 1.0)
        else:
             print(f"Warning: Unhandled audio dtype {audio_data.dtype}")
             return None, None

        # Convert to mono by averaging channels if stereo
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)

        return audio_data.astype(np.float32), sample_rate
    except Exception as e:
        print(f"Error loading WAV file with scipy: {e}")
        return None, None


def _load_audio_with_pydub(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load audio file using pydub.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate) or (None, None) if failed
    """
    try:
        from pydub import AudioSegment
        
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Get audio data as numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize based on sample width
        if audio.sample_width == 2: # 16-bit
            audio_data = audio_data / 32767.0
        elif audio.sample_width == 4: # 32-bit
            audio_data = audio_data / 2147483647.0
        elif audio.sample_width == 1: # 8-bit
            audio_data = (audio_data - 128.0) / 128.0

        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        sample_rate = audio.frame_rate
        
        return audio_data, sample_rate
        
    except ImportError:
        print("pydub not available for loading non-WAV files")
        return None, None
    except Exception as e:
        print(f"Error loading audio with pydub: {e}")
        return None, None


def get_audio_file_info(file_path: str) -> dict:
    """Get information about an audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        dict: Audio file information
    """
    info = {
        'file_path': file_path,
        'exists': False,
        'format': None,
        'duration': 0.0,
        'sample_rate': None,
        'channels': None,
        'file_size': 0
    }
    
    try:
        if not os.path.exists(file_path):
            return info
        
        info['exists'] = True
        info['file_size'] = os.path.getsize(file_path)
        info['format'] = Path(file_path).suffix.lower()
        
        # Try to get audio properties
        audio_data, sample_rate = load_audio_file(file_path)
        if audio_data is not None and sample_rate is not None:
            info['sample_rate'] = sample_rate
            info['channels'] = 1  # We convert everything to mono
            info['duration'] = len(audio_data) / sample_rate
        
    except Exception as e:
        print(f"Error getting audio file info: {e}")
    
    return info


def combine_audio_files(file_paths: List[str], output_path: str) -> bool:
    """Combine multiple audio files into one.
    
    Args:
        file_paths: List of audio file paths to combine
        output_path: Path for the combined output file
        
    Returns:
        bool: True if combination was successful
    """
    try:
        if not file_paths:
            return False
        
        combined_audio = []
        sample_rate = None
        
        for file_path in file_paths:
            audio_data, sr = load_audio_file(file_path)
            if audio_data is None:
                print(f"Failed to load audio file: {file_path}")
                continue
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                print(f"Warning: Sample rate mismatch in {file_path} ({sr} vs {sample_rate})")
                # Could resample here if needed
            
            combined_audio.append(audio_data)
        
        if not combined_audio:
            print("No audio files could be loaded")
            return False
        
        # Concatenate all audio
        final_audio = np.concatenate(combined_audio)
        
        # Save combined audio
        return save_audio_file(final_audio, sample_rate, output_path)
        
    except Exception as e:
        print(f"Error combining audio files: {e}")
        return False


def create_audio_manifest(project_dir: str) -> bool:
    """Create a manifest file listing all audio files in a project.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        bool: True if manifest was created successfully
    """
    try:
        audio_files = []
        
        # Scan for audio files
        for file_name in os.listdir(project_dir):
            if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(project_dir, file_name)
                file_info = get_audio_file_info(file_path)
                audio_files.append(file_info)
        
        # Sort by filename
        audio_files.sort(key=lambda x: x['file_path'])
        
        # Create manifest
        manifest = {
            'project_dir': project_dir,
            'created_at': os.path.getctime(project_dir),
            'total_files': len(audio_files),
            'total_duration': sum(f['duration'] for f in audio_files),
            'files': audio_files
        }
        
        # Save manifest
        manifest_path = os.path.join(project_dir, 'audio_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error creating audio manifest: {e}")
        return False 