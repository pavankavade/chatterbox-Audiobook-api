"""
Core audiobook generation module.

Handles the complete pipeline from text input to finished audiobook:
1. Text validation and chunking
2. TTS generation for each chunk  
3. Audio file management
4. Project metadata creation
5. Final audiobook compilation
"""

import os
import json
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Generator, Callable
from datetime import datetime

try:
    from ..text_processing.chunking import (
        chunk_text_by_sentences, 
        validate_audiobook_input,
        save_audio_chunks
    )
    from ..projects.management import create_project_directory
    from ..projects.metadata import save_project_metadata
    from ..voice_library.voice_management import load_voice_profile
    from ..models.tts_model import CHATTERBOX_AVAILABLE
except ImportError:
    from src.text_processing.chunking import (
        chunk_text_by_sentences, 
        validate_audiobook_input,
        save_audio_chunks
    )
    from src.projects.management import create_project_directory
    from src.projects.metadata import save_project_metadata
    from src.voice_library.voice_management import load_voice_profile
    from src.models.tts_model import CHATTERBOX_AVAILABLE


class AudiobookGenerator:
    """Class to handle audiobook generation with progress tracking."""
    
    def __init__(self, model_state, voice_library_path: str):
        """Initialize the audiobook generator.
        
        Args:
            model_state: Loaded TTS model
            voice_library_path: Path to voice library
        """
        self.model_state = model_state
        self.voice_library_path = voice_library_path
        self.is_generating = False
        self.current_chunk = 0
        self.total_chunks = 0
        self.project_dir = ""
        self.generated_files = []
        
    def generate_audiobook(
        self, 
        text: str, 
        voice_name: str, 
        project_name: str,
        max_words_per_chunk: int = 50,
        autosave_interval: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Generate audiobook with real-time progress updates.
        
        Args:
            text: Input text to convert
            voice_name: Name of the voice to use
            project_name: Name for the project
            max_words_per_chunk: Maximum words per chunk
            autosave_interval: Save progress every N chunks
            progress_callback: Optional callback for progress updates
            
        Yields:
            Dict with progress information
            
        Returns:
            Dict with final generation results
        """
        try:
            self.is_generating = True
            
            # Step 1: Validate input
            yield {
                'status': 'info',
                'message': '🔍 Validating input...',
                'progress': 0,
                'step': 'validation'
            }
            
            is_valid, validation_msg = validate_audiobook_input(text, voice_name, project_name)
            if not is_valid:
                yield {
                    'status': 'error',
                    'message': validation_msg,
                    'progress': 0,
                    'step': 'validation'
                }
                return
            
            # Step 2: Load voice profile
            yield {
                'status': 'info',
                'message': f'🎭 Loading voice profile: {voice_name}...',
                'progress': 5,
                'step': 'voice_loading'
            }
            
            voice_profile = load_voice_profile(self.voice_library_path, voice_name)
            if not voice_profile:
                yield {
                    'status': 'error',
                    'message': f'❌ Could not load voice profile: {voice_name}',
                    'progress': 5,
                    'step': 'voice_loading'
                }
                return
            
            # Step 3: Create project directory
            yield {
                'status': 'info',
                'message': f'📁 Creating project: {project_name}...',
                'progress': 10,
                'step': 'project_creation'
            }
            
            self.project_dir = create_project_directory(project_name)
            
            # Step 4: Chunk text
            yield {
                'status': 'info',
                'message': '✂️ Chunking text...',
                'progress': 15,
                'step': 'text_chunking'
            }
            
            chunks = chunk_text_by_sentences(text, max_words_per_chunk)
            self.total_chunks = len(chunks)
            
            if not chunks:
                yield {
                    'status': 'error',
                    'message': '❌ Failed to create text chunks',
                    'progress': 15,
                    'step': 'text_chunking'
                }
                return
            
            yield {
                'status': 'success',
                'message': f'✅ Created {self.total_chunks} text chunks',
                'progress': 20,
                'step': 'text_chunking'
            }
            
            # Step 5: Generate TTS for each chunk
            yield {
                'status': 'info',
                'message': '🎙️ Starting TTS generation...',
                'progress': 25,
                'step': 'tts_generation'
            }
            
            generated_audio_files = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks, 1):
                self.current_chunk = i
                
                # Generate TTS for this chunk
                chunk_start_time = time.time()
                
                yield {
                    'status': 'info',
                    'message': f'🎙️ Generating chunk {i}/{self.total_chunks}...',
                    'progress': 25 + (i / self.total_chunks) * 60,
                    'step': 'tts_generation',
                    'chunk_current': i,
                    'chunk_total': self.total_chunks,
                    'chunk_text': chunk[:100] + '...' if len(chunk) > 100 else chunk
                }
                
                # Generate audio for this chunk
                audio_result = self._generate_chunk_audio(chunk, voice_profile)
                
                if audio_result is None:
                    yield {
                        'status': 'error',
                        'message': f'❌ Failed to generate audio for chunk {i}',
                        'progress': 25 + (i / self.total_chunks) * 60,
                        'step': 'tts_generation'
                    }
                    return
                
                # Save chunk audio
                chunk_filename = f"chunk_{i:03d}.wav"
                chunk_filepath = os.path.join(self.project_dir, chunk_filename)
                
                self._save_audio_file(audio_result, chunk_filepath)
                generated_audio_files.append(chunk_filepath)
                
                # Track chunk metadata
                chunk_generation_time = time.time() - chunk_start_time
                chunk_metadata.append({
                    'chunk_number': i,
                    'text': chunk,
                    'audio_file': chunk_filename,
                    'generation_time': chunk_generation_time,
                    'word_count': len(chunk.split())
                })
                
                # Auto-save progress
                if i % autosave_interval == 0:
                    yield {
                        'status': 'info',
                        'message': f'💾 Auto-saving progress... ({i}/{self.total_chunks} chunks)',
                        'progress': 25 + (i / self.total_chunks) * 60,
                        'step': 'tts_generation'
                    }
                    
                    self._save_progress_metadata(project_name, text, voice_profile, chunk_metadata, i)
            
            # Step 6: Combine audio files
            yield {
                'status': 'info',
                'message': '🎵 Combining audio files...',
                'progress': 90,
                'step': 'audio_combination'
            }
            
            final_audio_path = self._combine_audio_files(generated_audio_files, project_name)
            
            # Step 7: Save final metadata
            yield {
                'status': 'info',
                'message': '💾 Saving project metadata...',
                'progress': 95,
                'step': 'metadata_saving'
            }
            
            final_metadata = self._create_final_metadata(
                project_name, text, voice_profile, chunk_metadata, final_audio_path
            )
            
            save_project_metadata(self.project_dir, final_metadata)
            
            # Step 8: Complete
            total_duration = sum(meta['generation_time'] for meta in chunk_metadata)
            total_words = sum(meta['word_count'] for meta in chunk_metadata)
            
            yield {
                'status': 'success',
                'message': f'🎉 Audiobook generation complete!',
                'progress': 100,
                'step': 'complete',
                'final_audio_path': final_audio_path,
                'project_dir': self.project_dir,
                'stats': {
                    'total_chunks': self.total_chunks,
                    'total_words': total_words,
                    'total_generation_time': total_duration,
                    'average_time_per_chunk': total_duration / self.total_chunks
                }
            }
            
            return {
                'success': True,
                'project_dir': self.project_dir,
                'final_audio_path': final_audio_path,
                'metadata': final_metadata
            }
            
        except Exception as e:
            yield {
                'status': 'error',
                'message': f'❌ Generation failed: {str(e)}',
                'progress': 0,
                'step': 'error'
            }
            return {'success': False, 'error': str(e)}
            
        finally:
            self.is_generating = False
    
    def _generate_chunk_audio(self, text: str, voice_profile: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate audio for a single text chunk.
        
        Args:
            text: Text to convert to speech
            voice_profile: Voice profile data
            
        Returns:
            Generated audio as numpy array or None if failed
        """
        if not CHATTERBOX_AVAILABLE or self.model_state is None:
            return None
        
        try:
            # Import here to avoid circular imports
            try:
                from ..models.tts_model import generate_for_gradio
            except ImportError:
                from src.models.tts_model import generate_for_gradio
            
            # Get audio reference file based on profile type
            profile_type = voice_profile.get('profile_type', 'unknown')
            
            if profile_type == 'subfolder':
                # Subfolder format: path points to subfolder, audio_file is relative
                voice_dir = Path(voice_profile['path'])
                audio_path = str(voice_dir / voice_profile['audio_file'])
            elif profile_type == 'legacy_json' or profile_type == 'raw_wav':
                # Legacy format: path points to main directory, audio_file is relative
                voice_dir = Path(voice_profile['path'])
                audio_path = str(voice_dir / voice_profile['audio_file'])
            else:
                # Fallback - try to construct path
                if voice_profile.get('legacy_format'):
                    audio_path = voice_profile['audio_file']
                else:
                    voice_dir = Path(voice_profile['path'])
                    audio_path = str(voice_dir / voice_profile['audio_file'])
            
            # Use voice profile settings for natural speech
            exaggeration = voice_profile.get('exaggeration', 0.5)
            temperature = voice_profile.get('temperature', 0.8)
            cfg_weight = voice_profile.get('cfg_weight', 0.5)
            
            print(f"🎙️ TTS Settings for '{voice_profile.get('name', 'Unknown')}': "
                  f"exaggeration={exaggeration:.2f}, temperature={temperature:.2f}, cfg_weight={cfg_weight:.2f}")
            
            # Generate audio using voice profile settings
            audio_result = generate_for_gradio(
                self.model_state,
                text,
                audio_path,
                exaggeration=exaggeration,
                temperature=temperature,
                seed_num=0,  # Use random seed
                cfg_weight=cfg_weight
            )
            
            if audio_result and len(audio_result) == 2:
                sample_rate, audio_data = audio_result
                return audio_data
            
            return None
            
        except Exception as e:
            print(f"Error generating chunk audio: {e}")
            return None
    
    def _save_audio_file(self, audio_data: np.ndarray, filepath: str, sample_rate: int = 24000):
        """Save audio data to WAV file.
        
        Args:
            audio_data: Audio numpy array
            filepath: Output file path
            sample_rate: Audio sample rate
        """
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
                
        except Exception as e:
            print(f"Error saving audio file {filepath}: {e}")
    
    def _combine_audio_files(self, audio_files: List[str], project_name: str) -> str:
        """Combine multiple audio files into a single audiobook.
        
        Args:
            audio_files: List of audio file paths
            project_name: Name of the project
            
        Returns:
            Path to the combined audio file
        """
        try:
            final_audio_path = os.path.join(self.project_dir, f"{project_name}_complete.wav")
            
            combined_audio = []
            sample_rate = 24000
            
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    with wave.open(audio_file, 'rb') as wav:
                        frames = wav.readframes(wav.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16)
                        combined_audio.extend(audio_data)
                        sample_rate = wav.getframerate()
            
            # Save combined audio
            if combined_audio:
                with wave.open(final_audio_path, 'wb') as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(sample_rate)
                    wav_out.writeframes(np.array(combined_audio, dtype=np.int16).tobytes())
            
            return final_audio_path
            
        except Exception as e:
            print(f"Error combining audio files: {e}")
            return ""
    
    def _save_progress_metadata(self, project_name: str, text: str, voice_profile: Dict[str, Any], 
                              chunk_metadata: List[Dict[str, Any]], chunks_completed: int):
        """Save progress metadata for resume functionality.
        
        Args:
            project_name: Name of the project
            text: Original text
            voice_profile: Voice profile data
            chunk_metadata: List of chunk metadata
            chunks_completed: Number of chunks completed
        """
        try:
            progress_metadata = {
                'project_name': project_name,
                'project_type': 'single_voice',
                'status': 'in_progress',
                'chunks_completed': chunks_completed,
                'total_chunks': self.total_chunks,
                'last_updated': datetime.now().isoformat(),
                'text_content': text,
                'voice_info': {
                    'voice_name': voice_profile.get('name', 'Unknown'),
                    'voice_path': voice_profile.get('path', ''),
                    'legacy_format': voice_profile.get('legacy_format', False)
                },
                'chunks': chunk_metadata[:chunks_completed]  # Only completed chunks
            }
            
            progress_file = os.path.join(self.project_dir, 'progress.json')
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving progress metadata: {e}")
    
    def _create_final_metadata(self, project_name: str, text: str, voice_profile: Dict[str, Any],
                             chunk_metadata: List[Dict[str, Any]], final_audio_path: str) -> Dict[str, Any]:
        """Create final project metadata.
        
        Args:
            project_name: Name of the project
            text: Original text
            voice_profile: Voice profile data
            chunk_metadata: List of all chunk metadata
            final_audio_path: Path to final combined audio
            
        Returns:
            Complete project metadata dictionary
        """
        return {
            'project_name': project_name,
            'project_type': 'single_voice',
            'status': 'completed',
            'created_date': datetime.now().isoformat(),
            'completed_date': datetime.now().isoformat(),
            'text_content': text,
            'voice_info': {
                'voice_name': voice_profile.get('name', 'Unknown'),
                'voice_path': voice_profile.get('path', ''),
                'legacy_format': voice_profile.get('legacy_format', False),
                'audio_file': voice_profile.get('audio_file', '')
            },
            'generation_info': {
                'total_chunks': len(chunk_metadata),
                'total_words': sum(meta['word_count'] for meta in chunk_metadata),
                'total_generation_time': sum(meta['generation_time'] for meta in chunk_metadata),
                'average_time_per_chunk': sum(meta['generation_time'] for meta in chunk_metadata) / len(chunk_metadata)
            },
            'files': {
                'final_audio': os.path.basename(final_audio_path),
                'chunks': [meta['audio_file'] for meta in chunk_metadata]
            },
            'chunks': chunk_metadata
        }


def generate_single_voice_audiobook(
    model_state,
    text: str,
    voice_name: str,
    project_name: str,
    voice_library_path: str,
    max_words_per_chunk: int = 50,
    autosave_interval: int = 10,
    progress_callback: Optional[Callable] = None
) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    """Generate a single-voice audiobook with progress tracking.
    
    Args:
        model_state: Loaded TTS model
        text: Input text to convert
        voice_name: Name of the voice to use
        project_name: Name for the project
        voice_library_path: Path to voice library
        max_words_per_chunk: Maximum words per chunk
        autosave_interval: Save progress every N chunks
        progress_callback: Optional callback for progress updates
        
    Yields:
        Dict with progress information
        
    Returns:
        Dict with final generation results
    """
    generator = AudiobookGenerator(model_state, voice_library_path)
    
    yield from generator.generate_audiobook(
        text=text,
        voice_name=voice_name,
        project_name=project_name,
        max_words_per_chunk=max_words_per_chunk,
        autosave_interval=autosave_interval,
        progress_callback=progress_callback
    ) 