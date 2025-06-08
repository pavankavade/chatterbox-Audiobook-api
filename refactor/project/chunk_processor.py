"""
# ==============================================================================
# CHUNK PROCESSOR MODULE
# ==============================================================================
# 
# This module provides sophisticated chunk processing capabilities for the
# Chatterbox Audiobook Studio refactored system. It handles text chunking,
# audio generation, and chunk-level operations.
# 
# **Key Features:**
# - **Text Chunking**: Intelligent text segmentation for audiobook processing
# - **Audio Generation**: Chunk-level TTS generation and processing
# - **Chunk Management**: Complete chunk lifecycle operations
# - **Performance Optimization**: Efficient processing of large texts
# - **Original Compatibility**: Full compatibility with existing chunk formats
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Import core modules
from core.tts_engine import RefactoredTTSEngine
from core.audio_processing import save_audio_file, combine_audio_arrays
from voice.voice_manager import VoiceManager

# ==============================================================================
# CHUNK DATA STRUCTURES
# ==============================================================================

@dataclass
class TextChunk:
    """
    Data structure for a text chunk with processing metadata.
    """
    chunk_id: str
    text: str
    chunk_index: int
    voice_profile: str = ""
    character_name: str = ""
    chunk_type: str = "narrative"  # "narrative", "dialogue", "description"
    processing_status: str = "pending"  # "pending", "processing", "completed", "failed"
    audio_file: str = ""
    created_date: str = ""
    processed_date: str = ""
    error_message: str = ""
    generation_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_options is None:
            self.generation_options = {}
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """Create from dictionary."""
        return cls(**data)

# ==============================================================================
# CHUNK PROCESSOR CLASS
# ==============================================================================

class ChunkProcessor:
    """
    Professional chunk processing system for audiobook generation.
    
    This class provides comprehensive chunk-level operations including
    text chunking, audio generation, and chunk management with integration
    to all system components.
    
    **Processing Features:**
    - **Intelligent Chunking**: Smart text segmentation based on content
    - **Audio Generation**: High-quality TTS generation per chunk
    - **Batch Processing**: Efficient processing of multiple chunks
    - **Error Recovery**: Robust error handling and retry mechanisms
    - **Progress Tracking**: Comprehensive processing status monitoring
    """
    
    def __init__(
        self,
        output_directory: str = "output",
        voice_manager: Optional[VoiceManager] = None,
        tts_engine: Optional[RefactoredTTSEngine] = None
    ):
        """
        Initialize the chunk processor.
        
        Args:
            output_directory (str): Directory for chunk output files
            voice_manager (Optional[VoiceManager]): Voice manager instance
            tts_engine (Optional[RefactoredTTSEngine]): TTS engine instance
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.voice_manager = voice_manager or VoiceManager()
        self.tts_engine = tts_engine or RefactoredTTSEngine()
        
        self.chunks: List[TextChunk] = []
        self.default_voice = ""
        self.narrator_voice = ""
        
        print(f"✅ Chunk Processor initialized - Output: {self.output_directory}")
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_method: str = "smart"
    ) -> List[TextChunk]:
        """
        Split text into intelligent chunks for processing.
        
        Args:
            text (str): Input text to chunk
            chunk_size (int): Target size for each chunk
            chunk_method (str): Chunking method ("smart", "sentence", "paragraph")
            
        Returns:
            List[TextChunk]: List of text chunks
            
        **Chunking Methods:**
        - **smart**: Intelligent chunking based on content structure
        - **sentence**: Split by sentences with size limits
        - **paragraph**: Split by paragraphs with size limits
        """
        chunks = []
        
        try:
            if chunk_method == "smart":
                chunks = self._smart_chunk_text(text, chunk_size)
            elif chunk_method == "sentence":
                chunks = self._sentence_chunk_text(text, chunk_size)
            elif chunk_method == "paragraph":
                chunks = self._paragraph_chunk_text(text, chunk_size)
            else:
                # Default to smart chunking
                chunks = self._smart_chunk_text(text, chunk_size)
            
            # Set chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.chunk_index = i
                chunk.chunk_id = f"chunk_{i:04d}"
                
                # Analyze chunk for voice assignment
                voice_info = self._analyze_chunk_for_voice(chunk.text)
                chunk.voice_profile = voice_info['voice_profile']
                chunk.character_name = voice_info['character_name']
                chunk.chunk_type = voice_info['chunk_type']
            
            self.chunks = chunks
            print(f"📝 Created {len(chunks)} chunks using {chunk_method} method")
            
        except Exception as e:
            print(f"❌ Error chunking text: {e}")
        
        return chunks
    
    def process_chunk(
        self,
        chunk: TextChunk,
        generation_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Process a single chunk to generate audio.
        
        Args:
            chunk (TextChunk): Chunk to process
            generation_options (Optional[Dict[str, Any]]): TTS generation options
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Update chunk status
            chunk.processing_status = "processing"
            
            # Merge generation options
            options = generation_options or {}
            options.update(chunk.generation_options)
            
            # Get voice profile path
            voice_path = self._get_voice_path(chunk.voice_profile)
            if not voice_path:
                voice_path = self._get_voice_path(self.default_voice)
            
            if not voice_path:
                chunk.processing_status = "failed"
                chunk.error_message = "No valid voice profile available"
                return False, f"❌ No voice profile for chunk {chunk.chunk_id}"
            
            # Generate audio
            sample_rate, audio_data = self.tts_engine.generate_speech(
                text=chunk.text,
                audio_prompt_path=voice_path,
                exaggeration=options.get('exaggeration', 1.0),
                temperature=options.get('temperature', 0.7),
                cfg_weight=options.get('cfg_weight', 3.0)
            )
            
            # Save audio file
            audio_filename = f"{chunk.chunk_id}.wav"
            audio_path = self.output_directory / audio_filename
            
            success = save_audio_file(audio_data, audio_path, sample_rate)
            
            if success:
                chunk.audio_file = str(audio_path)
                chunk.processing_status = "completed"
                chunk.processed_date = datetime.now().isoformat()
                chunk.error_message = ""
                
                return True, f"✅ Chunk {chunk.chunk_id} processed successfully"
            else:
                chunk.processing_status = "failed"
                chunk.error_message = "Failed to save audio file"
                return False, f"❌ Failed to save audio for chunk {chunk.chunk_id}"
            
        except Exception as e:
            chunk.processing_status = "failed"
            chunk.error_message = str(e)
            return False, f"❌ Error processing chunk {chunk.chunk_id}: {e}"
    
    def process_all_chunks(
        self,
        generation_options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process all chunks in the current batch.
        
        Args:
            generation_options (Optional[Dict[str, Any]]): TTS generation options
            progress_callback (Optional[callable]): Progress callback function
            
        Returns:
            Dict[str, Any]: Processing results summary
        """
        results = {
            'total_chunks': len(self.chunks),
            'processed_chunks': 0,
            'failed_chunks': 0,
            'successful_chunks': 0,
            'errors': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        for i, chunk in enumerate(self.chunks):
            try:
                # Process chunk
                success, message = self.process_chunk(chunk, generation_options)
                
                if success:
                    results['successful_chunks'] += 1
                else:
                    results['failed_chunks'] += 1
                    results['errors'].append({
                        'chunk_id': chunk.chunk_id,
                        'error': message
                    })
                
                results['processed_chunks'] += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress = {
                        'current_chunk': i + 1,
                        'total_chunks': len(self.chunks),
                        'progress_percent': ((i + 1) / len(self.chunks)) * 100,
                        'chunk_id': chunk.chunk_id,
                        'success': success,
                        'message': message
                    }
                    progress_callback(progress)
                
            except Exception as e:
                results['failed_chunks'] += 1
                results['errors'].append({
                    'chunk_id': chunk.chunk_id,
                    'error': f"Unexpected error: {str(e)}"
                })
                
                chunk.processing_status = "failed"
                chunk.error_message = str(e)
        
        results['end_time'] = datetime.now().isoformat()
        
        print(f"🎯 Processing complete: {results['successful_chunks']}/{results['total_chunks']} successful")
        
        return results
    
    def combine_chunk_audio(
        self,
        output_filename: str = "combined_audiobook.wav",
        pause_duration: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Combine all processed chunk audio files into a single audiobook.
        
        Args:
            output_filename (str): Name for the combined audio file
            pause_duration (float): Pause duration between chunks in seconds
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Get successfully processed chunks
            completed_chunks = [
                chunk for chunk in self.chunks
                if chunk.processing_status == "completed" and chunk.audio_file
            ]
            
            if not completed_chunks:
                return False, "❌ No completed chunks available for combination"
            
            # Load audio arrays
            audio_arrays = []
            sample_rate = 24000  # Default sample rate
            
            for chunk in completed_chunks:
                try:
                    from core.audio_processing import load_audio_file
                    audio_array, chunk_sample_rate = load_audio_file(chunk.audio_file)
                    audio_arrays.append(audio_array)
                    sample_rate = chunk_sample_rate  # Use the actual sample rate
                except Exception as e:
                    print(f"⚠️  Error loading audio for chunk {chunk.chunk_id}: {e}")
                    continue
            
            if not audio_arrays:
                return False, "❌ No valid audio arrays loaded"
            
            # Combine audio arrays
            combined_audio = combine_audio_arrays(
                audio_arrays,
                sample_rate=sample_rate,
                pause_duration=pause_duration
            )
            
            # Save combined audio
            output_path = self.output_directory / output_filename
            success = save_audio_file(combined_audio, output_path, sample_rate)
            
            if success:
                return True, f"✅ Combined audiobook saved: {output_path}"
            else:
                return False, "❌ Failed to save combined audiobook"
            
        except Exception as e:
            return False, f"❌ Error combining audio: {str(e)}"
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive chunk processing statistics.
        
        Returns:
            Dict[str, Any]: Chunk statistics
        """
        stats = {
            'total_chunks': len(self.chunks),
            'pending_chunks': 0,
            'processing_chunks': 0,
            'completed_chunks': 0,
            'failed_chunks': 0,
            'chunk_types': {},
            'voice_usage': {},
            'total_characters': 0,
            'total_words': 0
        }
        
        for chunk in self.chunks:
            # Count by status
            status = chunk.processing_status
            if status == "pending":
                stats['pending_chunks'] += 1
            elif status == "processing":
                stats['processing_chunks'] += 1
            elif status == "completed":
                stats['completed_chunks'] += 1
            elif status == "failed":
                stats['failed_chunks'] += 1
            
            # Count by type
            chunk_type = chunk.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            
            # Count voice usage
            voice = chunk.voice_profile
            if voice:
                stats['voice_usage'][voice] = stats['voice_usage'].get(voice, 0) + 1
            
            # Count text statistics
            stats['total_characters'] += len(chunk.text)
            stats['total_words'] += len(chunk.text.split())
        
        return stats
    
    def save_chunks_metadata(self, filename: str = "chunks_metadata.json") -> bool:
        """
        Save chunk metadata to a JSON file.
        
        Args:
            filename (str): Name of the metadata file
            
        Returns:
            bool: Success status
        """
        try:
            metadata_path = self.output_directory / filename
            
            metadata = {
                'chunk_count': len(self.chunks),
                'created_date': datetime.now().isoformat(),
                'statistics': self.get_chunk_statistics(),
                'chunks': [chunk.to_dict() for chunk in self.chunks]
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving chunk metadata: {e}")
            return False
    
    def load_chunks_metadata(self, filename: str = "chunks_metadata.json") -> bool:
        """
        Load chunk metadata from a JSON file.
        
        Args:
            filename (str): Name of the metadata file
            
        Returns:
            bool: Success status
        """
        try:
            metadata_path = self.output_directory / filename
            
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load chunks
            chunk_data = metadata.get('chunks', [])
            self.chunks = [TextChunk.from_dict(data) for data in chunk_data]
            
            print(f"📁 Loaded {len(self.chunks)} chunks from metadata")
            return True
            
        except Exception as e:
            print(f"❌ Error loading chunk metadata: {e}")
            return False
    
    def _smart_chunk_text(self, text: str, chunk_size: int) -> List[TextChunk]:
        """Smart text chunking based on content structure."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, create a new chunk
            if current_chunk and len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunks.append(TextChunk(
                        chunk_id="",
                        text=current_chunk.strip(),
                        chunk_index=0
                    ))
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(TextChunk(
                chunk_id="",
                text=current_chunk.strip(),
                chunk_index=0
            ))
        
        return chunks
    
    def _sentence_chunk_text(self, text: str, chunk_size: int) -> List[TextChunk]:
        """Sentence-based text chunking."""
        chunks = []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk.strip():
                    chunks.append(TextChunk(
                        chunk_id="",
                        text=current_chunk.strip(),
                        chunk_index=0
                    ))
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(TextChunk(
                chunk_id="",
                text=current_chunk.strip(),
                chunk_index=0
            ))
        
        return chunks
    
    def _paragraph_chunk_text(self, text: str, chunk_size: int) -> List[TextChunk]:
        """Paragraph-based text chunking."""
        chunks = []
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(paragraph) <= chunk_size:
                chunks.append(TextChunk(
                    chunk_id="",
                    text=paragraph,
                    chunk_index=0
                ))
            else:
                # Split large paragraphs by sentences
                sentence_chunks = self._sentence_chunk_text(paragraph, chunk_size)
                chunks.extend(sentence_chunks)
        
        return chunks
    
    def _analyze_chunk_for_voice(self, text: str) -> Dict[str, str]:
        """Analyze chunk text for appropriate voice assignment."""
        # Default to narrator voice
        voice_info = {
            'voice_profile': self.narrator_voice or self.default_voice,
            'character_name': 'Narrator',
            'chunk_type': 'narrative'
        }
        
        # Check for dialogue patterns
        if '"' in text or ''' in text or ''' in text:
            voice_info['chunk_type'] = 'dialogue'
            
            # Try to extract character name
            character_patterns = [
                r'"[^"]*",?\s*(\w+)\s+(?:said|replied|asked|whispered|shouted)',
                r'(\w+)\s+(?:said|replied|asked|whispered|shouted)',
                r'"[^"]*"\s*-\s*(\w+)',
                r'(\w+):\s*"[^"]*"'
            ]
            
            for pattern in character_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    character_name = match.group(1).strip().title()
                    if len(character_name) > 1:
                        voice_info['character_name'] = character_name
                        # Use character-specific voice if available
                        # (This would integrate with voice assignments from project)
                        break
        
        return voice_info
    
    def _get_voice_path(self, voice_profile: str) -> Optional[str]:
        """Get the path to a voice profile's audio file."""
        if not voice_profile:
            return None
        
        try:
            profile, message = self.voice_manager.load_voice_profile(voice_profile)
            if profile and profile.audio_file:
                return profile.audio_file
        except Exception as e:
            print(f"⚠️  Error loading voice profile {voice_profile}: {e}")
        
        return None

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def process_chunks(
    text: str,
    chunk_size: int = 1000,
    output_directory: str = "output"
) -> List[TextChunk]:
    """Process text chunks (convenience function)."""
    processor = ChunkProcessor(output_directory)
    return processor.chunk_text(text, chunk_size)

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Chunk Processor module loaded")
print("📝 Text chunking and audio generation ready for audiobook processing") 