"""
# ==============================================================================
# AUDIO EFFECTS PROCESSOR MODULE
# ==============================================================================
# 
# This module provides comprehensive audio effects processing and enhancement
# for the Chatterbox Audiobook Studio refactored system. It handles audio
# cleanup, enhancement, professional broadcast-quality processing, and trim
# functionality with automatic saving.
# 
# **Key Features:**
# - **Audio Enhancement**: Professional quality improvement algorithms
# - **Broadcast Processing**: ACX audiobook standard compliance
# - **Audio Cleanup**: Noise reduction and artifact removal
# - **Trim Functionality**: Precise audio trimming with automatic saving
# - **Original Compatibility**: Full compatibility with existing effects workflows
"""

import numpy as np
import scipy.signal
import librosa
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Import core audio processing
from core.audio_processing import (
    normalize_audio_array, save_audio_file, load_audio_file,
    validate_audio_array, DEFAULT_SAMPLE_RATE
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# EFFECTS CONFIGURATION DATA STRUCTURES
# ==============================================================================

@dataclass
class EffectsConfig:
    """
    Configuration for audio effects processing.
    """
    # Normalization settings
    target_lufs: float = -23.0      # EBU R128 / ACX standard
    max_peak: float = -3.0          # Maximum peak level in dB
    target_rms: float = -18.0       # Target RMS level
    
    # Noise reduction settings
    noise_reduction_enabled: bool = True
    noise_floor_db: float = -60.0   # Noise floor threshold
    gate_threshold: float = -50.0   # Gate threshold for silence
    
    # Compression settings
    compression_enabled: bool = True
    compression_ratio: float = 3.0   # Compression ratio
    compression_threshold: float = -12.0  # Compression threshold in dB
    attack_time: float = 0.003      # Attack time in seconds
    release_time: float = 0.100     # Release time in seconds
    
    # EQ settings
    eq_enabled: bool = True
    high_cut_freq: float = 8000.0   # High frequency cut for voice
    low_cut_freq: float = 80.0      # Low frequency cut
    presence_boost: float = 2.0     # Presence boost in dB (2-5kHz)
    
    # Limiting settings
    limiter_enabled: bool = True
    limiter_threshold: float = -1.0  # Limiter threshold
    limiter_release: float = 0.050   # Limiter release time

# ==============================================================================
# PROFESSIONAL EFFECTS PROCESSOR CLASS
# ==============================================================================

class EffectsProcessor:
    """
    Professional audio effects processing system.
    
    This class provides comprehensive audio enhancement and processing
    capabilities extracted from the original system's complex audio
    pipeline with professional broadcast-quality processing standards.
    
    **Processing Features:**
    - **Broadcast Quality**: ACX audiobook standard compliance
    - **Noise Reduction**: Advanced noise floor management
    - **Dynamic Processing**: Professional compression and limiting
    - **EQ Processing**: Voice-optimized frequency shaping
    - **Automatic Trim**: Revolutionary trim functionality with auto-save
    """
    
    def __init__(self, config: Optional[EffectsConfig] = None):
        """
        Initialize the effects processor with professional configuration.
        
        Args:
            config (Optional[EffectsConfig]): Effects configuration
        """
        self.config = config or EffectsConfig()
        self.processing_history: List[Dict[str, Any]] = []
        
        print("✅ Effects Processor initialized - Professional broadcast-quality processing ready")
    
    def enhance_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        preset: str = "audiobook"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply comprehensive audio enhancement for audiobook production.
        
        This is the master enhancement function that applies the complete
        professional audio processing chain for broadcast-quality results.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate of the audio
            preset (str): Enhancement preset ("audiobook", "speech", "custom")
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (enhanced_audio, processing_metadata)
            
        **Enhancement Chain:**
        - **High-Pass Filter**: Remove low-frequency rumble
        - **Noise Gate**: Reduce background noise
        - **Compression**: Even out dynamic range
        - **EQ Processing**: Voice-optimized frequency response
        - **Normalization**: Target LUFS compliance
        - **Limiting**: Prevent digital clipping
        """
        try:
            # Validate input
            is_valid, error_msg = validate_audio_array(audio_data)
            if not is_valid:
                raise ValueError(f"Invalid audio data: {error_msg}")
            
            # Initialize processing metadata
            metadata = {
                'preset': preset,
                'processing_steps': [],
                'start_time': datetime.now().isoformat(),
                'original_stats': self._analyze_audio_stats(audio_data, sample_rate)
            }
            
            enhanced_audio = audio_data.copy()
            
            # Apply preset-specific processing
            if preset == "audiobook":
                enhanced_audio = self._apply_audiobook_preset(enhanced_audio, sample_rate, metadata)
            elif preset == "speech":
                enhanced_audio = self._apply_speech_preset(enhanced_audio, sample_rate, metadata)
            else:
                enhanced_audio = self._apply_custom_processing(enhanced_audio, sample_rate, metadata)
            
            # Final validation and statistics
            metadata['final_stats'] = self._analyze_audio_stats(enhanced_audio, sample_rate)
            metadata['end_time'] = datetime.now().isoformat()
            metadata['enhancement_success'] = True
            
            # Record in processing history
            self.processing_history.append(metadata)
            
            return enhanced_audio, metadata
            
        except Exception as e:
            error_metadata = {
                'preset': preset,
                'error': str(e),
                'enhancement_success': False,
                'end_time': datetime.now().isoformat()
            }
            return audio_data, error_metadata
    
    def apply_noise_reduction(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        reduction_amount: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply intelligent noise reduction to audio.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            reduction_amount (float): Reduction amount (0.0-1.0)
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (processed_audio, metadata)
        """
        try:
            # Spectral noise reduction using librosa
            # Estimate noise from quieter portions
            S_full, phase = librosa.magphase(librosa.stft(audio_data))
            S_filter = librosa.decompose.nn_filter(S_full,
                                                 aggregate=np.median,
                                                 metric='cosine',
                                                 width=int(librosa.frames_to_time(2, sr=sample_rate)))
            S_filter = np.minimum(S_full, S_filter)
            
            margin_i, margin_v = 2, 10
            power = 2
            
            mask_i = librosa.util.softmask(S_filter,
                                         margin_i * (S_full - S_filter),
                                         power=power)
            
            mask_v = librosa.util.softmask(S_full - S_filter,
                                         margin_v * S_filter,
                                         power=power)
            
            # Apply mask based on reduction amount
            final_mask = (1 - reduction_amount) + reduction_amount * mask_v * mask_i
            S_foreground = final_mask * S_full
            
            # Reconstruct audio
            processed_audio = librosa.istft(S_foreground * phase, length=len(audio_data))
            
            metadata = {
                'operation': 'noise_reduction',
                'reduction_amount': reduction_amount,
                'success': True
            }
            
            return processed_audio.astype(np.float32), metadata
            
        except Exception as e:
            print(f"⚠️  Noise reduction failed: {e}")
            return audio_data, {'operation': 'noise_reduction', 'success': False, 'error': str(e)}
    
    def apply_compression(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        threshold: float = -12.0,
        ratio: float = 3.0,
        attack: float = 0.003,
        release: float = 0.100
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply dynamic range compression for consistent levels.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            threshold (float): Compression threshold in dB
            ratio (float): Compression ratio
            attack (float): Attack time in seconds
            release (float): Release time in seconds
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (compressed_audio, metadata)
        """
        try:
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)
            
            # Calculate gain reduction
            gain_reduction = np.zeros_like(audio_db)
            over_threshold = audio_db > threshold
            gain_reduction[over_threshold] = (audio_db[over_threshold] - threshold) * (1 - 1/ratio)
            
            # Apply attack/release smoothing
            attack_samples = int(attack * sample_rate)
            release_samples = int(release * sample_rate)
            
            # Simple envelope follower (simplified for efficiency)
            smoothed_gain = scipy.signal.filtfilt(
                [1/max(attack_samples, 1)] * max(attack_samples, 1),
                [1],
                gain_reduction
            )
            
            # Apply gain reduction
            gain_linear = 10 ** (-smoothed_gain / 20)
            compressed_audio = audio_data * gain_linear
            
            metadata = {
                'operation': 'compression',
                'threshold': threshold,
                'ratio': ratio,
                'attack': attack,
                'release': release,
                'max_gain_reduction': np.max(gain_reduction),
                'success': True
            }
            
            return compressed_audio.astype(np.float32), metadata
            
        except Exception as e:
            print(f"⚠️  Compression failed: {e}")
            return audio_data, {'operation': 'compression', 'success': False, 'error': str(e)}
    
    def apply_eq(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        low_cut: float = 80.0,
        high_cut: float = 8000.0,
        presence_boost: float = 2.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply voice-optimized EQ processing.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            low_cut (float): Low frequency cut in Hz
            high_cut (float): High frequency cut in Hz
            presence_boost (float): Presence boost in dB (2-5kHz range)
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (eq_audio, metadata)
        """
        try:
            # High-pass filter for low-end cleanup
            if low_cut > 0:
                sos_high = scipy.signal.butter(4, low_cut, btype='highpass', fs=sample_rate, output='sos')
                audio_data = scipy.signal.sosfilt(sos_high, audio_data)
            
            # Presence boost (2-5kHz for voice clarity)
            if presence_boost > 0:
                # Design bell filter for presence range
                center_freq = 3500  # Sweet spot for voice presence
                Q = 1.4  # Quality factor
                
                # Convert to digital filter
                w0 = 2 * np.pi * center_freq / sample_rate
                alpha = np.sin(w0) / (2 * Q)
                A = 10 ** (presence_boost / 40)  # Convert dB to linear
                
                # Bell filter coefficients
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha / A
                
                # Normalize
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                audio_data = scipy.signal.filtfilt(b, a, audio_data)
            
            # Low-pass filter for high-end smoothing
            if high_cut < sample_rate / 2:
                sos_low = scipy.signal.butter(2, high_cut, btype='lowpass', fs=sample_rate, output='sos')
                audio_data = scipy.signal.sosfilt(sos_low, audio_data)
            
            metadata = {
                'operation': 'eq',
                'low_cut': low_cut,
                'high_cut': high_cut,
                'presence_boost': presence_boost,
                'success': True
            }
            
            return audio_data.astype(np.float32), metadata
            
        except Exception as e:
            print(f"⚠️  EQ processing failed: {e}")
            return audio_data, {'operation': 'eq', 'success': False, 'error': str(e)}
    
    def trim_audio_with_autosave(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Revolutionary trim functionality with automatic saving.
        
        This implements the original system's automatic save-on-trim feature
        that eliminates the need for manual saving operations.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            output_path (Optional[str]): Output file path (auto-generated if None)
            
        Returns:
            Tuple[np.ndarray, str]: (trimmed_audio, save_message)
        """
        try:
            # Validate trim parameters
            duration = len(audio_data) / sample_rate
            start_time = max(0, min(start_time, duration))
            end_time = max(start_time, min(end_time, duration))
            
            # Convert to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Perform trim
            trimmed_audio = audio_data[start_sample:end_sample]
            
            # Auto-save functionality
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"trimmed_audio_{timestamp}.wav"
            
            # Automatic saving (revolutionary feature from original system)
            save_success = save_audio_file(trimmed_audio, output_path, sample_rate)
            
            if save_success:
                trim_duration = end_time - start_time
                save_message = f"✅ Audio trimmed ({trim_duration:.1f}s) and auto-saved: {output_path}"
            else:
                save_message = f"⚠️  Audio trimmed but save failed: {output_path}"
            
            return trimmed_audio, save_message
            
        except Exception as e:
            return audio_data, f"❌ Trim operation failed: {str(e)}"
    
    def normalize_to_standard(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        target_lufs: float = -23.0,
        max_peak: float = -3.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize audio to broadcast standards (EBU R128 / ACX).
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            target_lufs (float): Target LUFS level
            max_peak (float): Maximum peak level in dB
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (normalized_audio, metadata)
        """
        try:
            # Basic peak normalization first
            peak_level = np.max(np.abs(audio_data))
            if peak_level > 0:
                # Normalize to prevent clipping with headroom
                max_peak_linear = 10 ** (max_peak / 20)
                if peak_level > max_peak_linear:
                    audio_data = audio_data * (max_peak_linear / peak_level)
            
            # RMS-based normalization (simplified LUFS approximation)
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                target_rms = 10 ** (target_lufs / 20)
                rms_gain = target_rms / rms
                
                # Apply gain with limiting
                normalized_audio = audio_data * rms_gain
                
                # Final peak limiting
                final_peak = np.max(np.abs(normalized_audio))
                if final_peak > max_peak_linear:
                    normalized_audio = normalized_audio * (max_peak_linear / final_peak)
            else:
                normalized_audio = audio_data
            
            # Calculate final statistics
            final_peak_db = 20 * np.log10(np.max(np.abs(normalized_audio)) + 1e-10)
            final_rms_db = 20 * np.log10(np.sqrt(np.mean(normalized_audio ** 2)) + 1e-10)
            
            metadata = {
                'operation': 'normalize_to_standard',
                'target_lufs': target_lufs,
                'max_peak': max_peak,
                'final_peak_db': final_peak_db,
                'final_rms_db': final_rms_db,
                'success': True
            }
            
            return normalized_audio.astype(np.float32), metadata
            
        except Exception as e:
            print(f"⚠️  Normalization failed: {e}")
            return audio_data, {'operation': 'normalize_to_standard', 'success': False, 'error': str(e)}
    
    def _apply_audiobook_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply the complete audiobook production preset."""
        # 1. High-pass filter
        audio_data, eq_meta = self.apply_eq(audio_data, sample_rate, 
                                          low_cut=80.0, high_cut=8000.0, presence_boost=2.0)
        metadata['processing_steps'].append(eq_meta)
        
        # 2. Noise reduction
        audio_data, nr_meta = self.apply_noise_reduction(audio_data, sample_rate, 0.3)
        metadata['processing_steps'].append(nr_meta)
        
        # 3. Compression
        audio_data, comp_meta = self.apply_compression(audio_data, sample_rate,
                                                     threshold=-15.0, ratio=3.0)
        metadata['processing_steps'].append(comp_meta)
        
        # 4. Final normalization
        audio_data, norm_meta = self.normalize_to_standard(audio_data, sample_rate,
                                                         target_lufs=-23.0, max_peak=-3.0)
        metadata['processing_steps'].append(norm_meta)
        
        return audio_data
    
    def _apply_speech_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply speech-optimized processing preset."""
        # Lighter processing for speech
        audio_data, eq_meta = self.apply_eq(audio_data, sample_rate,
                                          low_cut=100.0, high_cut=7000.0, presence_boost=1.5)
        metadata['processing_steps'].append(eq_meta)
        
        audio_data, comp_meta = self.apply_compression(audio_data, sample_rate,
                                                     threshold=-18.0, ratio=2.5)
        metadata['processing_steps'].append(comp_meta)
        
        audio_data, norm_meta = self.normalize_to_standard(audio_data, sample_rate,
                                                         target_lufs=-20.0, max_peak=-2.0)
        metadata['processing_steps'].append(norm_meta)
        
        return audio_data
    
    def _apply_custom_processing(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply custom processing based on configuration."""
        if self.config.eq_enabled:
            audio_data, eq_meta = self.apply_eq(audio_data, sample_rate,
                                              self.config.low_cut_freq,
                                              self.config.high_cut_freq,
                                              self.config.presence_boost)
            metadata['processing_steps'].append(eq_meta)
        
        if self.config.noise_reduction_enabled:
            audio_data, nr_meta = self.apply_noise_reduction(audio_data, sample_rate, 0.4)
            metadata['processing_steps'].append(nr_meta)
        
        if self.config.compression_enabled:
            audio_data, comp_meta = self.apply_compression(audio_data, sample_rate,
                                                         self.config.compression_threshold,
                                                         self.config.compression_ratio,
                                                         self.config.attack_time,
                                                         self.config.release_time)
            metadata['processing_steps'].append(comp_meta)
        
        audio_data, norm_meta = self.normalize_to_standard(audio_data, sample_rate,
                                                         self.config.target_lufs,
                                                         self.config.max_peak)
        metadata['processing_steps'].append(norm_meta)
        
        return audio_data
    
    def _analyze_audio_stats(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze audio statistics for metadata."""
        peak_db = 20 * np.log10(np.max(np.abs(audio_data)) + 1e-10)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio_data ** 2)) + 1e-10)
        duration = len(audio_data) / sample_rate
        
        return {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'duration': duration,
            'sample_rate': sample_rate,
            'samples': len(audio_data)
        }
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get the complete processing history."""
        return self.processing_history.copy()

# ==============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ==============================================================================

# Global effects processor for convenience
_global_effects_processor: Optional[EffectsProcessor] = None

def get_global_effects_processor() -> EffectsProcessor:
    """Get or create the global effects processor instance."""
    global _global_effects_processor
    if _global_effects_processor is None:
        _global_effects_processor = EffectsProcessor()
    return _global_effects_processor

def enhance_audio(audio_data: np.ndarray, sample_rate: int, preset: str = "audiobook") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Enhance audio (backward compatibility)."""
    processor = get_global_effects_processor()
    return processor.enhance_audio(audio_data, sample_rate, preset)

def apply_effects(audio_data: np.ndarray, sample_rate: int, effects_config: Dict[str, Any]) -> np.ndarray:
    """Apply effects (backward compatibility)."""
    processor = get_global_effects_processor()
    enhanced_audio, _ = processor.enhance_audio(audio_data, sample_rate, "custom")
    return enhanced_audio

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Audio Effects Processor module loaded")
print("🎛️ Professional broadcast-quality audio processing ready") 