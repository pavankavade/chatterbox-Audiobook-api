"""
# ==============================================================================
# AUDIO ENHANCEMENT TOOLS MODULE
# ==============================================================================
# 
# This module provides advanced audio enhancement tools and professional
# finishing effects for the Chatterbox Audiobook Studio refactored system.
# It handles advanced cleanup algorithms, mastering-grade enhancement,
# and specialized audiobook production tools.
# 
# **Key Features:**
# - **Advanced Enhancement**: Mastering-grade audio improvement algorithms
# - **Professional Finishing**: Broadcast-quality final processing
# - **Specialized Tools**: Audiobook-specific enhancement features
# - **Cleanup Algorithms**: Advanced noise and artifact removal
# - **Original Compatibility**: Full compatibility with existing enhancement workflows
"""

import numpy as np
import scipy.signal
import scipy.ndimage
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Import core modules
from core.audio_processing import (
    normalize_audio_array, save_audio_file, validate_audio_array, 
    DEFAULT_SAMPLE_RATE
)

# ==============================================================================
# ENHANCEMENT CONFIGURATION
# ==============================================================================

@dataclass
class EnhancementConfig:
    """Configuration for advanced enhancement tools."""
    # Advanced processing
    spectral_repair: bool = True
    breath_reduction: bool = True
    plosive_reduction: bool = True
    
    # Mastering settings
    multiband_processing: bool = True
    stereo_enhancement: bool = False  # Usually false for audiobooks
    harmonic_enhancement: bool = True
    
    # Quality targets
    target_snr: float = 50.0
    target_clarity: float = 0.8
    target_consistency: float = 0.9

# ==============================================================================
# ENHANCEMENT TOOLS CLASS
# ==============================================================================

class EnhancementTools:
    """
    Advanced audio enhancement and finishing tools.
    
    This class provides mastering-grade enhancement capabilities
    for professional audiobook production finishing.
    
    **Enhancement Features:**
    - **Spectral Repair**: Advanced artifact removal and cleanup
    - **Breath Processing**: Intelligent breath reduction for speech
    - **Plosive Control**: Professional plosive and sibilant management
    - **Harmonic Enhancement**: Voice presence and clarity improvement
    - **Consistency Processing**: Multi-chapter consistency optimization
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        """Initialize enhancement tools with professional configuration."""
        self.config = config or EnhancementConfig()
        self.enhancement_history: List[Dict[str, Any]] = []
        
        print("✅ Enhancement Tools initialized - Mastering-grade audio finishing ready")
    
    def master_enhance_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        preset: str = "audiobook_master"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply complete mastering-grade enhancement chain.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            preset (str): Enhancement preset
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (enhanced_audio, processing_metadata)
        """
        try:
            metadata = {
                'preset': preset,
                'enhancement_steps': [],
                'start_time': datetime.now().isoformat()
            }
            
            enhanced_audio = audio_data.copy()
            
            # 1. Spectral repair
            if self.config.spectral_repair:
                enhanced_audio, step_meta = self.repair_spectral_artifacts(enhanced_audio, sample_rate)
                metadata['enhancement_steps'].append(step_meta)
            
            # 2. Breath reduction
            if self.config.breath_reduction:
                enhanced_audio, step_meta = self.reduce_breath_sounds(enhanced_audio, sample_rate)
                metadata['enhancement_steps'].append(step_meta)
            
            # 3. Plosive reduction
            if self.config.plosive_reduction:
                enhanced_audio, step_meta = self.reduce_plosives(enhanced_audio, sample_rate)
                metadata['enhancement_steps'].append(step_meta)
            
            # 4. Harmonic enhancement
            if self.config.harmonic_enhancement:
                enhanced_audio, step_meta = self.enhance_voice_harmonics(enhanced_audio, sample_rate)
                metadata['enhancement_steps'].append(step_meta)
            
            # 5. Final consistency processing
            enhanced_audio, step_meta = self.apply_consistency_processing(enhanced_audio, sample_rate)
            metadata['enhancement_steps'].append(step_meta)
            
            metadata['end_time'] = datetime.now().isoformat()
            metadata['success'] = True
            
            self.enhancement_history.append(metadata)
            
            return enhanced_audio, metadata
            
        except Exception as e:
            error_metadata = {
                'preset': preset,
                'error': str(e),
                'success': False
            }
            return audio_data, error_metadata
    
    def repair_spectral_artifacts(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advanced spectral artifact removal."""
        try:
            # Spectral gate for artifact removal
            stft = np.abs(np.fft.stft(audio_data, nperseg=2048)[2])
            
            # Identify spectral anomalies
            median_spectrum = np.median(stft, axis=1, keepdims=True)
            spectral_deviation = stft / (median_spectrum + 1e-10)
            
            # Create repair mask
            repair_mask = np.where(spectral_deviation > 3.0, 0.3, 1.0)
            
            # Apply spectral repair
            _, _, repaired_stft = np.fft.stft(audio_data, nperseg=2048)
            repaired_stft = repaired_stft * repair_mask
            
            repaired_audio = np.fft.istft(repaired_stft, nperseg=2048)[1]
            
            # Ensure correct length
            if len(repaired_audio) != len(audio_data):
                repaired_audio = np.pad(repaired_audio, (0, len(audio_data) - len(repaired_audio)), 'constant')[:len(audio_data)]
            
            metadata = {
                'operation': 'spectral_repair',
                'artifacts_detected': np.sum(repair_mask < 1.0),
                'success': True
            }
            
            return repaired_audio.astype(np.float32), metadata
            
        except Exception as e:
            return audio_data, {'operation': 'spectral_repair', 'success': False, 'error': str(e)}
    
    def reduce_breath_sounds(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        reduction_amount: float = 0.6
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Intelligent breath sound reduction for speech."""
        try:
            # Detect breath sounds using spectral characteristics
            window_size = int(0.05 * sample_rate)  # 50ms windows
            hop_size = window_size // 4
            
            breath_mask = np.ones(len(audio_data))
            
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                
                # Breath detection heuristics
                # High frequency content with low overall energy
                fft = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(len(window), 1/sample_rate)
                
                high_freq_energy = np.sum(np.abs(fft[freqs > 2000]) ** 2)
                total_energy = np.sum(np.abs(fft) ** 2)
                
                if total_energy > 0:
                    high_freq_ratio = high_freq_energy / total_energy
                    rms = np.sqrt(np.mean(window ** 2))
                    
                    # Breath characteristics: high freq ratio, low RMS
                    if high_freq_ratio > 0.3 and rms < 0.05:
                        # Reduce this section
                        breath_mask[i:i + window_size] *= (1 - reduction_amount)
            
            # Apply breath reduction
            processed_audio = audio_data * breath_mask
            
            metadata = {
                'operation': 'breath_reduction',
                'reduction_amount': reduction_amount,
                'breath_sections_detected': np.sum(breath_mask < 0.99),
                'success': True
            }
            
            return processed_audio.astype(np.float32), metadata
            
        except Exception as e:
            return audio_data, {'operation': 'breath_reduction', 'success': False, 'error': str(e)}
    
    def reduce_plosives(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Professional plosive and sibilant reduction."""
        try:
            # Multi-band approach for plosive control
            
            # Low frequency plosive reduction (P, B sounds)
            b_low, a_low = scipy.signal.butter(4, [20, 200], btype='bandpass', fs=sample_rate)
            low_band = scipy.signal.filtfilt(b_low, a_low, audio_data)
            
            # Apply gentle compression to low band
            low_band_compressed = self._apply_gentle_compression(low_band, threshold=-20, ratio=4.0)
            
            # High frequency sibilant reduction (S, T sounds)
            b_high, a_high = scipy.signal.butter(4, [5000, 10000], btype='bandpass', fs=sample_rate)
            high_band = scipy.signal.filtfilt(b_high, a_high, audio_data)
            
            # Apply de-esser to high band
            high_band_deessed = self._apply_deesser(high_band, sample_rate)
            
            # Reconstruct audio
            # Get the mid band
            b_mid, a_mid = scipy.signal.butter(4, [200, 5000], btype='bandpass', fs=sample_rate)
            mid_band = scipy.signal.filtfilt(b_mid, a_mid, audio_data)
            
            # Combine bands
            processed_audio = low_band_compressed + mid_band + high_band_deessed
            
            metadata = {
                'operation': 'plosive_reduction',
                'low_band_compression': True,
                'high_band_deessing': True,
                'success': True
            }
            
            return processed_audio.astype(np.float32), metadata
            
        except Exception as e:
            return audio_data, {'operation': 'plosive_reduction', 'success': False, 'error': str(e)}
    
    def enhance_voice_harmonics(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhance voice harmonics for clarity and presence."""
        try:
            # Harmonic enhancement using spectral processing
            stft_data = np.fft.stft(audio_data, nperseg=2048)
            freqs, times, stft = stft_data
            
            # Enhance harmonics in voice range (80Hz - 4kHz)
            voice_range = (freqs >= 80) & (freqs <= 4000)
            
            # Apply gentle harmonic enhancement
            enhancement_curve = np.ones_like(freqs)
            enhancement_curve[voice_range] = 1.1  # 10% boost
            
            # Apply frequency-dependent enhancement
            enhanced_stft = stft * enhancement_curve[:, np.newaxis]
            
            # Reconstruct audio
            enhanced_audio = np.fft.istft(enhanced_stft, nperseg=2048)[1]
            
            # Ensure correct length
            if len(enhanced_audio) != len(audio_data):
                enhanced_audio = np.pad(enhanced_audio, (0, len(audio_data) - len(enhanced_audio)), 'constant')[:len(audio_data)]
            
            metadata = {
                'operation': 'harmonic_enhancement',
                'voice_range_hz': '80-4000',
                'enhancement_amount': '10%',
                'success': True
            }
            
            return enhanced_audio.astype(np.float32), metadata
            
        except Exception as e:
            return audio_data, {'operation': 'harmonic_enhancement', 'success': False, 'error': str(e)}
    
    def apply_consistency_processing(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply consistency processing for uniform sound across chapters."""
        try:
            # Gentle leveling for consistency
            window_size = int(1.0 * sample_rate)  # 1-second windows
            leveled_audio = audio_data.copy()
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                
                # Calculate window RMS
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms > 0:
                    # Target RMS for consistency
                    target_rms = 0.1  # Adjust based on content
                    
                    # Gentle adjustment
                    adjustment = min(target_rms / rms, 2.0)  # Max 6dB boost
                    adjustment = max(adjustment, 0.5)       # Max 6dB cut
                    
                    leveled_audio[i:i + window_size] *= adjustment
            
            # Apply gentle smoothing to avoid artifacts
            leveled_audio = scipy.ndimage.gaussian_filter1d(leveled_audio, sigma=sample_rate * 0.001)
            
            metadata = {
                'operation': 'consistency_processing',
                'window_size_seconds': 1.0,
                'max_adjustment_db': 6.0,
                'success': True
            }
            
            return leveled_audio.astype(np.float32), metadata
            
        except Exception as e:
            return audio_data, {'operation': 'consistency_processing', 'success': False, 'error': str(e)}
    
    def _apply_gentle_compression(
        self,
        audio_data: np.ndarray,
        threshold: float = -20.0,
        ratio: float = 3.0
    ) -> np.ndarray:
        """Apply gentle compression to audio."""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)
        
        # Apply compression above threshold
        compressed_db = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        gain_db = compressed_db - audio_db
        gain_linear = 10 ** (gain_db / 20)
        
        return audio_data * gain_linear
    
    def _apply_deesser(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        frequency: float = 6000.0,
        threshold: float = -15.0
    ) -> np.ndarray:
        """Apply de-essing to reduce sibilants."""
        # Detect sibilant energy
        b, a = scipy.signal.butter(4, frequency, btype='highpass', fs=sample_rate)
        sibilant_signal = scipy.signal.filtfilt(b, a, audio_data)
        
        # Calculate sibilant level
        sibilant_level = 20 * np.log10(np.abs(sibilant_signal) + 1e-10)
        
        # Create reduction mask
        reduction_mask = np.where(
            sibilant_level > threshold,
            10 ** ((threshold - sibilant_level) / 40),  # Gentle reduction
            1.0
        )
        
        return audio_data * reduction_mask

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

_global_enhancement_tools: Optional[EnhancementTools] = None

def get_global_enhancement_tools() -> EnhancementTools:
    """Get or create the global enhancement tools instance."""
    global _global_enhancement_tools
    if _global_enhancement_tools is None:
        _global_enhancement_tools = EnhancementTools()
    return _global_enhancement_tools

def enhance_audio_master(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Master enhance audio (backward compatibility)."""
    tools = get_global_enhancement_tools()
    enhanced_audio, _ = tools.master_enhance_audio(audio_data, sample_rate)
    return enhanced_audio

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Audio Enhancement Tools module loaded")
print("🎭 Mastering-grade audio finishing and professional enhancement ready") 