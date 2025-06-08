"""
# ==============================================================================
# AUDIO QUALITY ANALYZER MODULE
# ==============================================================================
# 
# This module provides comprehensive audio quality analysis and validation
# for the Chatterbox Audiobook Studio refactored system. It handles volume
# normalization with RMS/Peak/LUFS measurements, professional audio validation,
# and broadcast standard compliance checking.
# 
# **Key Features:**
# - **Volume Normalization**: Professional RMS/Peak/LUFS analysis and correction
# - **Quality Metrics**: Comprehensive audio quality measurement and reporting
# - **Broadcast Standards**: ACX audiobook and EBU R128 compliance validation
# - **Audio Validation**: Professional quality checking and issue detection
# - **Original Compatibility**: Full compatibility with existing quality workflows
"""

import numpy as np
import librosa
import scipy.stats
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import core audio processing
from core.audio_processing import (
    load_audio_file, validate_audio_array, DEFAULT_SAMPLE_RATE
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# QUALITY ANALYSIS DATA STRUCTURES
# ==============================================================================

@dataclass
class AudioQualityMetrics:
    """
    Comprehensive audio quality metrics structure.
    """
    # Level measurements
    peak_db: float = 0.0
    rms_db: float = 0.0
    lufs_integrated: float = 0.0
    lufs_momentary_max: float = 0.0
    lufs_short_term_max: float = 0.0
    
    # Dynamic range
    dynamic_range_db: float = 0.0
    crest_factor: float = 0.0
    
    # Frequency analysis
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    zero_crossing_rate: float = 0.0
    
    # Noise and distortion
    snr_estimate: float = 0.0
    thd_estimate: float = 0.0
    noise_floor_db: float = 0.0
    
    # Timing and rhythm
    tempo: float = 0.0
    onset_rate: float = 0.0
    
    # Quality indicators
    clipping_detected: bool = False
    silence_percentage: float = 0.0
    
    # File metadata
    duration: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    bit_depth: int = 16
    file_size: int = 0
    
    # Analysis metadata
    analysis_time: str = field(default_factory=lambda: datetime.now().isoformat())
    analyzer_version: str = "1.0"

@dataclass
class QualityStandards:
    """
    Audio quality standards for different broadcast formats.
    """
    # ACX Audiobook Standards
    acx_peak_max: float = -3.0
    acx_rms_min: float = -18.0
    acx_rms_max: float = -23.0
    acx_noise_floor_max: float = -60.0
    
    # EBU R128 Standards
    ebu_lufs_target: float = -23.0
    ebu_lufs_tolerance: float = 1.0
    ebu_max_peak: float = -1.0
    
    # General podcast standards
    podcast_lufs_target: float = -16.0
    podcast_peak_max: float = -1.0
    
    # Quality thresholds
    min_dynamic_range: float = 10.0
    max_silence_percentage: float = 5.0
    min_snr: float = 40.0

# ==============================================================================
# PROFESSIONAL QUALITY ANALYZER CLASS
# ==============================================================================

class QualityAnalyzer:
    """
    Professional audio quality analysis and validation system.
    
    This class provides comprehensive audio quality analysis capabilities
    extracted and enhanced from the original system's quality management
    with professional broadcast standard compliance checking.
    
    **Analysis Features:**
    - **LUFS Measurement**: Professional loudness analysis (EBU R128)
    - **Dynamic Range**: Comprehensive dynamic range analysis
    - **Frequency Analysis**: Spectral content and voice characteristics
    - **Noise Analysis**: SNR estimation and noise floor detection
    - **Standards Compliance**: ACX audiobook and broadcast standard validation
    """
    
    def __init__(self, standards: Optional[QualityStandards] = None):
        """
        Initialize the quality analyzer with professional standards.
        
        Args:
            standards (Optional[QualityStandards]): Quality standards configuration
        """
        self.standards = standards or QualityStandards()
        self.analysis_history: List[AudioQualityMetrics] = []
        
        print("✅ Quality Analyzer initialized - Professional audio validation ready")
    
    def analyze_audio_quality(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        detailed: bool = True
    ) -> AudioQualityMetrics:
        """
        Perform comprehensive audio quality analysis.
        
        This is the master quality analysis function that performs complete
        audio quality assessment including all professional measurements.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate of the audio
            detailed (bool): Whether to perform detailed analysis
            
        Returns:
            AudioQualityMetrics: Complete audio quality metrics
            
        **Analysis Categories:**
        - **Level Analysis**: Peak, RMS, LUFS measurements
        - **Dynamic Analysis**: Dynamic range and crest factor
        - **Frequency Analysis**: Spectral characteristics and voice metrics  
        - **Noise Analysis**: Signal-to-noise ratio and noise floor
        - **Quality Issues**: Clipping detection and silence analysis
        """
        try:
            # Validate input
            is_valid, error_msg = validate_audio_array(audio_data)
            if not is_valid:
                raise ValueError(f"Invalid audio data: {error_msg}")
            
            # Initialize metrics
            metrics = AudioQualityMetrics()
            metrics.duration = len(audio_data) / sample_rate
            metrics.sample_rate = sample_rate
            
            # Core level measurements
            metrics = self._analyze_levels(audio_data, sample_rate, metrics)
            
            # Dynamic range analysis
            metrics = self._analyze_dynamics(audio_data, sample_rate, metrics)
            
            if detailed:
                # Detailed frequency analysis
                metrics = self._analyze_frequency_content(audio_data, sample_rate, metrics)
                
                # Noise and distortion analysis
                metrics = self._analyze_noise_and_distortion(audio_data, sample_rate, metrics)
                
                # Timing and rhythm analysis
                metrics = self._analyze_timing(audio_data, sample_rate, metrics)
            
            # Quality issue detection
            metrics = self._detect_quality_issues(audio_data, sample_rate, metrics)
            
            # Add to analysis history
            self.analysis_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Quality analysis failed: {e}")
            # Return basic metrics with error indication
            error_metrics = AudioQualityMetrics()
            error_metrics.analysis_time = f"ERROR: {str(e)}"
            return error_metrics
    
    def measure_lufs(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Measure LUFS (Loudness Units Full Scale) according to EBU R128.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate
            
        Returns:
            Dict[str, float]: LUFS measurements
            
        **LUFS Measurements:**
        - **Integrated LUFS**: Overall loudness across entire program
        - **Momentary LUFS**: Peak momentary loudness (400ms window)
        - **Short-term LUFS**: Peak short-term loudness (3s window)
        """
        try:
            # K-weighting filter (simplified approximation)
            # This is a simplified implementation - professional tools use exact EBU R128 filters
            
            # Pre-filter (shelf filter around 1500Hz)
            from scipy import signal
            b, a = signal.iirfilter(2, [1500], btype='highpass', ftype='butter', fs=sample_rate)
            audio_filtered = signal.filtfilt(b, a, audio_data)
            
            # RLB weighting (simplified)
            b, a = signal.iirfilter(1, [38], btype='highpass', ftype='butter', fs=sample_rate)
            audio_weighted = signal.filtfilt(b, a, audio_filtered)
            
            # Mean square calculation
            mean_square = np.mean(audio_weighted ** 2)
            
            # Convert to LUFS (simplified calculation)
            integrated_lufs = -0.691 + 10 * np.log10(mean_square + 1e-10)
            
            # Momentary (400ms windows)
            window_samples = int(0.4 * sample_rate)
            momentary_lufs = []
            
            for i in range(0, len(audio_weighted) - window_samples, window_samples // 4):
                window = audio_weighted[i:i + window_samples]
                ms = np.mean(window ** 2)
                lufs = -0.691 + 10 * np.log10(ms + 1e-10)
                momentary_lufs.append(lufs)
            
            momentary_max = max(momentary_lufs) if momentary_lufs else integrated_lufs
            
            # Short-term (3s windows)
            window_samples = int(3.0 * sample_rate)
            short_term_lufs = []
            
            for i in range(0, len(audio_weighted) - window_samples, window_samples // 4):
                window = audio_weighted[i:i + window_samples]
                ms = np.mean(window ** 2)
                lufs = -0.691 + 10 * np.log10(ms + 1e-10)
                short_term_lufs.append(lufs)
            
            short_term_max = max(short_term_lufs) if short_term_lufs else integrated_lufs
            
            return {
                'integrated': integrated_lufs,
                'momentary_max': momentary_max,
                'short_term_max': short_term_max
            }
            
        except Exception as e:
            print(f"⚠️  LUFS measurement failed: {e}")
            return {
                'integrated': -23.0,  # Default
                'momentary_max': -23.0,
                'short_term_max': -23.0
            }
    
    def validate_broadcast_standards(
        self,
        metrics: AudioQualityMetrics,
        standard: str = "acx"
    ) -> Dict[str, Any]:
        """
        Validate audio against broadcast standards.
        
        Args:
            metrics (AudioQualityMetrics): Audio quality metrics
            standard (str): Standard to validate against ("acx", "ebu", "podcast")
            
        Returns:
            Dict[str, Any]: Validation results with pass/fail and recommendations
        """
        validation = {
            'standard': standard,
            'overall_pass': True,
            'checks': [],
            'recommendations': [],
            'compliance_score': 100.0
        }
        
        deductions = 0
        
        if standard == "acx":
            # ACX Audiobook standards validation
            
            # Peak level check
            if metrics.peak_db > self.standards.acx_peak_max:
                validation['checks'].append({
                    'test': 'Peak Level',
                    'pass': False,
                    'value': metrics.peak_db,
                    'limit': self.standards.acx_peak_max,
                    'message': f'Peak level {metrics.peak_db:.1f}dB exceeds ACX maximum {self.standards.acx_peak_max}dB'
                })
                validation['recommendations'].append('Apply limiting to reduce peak levels')
                validation['overall_pass'] = False
                deductions += 20
            else:
                validation['checks'].append({
                    'test': 'Peak Level',
                    'pass': True,
                    'value': metrics.peak_db,
                    'limit': self.standards.acx_peak_max
                })
            
            # RMS level check
            if metrics.rms_db < self.standards.acx_rms_max or metrics.rms_db > self.standards.acx_rms_min:
                validation['checks'].append({
                    'test': 'RMS Level',
                    'pass': False,
                    'value': metrics.rms_db,
                    'range': f'{self.standards.acx_rms_max} to {self.standards.acx_rms_min}',
                    'message': f'RMS level {metrics.rms_db:.1f}dB outside ACX range {self.standards.acx_rms_max} to {self.standards.acx_rms_min}dB'
                })
                validation['recommendations'].append('Adjust overall level to meet ACX RMS requirements')
                validation['overall_pass'] = False
                deductions += 30
            else:
                validation['checks'].append({
                    'test': 'RMS Level',
                    'pass': True,
                    'value': metrics.rms_db,
                    'range': f'{self.standards.acx_rms_max} to {self.standards.acx_rms_min}'
                })
            
            # Noise floor check
            if metrics.noise_floor_db > self.standards.acx_noise_floor_max:
                validation['checks'].append({
                    'test': 'Noise Floor',
                    'pass': False,
                    'value': metrics.noise_floor_db,
                    'limit': self.standards.acx_noise_floor_max,
                    'message': f'Noise floor {metrics.noise_floor_db:.1f}dB exceeds ACX maximum {self.standards.acx_noise_floor_max}dB'
                })
                validation['recommendations'].append('Apply noise reduction to lower noise floor')
                validation['overall_pass'] = False
                deductions += 25
            else:
                validation['checks'].append({
                    'test': 'Noise Floor',
                    'pass': True,
                    'value': metrics.noise_floor_db,
                    'limit': self.standards.acx_noise_floor_max
                })
        
        elif standard == "ebu":
            # EBU R128 standards validation
            
            # LUFS level check
            lufs_diff = abs(metrics.lufs_integrated - self.standards.ebu_lufs_target)
            if lufs_diff > self.standards.ebu_lufs_tolerance:
                validation['checks'].append({
                    'test': 'LUFS Level',
                    'pass': False,
                    'value': metrics.lufs_integrated,
                    'target': self.standards.ebu_lufs_target,
                    'tolerance': self.standards.ebu_lufs_tolerance,
                    'message': f'LUFS {metrics.lufs_integrated:.1f} outside EBU R128 tolerance'
                })
                validation['recommendations'].append('Adjust loudness to meet EBU R128 standard')
                validation['overall_pass'] = False
                deductions += 30
            else:
                validation['checks'].append({
                    'test': 'LUFS Level',
                    'pass': True,
                    'value': metrics.lufs_integrated,
                    'target': self.standards.ebu_lufs_target
                })
            
            # True peak check
            if metrics.peak_db > self.standards.ebu_max_peak:
                validation['checks'].append({
                    'test': 'True Peak',
                    'pass': False,
                    'value': metrics.peak_db,
                    'limit': self.standards.ebu_max_peak,
                    'message': f'True peak {metrics.peak_db:.1f}dB exceeds EBU maximum {self.standards.ebu_max_peak}dB'
                })
                validation['recommendations'].append('Apply true peak limiting')
                validation['overall_pass'] = False
                deductions += 20
            else:
                validation['checks'].append({
                    'test': 'True Peak',
                    'pass': True,
                    'value': metrics.peak_db,
                    'limit': self.standards.ebu_max_peak
                })
        
        # Common quality checks for all standards
        
        # Clipping check
        if metrics.clipping_detected:
            validation['checks'].append({
                'test': 'Clipping Detection',
                'pass': False,
                'message': 'Digital clipping detected in audio'
            })
            validation['recommendations'].append('Remove clipping through limiting or level reduction')
            validation['overall_pass'] = False
            deductions += 40
        else:
            validation['checks'].append({
                'test': 'Clipping Detection',
                'pass': True
            })
        
        # Dynamic range check
        if metrics.dynamic_range_db < self.standards.min_dynamic_range:
            validation['checks'].append({
                'test': 'Dynamic Range',
                'pass': False,
                'value': metrics.dynamic_range_db,
                'minimum': self.standards.min_dynamic_range,
                'message': f'Dynamic range {metrics.dynamic_range_db:.1f}dB below minimum {self.standards.min_dynamic_range}dB'
            })
            validation['recommendations'].append('Reduce compression to preserve dynamic range')
            deductions += 15
        else:
            validation['checks'].append({
                'test': 'Dynamic Range',
                'pass': True,
                'value': metrics.dynamic_range_db,
                'minimum': self.standards.min_dynamic_range
            })
        
        # Calculate final compliance score
        validation['compliance_score'] = max(0, 100 - deductions)
        
        return validation
    
    def recommend_normalization(
        self,
        metrics: AudioQualityMetrics,
        target_standard: str = "acx"
    ) -> Dict[str, Any]:
        """
        Recommend normalization settings to meet target standard.
        
        Args:
            metrics (AudioQualityMetrics): Current audio metrics
            target_standard (str): Target standard ("acx", "ebu", "podcast")
            
        Returns:
            Dict[str, Any]: Normalization recommendations
        """
        recommendations = {
            'target_standard': target_standard,
            'adjustments_needed': [],
            'processing_chain': []
        }
        
        if target_standard == "acx":
            target_rms = (self.standards.acx_rms_min + self.standards.acx_rms_max) / 2  # -20.5 dB
            target_peak = self.standards.acx_peak_max + 0.5  # -2.5 dB (with headroom)
            
            # Level adjustment calculation
            rms_adjustment = target_rms - metrics.rms_db
            peak_adjustment = target_peak - metrics.peak_db
            
            # Use the more conservative adjustment
            level_adjustment = min(rms_adjustment, peak_adjustment)
            
            if abs(level_adjustment) > 0.5:
                recommendations['adjustments_needed'].append({
                    'type': 'level_adjustment',
                    'amount_db': level_adjustment,
                    'reason': f'Adjust overall level by {level_adjustment:+.1f}dB for ACX compliance'
                })
            
            # Processing chain recommendation
            recommendations['processing_chain'] = [
                {'step': 'high_pass_filter', 'frequency': 80, 'reason': 'Remove low-frequency rumble'},
                {'step': 'noise_reduction', 'amount': 0.3, 'reason': 'Meet ACX noise floor requirement'},
                {'step': 'compression', 'ratio': 3.0, 'threshold': -15.0, 'reason': 'Even out dynamic range'},
                {'step': 'eq', 'presence_boost': 2.0, 'reason': 'Enhance voice clarity'},
                {'step': 'normalization', 'target_rms': target_rms, 'target_peak': target_peak, 'reason': 'Meet ACX level requirements'},
                {'step': 'limiting', 'threshold': -3.0, 'reason': 'Prevent peak violations'}
            ]
        
        elif target_standard == "ebu":
            target_lufs = self.standards.ebu_lufs_target
            lufs_adjustment = target_lufs - metrics.lufs_integrated
            
            if abs(lufs_adjustment) > 0.5:
                recommendations['adjustments_needed'].append({
                    'type': 'lufs_adjustment',
                    'amount_lufs': lufs_adjustment,
                    'reason': f'Adjust loudness by {lufs_adjustment:+.1f} LUFS for EBU R128 compliance'
                })
            
            recommendations['processing_chain'] = [
                {'step': 'eq', 'low_cut': 40, 'reason': 'Clean up low end'},
                {'step': 'compression', 'ratio': 2.5, 'threshold': -18.0, 'reason': 'Gentle dynamic control'},
                {'step': 'lufs_normalization', 'target': target_lufs, 'reason': 'Meet EBU R128 standard'},
                {'step': 'true_peak_limiting', 'threshold': -1.0, 'reason': 'Prevent intersample peaks'}
            ]
        
        return recommendations
    
    def _analyze_levels(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Analyze audio levels (peak, RMS, LUFS)."""
        # Peak level
        peak_linear = np.max(np.abs(audio_data))
        metrics.peak_db = 20 * np.log10(peak_linear + 1e-10)
        
        # RMS level
        rms_linear = np.sqrt(np.mean(audio_data ** 2))
        metrics.rms_db = 20 * np.log10(rms_linear + 1e-10)
        
        # LUFS measurements
        lufs_measurements = self.measure_lufs(audio_data, sample_rate)
        metrics.lufs_integrated = lufs_measurements['integrated']
        metrics.lufs_momentary_max = lufs_measurements['momentary_max']
        metrics.lufs_short_term_max = lufs_measurements['short_term_max']
        
        return metrics
    
    def _analyze_dynamics(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Analyze dynamic range characteristics."""
        # Dynamic range (simplified DR14 approximation)
        window_size = int(3.0 * sample_rate)  # 3-second windows
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if rms_values:
            rms_values = np.array(rms_values)
            # Remove silent sections (bottom 20%)
            threshold = np.percentile(rms_values, 20)
            active_rms = rms_values[rms_values > threshold]
            
            if len(active_rms) > 0:
                peak_rms = np.max(active_rms)
                avg_rms = np.mean(active_rms)
                metrics.dynamic_range_db = 20 * np.log10(peak_rms / (avg_rms + 1e-10))
            else:
                metrics.dynamic_range_db = 0.0
        
        # Crest factor
        peak = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data ** 2))
        metrics.crest_factor = 20 * np.log10((peak / (rms + 1e-10)) + 1e-10)
        
        return metrics
    
    def _analyze_frequency_content(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Analyze frequency content and spectral characteristics."""
        try:
            # Spectral features using librosa
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            metrics.spectral_centroid = np.mean(spectral_centroids)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            metrics.spectral_bandwidth = np.mean(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            metrics.spectral_rolloff = np.mean(spectral_rolloff)
            
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            metrics.zero_crossing_rate = np.mean(zcr)
            
        except Exception as e:
            print(f"⚠️  Frequency analysis failed: {e}")
        
        return metrics
    
    def _analyze_noise_and_distortion(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Analyze noise floor and distortion characteristics."""
        # Estimate noise floor from quietest sections
        window_size = int(0.5 * sample_rate)  # 0.5-second windows
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if rms_values:
            # Noise floor estimate (bottom 10% of RMS values)
            noise_floor_linear = np.percentile(rms_values, 10)
            metrics.noise_floor_db = 20 * np.log10(noise_floor_linear + 1e-10)
            
            # SNR estimate
            signal_rms = np.sqrt(np.mean(audio_data ** 2))
            metrics.snr_estimate = 20 * np.log10((signal_rms / (noise_floor_linear + 1e-10)) + 1e-10)
        
        # THD estimate (simplified)
        try:
            # Basic harmonic distortion estimate using spectral analysis
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            
            # Find fundamental frequency (simplified)
            freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
            fundamental_idx = np.argmax(magnitude[10:]) + 10  # Skip DC and very low frequencies
            
            if fundamental_idx < len(magnitude) // 4:  # Only if reasonable fundamental
                fundamental_power = magnitude[fundamental_idx] ** 2
                
                # Look for harmonics
                harmonic_power = 0
                for h in range(2, 6):  # 2nd to 5th harmonics
                    harmonic_idx = int(fundamental_idx * h)
                    if harmonic_idx < len(magnitude):
                        harmonic_power += magnitude[harmonic_idx] ** 2
                
                if fundamental_power > 0:
                    metrics.thd_estimate = 100 * np.sqrt(harmonic_power / fundamental_power)
        
        except Exception as e:
            print(f"⚠️  THD analysis failed: {e}")
        
        return metrics
    
    def _analyze_timing(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Analyze timing and rhythm characteristics."""
        try:
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            metrics.tempo = float(tempo)
            
            # Onset detection rate
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
            metrics.onset_rate = len(onset_frames) / metrics.duration
            
        except Exception as e:
            print(f"⚠️  Timing analysis failed: {e}")
        
        return metrics
    
    def _detect_quality_issues(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Detect various audio quality issues."""
        # Clipping detection
        clipping_threshold = 0.99  # 99% of full scale
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        metrics.clipping_detected = clipped_samples > (len(audio_data) * 0.001)  # More than 0.1%
        
        # Silence detection
        silence_threshold = 0.001  # -60 dB roughly
        silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
        metrics.silence_percentage = (silent_samples / len(audio_data)) * 100
        
        return metrics
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analysis history."""
        if not self.analysis_history:
            return {'total_analyses': 0, 'message': 'No analyses performed yet'}
        
        return {
            'total_analyses': len(self.analysis_history),
            'latest_analysis': self.analysis_history[-1].analysis_time,
            'average_peak_db': np.mean([m.peak_db for m in self.analysis_history]),
            'average_rms_db': np.mean([m.rms_db for m in self.analysis_history]),
            'average_lufs': np.mean([m.lufs_integrated for m in self.analysis_history]),
            'total_duration_analyzed': sum([m.duration for m in self.analysis_history])
        }

# ==============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ==============================================================================

# Global quality analyzer for convenience
_global_quality_analyzer: Optional[QualityAnalyzer] = None

def get_global_quality_analyzer() -> QualityAnalyzer:
    """Get or create the global quality analyzer instance."""
    global _global_quality_analyzer
    if _global_quality_analyzer is None:
        _global_quality_analyzer = QualityAnalyzer()
    return _global_quality_analyzer

def analyze_audio_quality(audio_data: np.ndarray, sample_rate: int) -> AudioQualityMetrics:
    """Analyze audio quality (backward compatibility)."""
    analyzer = get_global_quality_analyzer()
    return analyzer.analyze_audio_quality(audio_data, sample_rate)

def normalize_volume(audio_data: np.ndarray, sample_rate: int, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize volume (backward compatibility)."""
    analyzer = get_global_quality_analyzer()
    metrics = analyzer.analyze_audio_quality(audio_data, sample_rate, detailed=False)
    
    # Simple level adjustment based on LUFS
    adjustment_db = target_lufs - metrics.lufs_integrated
    adjustment_linear = 10 ** (adjustment_db / 20)
    
    return (audio_data * adjustment_linear).astype(np.float32)

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Audio Quality Analyzer module loaded")
print("📊 Professional volume normalization and broadcast standards validation ready") 