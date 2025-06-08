"""Audio preprocessing for voice cloning and audiobook generation.

This module provides audio preprocessing functions that improve voice cloning quality,
including silence trimming, noise reduction, and audio normalization.
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")


def trim_silence(
    audio_data: np.ndarray, 
    sample_rate: int,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.1,
    fade_duration: float = 0.05
) -> Tuple[np.ndarray, dict]:
    """
    Trim silence from the beginning and end of audio data.
    
    This function improves voice cloning quality by removing silent regions
    that don't contribute to the voice characteristics.
    
    Args:
        audio_data (np.ndarray): Audio data as numpy array (float32, range -1 to 1)
        sample_rate (int): Sample rate of the audio
        silence_threshold (float): Amplitude threshold below which audio is considered silence (0-1)
        min_silence_duration (float): Minimum duration in seconds for silence to be trimmed
        fade_duration (float): Duration in seconds for fade in/out to avoid clicks
        
    Returns:
        tuple: (trimmed_audio_data, trim_info_dict)
            - trimmed_audio_data: Audio with silence trimmed
            - trim_info_dict: Information about the trimming operation
    """
    try:
        if len(audio_data) == 0:
            return audio_data, {"error": "Empty audio data"}
        
        # Convert to absolute values for threshold comparison
        abs_audio = np.abs(audio_data)
        
        # Calculate minimum silence samples
        min_silence_samples = int(min_silence_duration * sample_rate)
        fade_samples = int(fade_duration * sample_rate)
        
        # Simple approach: find first and last samples above threshold
        # Find first non-silent sample
        start_idx = 0
        for i in range(len(abs_audio)):
            if abs_audio[i] > silence_threshold:
                start_idx = max(0, i - fade_samples)  # Keep some lead-in
                break
        
        # Find last non-silent sample  
        end_idx = len(audio_data)
        for i in range(len(abs_audio) - 1, -1, -1):
            if abs_audio[i] > silence_threshold:
                end_idx = min(len(audio_data), i + fade_samples)  # Keep some lead-out
                break
        
        # Ensure we have valid indices
        if start_idx >= end_idx:
            # No audio found above threshold, return original
            return audio_data, {
                "success": True,
                "warning": "No audio found above threshold, kept original",
                "threshold": silence_threshold,
                "original_length": len(audio_data),
                "trimmed_length": len(audio_data),
                "original_duration": len(audio_data) / sample_rate,
                "trimmed_duration": len(audio_data) / sample_rate,
                "trimmed_start_seconds": 0,
                "trimmed_end_seconds": 0,
                "total_trimmed_seconds": 0,
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "fade_duration": fade_duration
            }
        
        # Trim the audio
        trimmed_audio = audio_data[start_idx:end_idx].copy()
        
        # Calculate durations for statistics
        original_duration = len(audio_data) / sample_rate
        trimmed_duration = len(trimmed_audio) / sample_rate
        
        # Apply gentle fade in/out to prevent clicks
        if len(trimmed_audio) > fade_samples * 2:
            # Gentle fade in (cosine fade for smoother transition)
            fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            trimmed_audio[:fade_samples] *= fade_in
            
            # Gentle fade out (cosine fade for smoother transition)
            fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            trimmed_audio[-fade_samples:] *= fade_out
        
        # Calculate trimming statistics
        original_duration = len(audio_data) / sample_rate
        trimmed_duration = len(trimmed_audio) / sample_rate
        trimmed_start = start_idx / sample_rate
        trimmed_end = (len(audio_data) - end_idx) / sample_rate
        
        trim_info = {
            "success": True,
            "original_length": len(audio_data),
            "trimmed_length": len(trimmed_audio),
            "original_duration": original_duration,
            "trimmed_duration": trimmed_duration,
            "trimmed_start_seconds": trimmed_start,
            "trimmed_end_seconds": trimmed_end,
            "total_trimmed_seconds": trimmed_start + trimmed_end,
            "silence_threshold": silence_threshold,
            "min_silence_duration": min_silence_duration,
            "fade_duration": fade_duration
        }
        
        return trimmed_audio, trim_info
        
    except Exception as e:
        return audio_data, {"error": f"Failed to trim silence: {str(e)}"}


def detect_silence_regions(
    audio_data: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.1
) -> list:
    """
    Detect regions of silence in audio data.
    
    Args:
        audio_data (np.ndarray): Audio data as numpy array
        sample_rate (int): Sample rate of the audio
        silence_threshold (float): Amplitude threshold for silence detection
        min_silence_duration (float): Minimum duration for a region to be considered silence
        
    Returns:
        list: List of tuples (start_time, end_time) for each silence region
    """
    try:
        abs_audio = np.abs(audio_data)
        min_silence_samples = int(min_silence_duration * sample_rate)
        
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, sample in enumerate(abs_audio):
            if sample <= silence_threshold:
                if not in_silence:
                    silence_start = i
                    in_silence = True
            else:
                if in_silence:
                    silence_length = i - silence_start
                    if silence_length >= min_silence_samples:
                        start_time = silence_start / sample_rate
                        end_time = i / sample_rate
                        silence_regions.append((start_time, end_time))
                    in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_length = len(abs_audio) - silence_start
            if silence_length >= min_silence_samples:
                start_time = silence_start / sample_rate
                end_time = len(abs_audio) / sample_rate
                silence_regions.append((start_time, end_time))
        
        return silence_regions
        
    except Exception as e:
        print(f"Error detecting silence regions: {e}")
        return []


def normalize_audio_level(
    audio_data: np.ndarray,
    target_level: float = -18.0,
    max_gain: float = 12.0
) -> Tuple[np.ndarray, dict]:
    """
    Normalize audio to a target level in dB.
    
    Args:
        audio_data (np.ndarray): Audio data as numpy array (float32, range -1 to 1)
        target_level (float): Target level in dB (negative value, e.g., -18.0)
        max_gain (float): Maximum gain to apply in dB to prevent over-amplification
        
    Returns:
        tuple: (normalized_audio, normalization_info)
    """
    try:
        if len(audio_data) == 0:
            return audio_data, {"error": "Empty audio data"}
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms == 0:
            return audio_data, {"error": "Audio contains only silence"}
        
        # Convert RMS to dB
        current_level_db = 20 * np.log10(rms)
        
        # Calculate required gain
        required_gain_db = target_level - current_level_db
        
        # Limit gain to prevent clipping
        applied_gain_db = min(required_gain_db, max_gain)
        
        # Convert gain to linear scale
        gain_linear = 10 ** (applied_gain_db / 20)
        
        # Apply gain
        normalized_audio = audio_data * gain_linear
        
        # Prevent clipping with headroom
        max_amplitude = np.max(np.abs(normalized_audio))
        headroom_threshold = 0.95  # Leave 5% headroom to prevent clipping artifacts
        
        if max_amplitude > headroom_threshold:
            clip_gain = headroom_threshold / max_amplitude
            normalized_audio *= clip_gain
            applied_gain_db += 20 * np.log10(clip_gain)
            
        # Additional safety: hard limit to prevent any values exceeding [-1, 1]
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        normalization_info = {
            "success": True,
            "original_rms_db": current_level_db,
            "target_level_db": target_level,
            "required_gain_db": required_gain_db,
            "applied_gain_db": applied_gain_db,
            "max_gain_limit": max_gain,
            "final_rms_db": target_level if max_amplitude <= 1.0 else current_level_db + applied_gain_db,
            "clipping_occurred": max_amplitude > 1.0
        }
        
        return normalized_audio, normalization_info
        
    except Exception as e:
        return audio_data, {"error": f"Failed to normalize audio: {str(e)}"}


def preprocess_voice_sample(
    audio_data: np.ndarray,
    sample_rate: int,
    trim_silence: bool = True,
    normalize_level: bool = True,
    target_level: float = -18.0,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.1,
    fade_duration: float = 0.05
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for voice cloning samples.
    
    This function applies the recommended preprocessing steps to improve
    voice cloning quality:
    1. Trim silence from beginning and end
    2. Normalize audio level
    3. Apply fade in/out
    
    Args:
        audio_data (np.ndarray): Input audio data
        sample_rate (int): Sample rate of the audio
        trim_silence (bool): Whether to trim silence
        normalize_level (bool): Whether to normalize audio level
        target_level (float): Target level in dB for normalization
        silence_threshold (float): Threshold for silence detection
        min_silence_duration (float): Minimum silence duration to trim
        fade_duration (float): Fade duration for smooth transitions
        
    Returns:
        tuple: (processed_audio, processing_info)
    """
    processing_info = {
        "steps_applied": [],
        "original_length": len(audio_data),
        "original_duration": len(audio_data) / sample_rate if sample_rate > 0 else 0
    }
    
    processed_audio = audio_data.copy()
    
    try:
        # Step 1: Trim silence
        if trim_silence:
            processed_audio, trim_info = trim_audio_silence(
                processed_audio, 
                sample_rate,
                silence_threshold=silence_threshold,
                min_silence_duration=min_silence_duration,
                fade_duration=fade_duration
            )
            processing_info["silence_trimming"] = trim_info
            if trim_info.get("success"):
                processing_info["steps_applied"].append("silence_trimming")
        
        # Step 2: Normalize audio level
        if normalize_level:
            processed_audio, norm_info = normalize_audio_level(
                processed_audio,
                target_level=target_level
            )
            processing_info["normalization"] = norm_info
            if norm_info.get("success"):
                processing_info["steps_applied"].append("normalization")
        
        # Final statistics
        processing_info.update({
            "final_length": len(processed_audio),
            "final_duration": len(processed_audio) / sample_rate if sample_rate > 0 else 0,
            "processing_success": len(processing_info["steps_applied"]) > 0
        })
        
        return processed_audio, processing_info
        
    except Exception as e:
        processing_info["error"] = f"Preprocessing failed: {str(e)}"
        return audio_data, processing_info


def analyze_audio_quality(
    audio_data: np.ndarray,
    sample_rate: int
) -> dict:
    """
    Analyze audio quality metrics for voice cloning suitability.
    
    Args:
        audio_data (np.ndarray): Audio data to analyze
        sample_rate (int): Sample rate of the audio
        
    Returns:
        dict: Quality analysis results
    """
    try:
        if len(audio_data) == 0:
            return {"error": "Empty audio data"}
        
        # Basic statistics
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data ** 2))
        peak = np.max(np.abs(audio_data))
        
        # Dynamic range
        dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))
        
        # Silence detection
        silence_regions = detect_silence_regions(audio_data, sample_rate)
        total_silence = sum(end - start for start, end in silence_regions)
        silence_percentage = (total_silence / duration) * 100 if duration > 0 else 0
        
        # Clipping detection
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(audio_data)) * 100
        
        # Quality assessment
        quality_score = 100
        quality_issues = []
        
        if duration < 5.0:
            quality_score -= 20
            quality_issues.append("Audio too short (< 5 seconds)")
        elif duration > 30.0:
            quality_score -= 10
            quality_issues.append("Audio quite long (> 30 seconds)")
        
        if silence_percentage > 30:
            quality_score -= 15
            quality_issues.append(f"High silence content ({silence_percentage:.1f}%)")
        
        if clipping_percentage > 1:
            quality_score -= 25
            quality_issues.append(f"Audio clipping detected ({clipping_percentage:.1f}%)")
        
        if rms < 0.01:
            quality_score -= 20
            quality_issues.append("Audio level too low")
        elif rms > 0.5:
            quality_score -= 15
            quality_issues.append("Audio level too high")
        
        quality_rating = "Excellent" if quality_score >= 90 else \
                        "Good" if quality_score >= 75 else \
                        "Fair" if quality_score >= 60 else \
                        "Poor"
        
        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "rms_level": rms,
            "rms_level_db": 20 * np.log10(rms + 1e-10),
            "peak_level": peak,
            "dynamic_range_db": dynamic_range_db,
            "silence_regions": len(silence_regions),
            "total_silence_seconds": total_silence,
            "silence_percentage": silence_percentage,
            "clipping_percentage": clipping_percentage,
            "quality_score": max(0, quality_score),
            "quality_rating": quality_rating,
            "quality_issues": quality_issues,
            "recommended_for_cloning": quality_score >= 60 and clipping_percentage < 5
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# Convenience function with correct name (fixing the local function name conflict)
def trim_audio_silence(audio_data: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, dict]:
    """Convenience wrapper for trim_silence function."""
    return trim_silence(audio_data, sample_rate, **kwargs) 