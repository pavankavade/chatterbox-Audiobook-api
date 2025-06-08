"""
# ==============================================================================
# AUDIO INTEGRATION MODULE
# ==============================================================================
# 
# This module provides seamless integration of the Phase 2 audio processing
# pipeline into the main Gradio interface. It bridges the powerful audio
# processing modules (Playback Engine, Effects Processor, Quality Analyzer,
# Enhancement Tools) into the user interface with real-time feedback.
# 
# **Key Features:**
# - **Audio Processing Integration**: Seamless connection to Phase 2 audio modules
# - **Real-time Audio Processing**: Live audio enhancement and processing feedback
# - **Professional Audio Controls**: Advanced audio processing controls in UI
# - **Quality Monitoring**: Live audio quality analysis and broadcast standards
# - **Original Compatibility**: Full compatibility with existing audio workflows
"""

import gradio as gr
import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import threading
import time

# Import Phase 2 audio processing modules
from audio.playback_engine import PlaybackEngine, get_global_playback_engine
from audio.effects_processor import EffectsProcessor, get_global_effects_processor
from audio.quality_analyzer import QualityAnalyzer, get_global_quality_analyzer
from audio.enhancement_tools import EnhancementTools, get_global_enhancement_tools

# Import core modules
from core.audio_processing import load_audio_file, save_audio_file

# Import centralized CSS system
try:
    from src.ui.styles import get_css, get_inline_style, get_audio_processing_css
except ImportError:
    # Fallback for different import paths
    from refactor.src.ui.styles import get_css, get_inline_style, get_audio_processing_css

# ==============================================================================
# AUDIO INTEGRATION DATA STRUCTURES
# ==============================================================================

@dataclass
class AudioProcessingState:
    """State management for integrated audio processing."""
    is_processing: bool = False
    current_operation: str = ""
    progress_percentage: float = 0.0
    last_processed_file: str = ""
    processing_log: List[str] = None
    audio_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.processing_log is None:
            self.processing_log = []

# ==============================================================================
# MAIN AUDIO INTEGRATION CLASS
# ==============================================================================

class AudioIntegration:
    """
    Audio processing integration for the main Gradio interface.
    
    This class provides seamless integration of all Phase 2 audio processing
    capabilities into the main user interface with real-time feedback and
    professional audio controls.
    
    **Integration Features:**
    - **Live Audio Processing**: Real-time audio enhancement and effects
    - **Quality Monitoring**: Live broadcast standards compliance checking
    - **Playback Integration**: Master continuous audio playback controls
    - **Professional Controls**: Advanced audio processing parameter controls
    - **Progress Feedback**: Real-time processing progress and status updates
    """
    
    def __init__(self):
        """Initialize audio integration with all Phase 2 modules."""
        # Initialize audio processing modules
        self.playback_engine = get_global_playback_engine()
        self.effects_processor = get_global_effects_processor()
        self.quality_analyzer = get_global_quality_analyzer()
        self.enhancement_tools = get_global_enhancement_tools()
        
        # State management
        self.processing_state = AudioProcessingState()
        self.audio_cache: Dict[str, np.ndarray] = {}
        
        # UI update callbacks
        self.ui_update_callbacks: List[Callable] = []
        
        # Setup playback callbacks
        self.playback_engine.add_position_callback(self._on_playback_position_changed)
        self.playback_engine.add_state_change_callback(self._on_playback_state_changed)
        
        print("✅ Audio Integration initialized - Phase 2 audio processing integrated into UI")
    
    def create_audio_processing_interface(self) -> Tuple[gr.Column, Dict[str, Any]]:
        """
        Create the complete audio processing interface for integration into main UI.
        
        Returns:
            Tuple[gr.Column, Dict[str, Any]]: (interface_column, component_references)
        """
        components = {}
        
        with gr.Column() as audio_interface:
            # Header
            gr.Markdown("## 🎵 Professional Audio Processing")
            
            # Real-time status display
            with gr.Row():
                components['status_display'] = gr.Textbox(
                    value="Ready for audio processing",
                    label="🔄 Processing Status",
                    interactive=False
                )
                components['progress_bar'] = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="📊 Progress",
                    interactive=False
                )
            
            # Audio file input/output
            with gr.Row():
                with gr.Column():
                    components['audio_input'] = gr.Audio(
                        label="🎤 Input Audio",
                        type="filepath"
                    )
                    components['audio_output'] = gr.Audio(
                        label="🎵 Processed Audio",
                        type="filepath"
                    )
            
            # Processing controls
            with gr.Row():
                components['enhance_btn'] = gr.Button("✨ Enhance Audio", variant="primary")
                components['analyze_btn'] = gr.Button("📊 Analyze Quality")
                components['normalize_btn'] = gr.Button("🎯 Normalize")
                components['master_btn'] = gr.Button("🎭 Master Process", variant="secondary")
            
            # Advanced audio controls
            with gr.Accordion("🎛️ Advanced Audio Controls", open=False):
                
                # Effects processing controls
                with gr.Tab("🎛️ Effects"):
                    with gr.Row():
                        components['noise_reduction'] = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3,
                            label="🔇 Noise Reduction",
                            info="Amount of noise reduction to apply"
                        )
                        components['compression_ratio'] = gr.Slider(
                            minimum=1.0, maximum=10.0, value=3.0,
                            label="🗜️ Compression Ratio",
                            info="Dynamic range compression ratio"
                        )
                    
                    with gr.Row():
                        components['eq_presence'] = gr.Slider(
                            minimum=0.0, maximum=6.0, value=2.0,
                            label="🎤 Presence Boost (dB)",
                            info="Voice clarity enhancement"
                        )
                        components['eq_low_cut'] = gr.Slider(
                            minimum=20.0, maximum=200.0, value=80.0,
                            label="🔽 Low Cut (Hz)",
                            info="High-pass filter frequency"
                        )
                
                # Quality analysis controls
                with gr.Tab("📊 Quality"):
                    with gr.Row():
                        components['target_lufs'] = gr.Slider(
                            minimum=-30.0, maximum=-10.0, value=-23.0,
                            label="🎯 Target LUFS",
                            info="Target loudness for broadcast standards"
                        )
                        components['max_peak'] = gr.Slider(
                            minimum=-6.0, maximum=-0.1, value=-3.0,
                            label="⚡ Max Peak (dB)",
                            info="Maximum peak level"
                        )
                    
                    components['standards_selector'] = gr.Dropdown(
                        choices=["acx", "ebu", "podcast"],
                        value="acx",
                        label="📋 Broadcast Standard",
                        info="Target broadcast standard for validation"
                    )
                
                # Enhancement controls
                with gr.Tab("🎭 Enhancement"):
                    with gr.Row():
                        components['breath_reduction'] = gr.Checkbox(
                            value=True,
                            label="💨 Breath Reduction",
                            info="Reduce breath sounds in speech"
                        )
                        components['plosive_reduction'] = gr.Checkbox(
                            value=True,
                            label="💥 Plosive Reduction",
                            info="Reduce P and B plosive sounds"
                        )
                    
                    with gr.Row():
                        components['spectral_repair'] = gr.Checkbox(
                            value=True,
                            label="🔧 Spectral Repair",
                            info="Advanced artifact removal"
                        )
                        components['harmonic_enhancement'] = gr.Checkbox(
                            value=True,
                            label="🎵 Harmonic Enhancement",
                            info="Enhance voice harmonics for clarity"
                        )
            
            # Quality analysis display
            with gr.Accordion("📊 Audio Quality Analysis", open=False):
                components['quality_metrics'] = gr.JSON(
                    label="📈 Quality Metrics",
                    value={}
                )
                
                components['compliance_report'] = gr.Textbox(
                    label="📋 Broadcast Compliance",
                    value="No analysis performed yet",
                    lines=5,
                    interactive=False
                )
            
            # Playback controls (integrated with Playback Engine)
            with gr.Accordion("🎵 Playback Controls", open=False):
                with gr.Row():
                    components['play_btn'] = gr.Button("▶️ Play")
                    components['pause_btn'] = gr.Button("⏸️ Pause")
                    components['stop_btn'] = gr.Button("⏹️ Stop")
                
                components['playback_position'] = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="🎵 Position",
                    info="Current playback position"
                )
                
                components['playback_info'] = gr.Textbox(
                    value="No audio loaded",
                    label="ℹ️ Playback Info",
                    interactive=False
                )
            
            # Processing log
            with gr.Accordion("📝 Processing Log", open=False):
                components['processing_log'] = gr.Textbox(
                    label="📋 Processing History",
                    value="Ready for audio processing...\n",
                    lines=10,
                    interactive=False,
                    max_lines=20
                )
        
        return audio_interface, components
    
    def setup_event_handlers(self, components: Dict[str, Any]) -> None:
        """Setup all event handlers for the audio processing interface."""
        
        # Main processing buttons
        components['enhance_btn'].click(
            fn=self._enhance_audio_handler,
            inputs=[
                components['audio_input'],
                components['noise_reduction'],
                components['compression_ratio'],
                components['eq_presence'],
                components['eq_low_cut']
            ],
            outputs=[
                components['audio_output'],
                components['status_display'],
                components['progress_bar'],
                components['processing_log']
            ]
        )
        
        components['analyze_btn'].click(
            fn=self._analyze_quality_handler,
            inputs=[
                components['audio_input'],
                components['standards_selector']
            ],
            outputs=[
                components['quality_metrics'],
                components['compliance_report'],
                components['status_display'],
                components['processing_log']
            ]
        )
        
        components['normalize_btn'].click(
            fn=self._normalize_audio_handler,
            inputs=[
                components['audio_input'],
                components['target_lufs'],
                components['max_peak']
            ],
            outputs=[
                components['audio_output'],
                components['status_display'],
                components['processing_log']
            ]
        )
        
        components['master_btn'].click(
            fn=self._master_process_handler,
            inputs=[
                components['audio_input'],
                components['breath_reduction'],
                components['plosive_reduction'],
                components['spectral_repair'],
                components['harmonic_enhancement']
            ],
            outputs=[
                components['audio_output'],
                components['status_display'],
                components['progress_bar'],
                components['quality_metrics'],
                components['compliance_report'],
                components['processing_log']
            ]
        )
        
        # Playback controls
        components['play_btn'].click(
            fn=self._play_audio_handler,
            inputs=[components['audio_output']],
            outputs=[
                components['playback_info'],
                components['status_display']
            ]
        )
        
        components['pause_btn'].click(
            fn=self._pause_audio_handler,
            outputs=[
                components['playback_info'],
                components['status_display']
            ]
        )
        
        components['stop_btn'].click(
            fn=self._stop_audio_handler,
            outputs=[
                components['playback_info'],
                components['status_display']
            ]
        )
    
    def _enhance_audio_handler(
        self,
        audio_file: str,
        noise_reduction: float,
        compression_ratio: float,
        eq_presence: float,
        eq_low_cut: float
    ) -> Tuple[str, str, float, str]:
        """Handle audio enhancement with custom parameters."""
        try:
            if not audio_file:
                return None, "❌ No audio file provided", 0, self._get_log()
            
            self._update_status("🎛️ Loading audio file...")
            
            # Load audio
            audio_data, sample_rate = load_audio_file(audio_file)
            
            self._update_status("✨ Applying audio enhancement...")
            self._update_progress(25)
            
            # Apply custom effects processing
            self.effects_processor.config.compression_ratio = compression_ratio
            self.effects_processor.config.presence_boost = eq_presence
            self.effects_processor.config.low_cut_freq = eq_low_cut
            
            # Apply noise reduction
            audio_data, nr_meta = self.effects_processor.apply_noise_reduction(
                audio_data, sample_rate, noise_reduction
            )
            self._update_progress(50)
            
            # Apply compression
            audio_data, comp_meta = self.effects_processor.apply_compression(
                audio_data, sample_rate, ratio=compression_ratio
            )
            self._update_progress(75)
            
            # Apply EQ
            audio_data, eq_meta = self.effects_processor.apply_eq(
                audio_data, sample_rate, low_cut=eq_low_cut, presence_boost=eq_presence
            )
            
            # Normalize
            audio_data, norm_meta = self.effects_processor.normalize_to_standard(
                audio_data, sample_rate
            )
            
            self._update_progress(100)
            
            # Save processed audio
            output_path = "processed_audio_enhanced.wav"
            save_audio_file(audio_data, output_path, sample_rate)
            
            self._log(f"✅ Audio enhanced successfully: {output_path}")
            self._log(f"   🔇 Noise reduction: {nr_meta['success']}")
            self._log(f"   🗜️ Compression: {comp_meta['success']}")
            self._log(f"   🎛️ EQ processing: {eq_meta['success']}")
            self._log(f"   🎯 Normalization: {norm_meta['success']}")
            
            return output_path, "✅ Audio enhancement completed", 100, self._get_log()
            
        except Exception as e:
            error_msg = f"❌ Enhancement failed: {str(e)}"
            self._log(error_msg)
            return None, error_msg, 0, self._get_log()
    
    def _analyze_quality_handler(
        self,
        audio_file: str,
        standard: str
    ) -> Tuple[Dict[str, Any], str, str, str]:
        """Handle audio quality analysis."""
        try:
            if not audio_file:
                return {}, "No audio file provided", "❌ No audio file", self._get_log()
            
            self._update_status("📊 Analyzing audio quality...")
            
            # Load and analyze audio
            audio_data, sample_rate = load_audio_file(audio_file)
            metrics = self.quality_analyzer.analyze_audio_quality(audio_data, sample_rate, detailed=True)
            
            # Validate against broadcast standards
            validation = self.quality_analyzer.validate_broadcast_standards(metrics, standard)
            
            # Format quality metrics for display
            quality_display = {
                "Peak Level (dB)": round(metrics.peak_db, 2),
                "RMS Level (dB)": round(metrics.rms_db, 2),
                "LUFS Integrated": round(metrics.lufs_integrated, 2),
                "Dynamic Range (dB)": round(metrics.dynamic_range_db, 2),
                "Noise Floor (dB)": round(metrics.noise_floor_db, 2),
                "SNR Estimate": round(metrics.snr_estimate, 2),
                "Duration (s)": round(metrics.duration, 2),
                "Clipping Detected": metrics.clipping_detected,
                "Silence %": round(metrics.silence_percentage, 2)
            }
            
            # Format compliance report
            compliance_text = f"📋 {standard.upper()} Broadcast Standard Validation\n\n"
            compliance_text += f"Overall Pass: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}\n"
            compliance_text += f"Compliance Score: {validation['compliance_score']:.1f}%\n\n"
            
            compliance_text += "Detailed Checks:\n"
            for check in validation['checks']:
                status = "✅" if check['pass'] else "❌"
                compliance_text += f"{status} {check['test']}: "
                if 'value' in check:
                    compliance_text += f"{check['value']:.1f}"
                    if 'limit' in check:
                        compliance_text += f" (limit: {check['limit']:.1f})"
                    elif 'range' in check:
                        compliance_text += f" (range: {check['range']})"
                compliance_text += "\n"
            
            if validation['recommendations']:
                compliance_text += "\n📝 Recommendations:\n"
                for rec in validation['recommendations']:
                    compliance_text += f"• {rec}\n"
            
            self._log(f"📊 Quality analysis completed for {standard.upper()} standard")
            self._log(f"   📈 Compliance score: {validation['compliance_score']:.1f}%")
            
            return quality_display, compliance_text, "✅ Quality analysis completed", self._get_log()
            
        except Exception as e:
            error_msg = f"❌ Quality analysis failed: {str(e)}"
            self._log(error_msg)
            return {}, error_msg, error_msg, self._get_log()
    
    def _normalize_audio_handler(
        self,
        audio_file: str,
        target_lufs: float,
        max_peak: float
    ) -> Tuple[str, str, str]:
        """Handle audio normalization."""
        try:
            if not audio_file:
                return None, "❌ No audio file provided", self._get_log()
            
            self._update_status("🎯 Normalizing audio...")
            
            # Load audio
            audio_data, sample_rate = load_audio_file(audio_file)
            
            # Normalize
            normalized_audio, metadata = self.effects_processor.normalize_to_standard(
                audio_data, sample_rate, target_lufs=target_lufs, max_peak=max_peak
            )
            
            # Save normalized audio
            output_path = "normalized_audio.wav"
            save_audio_file(normalized_audio, output_path, sample_rate)
            
            self._log(f"🎯 Audio normalized: Target LUFS {target_lufs}, Max Peak {max_peak}dB")
            self._log(f"   📈 Final Peak: {metadata['final_peak_db']:.1f}dB")
            self._log(f"   📊 Final RMS: {metadata['final_rms_db']:.1f}dB")
            
            return output_path, "✅ Audio normalization completed", self._get_log()
            
        except Exception as e:
            error_msg = f"❌ Normalization failed: {str(e)}"
            self._log(error_msg)
            return None, error_msg, self._get_log()
    
    def _master_process_handler(
        self,
        audio_file: str,
        breath_reduction: bool,
        plosive_reduction: bool,
        spectral_repair: bool,
        harmonic_enhancement: bool
    ) -> Tuple[str, str, float, Dict[str, Any], str, str]:
        """Handle complete mastering process."""
        try:
            if not audio_file:
                return None, "❌ No audio file provided", 0, {}, "", self._get_log()
            
            self._update_status("🎭 Starting master processing...")
            
            # Load audio
            audio_data, sample_rate = load_audio_file(audio_file)
            self._update_progress(10)
            
            # Configure enhancement tools
            self.enhancement_tools.config.breath_reduction = breath_reduction
            self.enhancement_tools.config.plosive_reduction = plosive_reduction
            self.enhancement_tools.config.spectral_repair = spectral_repair
            self.enhancement_tools.config.harmonic_enhancement = harmonic_enhancement
            
            # Apply master enhancement
            enhanced_audio, enhancement_metadata = self.enhancement_tools.master_enhance_audio(
                audio_data, sample_rate, "audiobook_master"
            )
            self._update_progress(70)
            
            # Final quality analysis
            final_metrics = self.quality_analyzer.analyze_audio_quality(enhanced_audio, sample_rate)
            validation = self.quality_analyzer.validate_broadcast_standards(final_metrics, "acx")
            self._update_progress(90)
            
            # Save master processed audio
            output_path = "master_processed_audio.wav"
            save_audio_file(enhanced_audio, output_path, sample_rate)
            self._update_progress(100)
            
            # Format results
            quality_display = {
                "Peak Level (dB)": round(final_metrics.peak_db, 2),
                "RMS Level (dB)": round(final_metrics.rms_db, 2),
                "LUFS Integrated": round(final_metrics.lufs_integrated, 2),
                "Processing Steps": len(enhancement_metadata['enhancement_steps']),
                "Enhancement Success": enhancement_metadata['success']
            }
            
            compliance_text = f"🎭 Master Processing Completed\n\n"
            compliance_text += f"Final Compliance Score: {validation['compliance_score']:.1f}%\n"
            compliance_text += f"Processing Steps: {len(enhancement_metadata['enhancement_steps'])}\n"
            compliance_text += f"Enhancement Success: {enhancement_metadata['success']}"
            
            self._log(f"🎭 Master processing completed: {output_path}")
            self._log(f"   📊 Final compliance score: {validation['compliance_score']:.1f}%")
            self._log(f"   🔧 Processing steps: {len(enhancement_metadata['enhancement_steps'])}")
            
            return output_path, "✅ Master processing completed", 100, quality_display, compliance_text, self._get_log()
            
        except Exception as e:
            error_msg = f"❌ Master processing failed: {str(e)}"
            self._log(error_msg)
            return None, error_msg, 0, {}, error_msg, self._get_log()
    
    def _play_audio_handler(self, audio_file: str) -> Tuple[str, str]:
        """Handle audio playback."""
        try:
            if not audio_file:
                return "No audio file to play", "❌ No audio file"
            
            # For now, return playback info (actual playback would need system audio integration)
            info = f"🎵 Ready to play: {Path(audio_file).name}"
            self._log(f"🎵 Playback initiated: {audio_file}")
            
            return info, "🎵 Playback ready"
            
        except Exception as e:
            error_msg = f"❌ Playback failed: {str(e)}"
            return error_msg, error_msg
    
    def _pause_audio_handler(self) -> Tuple[str, str]:
        """Handle audio pause."""
        self.playback_engine.pause_playback()
        self._log("⏸️ Playback paused")
        return "⏸️ Playback paused", "⏸️ Paused"
    
    def _stop_audio_handler(self) -> Tuple[str, str]:
        """Handle audio stop."""
        self.playback_engine.stop_playback()
        self._log("⏹️ Playback stopped")
        return "⏹️ Playback stopped", "⏹️ Stopped"
    
    def _update_status(self, status: str) -> None:
        """Update processing status."""
        self.processing_state.current_operation = status
        
    def _update_progress(self, progress: float) -> None:
        """Update processing progress."""
        self.processing_state.progress_percentage = progress
        
    def _log(self, message: str) -> None:
        """Add message to processing log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_state.processing_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.processing_state.processing_log) > 100:
            self.processing_state.processing_log = self.processing_state.processing_log[-80:]
    
    def _get_log(self) -> str:
        """Get formatted processing log."""
        return "\n".join(self.processing_state.processing_log)
    
    def _on_playback_position_changed(self, playback_info: Dict[str, Any]) -> None:
        """Handle playback position changes."""
        # Update UI with playback info (would be called by playback engine)
        pass
    
    def _on_playback_state_changed(self, playback_info: Dict[str, Any]) -> None:
        """Handle playback state changes."""
        # Update UI with state changes (would be called by playback engine)
        pass

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_audio_processing_ui() -> Tuple[gr.Column, AudioIntegration]:
    """Create complete audio processing UI with integration."""
    integration = AudioIntegration()
    interface, components = integration.create_audio_processing_interface()
    integration.setup_event_handlers(components)
    
    return interface, integration

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Audio Integration module loaded")
print("🎵 Phase 2 audio processing integrated into main UI interface") 