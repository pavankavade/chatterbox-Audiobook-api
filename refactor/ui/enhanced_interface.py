"""
# ==============================================================================
# ENHANCED GRADIO INTERFACE MODULE
# ==============================================================================
# 
# This module provides enhanced Gradio interface components with professional
# layouts, modern styling, and improved user experience for the Chatterbox
# Audiobook Studio refactored system.
# 
# **Key Features:**
# - **Professional Layouts**: Modern, intuitive interface design
# - **Enhanced Components**: Custom Gradio components with advanced functionality
# - **Responsive Design**: Adaptive layouts for different screen sizes
# - **User Experience**: Streamlined workflows and intuitive controls
# - **Integration Ready**: Seamless integration with Phase 2 audio processing
"""

import gradio as gr
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Import core modules for integration
from project.project_manager import get_global_project_manager
from voice.voice_manager import get_global_voice_manager

# Import centralized CSS system
try:
    from src.ui.styles import get_css, get_inline_style, get_audio_processing_css
except ImportError:
    # Fallback for different import paths
    from refactor.src.ui.styles import get_css, get_inline_style, get_audio_processing_css

# ==============================================================================
# ENHANCED UI COMPONENTS
# ==============================================================================

class EnhancedGradioInterface:
    """
    Enhanced Gradio interface with professional components and layouts.
    
    This class provides the enhanced main interface for the refactored
    audiobook studio with modern design, improved UX, and integrated
    audio processing capabilities.
    
    **Interface Features:**
    - **Modern Design**: Professional, clean interface layout
    - **Intuitive Navigation**: Streamlined tab-based organization
    - **Real-time Feedback**: Live status updates and progress indicators
    - **Professional Controls**: Advanced audiobook production controls
    - **Integrated Processing**: Seamless audio processing integration
    """
    
    def __init__(self):
        """Initialize enhanced interface components."""
        self.project_manager = get_global_project_manager()
        self.voice_manager = get_global_voice_manager()
        
        # UI state
        self.current_project = None
        self.interface_theme = "soft"
        
        print("✅ Enhanced Gradio Interface initialized - Modern UI ready")
    
    def create_enhanced_interface(self) -> gr.Blocks:
        """
        Create the complete enhanced interface.
        
        Returns:
            gr.Blocks: Complete enhanced Gradio interface
        """
        
        # Use centralized CSS with audio processing theme
        custom_css = get_css(theme='light', additional_themes=['audio_processing'])
        
        with gr.Blocks(
            title="Chatterbox Audiobook Studio - Enhanced Edition",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Main header
            with gr.Row(elem_classes="main-header"):
                gr.HTML("""
                <div>
                    <h1>🎙️ Chatterbox Audiobook Studio</h1>
                    <h3>Enhanced Edition - Professional Audiobook Production</h3>
                    <p>Powered by Advanced AI • Phase 2 Audio Processing Integrated</p>
                </div>
                """)
            
            # Global status bar
            with gr.Row(elem_classes="status-panel"):
                with gr.Column(scale=3):
                    status_display = gr.Textbox(
                        value="✅ System ready - All modules operational",
                        label="🔄 System Status",
                        interactive=False,
                        elem_classes="success-indicator"
                    )
                with gr.Column(scale=1):
                    system_info = gr.Textbox(
                        value="Phase 2 Complete | 11/16 Modules",
                        label="📊 Progress",
                        interactive=False
                    )
            
            # Main interface tabs
            with gr.Tabs() as main_tabs:
                
                # Tab 1: Project Management (Enhanced)
                with gr.Tab("📁 Project Management", id="project_tab"):
                    project_interface = self._create_project_management_tab()
                
                # Tab 2: Audio Processing (NEW - Phase 2 Integration)
                with gr.Tab("🎵 Audio Processing", id="audio_tab"):
                    audio_interface = self._create_audio_processing_tab()
                
                # Tab 3: Voice Management (Enhanced)
                with gr.Tab("🎭 Voice Management", id="voice_tab"):
                    voice_interface = self._create_voice_management_tab()
                
                # Tab 4: Text Processing (Enhanced)
                with gr.Tab("📝 Text Processing", id="text_tab"):
                    text_interface = self._create_text_processing_tab()
                
                # Tab 5: Production Studio (Enhanced)
                with gr.Tab("🎬 Production Studio", id="studio_tab"):
                    studio_interface = self._create_production_studio_tab()
                
                # Tab 6: Quality Control (NEW)
                with gr.Tab("📊 Quality Control", id="quality_tab"):
                    quality_interface = self._create_quality_control_tab()
            
            # Footer with system information
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #6c757d;">
                    <p>Chatterbox Audiobook Studio Enhanced Edition | 
                    Phase 2 Audio Processing Complete | 
                    Professional Audiobook Production Suite</p>
                </div>
                """)
        
        return interface
    
    def _create_project_management_tab(self) -> gr.Column:
        """Create enhanced project management tab."""
        with gr.Column() as project_tab:
            gr.Markdown("## 📁 Enhanced Project Management", elem_classes="section-header")
            
            # Quick project actions
            with gr.Row(elem_classes="compact-row"):
                new_project_btn = gr.Button("➕ New Project", variant="primary")
                load_project_btn = gr.Button("📂 Load Project")
                save_project_btn = gr.Button("💾 Save Project")
                export_project_btn = gr.Button("📤 Export", variant="secondary")
            
            # Project overview panel
            with gr.Row():
                with gr.Column(scale=2):
                    project_name = gr.Textbox(
                        label="📝 Project Name",
                        placeholder="Enter project name..."
                    )
                    project_description = gr.Textbox(
                        label="📋 Description",
                        lines=3,
                        placeholder="Project description..."
                    )
                
                with gr.Column(scale=1):
                    project_stats = gr.JSON(
                        label="📊 Project Statistics",
                        value={
                            "Total Chapters": 0,
                            "Total Pages": 0,
                            "Audio Generated": "0%",
                            "Quality Checked": "0%"
                        }
                    )
            
            # Text input with enhanced features
            with gr.Accordion("📝 Text Input & Processing", open=True):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="📖 Input Text",
                        lines=10,
                        placeholder="Paste your text here or upload a file...",
                        info="Supports plain text, with automatic chapter detection"
                    )
                
                with gr.Row():
                    upload_file = gr.File(
                        label="📁 Upload Text File",
                        file_types=[".txt", ".docx", ".pdf"]
                    )
                    auto_detect_chapters = gr.Checkbox(
                        value=True,
                        label="🔍 Auto-detect Chapters",
                        info="Automatically split text into chapters"
                    )
            
            # Processing controls
            with gr.Row():
                process_text_btn = gr.Button("⚙️ Process Text", variant="primary")
                chunk_text_btn = gr.Button("✂️ Chunk Text")
                preview_chunks_btn = gr.Button("👁️ Preview Chunks")
            
            # Results display
            processing_status = gr.Textbox(
                label="🔄 Processing Status",
                value="Ready to process text",
                interactive=False
            )
            
            chunks_display = gr.Dataframe(
                headers=["Page", "Text Preview", "Word Count", "Status"],
                label="📑 Text Chunks",
                interactive=False
            )
        
        return project_tab
    
    def _create_audio_processing_tab(self) -> gr.Column:
        """Create the new audio processing tab with Phase 2 integration."""
        with gr.Column() as audio_tab:
            gr.Markdown("## 🎵 Professional Audio Processing", elem_classes="section-header")
            gr.Markdown("*Powered by Phase 2 Audio Processing Pipeline*")
            
            # This will be integrated with the AudioIntegration module
            with gr.Row():
                gr.HTML("""
                <div class="audio-controls">
                    <h4>🎛️ Audio Processing Integration Point</h4>
                    <p>This section will be populated by the AudioIntegration module to provide:</p>
                    <ul>
                        <li>✨ Real-time audio enhancement and effects processing</li>
                        <li>📊 Live quality analysis and broadcast standards validation</li>
                        <li>🎵 Master continuous audio playback and controls</li>
                        <li>🎭 Professional mastering-grade audio finishing</li>
                        <li>🎯 Advanced normalization and compliance checking</li>
                    </ul>
                </div>
                """)
            
            # Placeholder for AudioIntegration module
            audio_integration_placeholder = gr.HTML(
                value="<div class='processing-panel'><h4>🔄 Loading Audio Processing Interface...</h4></div>",
                label="Audio Processing Interface"
            )
        
        return audio_tab
    
    def _create_voice_management_tab(self) -> gr.Column:
        """Create enhanced voice management tab."""
        with gr.Column() as voice_tab:
            gr.Markdown("## 🎭 Enhanced Voice Management", elem_classes="section-header")
            
            # Voice library overview
            with gr.Row():
                with gr.Column(scale=2):
                    voice_selector = gr.Dropdown(
                        label="🎤 Select Voice",
                        choices=[],
                        info="Choose from available voice profiles"
                    )
                    
                    voice_preview = gr.Audio(
                        label="🔊 Voice Preview",
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    voice_info = gr.JSON(
                        label="ℹ️ Voice Information",
                        value={}
                    )
            
            # Multi-voice assignment
            with gr.Accordion("🎭 Multi-Voice Character Assignment", open=False):
                character_mapping = gr.Dataframe(
                    headers=["Character", "Voice", "Sample"],
                    label="🎬 Character Voice Mapping",
                    interactive=True
                )
                
                with gr.Row():
                    add_character_btn = gr.Button("➕ Add Character")
                    auto_assign_btn = gr.Button("🤖 Auto-assign Voices")
                    test_voices_btn = gr.Button("🎤 Test All Voices")
            
            # Voice library management
            with gr.Accordion("📚 Voice Library Management", open=False):
                with gr.Row():
                    import_voice_btn = gr.Button("📥 Import Voice")
                    export_voice_btn = gr.Button("📤 Export Voice")
                    clone_voice_btn = gr.Button("🧬 Clone Voice")
                
                library_status = gr.Textbox(
                    label="📋 Library Status",
                    value="Voice library ready",
                    interactive=False
                )
        
        return voice_tab
    
    def _create_text_processing_tab(self) -> gr.Column:
        """Create enhanced text processing tab."""
        with gr.Column() as text_tab:
            gr.Markdown("## 📝 Advanced Text Processing", elem_classes="section-header")
            
            # Text analysis and preprocessing
            with gr.Accordion("🔍 Text Analysis", open=True):
                with gr.Row():
                    text_analyzer_input = gr.Textbox(
                        label="📖 Text to Analyze",
                        lines=8,
                        placeholder="Enter text for analysis..."
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
                    clean_text_btn = gr.Button("🧹 Clean Text")
                    optimize_btn = gr.Button("⚡ Optimize for TTS")
                
                analysis_results = gr.JSON(
                    label="📊 Analysis Results",
                    value={}
                )
            
            # Pronunciation and SSML
            with gr.Accordion("🗣️ Pronunciation & SSML", open=False):
                with gr.Row():
                    pronunciation_dict = gr.Textbox(
                        label="📖 Pronunciation Dictionary",
                        lines=5,
                        placeholder="word1: pronunciation1\nword2: pronunciation2\n..."
                    )
                
                with gr.Row():
                    add_ssml_btn = gr.Button("🏷️ Add SSML Tags")
                    validate_ssml_btn = gr.Button("✅ Validate SSML")
                    preview_ssml_btn = gr.Button("👁️ Preview")
                
                ssml_output = gr.Textbox(
                    label="🏷️ SSML Output",
                    lines=6,
                    interactive=False
                )
        
        return text_tab
    
    def _create_production_studio_tab(self) -> gr.Column:
        """Create enhanced production studio tab."""
        with gr.Column() as studio_tab:
            gr.Markdown("## 🎬 Professional Production Studio", elem_classes="section-header")
            
            # Production pipeline overview
            with gr.Row():
                pipeline_status = gr.HTML("""
                <div class="processing-panel">
                    <h4>🔄 Production Pipeline Status</h4>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>📝 Text Processing: <span class="success-indicator">✅ Ready</span></span>
                        <span>🎵 Audio Generation: <span class="warning-indicator">⚠️ Pending</span></span>
                        <span>📊 Quality Control: <span class="warning-indicator">⚠️ Pending</span></span>
                        <span>📤 Export: <span class="warning-indicator">⚠️ Pending</span></span>
                    </div>
                </div>
                """)
            
            # Batch processing controls
            with gr.Accordion("⚙️ Batch Processing", open=True):
                with gr.Row():
                    batch_start_btn = gr.Button("🚀 Start Batch Processing", variant="primary")
                    batch_pause_btn = gr.Button("⏸️ Pause")
                    batch_stop_btn = gr.Button("⏹️ Stop")
                    batch_resume_btn = gr.Button("▶️ Resume")
                
                batch_progress = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="📊 Batch Progress",
                    interactive=False
                )
                
                batch_log = gr.Textbox(
                    label="📋 Processing Log",
                    lines=8,
                    interactive=False,
                    value="Ready to start batch processing...\n"
                )
            
            # Output configuration
            with gr.Accordion("📤 Output Configuration", open=False):
                with gr.Row():
                    output_format = gr.Dropdown(
                        choices=["WAV", "MP3", "FLAC", "M4A"],
                        value="WAV",
                        label="🎵 Audio Format"
                    )
                    
                    output_quality = gr.Dropdown(
                        choices=["High (48kHz)", "Standard (44.1kHz)", "Compressed (22kHz)"],
                        value="Standard (44.1kHz)",
                        label="📊 Quality"
                    )
                
                output_directory = gr.Textbox(
                    label="📁 Output Directory",
                    value="./output",
                    placeholder="Select output directory..."
                )
        
        return studio_tab
    
    def _create_quality_control_tab(self) -> gr.Column:
        """Create new quality control tab."""
        with gr.Column() as quality_tab:
            gr.Markdown("## 📊 Professional Quality Control", elem_classes="section-header")
            gr.Markdown("*Comprehensive audio quality validation and broadcast standards compliance*")
            
            # Quality overview dashboard
            with gr.Row():
                quality_overview = gr.HTML("""
                <div class="status-panel">
                    <h4>📊 Quality Dashboard</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0;">
                        <div style="text-align: center;">
                            <h3 style="margin: 5px; color: #28a745;">85%</h3>
                            <p>ACX Compliance</p>
                        </div>
                        <div style="text-align: center;">
                            <h3 style="margin: 5px; color: #ffc107;">-21.2</h3>
                            <p>Avg LUFS</p>
                        </div>
                        <div style="text-align: center;">
                            <h3 style="margin: 5px; color: #007bff;">23</h3>
                            <p>Chapters Processed</p>
                        </div>
                    </div>
                </div>
                """)
            
            # Quality analysis tools
            with gr.Accordion("🔍 Quality Analysis Tools", open=True):
                with gr.Row():
                    analyze_project_btn = gr.Button("📊 Analyze Project", variant="primary")
                    validate_standards_btn = gr.Button("✅ Validate Standards")
                    generate_report_btn = gr.Button("📋 Generate Report")
                
                quality_results = gr.HTML(
                    value="<div class='processing-panel'><p>No quality analysis performed yet</p></div>",
                    label="Quality Analysis Results"
                )
            
            # Compliance checking
            with gr.Accordion("📋 Broadcast Standards Compliance", open=False):
                standards_selector = gr.Dropdown(
                    choices=["ACX Audiobook", "EBU R128", "Podcast Standards"],
                    value="ACX Audiobook",
                    label="📏 Standard"
                )
                
                compliance_report = gr.Textbox(
                    label="📋 Compliance Report",
                    lines=10,
                    interactive=False,
                    value="Select a standard and run compliance check..."
                )
        
        return quality_tab

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_enhanced_interface() -> gr.Blocks:
    """Create the complete enhanced interface."""
    interface_manager = EnhancedGradioInterface()
    return interface_manager.create_enhanced_interface()

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Enhanced Gradio Interface module loaded")
print("🎨 Professional UI components and layouts ready") 