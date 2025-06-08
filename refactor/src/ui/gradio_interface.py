"""
Chatterbox Audiobook Studio - Refactored Gradio Interface

This module provides the complete Gradio web interface for the audiobook studio,
integrating all refactored modules while maintaining the full feature set and 
professional UI of the original application.

Features:
- Text-to-Speech testing and voice selection
- Voice Library management and configuration  
- Single-voice and multi-voice audiobook creation
- Production Studio with advanced editing capabilities
- Listen & Edit mode with real-time chunk navigation
- Audio Enhancement with quality analysis and cleanup
"""

import gradio as gr
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import wave

# Import all refactored modules - use try/except for relative vs absolute imports
try:
    # Try relative imports first (when run as module)
    from ..config import config
    from ..models.tts_model import load_model, generate_for_gradio, CHATTERBOX_AVAILABLE as TTS_AVAILABLE, get_device_info
    from ..text_processing.chunking import (
        chunk_text_by_sentences,
        validate_audiobook_input,
        load_text_file
    )
    from ..text_processing.multi_voice import parse_multi_voice_text
    from ..projects.management import (
        create_project_directory,
        get_project_choices,
        get_existing_projects
    )
    from ..projects.metadata import (
        save_project_metadata,
        load_project_metadata
    )
    from ..projects.studio import (
        load_studio_project,
        render_interactive_chunk_editor,
        assemble_full_audio,
        MAX_CHUNKS_ON_PAGE,
        update_marked_chunks,
        regenerate_marked_chunks,
        regenerate_single_chunk
    )
    from ..voice_library.voice_management import (
        get_voice_choices,
        get_voice_choices_organized,
        load_voice_profile
    )
    from ..audiobook.generation import generate_single_voice_audiobook
    from .styles import get_css, get_inline_style
    from src.audio.preprocessing import get_audio_files
    from src.audio.trimming import trim_silence_from_file, create_test_report
except ImportError:
    # Fall back to absolute imports (when run from app.py)
    from src.config import config
    from src.models.tts_model import load_model, generate_for_gradio, CHATTERBOX_AVAILABLE as TTS_AVAILABLE, get_device_info
    from src.text_processing.chunking import (
        chunk_text_by_sentences,
        validate_audiobook_input,
        load_text_file
    )
    from src.text_processing.multi_voice import parse_multi_voice_text
    from src.projects.management import (
        create_project_directory,
        get_project_choices,
        get_existing_projects
    )
    from src.projects.metadata import (
        save_project_metadata,
        load_project_metadata
    )
    from src.projects.studio import (
        load_studio_project,
        render_interactive_chunk_editor,
        assemble_full_audio,
        MAX_CHUNKS_ON_PAGE,
        update_marked_chunks,
        regenerate_marked_chunks,
        regenerate_single_chunk
    )
    from src.voice_library.voice_management import (
        get_voice_choices,
        get_voice_choices_organized,
        load_voice_profile
    )
    from src.audiobook.generation import generate_single_voice_audiobook
    from src.ui.styles import get_css, get_inline_style

# ChatterboxTTS availability is handled by the models module
CHATTERBOX_AVAILABLE = TTS_AVAILABLE


# A helper function to create event handlers for the checkboxes.
# This needs to be at the module level to avoid scoping issues with Gradio.
def create_mark_chunk_handler(component_index: int):
    """Creates a closure to handle marking a chunk, capturing the component index."""
    def mark_chunk_handler(marked_chunks: List[int], is_checked: bool, page_num: int, chunks_per_page: int) -> List[int]:
        """Calculates the global chunk index and updates the marked list."""
        # Import here to resolve the NameError in Gradio's execution scope
        from src.projects.studio import update_marked_chunks
        global_chunk_index = (page_num - 1) * chunks_per_page + component_index
        return update_marked_chunks(marked_chunks, is_checked, global_chunk_index)
    return mark_chunk_handler


class ChatterboxGradioApp:
    """
    Main Gradio interface for the Chatterbox Audiobook Studio.
    
    This class encapsulates the entire web interface, providing a clean API
    for launching the application while keeping all UI logic organized.
    """
    
    def __init__(self):
        """Initialize the Gradio application with refactored modules."""
        self.demo = None
        self.model_state = None
        self.voice_library_path = config.get_voices_path()
        
        # Use centralized CSS system
        self.css = get_css(theme='light')
        
        self._create_interface()
    
    def _create_interface(self) -> None:
        """Create the main Gradio interface."""
        with gr.Blocks(css=self.css, title="Chatterbox TTS - Audiobook Edition (Refactored)") as self.demo:
            # Global state variables
            model_state = gr.State(None)
            voice_library_path_state = gr.State(self.voice_library_path)
            
            # Header
            gr.HTML("""
            <div class="voice-library-header">
                <h1>🎧 Chatterbox TTS - Audiobook Edition</h1>
                <p>Professional voice cloning for audiobook creation - Refactored Architecture</p>
            </div>
            """)
            
            with gr.Tabs():
                self._create_tts_tab(model_state, voice_library_path_state)
                self._create_voice_library_tab(voice_library_path_state)
                self._create_single_voice_tab(model_state, voice_library_path_state)
                self._create_multi_voice_tab(model_state, voice_library_path_state)
                self._create_production_studio_tab(model_state, voice_library_path_state)
    
    def _create_tts_tab(self, model_state: gr.State, voice_library_path_state: gr.State) -> None:
        """Create the Text-to-Speech testing tab."""
        with gr.TabItem("🎤 Text-to-Speech", id="tts"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(
                        value="Welcome to Chatterbox TTS Audiobook Edition. This tool will help you create amazing audiobooks with consistent character voices.",
                        label="Text to synthesize",
                        lines=3
                    )
                    
                    # Voice Selection Section
                    with gr.Group():
                        gr.HTML("<h4>🎭 Voice Selection</h4>")
                        with gr.Row():
                            tts_voice_selector = gr.Dropdown(
                                choices=self._get_voice_choices(),
                                label="Choose Voice",
                                value=None,
                                scale=4
                            )
                            tts_reload_voices_btn = gr.Button(
                                "🔄 Reload", 
                                size="sm", 
                                variant="secondary",
                                scale=1,
                                min_width=80
                            )
                        
                        # Voice status display
                        tts_voice_status = gr.HTML(
                            "<div class='voice-status'>📝 Manual input mode - upload your own audio file below</div>"
                        )
                    
                    # Audio input (conditionally visible)
                    ref_wav = gr.Audio(
                        sources=["upload", "microphone"], 
                        type="filepath", 
                        label="Reference Audio File (Manual Input)", 
                        value=None,
                        visible=True
                    )
                    
                    with gr.Row():
                        exaggeration = gr.Slider(
                            0.25, 2, step=.05, 
                            label="Exaggeration (Neutral = 0.5)", 
                            value=.5
                        )
                        cfg_weight = gr.Slider(
                            0.2, 1, step=.05, 
                            label="CFG/Pace", 
                            value=0.5
                        )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

                    with gr.Row():
                        run_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                        refresh_voices_btn = gr.Button("🔄 Refresh Voices", size="sm")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio")
                    
                    gr.HTML("""
                    <div class="instruction-box">
                        <h4>💡 TTS Tips:</h4>
                        <ul>
                            <li><strong>Voice Selection:</strong> Choose a saved voice for consistent character voices</li>
                            <li><strong>Reference Audio:</strong> 10-30 seconds of clear speech works best</li>
                            <li><strong>Exaggeration:</strong> 0.3-0.7 for most voices, higher for dramatic effect</li>
                            <li><strong>CFG/Pace:</strong> Lower values = slower, more deliberate speech</li>
                            <li><strong>Temperature:</strong> Higher values = more variation, lower = more consistent</li>
                        </ul>
                    </div>
                    """)
            
            # Event handlers for TTS tab
            run_btn.click(
                fn=self._generate_tts,
                inputs=[model_state, text, ref_wav, tts_voice_selector, exaggeration, cfg_weight, temp, seed_num, voice_library_path_state],
                outputs=[audio_output, model_state]
            )
            
            refresh_voices_btn.click(
                fn=self._refresh_voice_choices,
                inputs=[voice_library_path_state],
                outputs=[tts_voice_selector]
            )
            
            # TTS Reload voices button (prominent button next to dropdown)
            tts_reload_voices_btn.click(
                fn=self._refresh_voice_choices,
                inputs=[voice_library_path_state],
                outputs=[tts_voice_selector]
            )
            
            # TTS Voice selector change handler (load JSON settings)
            tts_voice_selector.change(
                fn=self._load_voice_settings,
                inputs=[voice_library_path_state, tts_voice_selector],
                outputs=[ref_wav, exaggeration, cfg_weight, temp, tts_voice_status]
            )
    
    def _create_voice_library_tab(self, voice_library_path_state: gr.State) -> None:
        """Create the Voice Library management tab."""
        with gr.TabItem("📚 Voice Library", id="voice_library"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h3>🗣️ Voice Profiles</h3>")
                    with gr.Row():
                        voice_profile_selector = gr.Dropdown(
                            choices=self._get_voice_choices(),
                            label="Select Voice Profile",
                            scale=4
                        )
                        reload_profiles_btn = gr.Button("🔄", size="sm", min_width=20, scale=1)

                    with gr.Group():
                        gr.Markdown("Create a new voice or edit an existing one.")
                        voice_profile_name = gr.Textbox(
                            label="Voice Name (e.g., 'Narrator_John')",
                            placeholder="A unique name for the voice profile",
                            interactive=True
                        )
                        voice_profile_display_name = gr.Textbox(
                            label="Display Name (e.g., 'John')",
                            placeholder="A shorter name for easy recognition",
                            interactive=True
                        )
                        voice_profile_description = gr.Textbox(
                            label="Description",
                            lines=2,
                            placeholder="Describe the voice's characteristics (e.g., 'Deep, resonant, with a slight British accent')",
                            interactive=True
                        )

                    with gr.Accordion("🎧 Audio Sample", open=True):
                        gr.Markdown("Provide a clean audio sample (10-30 seconds) of the voice.")
                        raw_voice_selector = gr.Dropdown(
                            label="Select Raw Audio File",
                            choices=self._get_raw_audio_files(self.voice_library_path),
                            interactive=True,
                            allow_custom_value=False
                        )
                        voice_profile_audio = gr.Audio(
                            label="Upload or Record Audio",
                            sources=["upload", "microphone"],
                            type="filepath"
                        )

                with gr.Column(scale=3):
                    gr.HTML("<h3>⚙️ Voice Settings & Processing</h3>")
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("#### TTS Generation Settings")
                                voice_profile_exaggeration = gr.Slider(
                                    label="Exaggeration",
                                    minimum=0.1,
                                    maximum=2.0,
                                    step=0.05,
                                    value=0.5,
                                    interactive=True
                                )
                                voice_profile_cfg_weight = gr.Slider(
                                    label="CFG/Pace",
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.05,
                                    value=0.5,
                                    interactive=True
                                )
                                voice_profile_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.05,
                                    maximum=5.0,
                                    step=0.05,
                                    value=0.8,
                                    interactive=True
                                )
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("#### Audio Processing Pipeline")
                                with gr.Accordion("⚙️ Advanced Audio Processing", open=False):
                                    gr.Markdown("Fine-tune how voice samples are processed for the library.")
                                    enable_normalization_checkbox = gr.Checkbox(
                                        label="Enable Volume Normalization",
                                        value=True,
                                        interactive=True
                                    )
                                    target_level_number = gr.Number(
                                        label="Target Volume (dB)",
                                        value=config.get_setting("default_volume_target"),
                                        interactive=True,
                                        step=0.5,
                                        info="Recommended: -18dB for audiobooks"
                                    )
                                    enable_silence_trimming_checkbox = gr.Checkbox(
                                        label="Enable Silence Trimming",
                                        value=True,
                                        interactive=True
                                    )
                                    silence_threshold_slider = gr.Slider(
                                        label="Silence Threshold (dBFS)",
                                        minimum=-60.0,
                                        maximum=-20.0,
                                        step=1,
                                        value=-40.0,
                                        interactive=True
                                    )
                                    min_silence_duration_slider = gr.Slider(
                                        label="Min Silence Duration (ms)",
                                        minimum=100,
                                        maximum=1000,
                                        step=25,
                                        value=300,
                                        interactive=True
                                    )

                                processing_quality_dropdown = gr.Dropdown(
                                    label="Processing Quality",
                                    choices=["Standard", "High", "Ultra"],
                                    value="High",
                                    interactive=True
                                )
                    
                    with gr.Row():
                        save_voice_profile_btn = gr.Button("💾 Save Voice Profile", variant="primary", size="lg")
                        delete_voice_profile_btn = gr.Button("🗑️ Delete Profile", variant="stop", size="lg")

                    with gr.Accordion("🧪 Test Voice Settings", open=False):
                        test_voice_output = gr.Audio(label="Test Output")
                        test_voice_profile_btn = gr.Button("Test Voice", variant="secondary")

            # Status Message
            voice_library_status = gr.HTML("<div id='voice_library_status' class='status-box-neutral'>Idle.</div>")

            # Event Handlers
            voice_profile_selector.change(
                fn=self._load_voice_into_form,
                inputs=[voice_library_path_state, voice_profile_selector],
                outputs=[
                    voice_profile_name,
                    voice_profile_display_name,
                    voice_profile_description,
                    voice_profile_audio,
                    voice_profile_exaggeration,
                    voice_profile_cfg_weight,
                    voice_profile_temperature,
                    enable_normalization_checkbox,
                    target_level_number,
                    voice_library_status
                ]
            )

            reload_profiles_btn.click(
                fn=self._refresh_voice_choices,
                inputs=[voice_library_path_state],
                outputs=[voice_profile_selector]
            )

            raw_voice_selector.change(
                fn=self._load_raw_voice,
                inputs=[raw_voice_selector, voice_library_path_state],
                outputs=[voice_profile_audio, voice_profile_name]
            )
            
            save_voice_profile_btn.click(
                fn=self._save_voice_profile,
                inputs=[
                    voice_library_path_state,
                    voice_profile_name,
                    voice_profile_display_name,
                    voice_profile_description,
                    voice_profile_audio,
                    voice_profile_exaggeration,
                    voice_profile_cfg_weight,
                    voice_profile_temperature,
                    enable_normalization_checkbox,
                    target_level_number,
                    enable_silence_trimming_checkbox,
                    silence_threshold_slider,
                    min_silence_duration_slider,
                    processing_quality_dropdown
                ],
                outputs=[voice_library_status]
            ).then(
                fn=self._refresh_voice_choices,
                inputs=[voice_library_path_state],
                outputs=[voice_profile_selector]
            )

            delete_voice_profile_btn.click(
                fn=self._delete_voice_profile,
                inputs=[voice_library_path_state, voice_profile_selector],
                outputs=[voice_library_status, voice_profile_selector]
            )
            
            test_voice_profile_btn.click(
                fn=self._test_voice_profile,
                inputs=[
                    voice_library_path_state,
                    voice_profile_name,
                    voice_profile_audio,
                    voice_profile_exaggeration,
                    voice_profile_cfg_weight,
                    voice_profile_temperature
                ],
                outputs=[test_voice_output, voice_library_status]
            )
    
    def _create_single_voice_tab(self, model_state: gr.State, voice_library_path_state: gr.State) -> None:
        """Create the Single Voice Audiobook tab."""
        with gr.TabItem("📖 Single Voice Audiobook", id="single"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>📚 Create Single Voice Audiobook</h3>")
                    
                    # Text input section
                    with gr.Group():
                        gr.HTML("<h4>📝 Text Input</h4>")
                        
                        with gr.Tabs():
                            with gr.TabItem("✍️ Direct Input"):
                                audiobook_text = gr.Textbox(
                                    label="Audiobook Text",
                                    placeholder="Enter or paste your audiobook text here...",
                                    lines=10
                                )
                            
                            with gr.TabItem("📄 File Upload"):
                                text_file = gr.File(
                                    label="Upload Text File",
                                    file_types=[".txt", ".md", ".rtf"]
                                )
                                load_file_btn = gr.Button("📥 Load File", size="sm")
                                file_status = gr.HTML("<div class='status-info'></div>")
                    
                    # Voice selection
                    with gr.Group():
                        gr.HTML("<h4>🎭 Voice Selection</h4>")
                        audiobook_voice = gr.Dropdown(
                            choices=self._get_voice_choices(),
                            label="Select Voice Profile"
                        )
                        
                        refresh_audiobook_voices = gr.Button("🔄 Refresh Voices", size="sm")
                    
                    # Project settings
                    with gr.Group():
                        gr.HTML("<h4>🎯 Project Settings</h4>")
                        project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="my_audiobook_project",
                            info="Name for your audiobook project (no spaces)"
                        )
                        
                        resume_project = gr.Checkbox(
                            label="Resume existing project",
                            value=False,
                            info="Continue from where you left off"
                        )
                        
                        autosave_interval = gr.Slider(
                            1, 50, step=1,
                            label="Auto-save interval (chunks)",
                            value=10,
                            info="Save progress every N chunks"
                        )
                
                with gr.Column():
                    # Generation controls
                    gr.HTML("<h3>🚀 Generation</h3>")
                    
                    # Status display
                    generation_status = gr.HTML(
                        "<div class='status-info'>Ready to generate audiobook</div>"
                    )
                    
                    # Action buttons
                    with gr.Row():
                        validate_btn = gr.Button("✅ Validate Input", size="lg")
                        generate_btn = gr.Button("🎬 Generate Audiobook", variant="primary", size="lg")
                    
                    # Progress and results
                    generation_progress = gr.HTML()
                    generation_results = gr.Audio(label="Generated Audio Preview")
                    
                    # Project info
                    with gr.Group():
                        gr.HTML("<h4>📊 Project Information</h4>")
                        project_info = gr.HTML()
            
            # Event handlers for single voice tab
            load_file_btn.click(
                fn=load_text_file,
                inputs=[text_file],
                outputs=[audiobook_text, file_status]
            )
            
            refresh_audiobook_voices.click(
                fn=self._refresh_voice_choices,
                inputs=[voice_library_path_state],
                outputs=[audiobook_voice]
            )
            
            validate_btn.click(
                fn=self._validate_single_voice_input,
                inputs=[audiobook_text, audiobook_voice, project_name],
                outputs=[generation_status]
            )
            
            generate_btn.click(
                fn=self._generate_single_voice_audiobook,
                inputs=[model_state, audiobook_text, audiobook_voice, project_name, resume_project, autosave_interval, voice_library_path_state],
                outputs=[generation_status, generation_progress, generation_results, project_info, model_state]
            )
    
    def _create_multi_voice_tab(self, model_state: gr.State, voice_library_path_state: gr.State) -> None:
        """Create the Multi-Voice Audiobook tab."""
        with gr.TabItem("🎭 Multi-Voice Audiobook", id="multi"):
            gr.HTML("<h3>🎭 Multi-Voice Audiobook Creation</h3>")
            gr.HTML("""
            <div class="instruction-box">
                <h4>💡 Multi-Voice Format:</h4>
                <p>Use [Character Name] at the beginning of lines to assign voices:</p>
                <pre>[Narrator] Once upon a time, in a land far away...
[Hero] I must save the kingdom!
[Villain] You'll never stop me!</pre>
</div>
""")
            
            with gr.Row():
                with gr.Column():
                    # Text input section
                    with gr.Group():
                        gr.HTML("<h4>📝 Text Input</h4>")
                    
                    with gr.Tabs():
                        with gr.TabItem("✍️ Direct Input"):
                            multi_voice_text = gr.Textbox(
                                label="Multi-Voice Text",
                                    placeholder="[Narrator] Enter your multi-voice text here...\n[Character1] Each character should be marked like this.\n[Character2] The system will detect all characters automatically.",
                                    lines=12,
                                    info="Use [Character Name] format to assign voices to different speakers"
                            )
                        
                        with gr.TabItem("📄 File Upload"):
                                multi_text_file = gr.File(
                                    label="Upload Multi-Voice Text File",
                                file_types=[".txt", ".md", ".rtf"]
                            )
                                load_multi_file_btn = gr.Button("📥 Load File", size="sm")
                                multi_file_status = gr.HTML("<div class='status-info'></div>")
                    
                    project_name_multi = gr.Textbox(
                        label="Project Name",
                        placeholder="my_multi_voice_audiobook",
                        info="Name for your multi-voice project"
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze Characters", size="lg")
                        generate_multi_btn = gr.Button("🎬 Generate Multi-Voice Audiobook", variant="primary", size="lg")
                
                with gr.Column():
                    # Character analysis and voice assignment
                    character_analysis = gr.HTML()
                    
                    # Voice assignment section - initially hidden
                    with gr.Group(visible=False) as voice_assignment_group:
                        gr.HTML("<h4>🎭 Assign Voices to Characters</h4>")
                        
                        # Dynamic voice assignment dropdowns - will be populated by analysis
                        voice_assignments = gr.JSON(value={}, visible=False)  # Hidden state for assignments
                        
                        # Character voice assignment interface - will be dynamically created
                        assignment_interface = gr.HTML()
                        
                        # Manual assignment dropdowns (up to 10 characters)
                        char1_name = gr.Textbox(visible=False, label="Character 1")
                        char1_voice = gr.Dropdown(choices=self._get_voice_choices(), visible=False, label="Voice for Character 1")
                        
                        char2_name = gr.Textbox(visible=False, label="Character 2") 
                        char2_voice = gr.Dropdown(choices=self._get_voice_choices(), visible=False, label="Voice for Character 2")
                        
                        char3_name = gr.Textbox(visible=False, label="Character 3")
                        char3_voice = gr.Dropdown(choices=self._get_voice_choices(), visible=False, label="Voice for Character 3")
                        
                        char4_name = gr.Textbox(visible=False, label="Character 4")
                        char4_voice = gr.Dropdown(choices=self._get_voice_choices(), visible=False, label="Voice for Character 4")
                        
                        char5_name = gr.Textbox(visible=False, label="Character 5")
                        char5_voice = gr.Dropdown(choices=self._get_voice_choices(), visible=False, label="Voice for Character 5")
                    
                    # Status and results
                    multi_generation_status = gr.HTML()
                    multi_generation_results = gr.Audio(label="Multi-Voice Preview")
            
            # Store all character components for easy access
            character_components = [
                (char1_name, char1_voice),
                (char2_name, char2_voice), 
                (char3_name, char3_voice),
                (char4_name, char4_voice),
                (char5_name, char5_voice)
            ]
            
            # Event handlers for multi-voice tab
            load_multi_file_btn.click(
                fn=load_text_file,
                inputs=[multi_text_file],
                outputs=[multi_voice_text, multi_file_status]
            )
            
            analyze_btn.click(
                fn=self._analyze_and_setup_voices,
                inputs=[multi_voice_text, voice_library_path_state],
                outputs=[character_analysis, voice_assignment_group, assignment_interface] + 
                        [comp for pair in character_components for comp in pair]
            )
            
            generate_multi_btn.click(
                fn=self._generate_multi_voice_audiobook_with_assignments,
                inputs=[model_state, multi_voice_text, project_name_multi, voice_library_path_state] +
                       [comp for pair in character_components for comp in pair],
                outputs=[multi_generation_status, multi_generation_results]
            )
    
    def _create_production_studio_tab(self, model_state: gr.State, voice_library_path_state: gr.State) -> None:
        """Create the Production Studio tab for advanced editing."""
        with gr.TabItem("🎬 Production Studio", id="studio"):
            gr.HTML("<h2>🎬 Production Studio</h2><p>Load a project to review, edit, and regenerate audio chunks.</p>")
            
            # State management for the studio
            studio_project_data = gr.State(None) # Will store { "name": str, "chunks": [{"audio": path, "text": str}] }
            studio_marked_chunks = gr.State([]) # Will store indices of marked chunks

            with gr.Row():
                # Left Column: Project Loading and Controls
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML("<h4>📁 1. Load Project</h4>")
                        with gr.Row():
                            studio_project_dropdown = gr.Dropdown(
                                choices=self._get_project_choices(),
                                label="Select Project",
                                scale=3
                            )
                            studio_refresh_projects_btn = gr.Button("🔄", size="sm", scale=1)
                        studio_load_project_btn = gr.Button("📂 Load Project", variant="primary")
                        studio_load_status = gr.HTML()

                    with gr.Group():
                        gr.HTML("<h4>▶️ 2. Full Playback</h4>")
                        with gr.Row():
                            studio_play_all_btn = gr.Button("🎵 Play All")
                            studio_mark_chunk_btn = gr.Button("🚩 Mark Chunk", interactive=False)
                        studio_full_audio_player = gr.Audio(label="Full Project Preview", interactive=False)
                        gr.HTML("<p><i>Real-time 'Mark' button coming in the next update. For now, use the checkboxes below to mark chunks for regeneration.</i></p>")
                    
                    with gr.Group():
                        gr.HTML("<h4>💾 3. Finalize & Regenerate</h4>")
                        studio_regenerate_btn = gr.Button("🔥 Regenerate Selected Chunks", variant="primary")
                        studio_download_btn = gr.Button("💾 Download Full Audiobook", variant="secondary")

                # Right Column: Chunk Editor
                with gr.Column(scale=2):
                    gr.HTML("<h4>✏️ 4. Chunk Editor</h4>")
                    with gr.Row():
                        studio_chunks_per_page = gr.Dropdown([10, 25], value=10, label="Chunks per Page", interactive=True)
                        studio_page_number = gr.Number(label="Page", value=1, interactive=True, minimum=1)
                    
                    # Create a fixed number of placeholder components for the chunk editor
                    MAX_CHUNKS_ON_PAGE = 25
                    chunk_editor_components = []
                    with gr.Column() as studio_chunk_editor_display:
                        for i in range(MAX_CHUNKS_ON_PAGE):
                            with gr.Group(visible=False) as chunk_group:
                                chunk_audio = gr.Audio(label=f"Chunk Audio {i+1}", interactive=True)
                                chunk_text = gr.Textbox(label="Chunk Text", lines=3, interactive=True)
                                chunk_marked = gr.Checkbox(label="Mark for Regeneration", interactive=True)
                                chunk_regen_btn = gr.Button(f"🔄 Regen #{i+1}", size="sm", variant="secondary", visible=False)
                                chunk_editor_components.extend([chunk_group, chunk_audio, chunk_text, chunk_marked, chunk_regen_btn])

            # Event handlers for Production Studio
            # Define a function that triggers re-rendering the editor
            def handle_editor_render(project_data, marked_chunks, chunks_per_page, page_num):
                return render_interactive_chunk_editor(project_data, marked_chunks, chunks_per_page, page_num)

            # Trigger rendering when project is loaded, or when pagination changes
            render_inputs = [studio_project_data, studio_marked_chunks, studio_chunks_per_page, studio_page_number]
            render_outputs = chunk_editor_components
            
            studio_load_project_btn.click(
                fn=load_studio_project,
                inputs=[studio_project_dropdown],
                outputs=[studio_project_data, studio_load_status]
            ).then(
                fn=handle_editor_render,
                inputs=render_inputs,
                outputs=render_outputs
            )
            
            studio_page_number.change(handle_editor_render, render_inputs, render_outputs)
            studio_chunks_per_page.change(handle_editor_render, render_inputs, render_outputs)

            # Handle checkbox marking logic  
            for i in range(MAX_CHUNKS_ON_PAGE):
                chunk_marked = chunk_editor_components[i * 5 + 3]  # Checkbox is now at index 3 (5 components per chunk)
                
                chunk_marked.change(
                    fn=create_mark_chunk_handler(i),
                    inputs=[studio_marked_chunks, chunk_marked, studio_page_number, studio_chunks_per_page],
                    outputs=[studio_marked_chunks]
                )

            # Handle individual regenerate button logic
            for i in range(MAX_CHUNKS_ON_PAGE):
                chunk_regen_btn = chunk_editor_components[i * 5 + 4]  # Regen button is at index 4
                chunk_text = chunk_editor_components[i * 5 + 2]       # Text box is at index 2
                
                def create_regen_handler(component_index):
                    def regen_chunk_handler(project_data, text_content, page_num, chunks_per_page, tts_engine):
                        global_chunk_index = (page_num - 1) * chunks_per_page + component_index
                        return regenerate_single_chunk(project_data, global_chunk_index, text_content, tts_engine)
                    return regen_chunk_handler

                chunk_regen_btn.click(
                    fn=create_regen_handler(i),
                    inputs=[studio_project_data, chunk_text, studio_page_number, studio_chunks_per_page, model_state],
                    outputs=[studio_load_status]
                ).then(
                    fn=handle_editor_render,
                    inputs=render_inputs,
                    outputs=render_outputs
                )

            studio_refresh_projects_btn.click(
                fn=self._refresh_project_choices,
                inputs=[],
                outputs=[studio_project_dropdown]
            )

            studio_play_all_btn.click(
                fn=assemble_full_audio,
                inputs=[studio_project_data],
                outputs=[studio_full_audio_player]
            )

            studio_regenerate_btn.click(
                fn=regenerate_marked_chunks,
                inputs=[studio_project_data, studio_marked_chunks, model_state],
                outputs=[studio_load_status] # Display status message
            ).then(
                fn=handle_editor_render,
                inputs=render_inputs,
                outputs=render_outputs
            )

            studio_download_btn.click(
                fn=assemble_full_audio,
                inputs=[studio_project_data],
                outputs=[studio_full_audio_player]
            )
    
    # Helper methods for the interface
    
    def _get_voice_choices(self) -> List[str]:
        """Get list of available voice profiles with organized layout."""
        try:
            voices = get_voice_choices_organized(self.voice_library_path)
            print(f"📋 Voice choices: Found {len(voices)} voices in {self.voice_library_path}")
            return voices
        except Exception as e:
            print(f"❌ Error getting voice choices: {e}")
            return []
    
    def _get_project_choices(self) -> List[str]:
        """Get list of existing projects."""
        try:
            projects = get_existing_projects()
            return [project['name'] for project in projects]
        except Exception as e:
            print(f"Error getting project choices: {e}")
            return []
    
    def _refresh_voice_choices(self, voice_library_path: str) -> gr.Dropdown:
        """Refresh the voice choices dropdown."""
        self.voice_library_path = voice_library_path
        try:
            choices = get_voice_choices_organized(voice_library_path)
            print(f"🔄 Refreshed voices: Found {len(choices)} voices")
            return gr.Dropdown(choices=choices)
        except Exception as e:
            print(f"❌ Error refreshing voice choices: {e}")
            return gr.Dropdown(choices=[])
    
    def _refresh_project_choices(self) -> gr.Dropdown:
        """Refresh the project choices dropdown."""
        return gr.Dropdown(choices=self._get_project_choices())
    
    def _load_voice_settings(self, voice_library_path: str, voice_name: str) -> Tuple[str, float, float, float, str]:
        """Load voice settings from JSON when a voice is selected."""
        try:
            if not voice_name or "─────── Voice Pool ───────" in voice_name:
                # Manual input mode
                return (
                    gr.Audio(visible=True),  # Show audio upload
                    0.5,  # Default exaggeration
                    0.5,  # Default cfg_weight 
                    0.8,  # Default temperature
                    "<div class='voice-status'>📝 Manual input mode - upload your own audio file below</div>"
                )
            
            # Load voice profile
            voice_profile = load_voice_profile(voice_library_path, voice_name)
            if voice_profile:
                # Extract settings from JSON
                exaggeration = voice_profile.get('exaggeration', 0.5)
                cfg_weight = voice_profile.get('cfg_weight', 0.5) 
                temperature = voice_profile.get('temperature', 0.8)
                display_name = voice_profile.get('display_name', voice_name)
                description = voice_profile.get('description', '')
                
                # Status message
                status_msg = f"<div class='voice-status'>✅ Using voice: {display_name}"
                if description:
                    status_msg += f" - {description}"
                status_msg += f"<br/>📄 Loaded from JSON: exag={exaggeration}, cfg={cfg_weight}, temp={temperature}</div>"
                
                return (
                    gr.Audio(visible=False),  # Hide manual audio upload
                    exaggeration,
                    cfg_weight,
                    temperature,
                    status_msg
                )
            else:
                return (
                    gr.Audio(visible=True),
                    0.5, 0.5, 0.8,
                    f"<div class='voice-status'>❌ Could not load voice profile: {voice_name}</div>"
                )
                
        except Exception as e:
            return (
                gr.Audio(visible=True),
                0.5, 0.5, 0.8,
                f"<div class='voice-status'>❌ Error loading voice settings: {str(e)}</div>"
            )
    
    def _load_voice_into_form(self, voice_library_path: str, voice_name: str) -> Tuple[str, str, str, str, float, float, float, bool, float, str]:
        """Load an existing voice profile's settings into the form for editing."""
        try:
            if not voice_name or "─────── Voice Pool ───────" in voice_name:
                return (
                    "",  # voice_name
                    "",  # voice_display_name
                    "",  # voice_description
                    None,  # voice_audio
                    0.5,  # voice_exaggeration
                    0.5,  # voice_cfg
                    0.8,  # voice_temp
                    True,  # enable_voice_normalization (default enabled)
                    -18.0,  # voice_target_level (ACX standard)
                    "<div class='voice-status'>❌ Please select a voice to load</div>"
                )
            
            # Load voice profile
            voice_profile = load_voice_profile(voice_library_path, voice_name)
            if voice_profile:
                # Extract all settings from JSON
                display_name = voice_profile.get('display_name', voice_name)
                description = voice_profile.get('description', '')
                exaggeration = voice_profile.get('exaggeration', 0.5)
                cfg_weight = voice_profile.get('cfg_weight', 0.5) 
                temperature = voice_profile.get('temperature', 0.8)
                
                # Volume normalization settings (enable by default if not specified)
                enable_normalization = voice_profile.get('enable_normalization', True)
                target_level = voice_profile.get('target_level_db', -18.0)
                
                # Get audio file path
                profile_type = voice_profile.get('profile_type', 'unknown')
                audio_path = None
                
                try:
                    if profile_type == 'subfolder':
                        voice_dir = Path(voice_profile['path'])
                        audio_path = str(voice_dir / voice_profile['audio_file'])
                    elif profile_type in ['legacy_json', 'raw_wav', 'legacy_folder']:
                        voice_dir = Path(voice_profile['path'])
                        audio_path = str(voice_dir / voice_profile['audio_file'])
                    
                    # Verify the audio file exists
                    if audio_path and not Path(audio_path).exists():
                        print(f"⚠️ Audio file not found: {audio_path}")
                        audio_path = None
                        
                except Exception as e:
                    print(f"⚠️ Error getting audio path: {e}")
                    audio_path = None
                
                # Status message
                status_msg = f"<div class='voice-status'>✅ Loaded voice settings for: {display_name}"
                if description:
                    status_msg += f" - {description}"
                status_msg += f"<br/>📄 Ready to edit and save over existing profile</div>"
                
                return (
                    voice_name,  # Keep original voice name for saving
                    display_name,
                    description,
                    audio_path,  # Load the audio file
                    exaggeration,
                    cfg_weight,
                    temperature,
                    enable_normalization,
                    target_level,
                    status_msg
                )
            else:
                return (
                    voice_name,
                    "",
                    "",
                    None,
                    0.5, 0.5, 0.8,
                    True, -18.0,  # Default normalization settings
                    f"<div class='voice-status'>❌ Could not load voice profile: {voice_name}</div>"
                )
                
        except Exception as e:
            return (
                "",
                "",
                "",
                None,
                0.5, 0.5, 0.8,
                True, -18.0,  # Default normalization settings
                f"<div class='voice-status'>❌ Error loading voice into form: {str(e)}</div>"
            )
    
    def _delete_voice_profile(self, voice_library_path: str, voice_name: str) -> Tuple[str, gr.Dropdown]:
        """Delete a voice profile."""
        try:
            if not voice_name or "─────── Voice Pool ───────" in voice_name:
                return (
                    "<div class='voice-status'>❌ Please select a voice to delete</div>",
                    gr.Dropdown(choices=self._get_voice_choices())
                )
            
            # Import the delete function from voice management
            try:
                from src.voice_library.voice_management import delete_voice_profile
            except ImportError:
                from voice_library.voice_management import delete_voice_profile
            
            # Delete the voice profile
            success, message = delete_voice_profile(voice_library_path, voice_name)
            
            if success:
                # Refresh the voice choices after deletion
                new_choices = self._get_voice_choices()
                return (
                    f"<div class='voice-status'>✅ {message}</div>",
                    gr.Dropdown(choices=new_choices, value=None)
                )
            else:
                return (
                    f"<div class='voice-status'>❌ {message}</div>",
                    gr.Dropdown(choices=self._get_voice_choices())
                )
                
        except Exception as e:
            return (
                f"<div class='voice-status'>❌ Error deleting voice: {str(e)}</div>",
                gr.Dropdown(choices=self._get_voice_choices())
            )
    
    def _update_voice_library_path(self, new_path: str) -> Tuple[str, str, gr.Dropdown, gr.Dropdown]:
        """Update the voice library path and refresh related components.
        
        Args:
            new_path: New voice library path
            
        Returns:
            Tuple of (updated_path_state, status_html, refreshed_voice_dropdown, refreshed_raw_voices_dropdown)
        """
        try:
            if not new_path.strip():
                return (
                    self.voice_library_path,
                    "<div class='config-status error'>❌ Please enter a valid path</div>",
                    gr.Dropdown(choices=self._get_voice_choices()),
                    gr.Dropdown(choices=[])
                )
            
            # Convert to absolute path and validate
            new_path = os.path.abspath(new_path.strip())
            
            # Create directory if it doesn't exist
            if not os.path.exists(new_path):
                try:
                    os.makedirs(new_path, exist_ok=True)
                    print(f"📁 Created voice library directory: {new_path}")
                except Exception as e:
                    return (
                        self.voice_library_path,
                        f"<div class='config-status error'>❌ Could not create directory: {str(e)}</div>",
                        gr.Dropdown(choices=self._get_voice_choices()),
                        gr.Dropdown(choices=[])
                    )
            
            # Update the instance variable
            self.voice_library_path = new_path
            
            # Update the config if possible
            try:
                from src.config.settings import config
                config.update_voices_path(new_path)
                print(f"🔧 Updated config with new voice library path: {new_path}")
            except Exception as e:
                print(f"⚠️ Could not update config: {e}")
            
            # Get new voice choices from the updated path
            try:
                # Import the voice management function
                try:
                    from src.voice_library.voice_management import get_voice_choices
                except ImportError:
                    from voice_library.voice_management import get_voice_choices
                
                new_choices = get_voice_choices(new_path)
                print(f"📋 Found {len(new_choices)} voices in new library")
                
            except Exception as e:
                print(f"⚠️ Error loading voices from new path: {e}")
                new_choices = []
            
            # Get raw audio files
            raw_audio_files = self._get_raw_audio_files(new_path)
            
            return (
                new_path,
                f"<div class='config-status success'>✅ Voice library updated: {new_path}<br/>Found {len(new_choices)} voice profiles, {len(raw_audio_files)} raw audio files</div>",
                gr.Dropdown(choices=new_choices, value=None),
                gr.Dropdown(choices=raw_audio_files, value=None)
            )
            
        except Exception as e:
            print(f"❌ Error updating voice library path: {e}")
            return (
                self.voice_library_path,
                f"<div class='config-status error'>❌ Error updating path: {str(e)}</div>",
                gr.Dropdown(choices=self._get_voice_choices()),
                gr.Dropdown(choices=[])
            )
    
    def _generate_tts(self, model_state, text: str, ref_wav: str, voice_selector: str, 
                     exaggeration: float, cfg_weight: float, temperature: float, 
                     seed: int, voice_library_path: str) -> Tuple[str, object]:
        """Generate TTS audio for testing."""
        try:
            if not CHATTERBOX_AVAILABLE:
                return None, model_state
            
            # Load model if needed
            if model_state is None:
                model_state = load_model()
            
            # Use voice selector or reference audio
            audio_path = ref_wav
            if voice_selector and "─────── Voice Pool ───────" not in voice_selector:
                # Load voice profile audio
                voice_profile = load_voice_profile(voice_library_path, voice_selector)
                if voice_profile:
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
                        # Fallback - try legacy format handling
                        if voice_profile.get('legacy_format'):
                            audio_path = voice_profile['audio_file']
                        else:
                            voice_dir = Path(voice_profile['path'])
                            audio_path = str(voice_dir / voice_profile['audio_file'])
            
            # Generate audio
            result = generate_for_gradio(
                model_state, text, audio_path, exaggeration, temperature, 0, cfg_weight
            )
            
            if result is not None:
                return result, model_state
            else:
                return None, model_state
                
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None, model_state
    
    def _validate_single_voice_input(self, text: str, voice: str, project_name: str) -> str:
        """Validate single voice audiobook input."""
        try:
            # Check if voice is the separator
            if voice and "─────── Voice Pool ───────" in voice:
                return f"<div class='status-error'>❌ Please select a voice, not the separator</div>"
                
            is_valid, message = validate_audiobook_input(text, voice, project_name)
            
            if is_valid:
                return f"<div class='status-success'>✅ {message}</div>"
            else:
                return f"<div class='status-error'>❌ {message}</div>"
                
        except Exception as e:
            return f"<div class='status-error'>❌ Validation error: {str(e)}</div>"
    
    def _generate_single_voice_audiobook(self, model_state, text: str, voice: str, 
                                       project_name: str, resume: bool, autosave_interval: int,
                                       voice_library_path: str) -> Tuple[str, str, str, str, object]:
        """Generate a single voice audiobook."""
        try:
            if not CHATTERBOX_AVAILABLE:
                return ("<div class='status-error'>❌ TTS engine not available</div>", 
                       "", None, "", model_state)
            
            # Check if voice is the separator
            if voice and "─────── Voice Pool ───────" in voice:
                return ("<div class='status-error'>❌ Please select a voice, not the separator</div>", 
                       "", None, "", model_state)
            
            # Load model if needed
            if model_state is None:
                model_state = load_model()
                if model_state is None:
                    return ("<div class='status-error'>❌ Failed to load TTS model</div>", 
                           "", None, "", model_state)
            
            # Start generation process
            status_html = "<div class='status-info'>🚀 Starting audiobook generation...</div>"
            progress_html = "<div class='progress-info'>Initializing audiobook generation...</div>"
            
            # Initialize variables for tracking
            final_audio_path = None
            project_info_html = ""
            
            # Create generator for real-time updates
            generation_generator = generate_single_voice_audiobook(
                model_state=model_state,
                text=text,
                voice_name=voice,
                project_name=project_name,
                voice_library_path=voice_library_path,
                max_words_per_chunk=50,
                autosave_interval=autosave_interval
            )
            
            # Process generation with progress tracking
            last_progress = None
            for progress_update in generation_generator:
                last_progress = progress_update
                
                # Update status based on progress
                step = progress_update.get('step', 'unknown')
                message = progress_update.get('message', 'Processing...')
                progress_percent = progress_update.get('progress', 0)
                status_type = progress_update.get('status', 'info')
                
                # Format status HTML
                if status_type == 'error':
                    status_html = f"<div class='status-error'>{message}</div>"
                    break
                elif status_type == 'success' and step == 'complete':
                    status_html = f"<div class='status-success'>{message}</div>"
                    
                    # Get final results
                    stats = progress_update.get('stats', {})
                    final_audio_path = progress_update.get('final_audio_path', '')
                    
                    # Create project info display
                    project_info_html = f"""
                    <div class='project-info-success'>
                        <h4>📊 Generation Complete!</h4>
                        <p><strong>Project:</strong> {project_name}</p>
                        <p><strong>Voice:</strong> {voice}</p>
                        <p><strong>Total Chunks:</strong> {stats.get('total_chunks', 0)}</p>
                        <p><strong>Total Words:</strong> {stats.get('total_words', 0)}</p>
                        <p><strong>Generation Time:</strong> {stats.get('total_generation_time', 0):.1f}s</p>
                        <p><strong>Final Audio:</strong> {os.path.basename(final_audio_path) if final_audio_path else 'N/A'}</p>
                    </div>
                    """
                else:
                    status_html = f"<div class='status-info'>{message}</div>"
                
                # Format progress HTML
                if step == 'tts_generation':
                    chunk_current = progress_update.get('chunk_current', 0)
                    chunk_total = progress_update.get('chunk_total', 0)
                    chunk_text = progress_update.get('chunk_text', '')
                    
                    progress_html = f"""
                    <div class='progress-detailed'>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: {progress_percent}%'></div>
                        </div>
                        <p><strong>Step:</strong> {step.replace('_', ' ').title()}</p>
                        <p><strong>Progress:</strong> {progress_percent:.1f}%</p>
                        <p><strong>Chunk:</strong> {chunk_current}/{chunk_total}</p>
                        <p><strong>Current Text:</strong> {chunk_text}</p>
                    </div>
                    """
                else:
                    progress_html = f"""
                    <div class='progress-simple'>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: {progress_percent}%'></div>
                        </div>
                        <p><strong>Step:</strong> {step.replace('_', ' ').title()}</p>
                        <p><strong>Progress:</strong> {progress_percent:.1f}%</p>
                    </div>
                    """
            
            # Handle final results
            if last_progress and last_progress.get('status') == 'success':
                # Return successful generation
                return (
                    status_html,
                    progress_html, 
                    final_audio_path if final_audio_path and os.path.exists(final_audio_path) else None,
                    project_info_html,
                    model_state
                )
            elif last_progress and last_progress.get('status') == 'error':
                # Return error state
                return (
                    status_html,
                    progress_html,
                    None,
                    "",
                    model_state
                )
            else:
                # Fallback case
                return (
                    "<div class='status-warning'>⚠️ Generation completed with unknown status</div>",
                    progress_html,
                    None,
                    "",
                    model_state
                )
            
        except Exception as e:
            error_msg = f"<div class='status-error'>❌ Generation error: {str(e)}</div>"
            return (error_msg, "", None, "", model_state)
    
    def _analyze_and_setup_voices(self, text: str, voice_library_path: str):
        """Analyze multi-voice text and setup character voice assignment."""
        try:
            if not text.strip():
                empty_returns = ("<div class='status-warning'>⚠️ Please enter multi-voice text to analyze</div>", 
                               gr.Group(visible=False), "", 
                               "", gr.Dropdown(visible=False),
                               "", gr.Dropdown(visible=False),
                               "", gr.Dropdown(visible=False))
                return empty_returns
            
            # Parse the text to find characters
            segments = parse_multi_voice_text(text)
            
            if not segments:
                empty_returns = ("<div class='status-warning'>⚠️ No character markers found. Use [Character Name] format.</div>", 
                               gr.Group(visible=False), "",
                               "", gr.Dropdown(visible=False),
                               "", gr.Dropdown(visible=False),
                               "", gr.Dropdown(visible=False))
                return empty_returns
            
            # Get unique characters and their counts
            characters = list(set(segment['character'] for segment in segments))
            character_counts = {char: sum(1 for seg in segments if seg['character'] == char) 
                              for char in characters}
            
            # Get available voices
            voices = self._get_voice_choices()
            
            # Create analysis display
            analysis_html = f"""
            <div class='status-success'>
                <h4>✅ Found {len(characters)} characters:</h4>
                <ul>
            """
            
            for char, count in character_counts.items():
                analysis_html += f"<li><strong>{char}</strong>: {count} segments</li>"
            
            analysis_html += """
                </ul>
                <p>Now assign voices to each character below:</p>
            </div>
            """
            
            # Create assignment interface description
            assignment_html = f"""
            <div class='instruction-box'>
                <h4>🎭 Character Voice Assignments</h4>
                <p>Select a voice for each character. Each character can have a different voice!</p>
                <p><strong>Characters found:</strong> {', '.join(characters)}</p>
            </div>
            """
            
            # Setup character dropdowns - show up to 5 characters
            char_updates = []
            voice_updates = []
            
            for i in range(5):
                if i < len(characters):
                    # Character exists - show it
                    char_name = characters[i]
                    char_updates.extend([char_name, gr.Dropdown(choices=voices, value=voices[0] if voices else None, visible=True, label=f"Voice for {char_name}")])
                else:
                    # No character - hide
                    char_updates.extend(["", gr.Dropdown(visible=False)])
            
            return (analysis_html, gr.Group(visible=True), assignment_html, *char_updates)
            
        except Exception as e:
            error_msg = f"<div class='status-error'>❌ Analysis error: {str(e)}</div>"
            empty_returns = (error_msg, gr.Group(visible=False), "",
                           "", gr.Dropdown(visible=False),
                           "", gr.Dropdown(visible=False),
                           "", gr.Dropdown(visible=False))
            return empty_returns
    
    def _test_voice_profile(self, voice_library_path: str, voice_name: str, audio_file: str,
                           exaggeration: float, cfg_weight: float, temperature: float) -> Tuple[str, Optional[Tuple]]:
        """Test a voice profile with current settings."""
        try:
            if not voice_name or not voice_name.strip():
                return "<div class='status-error'>⚠️ Please enter a voice name</div>", None
            
            if not audio_file:
                return "<div class='status-error'>⚠️ Please upload an audio file</div>", None
            
            # Import the test function
            try:
                from src.models.tts_model import generate_for_gradio, load_model
            except ImportError:
                from models.tts_model import generate_for_gradio, load_model
            
            # Load model
            model = load_model()
            if not model:
                return "<div class='status-error'>❌ Failed to load TTS model</div>", None
            
            # Generate test audio
            test_text = f"Hello, this is a test of the voice profile {voice_name}."
            result = generate_for_gradio(
                model, test_text, audio_file, exaggeration, temperature, 0, cfg_weight
            )
            
            if result:
                return f"<div class='status-success'>✅ Voice test successful for '{voice_name}'</div>", result
            else:
                return "<div class='status-error'>❌ Voice test failed</div>", None
                
        except Exception as e:
            return f"<div class='status-error'>❌ Error testing voice: {str(e)}</div>", None
    
    def _save_voice_profile(self, voice_library_path: str, voice_name: str, display_name: str,
                           description: str, audio_file: str, exaggeration: float, cfg_weight: float,
                           temperature: float, enable_normalization: bool, target_level: float,
                           enable_silence_trimming: bool, silence_threshold: float, min_silence_duration: float,
                           processing_quality: str) -> str:
        """Save a voice profile to the library."""
        try:
            # Debug logging
            print(f"🔧 DEBUG - Save voice called:")
            print(f"  voice_library_path: {voice_library_path}")
            print(f"  voice_name: '{voice_name}'")
            print(f"  display_name: '{display_name}'")
            print(f"  description: '{description}'")
            print(f"  audio_file: '{audio_file}'")
            print(f"  exaggeration: {exaggeration}")
            print(f"  cfg_weight: {cfg_weight}")
            print(f"  temperature: {temperature}")
            print(f"  enable_normalization: {enable_normalization}")
            print(f"  target_level: {target_level}")
            
            # Validate inputs
            if not voice_name or not voice_name.strip():
                return "<div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>⚠️ Please enter a voice name</div>"
            
            if not display_name or not display_name.strip():
                return "<div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>⚠️ Please enter a display name</div>"
            
            if not audio_file:
                return "<div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>⚠️ Please upload an audio file</div>"
            
            # Check if audio file exists
            if not os.path.exists(audio_file):
                return f"<div style='background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;'>⚠️ Audio file not found: {audio_file}</div>"
            
            # Preprocess audio if enabled
            processed_audio_file = audio_file
            preprocessing_status = []
            
            if enable_silence_trimming:
                print(f"🎛️ Applying audio preprocessing ({processing_quality} mode)...")
                try:
                    # Import audio preprocessing functions
                    from src.audio.preprocessing import preprocess_voice_sample
                    from src.audio.file_management import load_audio_file, save_audio_file
                    
                    # Load the original audio
                    audio_data, sample_rate = load_audio_file(audio_file)
                    
                    if audio_data is not None and sample_rate is not None:
                        # Adjust settings based on processing quality
                        if processing_quality == "Conservative":
                            actual_threshold = max(silence_threshold * 0.5, 0.002)  # More sensitive
                            actual_min_duration = min_silence_duration * 1.5  # Require longer silence
                            fade_time = 0.08  # Longer fade for smoother transitions
                        elif processing_quality == "Balanced":
                            actual_threshold = silence_threshold
                            actual_min_duration = min_silence_duration
                            fade_time = 0.05  # Standard fade
                        else:  # Aggressive
                            actual_threshold = min(silence_threshold * 2.0, 0.02)  # Less sensitive
                            actual_min_duration = min_silence_duration * 0.7  # Accept shorter silence
                            fade_time = 0.03  # Shorter fade
                        
                        print(f"   Using threshold: {actual_threshold:.4f}, min duration: {actual_min_duration:.2f}s")
                        
                        # Apply preprocessing
                        processed_audio, preprocessing_info = preprocess_voice_sample(
                            audio_data=audio_data,
                            sample_rate=sample_rate,
                            trim_silence=True,
                            normalize_level=enable_normalization,
                            target_level=target_level,
                            silence_threshold=actual_threshold,
                            min_silence_duration=actual_min_duration,
                            fade_duration=fade_time
                        )
                        
                        # Create processed audio filename
                        audio_dir = os.path.dirname(audio_file)
                        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
                        temp_processed_file = os.path.join(audio_dir, f"{audio_basename}_processed.wav")
                        
                        # Save processed audio
                        if save_audio_file(processed_audio, sample_rate, temp_processed_file):
                            print(f"✅ Audio preprocessing successful")
                            processed_audio_file = temp_processed_file  # Only use processed file if save succeeds
                            
                            # Build status message
                            if preprocessing_info.get("processing_success"):
                                steps = preprocessing_info.get("steps_applied", [])
                                if "silence_trimming" in steps:
                                    trim_info = preprocessing_info.get("silence_trimming", {})
                                    if trim_info.get("success"):
                                        trimmed_seconds = trim_info.get("total_trimmed_seconds", 0)
                                        preprocessing_status.append(f"Trimmed {trimmed_seconds:.2f}s silence")
                                
                                if "normalization" in steps:
                                    norm_info = preprocessing_info.get("normalization", {})
                                    if norm_info.get("success"):
                                        applied_gain = norm_info.get("applied_gain_db", 0)
                                        preprocessing_status.append(f"Normalized ({applied_gain:+.1f}dB)")
                        else:
                            print(f"⚠️ Failed to save processed audio, using original")
                            preprocessing_status.append("Preprocessing failed, using original")
                            # processed_audio_file remains as original audio_file
                    else:
                        print(f"⚠️ Failed to load audio for preprocessing, using original")
                        preprocessing_status.append("Could not load audio for preprocessing")
                        
                except ImportError as e:
                    print(f"⚠️ Audio preprocessing not available: {e}")
                    preprocessing_status.append("Preprocessing module not available")
                except Exception as e:
                    print(f"⚠️ Audio preprocessing error: {e}")
                    preprocessing_status.append(f"Preprocessing error: {str(e)}")
            
            print(f"🔧 Using audio file: {processed_audio_file}")
            if preprocessing_status:
                print(f"🎛️ Preprocessing: {', '.join(preprocessing_status)}")
            
            # Clean voice name (no spaces, special characters)
            clean_voice_name = voice_name.strip().replace(' ', '_').replace('-', '_')
            clean_voice_name = ''.join(c for c in clean_voice_name if c.isalnum() or c == '_')
            
            if not clean_voice_name:
                return "<div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>⚠️ Voice name must contain valid characters</div>"
            
            print(f"🔧 DEBUG - Clean voice name: '{clean_voice_name}'")
            
            # Import the save function
            try:
                from src.voice_library.voice_management import save_voice_profile
            except ImportError:
                from voice_library.voice_management import save_voice_profile
            
            print(f"🔧 DEBUG - About to call save_voice_profile function...")
            
            # Save the voice profile
            success, message = save_voice_profile(
                voice_library_path=voice_library_path,
                voice_name=clean_voice_name,
                display_name=display_name.strip(),
                description=description.strip() if description else f"Custom voice: {display_name}",
                audio_file_path=processed_audio_file,  # Use processed audio
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                enable_normalization=enable_normalization,
                target_level_db=target_level
            )
            
            print(f"🔧 DEBUG - Save result: success={success}, message='{message}'")
            
            if success:
                # Add preprocessing info to success message
                full_message = message
                if preprocessing_status:
                    full_message += f"<br><small>🎛️ Preprocessing: {', '.join(preprocessing_status)}</small>"
                return f"<div style='background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;'>{full_message}</div>"
            else:
                return f"<div style='background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;'>{message}</div>"
                
        except Exception as e:
            error_msg = f"❌ Error saving voice profile: {str(e)}"
            print(f"🔧 DEBUG - Exception occurred: {error_msg}")
            import traceback
            traceback.print_exc()
            return f"<div style='background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;'>{error_msg}</div>"
    
    def _generate_multi_voice_audiobook_with_assignments(self, model_state, text: str, project_name: str,
                                                       voice_library_path: str, char1_name: str, char1_voice: str,
                                                       char2_name: str, char2_voice: str,
                                                       char3_name: str, char3_voice: str,
                                                       char4_name: str, char4_voice: str,
                                                       char5_name: str, char5_voice: str) -> Tuple[str, str]:
        """Generate a multi-voice audiobook with assigned voices."""
        try:
            if not CHATTERBOX_AVAILABLE:
                return "<div class='status-error'>❌ TTS engine not available</div>", ""
            
            if not text.strip():
                return "<div class='status-warning'>⚠️ Please enter text to generate</div>", ""
                
            if not project_name.strip():
                return "<div class='status-warning'>⚠️ Please enter a project name</div>", ""
            
            # Parse the text to find characters
            segments = parse_multi_voice_text(text)
            
            if not segments:
                return "<div class='status-warning'>⚠️ No character markers found. Use [Character Name] format.</div>", ""
            
            # Build voice assignments from character inputs
            voice_assignments = {}
            character_inputs = [
                (char1_name, char1_voice),
                (char2_name, char2_voice),
                (char3_name, char3_voice),
                (char4_name, char4_voice),
                (char5_name, char5_voice)
            ]
            
            for char_name, char_voice in character_inputs:
                if char_name and char_name.strip() and char_voice:
                    voice_assignments[char_name.strip()] = char_voice
            
            if not voice_assignments:
                return "<div class='status-error'>❌ No voice assignments found. Please analyze characters first.</div>", ""
            
            # Load model if needed
            if model_state is None:
                model_state = load_model()
                if model_state is None:
                    return "<div class='status-error'>❌ Failed to load TTS model</div>", ""
            
            # Create output directory
            output_dir = os.path.join("audiobook_projects", f"{project_name}_multi_voice")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🔧 DEBUG - Multi-voice generation:")
            print(f"  Output directory: {os.path.abspath(output_dir)}")
            print(f"  Voice assignments: {voice_assignments}")
            print(f"  Total segments: {len(segments)}")
            
            try:
                # Import necessary functions
                try:
                    from src.models.tts_model import generate_for_gradio
                    from src.voice_library.voice_management import load_voice_profile
                except ImportError:
                    from models.tts_model import generate_for_gradio
                    from voice_library.voice_management import load_voice_profile
                
                # Generate each segment with appropriate voice
                segment_audio_files = []
                total_segments = len(segments)
                
                for i, segment in enumerate(segments):
                    character = segment['character']
                    text_content = segment['text'].strip()
                    
                    if not text_content:
                        continue
                    
                    # Get assigned voice for this character
                    voice_name = voice_assignments.get(character)
                    if not voice_name:
                        print(f"⚠️ No voice assigned for character '{character}', skipping segment")
                        continue
                    
                    print(f"🎙️ Generating segment {i+1}/{total_segments}: {character} → {voice_name}")
                    print(f"   Text: {text_content[:50]}...")
                    
                    # Load voice profile
                    voice_profile = load_voice_profile(voice_library_path, voice_name)
                    if not voice_profile:
                        print(f"❌ Failed to load voice profile for {voice_name}")
                        continue
                    
                    # Get audio reference file based on profile type
                    profile_type = voice_profile.get('profile_type', 'unknown')
                    
                    if profile_type == 'subfolder':
                        voice_dir = Path(voice_profile['path'])
                        audio_path = str(voice_dir / voice_profile['audio_file'])
                    elif profile_type == 'legacy_json' or profile_type == 'raw_wav':
                        voice_dir = Path(voice_profile['path'])
                        audio_path = str(voice_dir / voice_profile['audio_file'])
                    else:
                        if voice_profile.get('legacy_format'):
                            audio_path = voice_profile['audio_file']
                        else:
                            voice_dir = Path(voice_profile['path'])
                            audio_path = str(voice_dir / voice_profile['audio_file'])
                    
                    # Use voice profile settings
                    exaggeration = voice_profile.get('exaggeration', 0.5)
                    temperature = voice_profile.get('temperature', 0.8)
                    cfg_weight = voice_profile.get('cfg_weight', 0.5)
                    
                    print(f"🎛️ Voice settings: exaggeration={exaggeration:.2f}, temp={temperature:.2f}, cfg={cfg_weight:.2f}")
                    
                    # Generate audio for this segment
                    audio_result = generate_for_gradio(
                        model_state,
                        text_content,  # Text WITHOUT character brackets
                        audio_path,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        seed_num=0,
                        cfg_weight=cfg_weight
                    )
                    
                    if audio_result and len(audio_result) == 2:
                        sample_rate, audio_data = audio_result
                        
                        # Save segment audio
                        segment_filename = f"segment_{i:03d}_{character.replace(' ', '_')}.wav"
                        segment_filepath = os.path.join(output_dir, segment_filename)
                        
                        # Save audio file
                        with wave.open(segment_filepath, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            
                            # Convert float32 to int16
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            wav_file.writeframes(audio_int16.tobytes())
                        
                        segment_audio_files.append(segment_filepath)
                        print(f"✅ Generated segment audio: {segment_filename}")
                    else:
                        print(f"❌ Failed to generate audio for segment {i+1}")
                
                if not segment_audio_files:
                    return "<div class='status-error'>❌ No audio segments were generated successfully</div>", ""
                
                # Combine all segment audio files
                print(f"🎵 Combining {len(segment_audio_files)} audio segments...")
                final_audio_path = os.path.join(output_dir, f"{project_name}_multi_voice_complete.wav")
                
                combined_audio = []
                sample_rate = 24000
                
                for audio_file in segment_audio_files:
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
                
                print(f"🔧 DEBUG - Multi-voice generation complete:")
                print(f"  Final audio path: {final_audio_path}")
                print(f"  Audio file exists: {os.path.exists(final_audio_path)}")
                print(f"  Segments processed: {len(segment_audio_files)}")
                
                # Create success message
                assignment_summary = []
                for char, voice in voice_assignments.items():
                    assignment_summary.append(f"{char} → {voice}")
                
                success_message = f"""
                <div class='status-success'>
                    <h4>✅ Multi-Voice Audiobook Generated!</h4>
                    <p><strong>Project:</strong> {project_name}_multi_voice</p>
                    <p><strong>Voice Assignments:</strong></p>
                    <ul>
                        {''.join(f'<li>{assignment}</li>' for assignment in assignment_summary)}
                    </ul>
                    <p><strong>Segments Processed:</strong> {len(segment_audio_files)}</p>
                    <p><strong>Total Characters:</strong> {len(voice_assignments)}</p>
                    <p><em>✨ True multi-voice processing with character-specific voices!</em></p>
                </div>
                """
                
                # Verify the audio file exists and is accessible
                if final_audio_path and os.path.exists(final_audio_path):
                    return success_message, final_audio_path
                else:
                    return success_message, ""
                
            except Exception as generation_error:
                print(f"Generation error: {generation_error}")
                return f"<div class='status-error'>❌ Generation error: {str(generation_error)}</div>", ""
                
        except Exception as e:
            import traceback
            print(f"Multi-voice generation error: {e}")
            traceback.print_exc()
            return f"<div class='status-error'>❌ Generation error: {str(e)}</div>", ""
    
    def _reset_voice_library_path(self) -> Tuple[str, str, str, gr.Dropdown, gr.Dropdown]:
        """Reset the voice library path to its default value."""
        default_path = config.get_voices_path()  # Use new config system
        self.voice_library_path = default_path
        status_msg = f"<div class='config-status'>✅ Library path reset to default: {default_path}</div>"
        
        # Refresh dropdowns
        voice_choices = self._get_voice_choices()
        raw_voices = self._get_raw_audio_files(default_path)

        return (
            default_path,  # Update the textbox
            status_msg,
            "",  # Clear the form status
            gr.Dropdown(choices=voice_choices),
            gr.Dropdown(choices=raw_voices)
        )
    
    def _get_raw_audio_files(self, voice_library_path: str) -> List[str]:
        """Get list of raw audio files that don't have voice profiles.
        
        Args:
            voice_library_path: Path to voice library
            
        Returns:
            List of raw audio file names
        """
        try:
            if not os.path.exists(voice_library_path):
                return []
            
            # Audio file extensions to look for
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            
            # Get all processed voice profile names to exclude
            try:
                from src.voice_library.voice_management import get_voice_choices
                processed_voices = set(get_voice_choices(voice_library_path))
            except:
                processed_voices = set()
            
            raw_files = []
            
            # Scan for audio files
            for item in os.listdir(voice_library_path):
                item_path = os.path.join(voice_library_path, item)
                
                # Check if it's an audio file
                if os.path.isfile(item_path):
                    name, ext = os.path.splitext(item)
                    if ext.lower() in audio_extensions:
                        # Check if this file already has a voice profile
                        if name not in processed_voices:
                            raw_files.append(item)
                
                # Also check subdirectories for audio files
                elif os.path.isdir(item_path):
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isfile(subitem_path):
                            name, ext = os.path.splitext(subitem)
                            if ext.lower() in audio_extensions:
                                # Use folder/filename format
                                full_name = f"{item}/{subitem}"
                                if item not in processed_voices:  # Check folder name
                                    raw_files.append(full_name)
            
            return sorted(raw_files)
            
        except Exception as e:
            print(f"Error scanning for raw audio files: {e}")
            return []
    
    def _refresh_raw_voices(self, voice_library_path: str) -> gr.Dropdown:
        """Refresh the raw voices dropdown.
        
        Args:
            voice_library_path: Path to voice library
            
        Returns:
            Updated dropdown with raw audio files
        """
        raw_files = self._get_raw_audio_files(voice_library_path)
        print(f"🔄 Refreshed raw voices: Found {len(raw_files)} raw audio files")
        return gr.Dropdown(choices=raw_files, value=None)
    
    def _load_raw_voice(self, selected_file: str, voice_library_path: str) -> Tuple[str, str]:
        """Load a raw audio file into the voice configuration.
        
        Args:
            selected_file: Selected raw audio file
            voice_library_path: Path to voice library
            
        Returns:
            Tuple of (audio_file_path, suggested_voice_name)
        """
        try:
            if not selected_file:
                return None, ""
            
            # Construct full path to audio file
            audio_file_path = os.path.join(voice_library_path, selected_file)
            
            if not os.path.exists(audio_file_path):
                print(f"❌ Raw audio file not found: {audio_file_path}")
                return None, ""
            
            # Generate suggested voice name from filename
            if "/" in selected_file:
                # Handle subdirectory format: folder/file.wav
                folder, filename = selected_file.split("/", 1)
                name_base = folder
            else:
                # Handle direct file: file.wav
                name_base = os.path.splitext(selected_file)[0]
            
            # Clean up name for voice profile
            suggested_name = name_base.replace(" ", "_").replace("-", "_")
            suggested_name = ''.join(c for c in suggested_name if c.isalnum() or c == '_')
            
            print(f"📁 Loaded raw audio: {selected_file} → suggested name: {suggested_name}")
            
            return audio_file_path, suggested_name
            
        except Exception as e:
            print(f"❌ Error loading raw audio file: {e}")
            return None, ""
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
        """Launch the Gradio interface."""
        if self.demo is None:
            raise ValueError("Interface not created. Call _create_interface() first.")
        
        print("🚀 Launching Chatterbox Audiobook Studio (Refactored)...")
        print(f"📍 Server: http://{server_name}:{server_port}")
        print("🎧 Ready for audiobook creation!")
        
        # Add voice library path to allowed paths for security
        allowed_paths = [self.voice_library_path]
        
        # Also add the parent speakers directory if different
        speakers_dir = os.path.abspath("../speakers")
        if speakers_dir not in allowed_paths:
            allowed_paths.append(speakers_dir)
            
        print(f"🔒 Allowed paths: {allowed_paths}")
        
        self.demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False,
            allowed_paths=allowed_paths
        )

    def _test_trimming(self, audio_file: str, enable_trimming: bool, quality: str, threshold: float, min_duration: float) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """Runs only the trimming process on an uploaded audio file for quick testing."""
        if not enable_trimming:
            return None, "<div class='status-warning'>⚠️ Silence Trimming is not enabled.</div>"
        
        if not audio_file or not os.path.exists(audio_file):
            return None, "<div class='status-error'>❌ Please upload an audio file first.</div>"
            
        try:
            from src.audio.file_management import load_audio_file
            from src.audio.preprocessing import preprocess_voice_sample

            print(f"✂️ Testing trimming with settings: Quality='{quality}', Threshold={threshold}, Min Duration={min_duration}")

            audio_data, sample_rate = load_audio_file(audio_file)

            if audio_data is None or sample_rate is None:
                return None, "<div class='status-error'>❌ Failed to load audio file for trimming test.</div>"

            # Adjust settings based on processing quality
            if quality == "Conservative":
                actual_threshold = max(threshold * 0.5, 0.002)
                actual_min_duration = min_duration * 1.5
                fade_time = 0.08
            elif quality == "Balanced":
                actual_threshold = threshold
                actual_min_duration = min_duration
                fade_time = 0.05
            else:  # Aggressive
                actual_threshold = min(threshold * 2.0, 0.02)
                actual_min_duration = min_duration * 0.7
                fade_time = 0.03
            
            print(f"   Using effective settings: Threshold={actual_threshold:.4f}, Duration={actual_min_duration:.2f}s")

            # Perform trimming ONLY
            processed_audio, processing_info = preprocess_voice_sample(
                audio_data=audio_data,
                sample_rate=sample_rate,
                trim_silence=True,
                normalize_level=False, # We only want to test trimming here
                silence_threshold=actual_threshold,
                min_silence_duration=actual_min_duration,
                fade_duration=fade_time
            )

            trim_info = processing_info.get("silence_trimming", {})
            if not trim_info.get("success"):
                error_msg = trim_info.get("error", "Unknown trimming error")
                return None, f"<div class='status-error'>❌ Trimming failed: {error_msg}</div>"

            original_duration = len(audio_data) / sample_rate
            trimmed_duration = len(processed_audio) / sample_rate
            removed_seconds = original_duration - trimmed_duration

            status_msg = f"✅ Trim test successful. Removed {removed_seconds:.2f}s of silence. ({original_duration:.2f}s → {trimmed_duration:.2f}s)"
            
            if 'warning' in trim_info:
                status_msg += f"<br><small>⚠️ {trim_info['warning']}</small>"

            print(status_msg.replace('<br>', '\n'))

            return (sample_rate, processed_audio), f"<div class='status-info'>{status_msg}</div>"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"<div class='status-error'>❌ An error occurred during trim test: {e}</div>"


# Factory function for easy creation
def create_chatterbox_app() -> ChatterboxGradioApp:
    """Create and return a new ChatterboxGradioApp instance."""
    return ChatterboxGradioApp()


if __name__ == "__main__":
    # Create and launch the app directly
    app = create_chatterbox_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860) 