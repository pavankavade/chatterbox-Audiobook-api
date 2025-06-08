"""
Production Studio Logic

This module contains the backend functions for the Production Studio, including:
- Loading project data (audio chunks and text)
- Assembling audio for full playback
- Rendering the paginated chunk editor UI
- Handling chunk marking and regeneration
"""
import gradio as gr
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Any
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import numpy as np

# Constants
MAX_CHUNKS_ON_PAGE = 25  # Maximum number of chunks that can be displayed on one page

try:
    from .metadata import load_project_metadata
    from .management import get_projects_path, get_existing_projects
    from ..models.tts_model import generate_for_gradio, load_model, CHATTERBOX_AVAILABLE
except ImportError:
    from src.projects.metadata import load_project_metadata
    from src.projects.management import get_projects_path, get_existing_projects
    from src.models.tts_model import generate_for_gradio, load_model, CHATTERBOX_AVAILABLE

def load_studio_project(project_name: str) -> Tuple[Dict[str, Any], str]:
    """
    Loads all necessary data for a project to be edited in the Production Studio.

    Args:
        project_name: The name of the project to load.

    Returns:
        A tuple containing:
        - A dictionary with project data (name, chunks).
        - A status message for the UI.
    """
    if not project_name:
        return None, "<p class='status-error'>❌ Please select a project to load.</p>"
    
    try:
        project_path = Path(get_projects_path()) / project_name
        metadata = load_project_metadata(str(project_path))
        
        if not metadata:
            return None, f"<p class='status-error'>❌ Could not load metadata for '{project_name}'. Project may not exist or have invalid metadata.</p>"

        # Extract just the 'text' from each chunk dictionary
        text_chunks = [chunk.get('text', '') for chunk in metadata.get("chunks", [])]
        audio_files = sorted(project_path.glob("*.wav"))

        if len(text_chunks) != len(audio_files):
            warning_msg = (
                f"⚠️ Mismatch between text chunks ({len(text_chunks)}) and audio files ({len(audio_files)}). "
                "The editor may not function correctly."
            )
            print(warning_msg)
        
        project_data = {
            "name": project_name,
            "path": str(project_path),
            "chunks": [
                {
                    "text": text,
                    "audio": str(audio_files[i]) if i < len(audio_files) else None,
                }
                for i, text in enumerate(text_chunks)
            ]
        }
        
        status_msg = f"<p class='status-success'>✅ Successfully loaded project '{project_name}' with {len(project_data['chunks'])} chunks.</p>"
        return project_data, status_msg

    except Exception as e:
        error_msg = f"<p class='status-error'>❌ An unexpected error occurred while loading project '{project_name}': {e}</p>"
        print(error_msg)
        return None, error_msg

def render_interactive_chunk_editor(project_data: Dict[str, Any], marked_chunks: List[int], chunks_per_page: int, page_num: int) -> List[Any]:
    """
    Generates a list of Gradio updates to populate the interactive chunk editor.
    Now includes individual regenerate buttons for each chunk.

    Args:
        project_data: The dictionary containing all project chunk data.
        marked_chunks: A list of indices of the chunks that are currently marked.
        chunks_per_page: The number of chunks to display on one page.
        page_num: The current page number to display.

    Returns:
        A list of gr.update() values for all the components in the editor.
    """
    # The maximum number of chunk editors we can display on a page.
    # Now 5 components per chunk: Group, Audio, Textbox, Checkbox, Button
    MAX_COMPONENTS = 25 
    
    # Start with a list of updates to hide all components
    outputs = []
    for _ in range(MAX_COMPONENTS):
        outputs.extend([
            gr.update(visible=False),  # Group
            gr.update(value=None),     # Audio
            gr.update(value=""),       # Textbox
            gr.update(value=False),    # Checkbox
            gr.update(visible=False)   # Regen Button
        ])

    if not project_data:
        # No project loaded, so return the list that hides everything
        return outputs

    all_chunks = project_data.get("chunks", [])
    total_chunks = len(all_chunks)
    
    if total_chunks == 0:
        return outputs

    start_index = (page_num - 1) * chunks_per_page
    end_index = start_index + min(chunks_per_page, MAX_COMPONENTS)

    chunks_to_display = all_chunks[start_index:end_index]
    
    for i, chunk_data in enumerate(chunks_to_display):
        if i >= MAX_COMPONENTS:
            break
            
        chunk_index = start_index + i
        
        # Calculate the base index for the `outputs` list (5 components per chunk)
        base_idx = i * 5 
        
        # Update the components to be visible and populated with data
        outputs[base_idx] = gr.update(visible=True)  # Group
        outputs[base_idx + 1] = gr.update(          # Audio
            value=chunk_data.get("audio"),
            label=f"Chunk #{chunk_index + 1}",
            waveform_options=gr.WaveformOptions(
                waveform_color="#017BFF",
                waveform_progress_color="#0056B3",
            )
        )
        outputs[base_idx + 2] = gr.update(          # Textbox
            value=chunk_data.get("text", ""),
            label=f"Text Chunk #{chunk_index + 1}"
        )
        outputs[base_idx + 3] = gr.update(          # Checkbox
            label=f"Mark Chunk {chunk_index + 1}",
            value=(chunk_index in marked_chunks)
        )
        outputs[base_idx + 4] = gr.update(          # Regen Button
            visible=True,
            value=f"🔄 Regen #{chunk_index + 1}"
        )
        
    return outputs

def update_marked_chunks(marked_chunks: List[int], is_checked: bool, global_chunk_index: int) -> List[int]:
    """
    Updates the list of marked chunk indices based on user interaction.

    Args:
        marked_chunks: The current list of marked chunk indices.
        is_checked: The boolean state of the checkbox that was changed.
        global_chunk_index: The global index of the chunk corresponding to the checkbox.

    Returns:
        The updated list of marked chunk indices.
    """
    if is_checked:
        if global_chunk_index not in marked_chunks:
            marked_chunks.append(global_chunk_index)
    else:
        if global_chunk_index in marked_chunks:
            marked_chunks.remove(global_chunk_index)
            
    return sorted(marked_chunks)

def assemble_full_audio(project_data: Dict[str, Any]) -> str:
    """
    Assembles all audio chunks from a project into a single playable audio file.

    Args:
        project_data: The dictionary containing all project chunk data.

    Returns:
        The filepath to the temporary concatenated audio file, or None on failure.
    """
    if not project_data or "chunks" not in project_data:
        print("Assembly failed: No project data provided.")
        return None

    audio_files = [chunk["audio"] for chunk in project_data["chunks"] if chunk.get("audio") and os.path.exists(chunk["audio"])]

    if not audio_files:
        print("Assembly failed: No valid audio files found in project.")
        return None

    try:
        # Concatenate all audio files
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            try:
                segment = AudioSegment.from_wav(audio_file)
                combined_audio += segment
            except Exception as e:
                print(f"Warning: Could not process file {audio_file}, skipping. Error: {e}")
                continue
        
        if len(combined_audio) == 0:
            print("Assembly failed: All audio files were invalid or could not be processed.")
            return None

        # Export to a temporary file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"full_preview_{project_data['name']}.wav"
        
        combined_audio.export(temp_file_path, format="wav")
        
        print(f"Assembled full audio preview for '{project_data['name']}' at {temp_file_path}")
        return str(temp_file_path)

    except Exception as e:
        error_msg = f"An unexpected error occurred during audio assembly: {e}"
        print(error_msg)
        return None

def regenerate_single_chunk(project_data: Dict[str, Any], chunk_index: int, updated_text: str, tts_engine: Any) -> str:
    """
    Regenerates audio for a single chunk with updated text.

    Args:
        project_data: The dictionary containing all project data.
        chunk_index: The index of the chunk to regenerate.
        updated_text: The new text content for the chunk.
        tts_engine: The text-to-speech engine instance.

    Returns:
        A status message for the UI.
    """
    print(f"🔄 DEBUG: Starting regeneration for chunk {chunk_index}")
    print(f"🔄 DEBUG: Text to generate: '{updated_text[:100]}...'")
    print(f"🔄 DEBUG: TTS Engine type: {type(tts_engine)}")
    
    if not project_data or chunk_index < 0:
        return "<p class='status-error'>❌ Invalid chunk data.</p>"
    
    if tts_engine is None:
        print("🔄 DEBUG: TTS engine is None, attempting to load model...")
        if not CHATTERBOX_AVAILABLE:
            return "<p class='status-error'>❌ TTS engine not available. Please check installation.</p>"
        try:
            tts_engine = load_model()
            if tts_engine is None:
                return "<p class='status-error'>❌ Failed to load TTS model. Please restart the application.</p>"
            print("🔄 DEBUG: TTS model loaded successfully")
        except Exception as e:
            return f"<p class='status-error'>❌ Error loading TTS model: {e}</p>"

    all_chunks = project_data.get("chunks", [])
    if chunk_index >= len(all_chunks):
        return f"<p class='status-error'>❌ Chunk index {chunk_index + 1} is out of range.</p>"

    try:
        project_path = Path(project_data["path"])
        print(f"🔄 DEBUG: Project path: {project_path}")
        
        metadata = load_project_metadata(str(project_path))
        print(f"🔄 DEBUG: Loaded metadata: {metadata is not None}")
        if metadata:
            print(f"🔄 DEBUG: Metadata keys: {list(metadata.keys())}")
            print(f"🔄 DEBUG: Project type: {metadata.get('project_type')}")
            if 'chunks' in metadata:
                print(f"🔄 DEBUG: Number of chunks in metadata: {len(metadata['chunks'])}")
                if metadata['chunks']:
                    first_chunk = metadata['chunks'][0]
                    print(f"🔄 DEBUG: First chunk keys: {list(first_chunk.keys())}")
                    print(f"🔄 DEBUG: First chunk content: {first_chunk}")
        
        # --- START REFINED VOICE DETECTION LOGIC ---
        voice_choice = None
        if metadata:
            print(f"🔄 DEBUG: Metadata keys: {list(metadata.keys())}")
            if 'voice_info' in metadata:
                print(f"🔄 DEBUG: Found 'voice_info' key. Contents: {metadata['voice_info']}")

            # Strategy 1: Check the 'voice_info' dictionary (most reliable)
            voice_info = metadata.get('voice_info')
            if isinstance(voice_info, dict):
                # Look for common keys that store the voice name
                voice_choice = voice_info.get('voice_name') or voice_info.get('name') or voice_info.get('voice')
                print(f"🔄 DEBUG: [Strategy 1] Extracted '{voice_choice}' from 'voice_info' dictionary.")

            # Strategy 2: Check for top-level voice keys
            if not voice_choice:
                voice_choice = metadata.get("voice_choice") or metadata.get("voice_name") or metadata.get("voice")
                if voice_choice:
                    print(f"🔄 DEBUG: [Strategy 2] Found top-level voice key: '{voice_choice}'.")

            # Strategy 3: Check voice info inside the first chunk
            if not voice_choice and "chunks" in metadata and metadata["chunks"]:
                first_chunk = metadata["chunks"][0]
                voice_choice = first_chunk.get("voice_name") or first_chunk.get("voice")
                if voice_choice:
                    print(f"🔄 DEBUG: [Strategy 3] Found voice in first chunk: '{voice_choice}'.")
            
            # Strategy 4: Check alternative metadata file (project_info.json)
            if not voice_choice:
                try:
                    project_info_path = project_path / "project_info.json"
                    if project_info_path.exists():
                        print(f"🔄 DEBUG: [Strategy 4] Checking project_info.json...")
                        with open(project_info_path, 'r', encoding='utf-8') as f:
                            project_info = json.load(f)
                        
                        voice_assignments = project_info.get("voice_assignments", {})
                        if voice_assignments:
                            voice_choice = list(voice_assignments.values())[0]
                            print(f"🔄 DEBUG: [Strategy 4] Found voice in voice_assignments: '{voice_choice}'.")
                        elif "chunks" in project_info and project_info["chunks"]:
                            first_chunk = project_info["chunks"][0]
                            voice_choice = first_chunk.get("voice_name") or first_chunk.get("voice")
                            print(f"🔄 DEBUG: [Strategy 4] Found voice in project_info chunks: '{voice_choice}'.")
                except Exception as e:
                    print(f"🔄 DEBUG: [Strategy 4] Could not read project_info.json: {e}")
        
        # Fallback Strategy: Guess from filename
        if not voice_choice and all_chunks:
            first_audio = Path(all_chunks[0]["audio"])
            voice_choice = _detect_voice_from_filename(first_audio.name)
            print(f"🔄 DEBUG: [Fallback] Detected voice from filename: '{voice_choice}'.")
        
        # Final Fallback: Use a default voice
        if not voice_choice:
            voice_choice = "M_Frank"
            print(f"🔄 DEBUG: [Final Fallback] Using default voice: '{voice_choice}'.")
        
        print(f"✅ Final voice choice: {voice_choice}")
        # --- END REFINED VOICE DETECTION LOGIC ---
        
        # Get voice settings with better defaults
        voice_settings = metadata.get("voice_settings", {})
        exaggeration = voice_settings.get("exaggeration", 0.5)
        temperature = voice_settings.get("temperature", 0.5) 
        cfg_weight = voice_settings.get("cfg_weight", 0.5)
        print(f"🔄 DEBUG: Voice settings - exag: {exaggeration}, temp: {temperature}, cfg: {cfg_weight}")
        
        # Find the voice file with multiple search strategies
        voice_file = _find_voice_file(voice_choice)
        print(f"🔄 DEBUG: Looking for voice file: {voice_choice}")
        print(f"🔄 DEBUG: Voice file exists: {voice_file.exists() if voice_file else False}")
        
        if not voice_file or not voice_file.exists():
            return f"<p class='status-error'>❌ Voice file not found for '{voice_choice}'. Available voices are in the speakers folder.</p>"

        chunk_to_update = all_chunks[chunk_index]
        audio_path = Path(chunk_to_update["audio"])
        print(f"🔄 DEBUG: Audio path to overwrite: {audio_path}")

        print(f"🔄 DEBUG: Calling TTS engine...")

        # Call the TTS engine
        result = generate_for_gradio(
            tts_engine, updated_text, str(voice_file), 
            exaggeration, temperature, 0, cfg_weight
        )
        
        print(f"🔄 DEBUG: TTS result: {result is not None}")
        
        if result is None:
            return f"<p class='status-error'>❌ Failed to generate audio for chunk {chunk_index + 1}.</p>"
        
        sample_rate, audio_array = result
        print(f"🔄 DEBUG: Generated audio - sample_rate: {sample_rate}, array_shape: {audio_array.shape}")
        
        # Save the new audio file
        print(f"🔄 DEBUG: Saving audio to: {audio_path}")
        wavfile.write(str(audio_path), sample_rate, audio_array.astype(np.int16))
        print(f"🔄 DEBUG: Audio file saved successfully")
        
        # Update the project metadata with new text AND voice info if missing
        if "chunks" not in metadata:
            metadata["chunks"] = []
        while len(metadata["chunks"]) <= chunk_index:
            metadata["chunks"].append({})
        
        metadata["chunks"][chunk_index]["text"] = updated_text
        
        # Save voice choice for future regenerations if not already stored
        if not metadata.get("voice_choice"):
            metadata["voice_choice"] = voice_choice
            metadata["voice_settings"] = voice_settings
        
        metadata_path = project_path / "metadata.json"
        print(f"🔄 DEBUG: Updating metadata at: {metadata_path}")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"🔄 DEBUG: Metadata updated successfully")
        
        print(f"✅ Successfully regenerated chunk {chunk_index + 1}")
        return f"<p class='status-success'>✅ Successfully regenerated chunk {chunk_index + 1}.</p>"

    except Exception as e:
        error_msg = f"<p class='status-error'>❌ Error regenerating chunk {chunk_index + 1}: {e}</p>"
        print(f"🔄 DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        return error_msg

def _detect_voice_from_filename(filename: str) -> str:
    """
    Try to detect the voice name from an audio filename.
    Most audio files follow pattern: projectname_chunknumber.wav
    But sometimes they might include voice info.
    """
    # Remove common suffixes and chunk numbers
    base_name = filename.replace('.wav', '').replace('.mp3', '')
    
    # If filename contains known voice patterns, extract them
    common_voices = ["M_Frank", "M_Sam", "M_Andrew", "Frank", "Sam", "Andrew", 
                    "F_Elizabeth", "Elizabeth", "Natalie", "Rose"]
    
    for voice in common_voices:
        if voice.lower() in base_name.lower():
            return voice
    
    # Default fallback
    return "M_Frank"

def _find_voice_file(voice_choice: str) -> Path:
    """
    Find the voice file using multiple search strategies.
    """
    # Strategy 1: Check if it's a directory-based voice profile (subfolder format)
    speakers_path = Path(get_projects_path()).parent / "speakers"
    voice_dir = speakers_path / voice_choice
    if voice_dir.is_dir():
        # Look for reference.wav in the voice directory
        reference_file = voice_dir / "reference.wav"
        if reference_file.exists():
            return reference_file
        # Also try other common audio file names in the directory
        for audio_file in voice_dir.glob("*.wav"):
            return audio_file  # Return the first wav file found
    
    # Strategy 2: Standard speakers directory (single file format)
    voice_file = speakers_path / f"{voice_choice}.wav"
    if voice_file.exists():
        return voice_file
    
    # Strategy 3: Relative speakers directory
    alt_voice_file = Path("speakers") / f"{voice_choice}.wav"
    if alt_voice_file.exists():
        return alt_voice_file
    
    # Strategy 4: Try without .wav extension in case it's included
    if voice_choice.endswith('.wav'):
        clean_name = voice_choice[:-4]
        voice_file = speakers_path / f"{clean_name}.wav"
        if voice_file.exists():
            return voice_file
        # Also check directory format for the clean name
        voice_dir = speakers_path / clean_name
        if voice_dir.is_dir():
            reference_file = voice_dir / "reference.wav"
            if reference_file.exists():
                return reference_file
    
    # Strategy 5: Case insensitive search in speakers directory (both files and directories)
    if speakers_path.exists():
        # Check for directory matches first
        for item in speakers_path.iterdir():
            if item.is_dir() and item.name.lower() == voice_choice.lower():
                reference_file = item / "reference.wav"
                if reference_file.exists():
                    return reference_file
                # Return first wav file in the directory
                for audio_file in item.glob("*.wav"):
                    return audio_file
        
        # Then check for file matches
        for file in speakers_path.glob("*.wav"):
            if file.stem.lower() == voice_choice.lower():
                return file
    
    # Strategy 6: Try common voice name transformations
    if speakers_path.exists():
        # Try adding common prefixes if the voice name doesn't have them
        common_prefixes = ["M_", "F_", ""]
        for prefix in common_prefixes:
            for file in speakers_path.glob("*.wav"):
                # Check if the voice name matches without prefix
                if file.stem.lower() == f"{prefix}{voice_choice}".lower():
                    return file
                # Also check if removing the prefix from filename matches
                if file.stem.lower().startswith(prefix.lower()) and file.stem[len(prefix):].lower() == voice_choice.lower():
                    return file
            
            # Also check directories with prefixes
            for item in speakers_path.iterdir():
                if item.is_dir() and item.name.lower() == f"{prefix}{voice_choice}".lower():
                    reference_file = item / "reference.wav"
                    if reference_file.exists():
                        return reference_file
    
    # Strategy 7: Absolute path from working directory
    abs_speakers_path = Path.cwd() / "speakers"
    if abs_speakers_path.exists() and abs_speakers_path != speakers_path:
        # Check directory format first
        voice_dir = abs_speakers_path / voice_choice
        if voice_dir.is_dir():
            reference_file = voice_dir / "reference.wav"
            if reference_file.exists():
                return reference_file
        
        voice_file = abs_speakers_path / f"{voice_choice}.wav"
        if voice_file.exists():
            return voice_file
        
        # Case insensitive search in absolute path
        for file in abs_speakers_path.glob("*.wav"):
            if file.stem.lower() == voice_choice.lower():
                return file
    
    print(f"🔄 DEBUG: Searched for voice '{voice_choice}' in:")
    print(f"  - Directory format: {speakers_path / voice_choice}")
    print(f"  - File format: {speakers_path / f'{voice_choice}.wav'}")
    if speakers_path.exists():
        available_dirs = [d.name for d in speakers_path.iterdir() if d.is_dir()]
        available_files = [f.stem for f in speakers_path.glob("*.wav")]
        print(f"  - Available voice directories: {available_dirs[:10]}...")  # Show first 10
        print(f"  - Available voice files: {available_files[:10]}...")  # Show first 10
    
    # Return the primary path even if it doesn't exist (for error reporting)
    return speakers_path / f"{voice_choice}.wav"

def regenerate_marked_chunks(project_data: Dict[str, Any], marked_chunks: List[int], tts_engine: Any) -> str:
    """
    Regenerates the audio for the selected (marked) chunks.

    Args:
        project_data: The dictionary containing all project data.
        marked_chunks: A list of indices of the chunks to regenerate.
        tts_engine: The text-to-speech engine instance.

    Returns:
        A status message for the UI.
    """
    if not project_data or not marked_chunks:
        return "<p class='status-warning'>⚠️ No chunks marked for regeneration.</p>"

    if tts_engine is None:
        print("🔄 DEBUG: TTS engine is None, attempting to load model...")
        if not CHATTERBOX_AVAILABLE:
            return "<p class='status-error'>❌ TTS engine not available. Please check installation.</p>"
        try:
            tts_engine = load_model()
            if tts_engine is None:
                return "<p class='status-error'>❌ Failed to load TTS model. Please restart the application.</p>"
            print("🔄 DEBUG: TTS model loaded successfully")
        except Exception as e:
            return f"<p class='status-error'>❌ Error loading TTS model: {e}</p>"

    try:
        project_path = Path(project_data["path"])
        metadata = load_project_metadata(str(project_path))
        voice_choice = metadata.get("voice_choice", "default_voice")
        
        # Get voice settings
        voice_settings = metadata.get("voice_settings", {})
        exaggeration = voice_settings.get("exaggeration", 0.5)
        temperature = voice_settings.get("temperature", 0.5)
        cfg_weight = voice_settings.get("cfg_weight", 0.5)
        
        # Find the voice file
        speakers_path = Path(get_projects_path()).parent / "speakers"
        voice_file = speakers_path / f"{voice_choice}.wav"
        
        if not voice_file.exists():
            return f"<p class='status-error'>❌ Voice file not found: {voice_file}</p>"

        all_chunks = project_data["chunks"]
        regenerated_count = 0

        for chunk_index in marked_chunks:
            if 0 <= chunk_index < len(all_chunks):
                chunk_to_update = all_chunks[chunk_index]
                text_to_generate = chunk_to_update["text"]
                audio_path = Path(chunk_to_update["audio"])

                print(f"Regenerating marked chunk {chunk_index + 1}: {audio_path.name}...")

                # Call the TTS engine
                result = generate_for_gradio(
                    tts_engine, text_to_generate, str(voice_file),
                    exaggeration, temperature, 0, cfg_weight
                )
                
                if result is not None:
                    sample_rate, audio_array = result
                    wavfile.write(str(audio_path), sample_rate, audio_array.astype(np.int16))
                    regenerated_count += 1
                    print(f"✅ Successfully regenerated chunk {chunk_index + 1}")
                else:
                    print(f"❌ Failed to regenerate chunk {chunk_index + 1}")
        
        if regenerated_count > 0:
            return f"<p class='status-success'>✅ Successfully regenerated {regenerated_count} chunk(s).</p>"
        else:
            return "<p class='status-warning'>⚠️ No chunks were regenerated. Please check the logs.</p>"

    except Exception as e:
        error_msg = f"<p class='status-error'>❌ An error occurred during regeneration: {e}</p>"
        print(error_msg)
        return error_msg