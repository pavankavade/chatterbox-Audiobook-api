"""# Chatterbox TTS - Audiobook Edition 🎧📚

Welcome to the Audiobook Edition of Chatterbox TTS! This project extends the powerful open-source Text-to-Speech and Voice Conversion capabilities of Resemble AI's Chatterbox, specifically tailored for creating high-quality audiobooks with consistent character voices and robust regeneration features.

This version is designed to run locally on your machine, giving you full control over your audiobook production workflow.

## ✨ Key Features

*   **🎤 Advanced Voice Cloning:** Clone voices from short audio samples to narrate your audiobooks.
*   **🎭 Multi-Voice Management:**
    *   Create and manage a library of distinct voice profiles.
    *   Assign voices to different characters in your script using simple tags (e.g., `[character_name]`).
    *   Generate audiobooks with multiple, consistent character voices.
*   **📖 Audiobook Creation Studio:**
    *   **Single-Voice Mode:** Generate entire audiobooks using a single, consistent cloned voice.
    *   **Multi-Voice Mode:** Process scripts with embedded character tags to produce dynamic multi-character narrations.
    *   **Smart Text Chunking:** Text is automatically split into manageable chunks for efficient processing and regeneration.
*   **🔄 Project & Chunk Regeneration:**
    *   Save and load audiobook projects.
    *   View individual audio chunks with their corresponding text and voice information.
    *   **Visual Audio Trimming:** Interactively trim silence or unwanted parts from audio chunks directly in the UI. Saved trims are reflected in the final download.
    *   Regenerate individual audio chunks if corrections are needed, without re-processing the entire book.
    *   Accept or decline regenerated chunks.
*   **📁 Local Project Management:** All projects, audio files, and metadata are stored locally on your machine.
*   **🚀 GPU Accelerated (with CPU Fallback):** Utilizes NVIDIA GPUs for fast processing, with automatic fallback to CPU if CUDA errors occur or if a GPU is not available.
*   **⚙️ Customizable Settings:** Fine-tune voice characteristics like exaggeration, pace (CFG weight), and temperature.

## 🛠️ Setup & Installation

This project is designed to be run from the `chatterbox-audiobook` directory, which contains this README and the main launch scripts. The core Chatterbox application and its virtual environment are located within the `chatterbox-audiobook/chatterbox/` subdirectory.

**Prerequisites:**

*   **Python:** Version 3.10 or higher recommended.
*   **NVIDIA GPU:** Strongly recommended for performance (CUDA support needed). The scripts are configured for CUDA 12.1.
*   **Git:** For cloning the repository and managing updates.

**Installation Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd chatterbox-audiobook
    ```

2.  **Run the Installation Script:**
    Navigate into the `chatterbox` subdirectory and run the installer:
    ```bash
    cd chatterbox
    install-audiobook.bat 
    ```
    This script will:
    *   Check your Python version.
    *   Create a Python virtual environment (`venv`) inside the `chatterbox` directory.
    *   Activate the virtual environment.
    *   Install specific versions of PyTorch, Torchvision, and Torchaudio compatible with CUDA 12.1.
    *   Install Chatterbox TTS and other necessary dependencies (like Gradio and NumPy).
    *   Test the installation.

    *If you encounter issues related to PyTorch or Torchvision after installation or an update, you can try running the `quick_fix.bat` script located in the parent `chatterbox-audiobook` directory. First, navigate back to `chatterbox-audiobook` then run `quick_fix.bat`.*
    ```bash
    cd .. 
    quick_fix.bat
    ```
    Then, try running `install-audiobook.bat` again from the `chatterbox` directory if problems persist.

## 🚀 Launching the Application

Once installation is complete, you can launch the Chatterbox Audiobook Edition application using the provided batch script from the **root `chatterbox-audiobook` directory**:

```bash
launch_chatterbox.bat
```

This script will:
*   Activate the correct virtual environment.
*   Set necessary environment variables (like `CUDA_LAUNCH_BLOCKING=1` for better error reporting).
*   Navigate to the `chatterbox` subdirectory.
*   Start the Gradio web interface.

You can then access the application through the local URL provided in your terminal (usually `http://127.0.0.1:7860`).

## 📁 Project Structure

*   `chatterbox-audiobook/` (Root directory)
    *   `launch_chatterbox.bat`: Main script to launch the application.
    *   `quick_fix.bat`: Script to resolve common PyTorch/Torchvision compatibility issues.
    *   `README.md`: This file.
    *   `chatterbox/`: Contains the core Chatterbox application, virtual environment, and specific project files.
        *   `install-audiobook.bat`: Installation script for dependencies.
        *   `gradio_tts_app_audiobook.py`: The main Gradio application file for the audiobook edition.
        *   `src/`: Source code for the Chatterbox library.
        *   `venv/`: Python virtual environment (created by the install script).
        *   `audiobook_projects/`: Default directory where your audiobook projects (audio chunks, metadata) will be saved.
        *   `voice_library/`: Default directory where your cloned voice profiles will be stored.
        *   `pyproject.toml`: Defines project dependencies for Chatterbox.
        *   ... (other Chatterbox core files)

## 🎤 Workflow Overview

1.  **Manage Voices (📚 Voice Library Tab):**
    *   Create new voice profiles by providing a name, display name, description, and a reference audio file.
    *   Adjust and test voice settings (exaggeration, CFG/pace, temperature).
    *   Save voice profiles to your library.

2.  **Create Audiobooks:**
    *   **Single-Voice (📖 Audiobook Creation - Single Sample Tab):**
        *   Load or paste your text.
        *   Select a voice from your library.
        *   Set a project name.
        *   Validate and create the audiobook. Audio chunks will be saved in `audiobook_projects/your_project_name/`.
    *   **Multi-Voice (🎭 Audiobook Creation - Multi-Sample Tab):**
        *   Prepare your script with character tags, e.g., `[Bilbo] In a hole in the ground there lived a hobbit. [Gandalf] Not a nasty, dirty, wet hole...`
        *   Load or paste the tagged text.
        *   Analyze the text to identify characters.
        *   Assign a voice from your library to each identified character.
        *   Set a project name.
        *   Validate and create the audiobook. Chunks will be saved with character identifiers.

3.  **Regenerate & Refine (🔄 Regenerate Sample Tab):**
    *   Load an existing project.
    *   View all audio chunks with their text and voice info.
    *   Listen to individual chunks.
    *   **Trim Audio:** Use the visual waveform editor to select a portion of an audio chunk. Click "💾 Save Trimmed Chunk" to apply the trim. The original is backed up.
    *   Edit the text for a specific chunk if needed.
    *   Regenerate the selected chunk using its original voice settings.
    *   Compare the original and regenerated audio.
    *   Accept the regenerated version (replaces the original chunk, backs up the previous) or decline it.
    *   Download the full, combined project audio (includes any saved trims).

## 💡 Tips & Best Practices

*   **Clear Reference Audio:** For voice cloning, use 10-30 seconds of clear speech with minimal background noise for the best results.
*   **Consistent Environment:** Ensure your recording environment for reference audio is consistent if you plan to add more samples to a voice later.
*   **Experiment with Settings:** Different voices may benefit from different exaggeration, pace, and temperature settings. Test them out in the Voice Library.
*   **Save Projects Regularly:** Your work is saved locally.
*   **Backup:** Consider backing up your `audiobook_projects/` and `voice_library/` directories periodically.
*   **CUDA Errors:** If you encounter CUDA errors, ensure your NVIDIA drivers are up-to-date. The `CUDA_LAUNCH_BLOCKING=1` setting in `launch_chatterbox.bat` helps provide more specific error messages. The `quick_fix.bat` can resolve common PyTorch/Torchvision version conflicts.

## 🤝 Contributing

This project builds upon the open-source Chatterbox TTS. Contributions to improve the audiobook-specific features are welcome. Please refer to the main Chatterbox repository for general contribution guidelines. For this edition, consider enhancements to:

*   UI/UX for audiobook workflow.
*   Advanced batch processing features.
*   Metadata management.

## 📝 License

This project inherits the license from the core Chatterbox TTS project. Please see the `LICENSE` file in the `chatterbox/` subdirectory.

---

Happy Audiobook Crafting! 🎉
"" 