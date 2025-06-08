#!/usr/bin/env python3
"""
Chatterbox Audiobook Studio - App Launcher

This is the main launcher for the refactored Chatterbox Audiobook Studio.
It creates and launches the Gradio web interface.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.gradio_interface import ChatterboxGradioApp

def main():
    """Main entry point for launching the Chatterbox Audiobook Studio."""
    print("🎧 Starting Chatterbox Audiobook Studio...")
    print("Creating Gradio interface...")
    
    try:
        # Create the application
        app = ChatterboxGradioApp()
        
        print("✅ Application created successfully!")
        print("🚀 Launching web interface...")
        
        # Launch the interface
        app.launch(
            share=False,
            server_name="0.0.0.0", 
            server_port=7860
        )
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 