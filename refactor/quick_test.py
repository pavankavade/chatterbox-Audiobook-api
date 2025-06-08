#!/usr/bin/env python3
"""
Quick test to verify the interface works with legacy voices
"""

import gradio as gr
from src.voice_library.voice_management import get_voice_choices_organized
from src.config.settings import config

def test_interface():
    """Create a simple test interface to verify voice loading works"""
    print("🔍 Testing voice loading...")
    
    # Get voices using the same method as the main app
    voice_library_path = config.get_voices_path()
    print(f"Using voice library path: {voice_library_path}")
    
    voices = get_voice_choices_organized(voice_library_path)
    print(f"Found {len(voices)} voices")
    print("First 10 voices:", voices[:10])
    
    # Create a simple interface
    with gr.Blocks(title="Voice Test") as demo:
        gr.Markdown("# 🎤 Voice Loading Test")
        
        with gr.Row():
            voice_dropdown = gr.Dropdown(
                choices=voices,
                label=f"Select Voice ({len(voices)} available)",
                value=voices[0] if voices else None
            )
            
            refresh_btn = gr.Button("🔄 Refresh")
        
        status = gr.HTML(f"<p>✅ Loaded {len(voices)} voices successfully!</p>")
        
        def refresh_voices():
            new_voices = get_voice_choices_organized(voice_library_path)
            return (
                gr.Dropdown(choices=new_voices, value=new_voices[0] if new_voices else None),
                f"<p>✅ Refreshed: {len(new_voices)} voices loaded</p>"
            )
        
        refresh_btn.click(
            refresh_voices,
            outputs=[voice_dropdown, status]
        )
    
    print("✅ Interface created successfully!")
    return demo

if __name__ == "__main__":
    print("🚀 Starting voice loading test...")
    demo = test_interface()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False) 