#!/usr/bin/env python3
"""
Debug script to test voice loading in the Gradio interface
"""

import gradio as gr
from src.voice_library.voice_management import get_voice_choices_organized

def test_voice_loading():
    """Test voice loading functions"""
    print("🔍 Testing voice loading...")
    
    # Test basic voice loading
    voices = get_voice_choices_organized('../speakers')
    print(f"Found {len(voices)} voices from function")
    
    # Test Gradio dropdown creation
    dropdown = gr.Dropdown(choices=voices, label="Test Voices")
    print(f"Created dropdown with {len(voices)} choices")
    
    return voices, dropdown

def create_test_interface():
    """Create a simple test interface to verify voice loading"""
    print("🎮 Creating test interface...")
    
    voices = get_voice_choices_organized('../speakers')
    print(f"Loading {len(voices)} voices into interface")
    
    with gr.Blocks(title="Voice Loading Test") as interface:
        gr.Markdown("# Voice Loading Test")
        
        voice_dropdown = gr.Dropdown(
            choices=voices,
            label="Select Voice",
            value=voices[0] if voices else None
        )
        
        refresh_btn = gr.Button("Refresh Voices")
        status = gr.HTML()
        
        def refresh_voices():
            new_voices = get_voice_choices_organized('../speakers')
            return gr.Dropdown(choices=new_voices), f"Refreshed: {len(new_voices)} voices found"
        
        refresh_btn.click(
            refresh_voices,
            outputs=[voice_dropdown, status]
        )
    
    return interface

if __name__ == "__main__":
    print("🧪 Running voice loading tests...")
    
    # Test 1: Basic loading
    voices, dropdown = test_voice_loading()
    print(f"✅ Basic test: {len(voices)} voices")
    
    # Test 2: Interface creation
    interface = create_test_interface()
    print("✅ Interface created")
    
    # Launch test interface
    print("🚀 Launching test interface...")
    interface.launch(server_name="0.0.0.0", server_port=7861, share=False) 