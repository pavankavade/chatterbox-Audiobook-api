#!/usr/bin/env python3
"""
Debug script to test load project button functionality
"""

import gradio as gr
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the functions we need
from gradio_tts_app_audiobook import (
    get_project_choices, 
    load_single_voice_project,
    get_audiobook_voice_choices
)

SAVED_VOICE_LIBRARY_PATH = "voice_library"

def test_load_project_function():
    """Test the load project function directly"""
    print("Testing load project function...")
    
    # Get available projects
    projects = get_project_choices()
    print(f"Found {len(projects)} projects")
    
    if not projects:
        return "❌ No projects found to test with"
    
    # Test with first project (extract project name from tuple)
    test_project_name = projects[0][1] if len(projects[0]) > 1 else projects[0]
    print(f"Testing with project: {test_project_name}")
    
    try:
        # Call the load function
        result = load_single_voice_project(test_project_name)
        text, voice, proj_name, status = result
        
        return f"""✅ Load Project Test Results:
        📄 Text Length: {len(text) if text else 0} characters
        🎭 Voice: {voice if voice else 'None'}
        📁 Project: {proj_name if proj_name else 'None'}
        📊 Status: {status if status else 'None'}
        """
    except Exception as e:
        return f"❌ Error loading project: {str(e)}"

def create_test_interface():
    """Create a simplified test interface for the load project functionality"""
    
    with gr.Blocks(title="Load Project Debug Test") as demo:
        gr.HTML("<h1>🔧 Load Project Debug Test</h1>")
        
        # Test the function directly
        test_result = gr.HTML()
        
        test_btn = gr.Button("🧪 Test Load Function", variant="primary")
        test_btn.click(
            fn=test_load_project_function,
            inputs=[],
            outputs=[test_result]
        )
        
        gr.HTML("<hr>")
        
        # Now test the actual UI components
        gr.HTML("<h2>📋 UI Component Test</h2>")
        
        project_dropdown = gr.Dropdown(
            choices=get_project_choices(),
            label="Select Project",
            value=None
        )
        
        load_btn = gr.Button("📂 Load Project", variant="secondary")
        
        # Output components
        output_text = gr.Textbox(label="Loaded Text", lines=5)
        output_voice = gr.Textbox(label="Voice Info")
        output_project = gr.Textbox(label="Project Name")
        output_status = gr.HTML(label="Status")
        
        # Wire up the load button
        load_btn.click(
            fn=load_single_voice_project,
            inputs=[project_dropdown],
            outputs=[output_text, output_voice, output_project, output_status]
        )
        
        # Add refresh button
        refresh_btn = gr.Button("🔄 Refresh Projects", size="sm")
        refresh_btn.click(
            fn=lambda: gr.Dropdown(choices=get_project_choices()),
            inputs=[],
            outputs=[project_dropdown]
        )
    
    return demo

if __name__ == "__main__":
    print("🔧 Starting Load Project Debug Test...")
    demo = create_test_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False) 