#!/usr/bin/env python3
"""
Fix the missing load project button functionality for multi-voice projects
"""

import re

print("🔧 Fixing Load Project Button Issue...")
print("=" * 50)

# Read the main file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Add the missing load_multi_voice_project function
print("📝 Adding load_multi_voice_project function...")

# Find the position to insert the function (after load_single_voice_project)
function_insertion_pattern = r'(    def load_single_voice_project\(project_name: str\):\s+"""Load project info and update UI fields for single-voice tab\."""[^#]+return text, selected_voice, proj_name, status\s+)'

multi_voice_function = r'''\1
    def load_multi_voice_project(project_name: str):
        """Load project info and update UI fields for multi-voice tab."""
        text, voice_info, proj_name, _, status = load_project_for_regeneration(project_name)
        return text, status

'''

if re.search(function_insertion_pattern, content, re.DOTALL):
    content = re.sub(function_insertion_pattern, multi_voice_function, content, flags=re.DOTALL)
    print("✅ Added load_multi_voice_project function")
else:
    print("❌ Could not find location to insert load_multi_voice_project function")

# Step 2: Add the missing click handler for multi-voice load button
print("📝 Adding load_multi_project_btn click handler...")

# Find where to insert the handler (after the restart handlers)
handler_insertion_pattern = r'(    # Restart multi-voice project button  \s+restart_multi_project_btn\.click\(\s+fn=restart_project_generation,\s+inputs=\[multi_project_dropdown\],\s+outputs=\[multi_project_progress\]\s+\))'

multi_load_handler = r'''\1

    # Load multi-voice project button
    load_multi_project_btn.click(
        fn=load_multi_voice_project,
        inputs=[multi_project_dropdown],
        outputs=[multi_audiobook_text, multi_project_progress]
    )'''

if re.search(handler_insertion_pattern, content, re.DOTALL):
    content = re.sub(handler_insertion_pattern, multi_load_handler, content, flags=re.DOTALL)
    print("✅ Added load_multi_project_btn click handler")
else:
    print("❌ Could not find location to insert load_multi_project_btn handler")

# Step 3: Also need to find the resume multi-voice button handler if it's missing
print("📝 Checking for resume_multi_project_btn handler...")

if "resume_multi_project_btn.click" not in content:
    print("❌ Missing resume_multi_project_btn handler - will add it")
    
    # Add resume handler after load handler
    resume_insertion_pattern = r'(    # Load multi-voice project button\s+load_multi_project_btn\.click\(\s+fn=load_multi_voice_project,\s+inputs=\[multi_project_dropdown\],\s+outputs=\[multi_audiobook_text, multi_project_progress\]\s+\))'
    
    resume_handler = r'''\1

    # Resume multi-voice project button
    resume_multi_project_btn.click(
        fn=lambda model, project_name, voice_library_path: create_multi_voice_audiobook_with_assignments(
            model, "", voice_library_path, project_name, {}, resume=True
        ),
        inputs=[model_state, multi_project_dropdown, voice_library_path_state],
        outputs=[multi_audiobook_output, multi_project_progress]
    )'''
    
    if re.search(resume_insertion_pattern, content, re.DOTALL):
        content = re.sub(resume_insertion_pattern, resume_handler, content, flags=re.DOTALL)
        print("✅ Added resume_multi_project_btn click handler")
    else:
        print("❌ Could not find location to insert resume_multi_project_btn handler")
else:
    print("✅ resume_multi_project_btn handler already exists")

# Write the updated content back
with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n" + "=" * 50)
print("🎉 Load Project Button Fix Complete!")
print("\n📋 What was fixed:")
print("   • Added load_multi_voice_project() function")
print("   • Added load_multi_project_btn click handler")
print("   • Added resume_multi_project_btn handler if missing")
print("\n🚀 The Load Project buttons should now work for both single-voice and multi-voice projects!") 