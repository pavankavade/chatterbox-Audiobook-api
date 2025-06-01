#!/usr/bin/env python3
"""
Adds the resume_multi_voice_project_data helper function to gradio_tts_app_audiobook.py
"""
import re

print("🚀 Adding resume_multi_voice_project_data function...")

main_file_path = "gradio_tts_app_audiobook.py"

new_function_code = """
def resume_multi_voice_project_data(project_name: str) -> tuple:
    \"\"\"Load text_content and voice_assignments from a multi-voice project's metadata for resuming.\"\"\"
    if not project_name:
        return None, None, "❌ No project selected for resuming data load."

    # Need to import os and json if not already globally available in this scope
    import os
    import json

    projects_dir = "audiobook_projects"
    project_path = os.path.join(projects_dir, project_name)
    metadata_path = os.path.join(project_path, "project_metadata.json")

    if not os.path.exists(metadata_path):
        return None, None, f"❌ Metadata file not found for project '{project_name}'."

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return None, None, f"❌ Error loading metadata for '{project_name}': {str(e)}"

    text_content = metadata.get("text_content")
    # Voice assignments are stored within voice_info for multi-voice projects
    voice_assignments = metadata.get("voice_info") 

    if not text_content or not voice_assignments:
        missing_items = []
        if not text_content: missing_items.append("text content")
        if not voice_assignments: missing_items.append("voice assignments")
        return None, None, f"❌ Project '{project_name}' is missing: {', '.join(missing_items)} in its metadata."
    
    if metadata.get("project_type") != "multi_voice":
        return None, None, f"❌ Project '{project_name}' is not a multi-voice project. Cannot resume with multi-voice settings."

    return text_content, voice_assignments, f"✅ Data loaded for '{project_name}'"

"""

# Anchor pattern to insert before: the definition of create_continuous_playback_audio
anchor_pattern = r"def create_continuous_playback_audio\(project_name: str\) -> tuple:"

with open(main_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if re.search(anchor_pattern, content):
    # Insert the new function code before the anchor pattern
    modified_content = re.sub(anchor_pattern, new_function_code + "\n\n" + anchor_pattern, content, 1)
    
    with open(main_file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    print(f"✅ Successfully added resume_multi_voice_project_data function to {main_file_path}")
else:
    print(f"❌ Anchor pattern not found in {main_file_path}. Function not added.")

print("�� Script finished.") 