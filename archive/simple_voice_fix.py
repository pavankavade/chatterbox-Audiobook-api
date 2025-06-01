#!/usr/bin/env python3
"""
Simple fix for voice_assignments normalization issue.
"""

print("🚀 Applying simple voice_assignments fix...")

# Read the file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace just the voice_assignments line with normalization logic
lines = content.split('\n')

# Find the resume function
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'def resume_multi_voice_project_data(project_name: str) -> tuple:' in line:
        start_idx = i
    elif start_idx is not None and line.strip() and not line.startswith('    ') and not line.startswith('\t'):
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    # Replace the function with the fixed version
    new_function = '''def resume_multi_voice_project_data(project_name: str) -> tuple:
    """Load text_content and voice_assignments from a multi-voice project's metadata for resuming."""
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
    loaded_voice_info = metadata.get("voice_info")

    if not text_content or not loaded_voice_info:
        missing_items = []
        if not text_content: missing_items.append("text content")
        if not loaded_voice_info: missing_items.append("voice assignments (voice_info)")
        return None, None, f"❌ Project '{project_name}' is missing: {', '.join(missing_items)} in its metadata."
    
    if metadata.get("project_type") != "multi_voice":
        return None, None, f"❌ Project '{project_name}' is not a multi-voice project. Cannot resume with multi-voice settings."

    # Normalize voice_assignments to be Dict[char_name, voice_profile_name_str]
    normalized_voice_assignments = {}
    if isinstance(loaded_voice_info, dict):
        for char_name, voice_data in loaded_voice_info.items():
            if isinstance(voice_data, str):
                # Already in the desired format (voice_profile_name_str)
                normalized_voice_assignments[char_name] = voice_data
            elif isinstance(voice_data, dict):
                # Legacy format: voice_data is a dictionary of details
                # Use 'display_name' as the canonical voice profile name string
                profile_name = voice_data.get('display_name')
                if profile_name:
                    normalized_voice_assignments[char_name] = profile_name
                else:
                    # If display_name is missing, try 'name' field as fallback
                    profile_name = voice_data.get('name')
                    if profile_name:
                        normalized_voice_assignments[char_name] = profile_name
                    else:
                        print(f"⚠️ CRITICAL: Could not determine voice profile name for character '{char_name}' from voice_info dict: {voice_data}")
                        normalized_voice_assignments[char_name] = f"UNKNOWN_PROFILE_FOR_{char_name}"
            else:
                # Unexpected format for voice_data for a character
                return None, None, f"❌ Unexpected voice data format for character '{char_name}' in project '{project_name}' (expected str or dict, got {type(voice_data)})."
    else:
        return None, None, f"❌ Voice assignments (voice_info) in project '{project_name}' is not a dictionary as expected (got {type(loaded_voice_info)})."

    if not normalized_voice_assignments:
        return None, None, f"❌ Could not derive any valid voice assignments for project '{project_name}'. Check metadata integrity."

    return text_content, normalized_voice_assignments, f"✅ Data loaded for '{project_name}'"'''

    # Replace the function
    new_lines = lines[:start_idx] + new_function.split('\n') + lines[end_idx:]
    
    # Write back to file
    with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Successfully fixed voice_assignments normalization!")
else:
    print("❌ Could not find the resume_multi_voice_project_data function.")

print("�� Script finished.") 