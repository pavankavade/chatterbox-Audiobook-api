#!/usr/bin/env python3
"""
Fix voice_assignments normalization in resume_multi_voice_project_data function.
"""

print("🚀 Fixing voice_assignments normalization...")

main_file_path = "gradio_tts_app_audiobook.py"

# Read the file
with open(main_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic section (exact text from the file)
old_section = """    text_content = metadata.get("text_content")
    # Voice assignments are stored within voice_info for multi-voice projects
    voice_assignments = metadata.get("voice_info") 

    if not text_content or not voice_assignments:
        missing_items = []
        if not text_content: missing_items.append("text content")
        if not voice_assignments: missing_items.append("voice assignments")
        return None, None, f"❌ Project '{project_name}' is missing: {', '.join(missing_items)} in its metadata."
    
    if metadata.get("project_type") != "multi_voice":
        return None, None, f"❌ Project '{project_name}' is not a multi-voice project. Cannot resume with multi-voice settings."

    return text_content, voice_assignments, f"✅ Data loaded for '{project_name}'" """

new_section = """    text_content = metadata.get("text_content")
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

    return text_content, normalized_voice_assignments, f"✅ Data loaded for '{project_name}'" """

# Perform replacement
if old_section in content:
    modified_content = content.replace(old_section, new_section)
    with open(main_file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    print("✅ Successfully fixed voice_assignments normalization!")
else:
    print("❌ Could not find the exact section to replace. Check the function manually.")

print("�� Script finished.") 