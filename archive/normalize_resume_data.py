#!/usr/bin/env python3
"""
Normalizes voice_assignments in resume_multi_voice_project_data.
"""
import re

print("🚀 Normalizing voice_assignments in resume_multi_voice_project_data...")

main_file_path = "gradio_tts_app_audiobook.py"

# --- Code to be replaced --- (The section that loads and initially checks voice_info)
old_code_section = r"""    text_content = metadata.get\("text_content"\)
    # Voice assignments are stored within voice_info for multi-voice projects
    voice_assignments = metadata.get\("voice_info"\) 

    if not text_content or not voice_assignments:
        missing_items = \[\]
        if not text_content: missing_items.append\("text content"\)
        if not voice_assignments: missing_items.append\("voice assignments"\)
        return None, None, f"❌ Project '{project_name}' is missing: \{', '.join\(missing_items\)\} in its metadata."
    
    if metadata.get\("project_type"\) != "multi_voice":
        return None, None, f"❌ Project '{project_name}' is not a multi-voice project. Cannot resume with multi-voice settings."

    return text_content, voice_assignments, f"✅ Data loaded for '{project_name}'""" 
# Note the space after voice_assignments = metadata.get("voice_info") # Important for exact match

# --- New code section with normalization logic --- 
new_code_section = """    text_content = metadata.get("text_content")
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
                # Use 'display_name' as the canonical voice profile name string.
                # This assumes 'display_name' was reliably the key/identifier for the voice profile.
                profile_name = voice_data.get('display_name')
                if profile_name:
                    normalized_voice_assignments[char_name] = profile_name
                else:
                    # If display_name is missing, this is a more problematic legacy format.
                    # For now, we have to signal this clearly as it will likely fail later.
                    print(f"⚠️ CRITICAL: Could not determine voice profile name for character '{char_name}' from legacy voice_info dict: {voice_data}. Assigning a placeholder that WILL LIKELY FAIL.")
                    normalized_voice_assignments[char_name] = f"UNKNOWN_PROFILE_FOR_{char_name}"
            else:
                # Unexpected format for voice_data for a character
                return None, None, f"❌ Unexpected voice data format for character '{char_name}' in project '{project_name}' (expected str or dict, got {type(voice_data)})."
    else:
        return None, None, f"❌ Voice assignments (voice_info) in project '{project_name}' is not a dictionary as expected (got {type(loaded_voice_info)})."

    if not normalized_voice_assignments:
        # This case should ideally be caught by earlier checks if loaded_voice_info was empty or not a dict.
        return None, None, f"❌ Could not derive any valid voice assignments for project '{project_name}'. Check metadata integrity."

    return text_content, normalized_voice_assignments, f"✅ Data loaded for '{project_name}'"""

with open(main_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Perform the replacement
# Ensure the regex matches the exact old code block including indentation and newlines
# Need to escape special regex characters in old_code_section
escaped_old_code = re.escape(old_code_section)
# And then replace the escaped newlines with \n for the regex engine
escaped_old_code = escaped_old_code.replace(re.escape("\n"), "\\n")
# This approach with re.escape for the whole block is often problematic for multiline. 
# A more direct string replace might be safer if the block is unique enough.

# Let's try direct string replacement if the block is highly specific:
if old_code_section.strip() in content: # Looser check for presence before attempting direct replace
    modified_content = content.replace(old_code_section, new_code_section)
    if modified_content != content:
        with open(main_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"✅ Successfully normalized voice_assignments logic in {main_file_path}.")
    else:
        print("❌ Code section for voice_assignments not found or not replaced (direct replace). Manually check `resume_multi_voice_project_data`.")
else:
    print("❌ Old code section not found for direct replacement. Please check the script or apply manually.")

print("�� Script finished.") 