#!/usr/bin/env python3
"""
Fix voice normalization to extract voice_name instead of display_name.
"""

print("🚀 Fixing voice_name extraction in normalization...")

# Read the file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic section
old_section = """                # Legacy format: voice_data is a dictionary of details
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
                        normalized_voice_assignments[char_name] = f"UNKNOWN_PROFILE_FOR_{char_name}" """

new_section = """                # Legacy format: voice_data is a dictionary of details
                # Use 'voice_name' as the canonical voice profile name string (this is the folder name)
                profile_name = voice_data.get('voice_name')
                if profile_name:
                    normalized_voice_assignments[char_name] = profile_name
                else:
                    # If voice_name is missing, try 'display_name' as fallback (though less reliable)
                    profile_name = voice_data.get('display_name')
                    if profile_name:
                        normalized_voice_assignments[char_name] = profile_name
                    else:
                        # Final fallback to 'name' field
                        profile_name = voice_data.get('name')
                        if profile_name:
                            normalized_voice_assignments[char_name] = profile_name
                        else:
                            print(f"⚠️ CRITICAL: Could not determine voice profile name for character '{char_name}' from voice_info dict: {voice_data}")
                            normalized_voice_assignments[char_name] = f"UNKNOWN_PROFILE_FOR_{char_name}" """

# Perform replacement
if old_section in content:
    modified_content = content.replace(old_section, new_section)
    with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
        f.write(modified_content)
    print("✅ Successfully fixed voice_name extraction!")
else:
    print("❌ Could not find the exact section to replace. The function may have been changed.")
    print("🔍 Trying to locate the relevant function...")
    if "normalized_voice_assignments" in content:
        print("✅ Found normalized_voice_assignments function - manual inspection may be needed.")
    else:
        print("❌ Function not found - please check manually.")

print("�� Script finished.") 