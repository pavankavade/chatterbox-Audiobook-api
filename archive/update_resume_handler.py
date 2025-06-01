#!/usr/bin/env python3
"""
Modifies the resume_multi_project_btn click handler to correctly load data.
"""
import re

print("🚀 Modifying resume_multi_project_btn click handler...")

main_file_path = "gradio_tts_app_audiobook.py"

# The new intermediate handler function
intermediate_handler_code = """
    def handle_resume_multi_voice(model, project_name, voice_library_path):
        \"\"\"Handles loading data and then calling the main multi-voice creation function for resume.\"\"\"
        if not project_name:
            return None, "<div class='audiobook-status'>❌ Please select a project to resume.</div>"

        text_content, voice_assignments, load_status = resume_multi_voice_project_data(project_name)

        if text_content is None or voice_assignments is None:
            # resume_multi_voice_project_data already returns a Gradio HTML formatted error string
            return None, load_status 

        # Log the loaded data for debugging
        print(f"[Resume Multi] Loaded text: {len(text_content)} chars, Voice Assignments: {len(voice_assignments)} characters")
        # print(f"[Resume Multi] Voice Assignments details: {voice_assignments}") # Potentially very long

        return create_multi_voice_audiobook_with_assignments(
            model, 
            text_content, 
            voice_library_path, 
            project_name, 
            voice_assignments, 
            resume=True
        )
"""

# Anchor pattern: The line defining the start of the Gradio Blocks `with demo:`
# We'll insert the new handler function definition within the app's main scope, before the UI event handlers.
# A safe place is usually right before the ` # --- Wire up the buttons in the UI logic ---` comment, or similar.
# Let's find the beginning of the UI wiring section.
ui_wiring_anchor = r"# --- Wire up the buttons in the UI logic ---"

# Pattern for the existing resume_multi_project_btn.click handler
old_handler_pattern = r"""(
    # Resume multi-voice project button
    resume_multi_project_btn\.click\(
        fn=lambda model, project_name, voice_library_path: create_multi_voice_audiobook_with_assignments\(
            model, "", voice_library_path, project_name, \{\}, resume=True
        \),
        inputs=\[model_state, multi_project_dropdown, voice_library_path_state\],
        outputs=\[multi_audiobook_output, multi_project_progress\]
    \)
)"""

# New handler pointing to our intermediate function
new_handler_replacement = """

    # Resume multi-voice project button
    resume_multi_project_btn.click(
        fn=handle_resume_multi_voice,
        inputs=[model_state, multi_project_dropdown, voice_library_path_state],
        outputs=[multi_audiobook_output, multi_project_progress]
    )
"""

with open(main_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

modified_content = content
function_inserted = False
handler_updated = False

# 1. Insert the intermediate_handler_code
if re.search(ui_wiring_anchor, modified_content):
    modified_content = re.sub(ui_wiring_anchor, intermediate_handler_code + "\n\n    " + ui_wiring_anchor, modified_content, 1)
    print("✅ Inserted handle_resume_multi_voice function definition.")
    function_inserted = True
else:
    print(f"❌ UI Wiring Anchor pattern ('{ui_wiring_anchor}') not found. Cannot insert helper function automatically.")

# 2. Replace the old handler with the new one
match = re.search(old_handler_pattern, modified_content, flags=re.DOTALL)
if match:
    modified_content = modified_content.replace(match.group(1), new_handler_replacement)
    print("✅ Updated resume_multi_project_btn.click() to use new handler.")
    handler_updated = True
else:
    print("❌ Existing resume_multi_project_btn.click() handler pattern not found. Check the script's regex.")

if function_inserted and handler_updated:
    with open(main_file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    print(f"✅ Successfully modified resume_multi_project_btn handler in {main_file_path}")
else:
    print("❌ No changes made to the file due to missing patterns or errors.")

print("�� Script finished.") 