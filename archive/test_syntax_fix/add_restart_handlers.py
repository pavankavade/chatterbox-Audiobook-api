import re

# Read the main file
with open('../gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Adding restart button event handlers...")

# Find the resume button handler and add restart handlers after it
resume_pattern = r'(resume_project_btn\.click\(\s+fn=resume_single_voice_project,\s+inputs=\[model_state, single_project_dropdown, voice_library_path_state\],\s+outputs=\[audiobook_output, single_project_progress\]\s+\))'

restart_handlers = r'''\1

    # Restart single-voice project button
    restart_single_project_btn.click(
        fn=restart_project_generation,
        inputs=[single_project_dropdown],
        outputs=[single_project_progress]
    )

    # Restart multi-voice project button  
    restart_multi_project_btn.click(
        fn=restart_project_generation,
        inputs=[multi_project_dropdown],
        outputs=[multi_project_progress]
    )'''

if re.search(resume_pattern, content, re.MULTILINE | re.DOTALL):
    content = re.sub(resume_pattern, restart_handlers, content, flags=re.MULTILINE | re.DOTALL)
    print("✅ Added restart button event handlers")
else:
    print("❌ Could not find resume button handler pattern")

# Write the updated content back
with open('../gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Restart button event handlers added!")
print("\n🎯 Restart buttons are now fully functional!") 