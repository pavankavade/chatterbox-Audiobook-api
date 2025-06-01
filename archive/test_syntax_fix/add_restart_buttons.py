import re

# Read the main file
with open('../gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Adding restart buttons to project loading sections...")

# Pattern 1: Single-voice section - add restart button after resume button
single_voice_pattern = r'(load_project_btn = gr\.Button\("📂 Load Project", size="sm", variant="secondary"\)\s+resume_project_btn = gr\.Button\("▶️ Resume Project", size="sm", variant="primary"\))'
single_voice_replacement = r'\1\n                            restart_single_project_btn = gr.Button("🔄 Restart Project", size="sm", variant="primary")'

if re.search(single_voice_pattern, content):
    content = re.sub(single_voice_pattern, single_voice_replacement, content)
    print("✅ Added restart button to single-voice section")
else:
    print("❌ Could not find single-voice pattern")

# Pattern 2: Multi-voice section - add restart button after resume button  
multi_voice_pattern = r'(load_multi_project_btn = gr\.Button\("📂 Load Project", size="sm", variant="secondary"\)\s+resume_multi_project_btn = gr\.Button\("▶️ Resume Project", size="sm", variant="primary"\))'
multi_voice_replacement = r'\1\n                            restart_multi_project_btn = gr.Button("🔄 Restart Project", size="sm", variant="primary")'

if re.search(multi_voice_pattern, content):
    content = re.sub(multi_voice_pattern, multi_voice_replacement, content)
    print("✅ Added restart button to multi-voice section")
else:
    print("❌ Could not find multi-voice pattern")

# Write the updated content back
with open('../gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Restart buttons added to both sections!")
print("\n🎯 Next step: Add event handlers for the restart buttons") 