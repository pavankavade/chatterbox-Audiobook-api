#!/usr/bin/env python3
"""
Fixes a syntax error caused by escaped parentheses in a function definition.
"""

print("🔧 Fixing syntax error in function definition...")

main_file_path = "gradio_tts_app_audiobook.py"

with open(main_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# The problematic line often looks like: def create_continuous_playback_audio\(project_name: str\) -> tuple:
# We want to change it to: def create_continuous_playback_audio(project_name: str) -> tuple:

fixed_content = content.replace(
    "def create_continuous_playback_audio\\(project_name: str\\) -> tuple:",
    "def create_continuous_playback_audio(project_name: str) -> tuple:"
)

if fixed_content != content:
    with open(main_file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print("✅ Syntax error in create_continuous_playback_audio definition likely fixed.")
else:
    print("❌ Problematic pattern not found. Manual check might be needed if syntax error persists.")

print("�� Script finished.") 