#!/usr/bin/env python3
"""
Quick fix to replace display_name with voice_name in normalization.
"""

print("🚀 Quick voice_name fix...")

# Read the file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the specific line
for i, line in enumerate(lines):
    if "profile_name = voice_data.get('display_name')" in line:
        lines[i] = line.replace("voice_data.get('display_name')", "voice_data.get('voice_name')")
        print(f"✅ Fixed line {i+1}: {lines[i].strip()}")
        break

# Write back
with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("🏁 Quick fix completed!") 