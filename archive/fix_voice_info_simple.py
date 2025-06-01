#!/usr/bin/env python3
"""
Fix the voice_info bug by replacing specific problematic lines
"""

print("🔧 Fixing voice_info bug...")

# Read the main file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the problematic lines
fixed = False
for i, line in enumerate(lines):
    if 'voice_display += f"  • {voice_name}: {info.get(' in line:
        lines[i] = line.replace(
            'voice_display += f"  • {voice_name}: {info.get(\'display_name\', voice_name)}\\n"',
            'voice_display += f"  • {voice_name}: {info.get(\'display_name\', voice_name) if isinstance(info, dict) else info}\\n"'
        )
        print(f"✅ Fixed line {i+1}: Multi-voice info handling")
        fixed = True
    elif 'voice_display = f"🎤 Single voice: {voice_info.get(' in line:
        lines[i] = line.replace(
            'voice_display = f"🎤 Single voice: {voice_info.get(\'display_name\', \'Unknown\')}"',
            'voice_display = f"🎤 Single voice: {voice_info.get(\'display_name\', \'Unknown\') if isinstance(voice_info, dict) else voice_info}"'
        )
        print(f"✅ Fixed line {i+1}: Single-voice info handling")
        fixed = True

if not fixed:
    print("❌ Could not find the exact lines to fix")
    # Let's try a different approach - find the lines by content
    for i, line in enumerate(lines):
        if 'info.get(' in line and 'voice_display' in line:
            print(f"Found potential line {i+1}: {line.strip()}")

# Write the updated content back
with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("🎉 Bug fix attempt complete!") 