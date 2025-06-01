#!/usr/bin/env python3
"""
Fix the voice_info bug in load_project_for_regeneration function
"""

import re

print("🔧 Fixing voice_info bug...")

# Read the main file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the problematic code
old_pattern = r'''    # Format voice info display
    if metadata\.get\('project_type'\) == 'multi_voice':
        voice_display = "🎭 Multi-voice project:\\n"
        for voice_name, info in voice_info\.items\(\):
            voice_display \+= f"  • \{voice_name\}: \{info\.get\('display_name', voice_name\)\}\\n"
    else:
        voice_display = f"🎤 Single voice: \{voice_info\.get\('display_name', 'Unknown'\)\}"'''

new_pattern = '''    # Format voice info display
    if metadata.get('project_type') == 'multi_voice':
        voice_display = "🎭 Multi-voice project:\\n"
        for voice_name, info in voice_info.items():
            if isinstance(info, dict):
                voice_display += f"  • {voice_name}: {info.get('display_name', voice_name)}\\n"
            else:
                voice_display += f"  • {voice_name}: {info}\\n"
    else:
        if isinstance(voice_info, dict):
            voice_display = f"🎤 Single voice: {voice_info.get('display_name', 'Unknown')}"
        else:
            voice_display = f"🎤 Single voice: {voice_info}"'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("✅ Fixed voice_info bug")
else:
    print("❌ Could not find exact pattern to fix")

# Write the updated content back
with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("🎉 Bug fix complete!") 