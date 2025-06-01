#!/usr/bin/env python3
"""
Add the missing load_multi_voice_project function
"""

import re

print("🔧 Adding missing load_multi_voice_project function...")

# Read the main file
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the exact location after load_single_voice_project
pattern = r'(    def load_single_voice_project\(project_name: str\):\s+"""Load project info and update UI fields for single-voice tab\."""\s+text, voice_info, proj_name, _, status = load_project_for_regeneration\(project_name\)\s+# Try to extract voice name from voice_info string\s+import re\s+voice_match = re\.search\(r\'\\\\\\(\\[\\^\\)\\]\\+\\\\\\)\', voice_info\)\s+selected_voice = None\s+if voice_match:\s+selected_voice = voice_match\.group\(1\)\s+return text, selected_voice, proj_name, status\s+)'

replacement = r'''\1
    def load_multi_voice_project(project_name: str):
        """Load project info and update UI fields for multi-voice tab."""
        text, voice_info, proj_name, _, status = load_project_for_regeneration(project_name)
        return text, status

'''

if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    print("✅ Added load_multi_voice_project function")
else:
    print("❌ Could not find exact pattern, trying simpler approach...")
    
    # Try a simpler pattern
    simple_pattern = r'(return text, selected_voice, proj_name, status\s+\s+# Handler to resume single-voice project generation)'
    simple_replacement = r'''return text, selected_voice, proj_name, status

    def load_multi_voice_project(project_name: str):
        """Load project info and update UI fields for multi-voice tab."""
        text, voice_info, proj_name, _, status = load_project_for_regeneration(project_name)
        return text, status

    # Handler to resume single-voice project generation'''
    
    if re.search(simple_pattern, content, re.DOTALL):
        content = re.sub(simple_pattern, simple_replacement, content, flags=re.DOTALL)
        print("✅ Added load_multi_voice_project function (simple pattern)")
    else:
        print("❌ Could not find location to insert function")

# Write the updated content back
with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("🎉 Function addition complete!") 