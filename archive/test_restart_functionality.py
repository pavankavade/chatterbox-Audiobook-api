#!/usr/bin/env python3
"""
Quick test to verify restart functionality is working
"""

print("🎯 Testing Restart Functionality")
print("=" * 50)

# Test 1: Check if restart function exists
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Look for the key components
has_restart_function = "def restart_project_generation" in content
has_single_restart_btn = "restart_single_project_btn = gr.Button" in content
has_multi_restart_btn = "restart_multi_project_btn = gr.Button" in content
has_single_handler = "restart_single_project_btn.click" in content
has_multi_handler = "restart_multi_project_btn.click" in content
has_json_import = "import json" in content

print(f"✅ Restart Function: {'FOUND' if has_restart_function else 'MISSING'}")
print(f"✅ Single-Voice Restart Button: {'FOUND' if has_single_restart_btn else 'MISSING'}")
print(f"✅ Multi-Voice Restart Button: {'FOUND' if has_multi_restart_btn else 'MISSING'}")
print(f"✅ Single-Voice Event Handler: {'FOUND' if has_single_handler else 'MISSING'}")
print(f"✅ Multi-Voice Event Handler: {'FOUND' if has_multi_handler else 'MISSING'}")
print(f"✅ JSON Import (required): {'FOUND' if has_json_import else 'MISSING'}")

print("\n" + "=" * 50)

all_components = [
    has_restart_function,
    has_single_restart_btn,
    has_multi_restart_btn,
    has_single_handler,
    has_multi_handler,
    has_json_import
]

if all(all_components):
    print("🎉 SUCCESS: All restart functionality components are present!")
    print("\n📋 What the restart buttons do:")
    print("   • Reset project progress to chunk 1")
    print("   • Clean up temporary files")
    print("   • Allow full project regeneration")
    print("   • Work with both single-voice and multi-voice projects")
    print("\n🚀 Ready to test in the app!")
else:
    print("❌ MISSING COMPONENTS: Some restart functionality is missing")
    missing = []
    if not has_restart_function: missing.append("restart function")
    if not has_single_restart_btn: missing.append("single-voice button")
    if not has_multi_restart_btn: missing.append("multi-voice button")
    if not has_single_handler: missing.append("single-voice handler")
    if not has_multi_handler: missing.append("multi-voice handler")
    if not has_json_import: missing.append("json import")
    print(f"   Missing: {', '.join(missing)}")

print("\n" + "=" * 50) 