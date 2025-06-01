#!/usr/bin/env python3
"""
Test script to verify load project functionality is working
"""

import sys
sys.path.append('.')

from gradio_tts_app_audiobook import (
    get_project_choices,
    load_single_voice_project,
    load_multi_voice_project,
    restart_project_generation
)

def test_project_functions():
    """Test all project loading functions"""
    print("🧪 Testing Project Loading Functions")
    print("=" * 50)
    
    # Test 1: Get project choices
    print("📋 Test 1: Getting project choices...")
    projects = get_project_choices()
    print(f"✅ Found {len(projects)} projects")
    if projects:
        print(f"   First few projects: {[p[1] if isinstance(p, tuple) else p for p in projects[:3]]}")
    
    if not projects:
        print("❌ No projects found to test with")
        return
    
    # Get a test project name
    test_project = projects[0][1] if isinstance(projects[0], tuple) else projects[0]
    print(f"\n🎯 Using test project: {test_project}")
    
    # Test 2: Single-voice load function
    print("\n📋 Test 2: Testing single-voice load function...")
    try:
        result = load_single_voice_project(test_project)
        text, voice, proj_name, status = result
        print(f"✅ Single-voice load successful:")
        print(f"   📄 Text length: {len(text) if text else 0} characters")
        print(f"   🎭 Voice: {voice if voice else 'None'}")
        print(f"   📁 Project: {proj_name if proj_name else 'None'}")
        print(f"   📊 Status: {status[:100] if status else 'None'}...")
    except Exception as e:
        print(f"❌ Single-voice load failed: {e}")
    
    # Test 3: Multi-voice load function
    print("\n📋 Test 3: Testing multi-voice load function...")
    try:
        result = load_multi_voice_project(test_project)
        text, status = result
        print(f"✅ Multi-voice load successful:")
        print(f"   📄 Text length: {len(text) if text else 0} characters")
        print(f"   📊 Status: {status[:100] if status else 'None'}...")
    except Exception as e:
        print(f"❌ Multi-voice load failed: {e}")
    
    # Test 4: Restart function
    print("\n📋 Test 4: Testing restart function...")
    try:
        result = restart_project_generation(test_project)
        print(f"✅ Restart function successful:")
        print(f"   📊 Result: {result[:100] if result else 'None'}...")
    except Exception as e:
        print(f"❌ Restart function failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    test_project_functions() 