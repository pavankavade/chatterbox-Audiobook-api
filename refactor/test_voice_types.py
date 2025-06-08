#!/usr/bin/env python3
"""
Test script to show legacy voice detection and types
"""

from src.voice_library.voice_management import get_voice_profiles

def test_voice_types():
    print("🎤 Testing Legacy Voice Detection")
    print("=" * 50)
    
    voices = get_voice_profiles('../speakers')
    print(f"Found {len(voices)} total voices")
    
    # Group by type
    by_type = {}
    legacy_examples = []
    modern_examples = []
    
    for voice in voices:
        voice_type = voice.get('profile_type', 'unknown')
        by_type[voice_type] = by_type.get(voice_type, 0) + 1
        
        if voice_type == 'legacy_json':
            if len(legacy_examples) < 5:
                legacy_examples.append(voice['name'])
        elif voice_type == 'subfolder':
            if len(modern_examples) < 5:
                modern_examples.append(voice['name'])
    
    print(f"\nVoice profile types:")
    for voice_type, count in by_type.items():
        print(f"  📁 {voice_type}: {count} voices")
    
    if legacy_examples:
        print(f"\n🔍 Legacy JSON examples: {', '.join(legacy_examples)}")
    
    if modern_examples:
        print(f"🔍 Modern config examples: {', '.join(modern_examples)}")
    
    # Test loading a specific legacy voice
    if legacy_examples:
        from src.voice_library.voice_management import load_voice_profile
        test_voice = legacy_examples[0]
        print(f"\n📋 Testing legacy voice '{test_voice}':")
        
        try:
            profile = load_voice_profile('../speakers', test_voice)
            if profile:
                print(f"   ✅ Loaded successfully")
                print(f"   📝 Type: {profile.get('profile_type', 'unknown')}")
                if 'display_name' in profile:
                    print(f"   🏷️  Display: {profile['display_name']}")
                if 'audio_file' in profile:
                    print(f"   🎵 Audio: {profile['audio_file']}")
            else:
                print(f"   ❌ Failed to load")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_voice_types() 