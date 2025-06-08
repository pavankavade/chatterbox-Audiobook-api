#!/usr/bin/env python3
"""
Test script for legacy voice detection
"""

from src.voice_library.voice_management import get_voice_profiles

def test_legacy_voices():
    print("🔍 Testing legacy voice detection...")
    
    voices = get_voice_profiles('../speakers')
    print(f"Found {len(voices)} total voice profiles")
    
    # Count by profile type
    types = {}
    legacy_examples = []
    
    for voice in voices:
        profile_type = voice.get('profile_type', 'unknown')
        types[profile_type] = types.get(profile_type, 0) + 1
        
        if profile_type == 'legacy_folder':
            legacy_examples.append(voice['name'])
    
    print("\nVoice profile types:")
    for profile_type, count in types.items():
        print(f"  {profile_type}: {count}")
    
    if legacy_examples:
        print(f"\n✅ Legacy folder voices detected: {len(legacy_examples)}")
        print("Examples:")
        for name in legacy_examples[:5]:
            print(f"  - {name}")
    else:
        print("❌ No legacy folder voices detected")

if __name__ == "__main__":
    test_legacy_voices() 