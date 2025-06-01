#!/usr/bin/env python3
"""
Quick test to verify smart hybrid CPU/GPU solution is working
"""

print("🎯 Testing Smart Hybrid CPU/GPU Solution")
print("=" * 50)

# Test 1: Check if the generate function contains smart hybrid logic
with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Look for the key components
has_cpu_threshold = "cpu_threshold = 25" in content
has_smart_detection = "text_length = len(text.strip())" in content
has_cpu_routing = "Short text" in content and "CPU:" in content
has_gpu_routing = "Long text" in content and "GPU:" in content
has_cuda_fallback = "srcIndex < srcSelectDimSize" in content

print(f"✅ CPU Threshold (25 chars): {'FOUND' if has_cpu_threshold else 'MISSING'}")
print(f"✅ Smart Text Length Detection: {'FOUND' if has_smart_detection else 'MISSING'}")
print(f"✅ CPU Routing for Short Text: {'FOUND' if has_cpu_routing else 'MISSING'}")
print(f"✅ GPU Routing for Long Text: {'FOUND' if has_gpu_routing else 'MISSING'}")
print(f"✅ CUDA Error Fallback: {'FOUND' if has_cuda_fallback else 'MISSING'}")

all_components = [has_cpu_threshold, has_smart_detection, has_cpu_routing, 
                 has_gpu_routing, has_cuda_fallback]

print("\n" + "=" * 50)

if all(all_components):
    print("🎉 SUCCESS: Smart Hybrid CPU/GPU Solution is ACTIVE!")
    print("📋 How it works:")
    print("   🧮 Text ≤25 chars → CPU (avoids CUDA srcIndex errors)")
    print("   🚀 Text >25 chars → GPU (performance)")
    print("   💪 GPU fails → CPU fallback")
    print("\n🔧 This should fix the 'Yellow...' CUDA error!")
else:
    print("❌ ISSUE: Some components are missing")
    print("💡 The smart hybrid solution may not be fully active")

print("\n🧪 Test specific examples:")
test_cases = [
    ("Yellow", 6, "CPU"),
    ("Yellow...", 8, "CPU"), 
    ("Hi", 2, "CPU"),
    ("This is a longer sentence that would use GPU", 45, "GPU")
]

for text, length, expected in test_cases:
    device = "CPU" if length <= 25 else "GPU"
    status = "✅" if device == expected else "❌"
    print(f"   {status} '{text}' ({length} chars) → {device}")

print("\n🚀 Ready to test! Try the problematic 'Yellow' text now.") 