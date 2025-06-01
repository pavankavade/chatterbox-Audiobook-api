import re

# Read the main file
with open('../gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the generate function and replace it
old_function = '''def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = models_load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = models_generate(
        model,
        text,
        audio_prompt_path,
        exaggeration,
        temperature,
        seed_num,
        cfgw
    )
    return wav'''

new_function = '''def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection
    
    Generate audio with automatic CPU selection for short text to avoid CUDA errors.
    """
    if model is None:
        model = models_load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    text_length = len(text.strip())
    cpu_threshold = 25  # Characters below this use CPU automatically
    
    # 🎯 SMART HYBRID: Force CPU for very short text (avoids CUDA srcIndex errors)
    if text_length <= cpu_threshold:
        print(f"🧮 Short text ({text_length} chars) → CPU: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        try:
            cpu_model = models_load_model_cpu()
            wav = cpu_model.generate(
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                cfgw,
            )
            print(f"✅ CPU generation successful for short text ({text_length} chars)")
            return (cpu_model.sr, wav.squeeze(0).numpy())
        except Exception as e:
            raise RuntimeError(f"CPU generation failed for short text: {str(e)}")
    
    # Use GPU for longer text
    print(f"🚀 Long text ({text_length} chars) → GPU: '{text[:30]}{'...' if len(text) > 30 else ''}'")
    
    try:
        wav = models_generate(
            model,
            text,
            audio_prompt_path,
            exaggeration,
            temperature,
            seed_num,
            cfgw
        )
        print(f"✅ GPU generation successful for long text ({text_length} chars)")
        return wav
    except RuntimeError as e:
        if ("srcIndex < srcSelectDimSize" in str(e) or 
            "CUDA" in str(e) or 
            "out of memory" in str(e).lower()):
            
            print(f"❌ GPU failed with CUDA error, falling back to CPU...")
            # Final fallback to CPU
            try:
                cpu_model = models_load_model_cpu()
                wav = cpu_model.generate(
                    text,
                    audio_prompt_path,
                    exaggeration,
                    temperature,
                    cfgw,
                )
                print(f"✅ CPU fallback successful for long text ({text_length} chars)")
                return (cpu_model.sr, wav.squeeze(0).numpy())
            except Exception as cpu_error:
                raise RuntimeError(f"Both GPU and CPU failed: GPU: {str(e)}, CPU: {str(cpu_error)}")
        else:
            # Non-CUDA error, re-raise
            raise e'''

# Apply the replacement
if old_function in content:
    updated_content = content.replace(old_function, new_function)
    
    # Write back to the main file
    with open('../gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Successfully applied smart hybrid CPU/GPU solution!")
    print("🎯 Short text (≤25 chars) will now use CPU automatically")
    print("🚀 Long text will use GPU with CPU fallback if needed")
else:
    print("❌ Could not find the exact function to replace")
    print("The function may have been modified already") 