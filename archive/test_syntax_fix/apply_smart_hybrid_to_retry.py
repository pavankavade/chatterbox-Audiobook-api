import re

# Read the main file
with open('../gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the generate_with_retry function and replace it
old_function = '''def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """Generate audio with retry logic for CUDA errors"""
    for retry in range(max_retries):
        try:
            # Clear memory before generation
            if retry > 0:
                models_clear_gpu_memory()
                print(f"🔄 Retry {retry}/{max_retries}: Cleared GPU memory, attempting generation again...")
            
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
            )
            
            # Success message for retries
            if retry > 0:
                print(f"✅ Generation successful on retry {retry}/{max_retries}!")
            
            return wav
            
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or 
                "CUDA" in str(e) or 
                "out of memory" in str(e).lower()):
                
                if retry < max_retries - 1:
                    print(f"⚠️ GPU error, retry {retry + 1}/{max_retries}: {str(e)[:100]}...")
                    print(f"🔧 Attempting to clear GPU memory and retry...")
                    models_clear_gpu_memory()
                    continue
                else:
                    print(f"❌ CUDA error persisted after {max_retries} retries: {str(e)[:150]}...")
                    print(f"🔧 Consider switching to CPU processing or reducing text complexity")
                    raise RuntimeError(f"Failed after {max_retries} retries: {str(e)}")
            else:
                # Non-CUDA error, don't retry
                print(f"❌ Non-CUDA error occurred: {str(e)[:150]}...")
                raise e
    
    raise RuntimeError("Generation failed after all retries")'''

new_function = '''def generate_with_retry(model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, max_retries=3):
    """
    🎯 ENHANCED with Smart Hybrid CPU/GPU Selection + Retry Logic
    
    Generate audio with automatic CPU selection for short text AND retry logic for CUDA errors.
    This solves the CUDA srcIndex error by avoiding GPU for very short chunks.
    """
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
                cfg_weight,
            )
            print(f"✅ CPU generation successful for short text ({text_length} chars)")
            return (cpu_model.sr, wav.squeeze(0).numpy())
        except Exception as e:
            raise RuntimeError(f"CPU generation failed for short text: {str(e)}")
    
    # Use GPU for longer text with retry logic
    print(f"🚀 Long text ({text_length} chars) → GPU: '{text[:30]}{'...' if len(text) > 30 else ''}'")
    
    for retry in range(max_retries):
        try:
            # Clear memory before generation
            if retry > 0:
                models_clear_gpu_memory()
                print(f"🔄 Retry {retry}/{max_retries}: Cleared GPU memory, attempting generation again...")
            
            wav = models_generate(
                model,
                text,
                audio_prompt_path,
                exaggeration,
                temperature,
                0,  # seed_num - set to 0 for no specific seed
                cfg_weight
            )
            
            # Success message for retries
            if retry > 0:
                print(f"✅ GPU generation successful on retry {retry}/{max_retries} for long text ({text_length} chars)!")
            else:
                print(f"✅ GPU generation successful for long text ({text_length} chars)")
            
            return wav
            
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or 
                "CUDA" in str(e) or 
                "out of memory" in str(e).lower()):
                
                if retry < max_retries - 1:
                    print(f"⚠️ GPU error, retry {retry + 1}/{max_retries}: {str(e)[:100]}...")
                    print(f"🔧 Attempting to clear GPU memory and retry...")
                    models_clear_gpu_memory()
                    continue
                else:
                    print(f"❌ GPU failed after {max_retries} retries with CUDA error, falling back to CPU...")
                    # Final fallback to CPU for longer text
                    try:
                        cpu_model = models_load_model_cpu()
                        wav = cpu_model.generate(
                            text,
                            audio_prompt_path,
                            exaggeration,
                            temperature,
                            cfg_weight,
                        )
                        print(f"✅ CPU fallback successful for long text ({text_length} chars)")
                        return (cpu_model.sr, wav.squeeze(0).numpy())
                    except Exception as cpu_error:
                        raise RuntimeError(f"Both GPU and CPU failed: GPU: {str(e)}, CPU: {str(cpu_error)}")
            else:
                # Non-CUDA error, don't retry
                print(f"❌ Non-CUDA error occurred: {str(e)[:150]}...")
                raise e
    
    raise RuntimeError("Generation failed after all retries")'''

# Apply the replacement
if old_function in content:
    updated_content = content.replace(old_function, new_function)
    
    # Write back to the main file
    with open('../gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Successfully applied smart hybrid CPU/GPU solution to generate_with_retry!")
    print("🎯 Short text (≤25 chars) will now use CPU automatically")
    print("🚀 Long text will use GPU with CPU fallback if needed")
    print("🔄 CUDA error retry logic preserved")
else:
    print("❌ Could not find the exact function to replace")
    print("The function may have been modified already")
    
    # Let's search for the function start to debug
    if "def generate_with_retry(" in content:
        print("✓ Function exists but content doesn't match exactly")
        # Find the function and print a snippet
        start_idx = content.find("def generate_with_retry(")
        if start_idx != -1:
            snippet = content[start_idx:start_idx+500]
            print("Current function start:")
            print(snippet)
    else:
        print("❌ Function not found at all") 