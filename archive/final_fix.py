with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove the problematic line 5838 (index 5837)
if len(lines) > 5837:
    print(f"Removing problematic line 5838: {repr(lines[5837])}")
    lines[5837] = ''

with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Final fix applied!")