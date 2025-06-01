with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the indentation on line 5829 (index 5828)
if len(lines) > 5828:
    if 'else:' in lines[5828]:
        lines[5828] = '                        else:\n'
        print(f"Fixed line 5829: {repr(lines[5828])}")

with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Indentation fix applied!")