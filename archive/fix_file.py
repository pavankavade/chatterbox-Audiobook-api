#!/usr/bin/env python3
"""Simple script to remove problematic lines from the gradio app file"""

def fix_gradio_file():
    with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter out the problematic lines
    fixed_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        # Skip lines that reference removed UI elements
        if ("chunk_interface['get_duration_btn']" in line or
            "chunk_interface['apply_trim_btn']" in line or
            "outputs=[chunk_interface['trim_end']" in line or
            "inputs=[chunk_interface['trim_start'], chunk_interface['trim_end']]" in line):
            # Skip this line and the closing bracket lines that follow
            skip_next = 2  # Skip this line and next 2 lines (closing brackets)
            continue
        
        if skip_next > 0:
            skip_next -= 1
            continue
            
        fixed_lines.append(line)
    
    # Write the fixed content back
    with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed gradio_tts_app_audiobook.py - removed problematic button references")

if __name__ == "__main__":
    fix_gradio_file() 