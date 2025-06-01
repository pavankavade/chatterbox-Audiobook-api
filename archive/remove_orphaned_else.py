with open('gradio_tts_app_audiobook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove orphaned else statements
fixes_applied = 0

# Line 5838 (index 5837) - orphaned else
if len(lines) > 5837 and 'else:' in lines[5837]:
    print(f"Removing orphaned else on line 5838: {repr(lines[5837])}")
    lines[5837] = ''  # Remove the line
    fixes_applied += 1

# Line 5841 (index 5840) - orphaned else  
if len(lines) > 5840 and 'else:' in lines[5840]:
    print(f"Removing orphaned else on line 5841: {repr(lines[5840])}")
    lines[5840] = ''  # Remove the line
    fixes_applied += 1

print(f"Removed {fixes_applied} orphaned else statements")

with open('gradio_tts_app_audiobook.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Orphaned else statements removed!")