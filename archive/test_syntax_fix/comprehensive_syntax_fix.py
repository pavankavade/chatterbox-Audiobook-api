with open('gradio_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Comprehensive syntax fix...")

fixes_applied = 0

# Go through each line and fix syntax issues
for i, line in enumerate(lines):
    line_num = i + 1
    stripped_line = line.strip()
    
    # Fix line 122 where there's a for loop followed by try instead of matching try block
    if (115 <= line_num <= 125 and 
        'for retry in range(max_retries):' in line):
        # This line should have its try block properly indented
        current_indent = len(line) - len(line.lstrip())
        
        # Look ahead to see if we have indentation issues
        for j in range(i+1, min(len(lines), i+10)):
            next_line = lines[j]
            if next_line.strip() and not next_line.startswith(' ' * (current_indent + 4)):
                # Fix the indentation
                if 'try:' not in next_line and 'except' not in next_line:
                    # Add missing try block
                    insert_line = ' ' * (current_indent + 4) + 'try:\n'
                    lines.insert(j, insert_line)
                    fixes_applied += 1
                    print(f"Added missing try block after line {line_num}")
                    break
    
    # Remove line 116-117 which has problematic return/raise structure
    if (115 <= line_num <= 118 and 
        ('return wav' in line or 'raise RuntimeError' in line) and
        'CPU generation failed' in line):
        print(f"Removing problematic return/raise on line {line_num}")
        lines[i] = ''
        fixes_applied += 1
    
    # Fix line 166 - orphaned except
    if (165 <= line_num <= 168 and stripped_line == 'except Exception as e:'):
        print(f"Removing orphaned except on line {line_num}")
        lines[i] = ''
        fixes_applied += 1
    
    # Fix missing try blocks in get_voice_profiles function
    if (200 <= line_num <= 220 and 
        'with open(config_file' in line and 
        'try:' not in lines[i-1]):
        # Add try block before the with statement
        current_indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * current_indent + 'try:\n' + line
        fixes_applied += 1
        print(f"Added missing try block before line {line_num}")
    
    # Remove orphaned continue statements
    if stripped_line == 'continue' and i > 0:
        # Check if this continue is part of a valid loop
        found_loop = False
        for j in range(i-1, max(0, i-10), -1):
            prev_line = lines[j].strip()
            if prev_line.startswith('for ') or prev_line.startswith('while '):
                found_loop = True
                break
        if not found_loop:
            print(f"Removing orphaned continue on line {line_num}")
            lines[i] = ''
            fixes_applied += 1

print(f"Applied {fixes_applied} comprehensive fixes")

# Now write the cleaned file
with open('gradio_test.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Comprehensive syntax fixes applied!") 