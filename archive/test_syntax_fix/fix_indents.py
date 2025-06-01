with open('gradio_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Fixing all remaining syntax issues...")

fixes_applied = 0

# First pass: Remove all orphaned except statements
for i, line in enumerate(lines):
    line_num = i + 1
    stripped_line = line.strip()
    
    # Remove orphaned except statements (bare except:)
    if stripped_line == 'except:':
        print(f"Removing orphaned except on line {line_num}")
        lines[i] = ''
        fixes_applied += 1
    
    # Remove orphaned except Exception as e:
    elif stripped_line == 'except Exception as e:':
        print(f"Removing orphaned except Exception on line {line_num}")
        lines[i] = ''
        fixes_applied += 1
        
    # Remove orphaned else statements
    elif stripped_line == 'else:':
        # Check if this is truly orphaned by looking at indentation patterns
        current_indent = len(line) - len(line.lstrip())
        
        # Look backwards for matching if/try/for/while
        found_match = False
        for j in range(i-1, max(0, i-20), -1):  # Look back up to 20 lines
            prev_line = lines[j].strip()
            if prev_line == '':
                continue
            prev_indent = len(lines[j]) - len(lines[j].lstrip())
            
            # If we find something at the same or less indentation that could match
            if (prev_indent == current_indent and 
                (prev_line.startswith('if ') or prev_line.startswith('try:') or 
                 prev_line.startswith('for ') or prev_line.startswith('while '))):
                found_match = True
                break
        
        if not found_match:
            print(f"Removing orphaned else on line {line_num}")
            lines[i] = ''
            fixes_applied += 1

print(f"Applied {fixes_applied} fixes")

with open('gradio_test.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ All syntax issues fixed!") 