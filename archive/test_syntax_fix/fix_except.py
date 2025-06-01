with open('gradio_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Scanning for syntax issues...")

fixes_applied = 0

# Look for orphaned except statements and malformed try/except blocks
for i, line in enumerate(lines):
    line_num = i + 1
    
    # Check for orphaned 'except Exception as e:' around line 6096
    if (6090 <= line_num <= 6100 and 
        'except Exception as e:' in line and 
        line.strip() == 'except Exception as e:'):
        print(f"Found orphaned except on line {line_num}: {repr(line)}")
        # Remove this orphaned except
        lines[i] = ''
        fixes_applied += 1
    
    # Check for malformed try blocks
    if ('try:' in line and line.strip() == 'try:' and 
        i > 0 and lines[i-1].strip().endswith(':')):
        print(f"Found malformed try on line {line_num}: {repr(line)}")
        # Remove orphaned try
        lines[i] = ''
        fixes_applied += 1

print(f"Applied {fixes_applied} fixes")

with open('gradio_test.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Orphaned except statements fixed!") 