with open('gradio_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Before fixing:")
for i in range(5835, min(len(lines), 5845)):
    print(f"Line {i+1}: {repr(lines[i])}")

# The problem lines are:
# Line 5838: "                    return (sr, audio), None, f"{settings_info}⚠️ No audio segments detected with these settings"\n"
# Line 5839: "                return (sr, audio), None, f"{settings_info}⚠️ File appears completely silent with threshold {silence_threshold} dB"\n"

# These are orphaned return statements that need to be removed
fixes_applied = 0

# Remove line 5838 (index 5837) - orphaned return statement
if len(lines) > 5837 and 'return (sr, audio), None,' in lines[5837] and 'No audio segments detected' in lines[5837]:
    print(f"Removing orphaned return on line 5838: {repr(lines[5837])}")
    lines[5837] = ''
    fixes_applied += 1

# Remove line 5839 (index 5838) - orphaned return statement  
if len(lines) > 5838 and 'return (sr, audio), None,' in lines[5838] and 'File appears completely silent' in lines[5838]:
    print(f"Removing orphaned return on line 5839: {repr(lines[5838])}")
    lines[5838] = ''
    fixes_applied += 1

print(f"\nRemoved {fixes_applied} orphaned return statements")

with open('gradio_test.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nAfter fixing:")
for i in range(5835, min(len(lines), 5845)):
    if lines[i].strip():  # Only show non-empty lines
        print(f"Line {i+1}: {repr(lines[i])}")

print("✅ Orphaned return statements removed!") 