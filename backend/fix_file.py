#!/usr/bin/env python3
"""Fix corrupted heartbeat_sim.py file"""

with open('heartbeat_sim.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the first occurrence of the closing HTML tag and function return
first_return_idx = None
for i, line in enumerate(lines):
    if 'return HTMLResponse(content=html_content)' in line:
        first_return_idx = i
        break

# Find the second occurrence (duplicate)
second_return_idx = None
if first_return_idx:
    for i in range(first_return_idx + 1, len(lines)):
        if 'return HTMLResponse(content=html_content)' in lines[i]:
            second_return_idx = i
            break

if first_return_idx and second_return_idx:
    # Keep lines up to and including first return
    # Then skip to second return + 2 (to skip blank lines)
    fixed_lines = lines[:first_return_idx+1]
    fixed_lines.append('\n\n')
    fixed_lines.extend(lines[second_return_idx+3:])
    
    with open('heartbeat_sim.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed! Removed lines {first_return_idx+1} to {second_return_idx+2}")
    print(f"Total lines removed: {second_return_idx - first_return_idx + 2}")
else:
    print(f"Could not find duplicate returns. First at {first_return_idx}, Second at {second_return_idx}")
