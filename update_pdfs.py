import os
import glob

replacements = {
    "2311.pdf": "2311.pdf",
    "2606.12683v1.pdf": "2606.12683v1.pdf",
    "Levels of AGI": "Levels of AGI",
    "Levels of AGI": "Levels of AGI",
    "AGI performance, generality, and autonomy": "AGI performance, generality, and autonomy",
    "agi_pdf": "agi_pdf",
    "agi_collection": "agi_collection",
    "agi_research_search": "agi_research_search",
    "AGI": "AGI",
    "agi": "agi",
    "What are the levels of AGI performance?": "What are the levels of AGI performance?",
    "AGI": "AGI",
}

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, _, files in os.walk('.'):
    if 'venv' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.md'):
            update_file(os.path.join(root, file))

