import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 13 (baseline training) - remove the file rewrite, just run train.py
cell13 = nb['cells'][13]
cell13['source'] = [
    'import subprocess, sys\n',
    'print("Running baseline YOLOv4 training (custom=False)...")\n',
    '!python train.py\n',
]

# Fix cell 15 (custom training) - just flip custom flag cleanly and run
cell15 = nb['cells'][15]
src15 = ''.join(cell15['source'])

# Replace the broken rewrite block with a clean one
new_src15 = [
    'import re\n',
    '# Flip custom flag to True for MSA-YOLO run\n',
    'with open("train.py", "r", encoding="utf-8") as f:\n',
    '    content = f.read()\n',
    'content = re.sub(r"custom\\s*=\\s*(True|False)", "custom = True", content)\n',
    'with open("train.py", "w", encoding="utf-8") as f:\n',
    '    f.write(content)\n',
    'print("Running MSA-YOLO training (custom=True)...")\n',
    '!python train.py\n',
    '# Reset back to False after run\n',
    'with open("train.py", "r", encoding="utf-8") as f:\n',
    '    content = f.read()\n',
    'content = re.sub(r"custom\\s*=\\s*(True|False)", "custom = False", content)\n',
    'with open("train.py", "w", encoding="utf-8") as f:\n',
    '    f.write(content)\n',
]
cell15['source'] = new_src15

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed cells 13 and 15.')
