import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4 currently contains a hardcoded inline script that creates a broken dataset file
# Instead, we will replace its content to run generate_dataset.py directly from Code dir.

nb['cells'][4]['source'] = [
    'import os\n',
    'import sys\n',
    'print("Generating dataset annotation files...")\n',
    '!python generate_dataset.py\n',
]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched dataset generation cell in notebook')
