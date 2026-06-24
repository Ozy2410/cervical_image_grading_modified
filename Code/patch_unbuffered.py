import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 13 - use python -u for unbuffered real-time output
nb['cells'][13]['source'] = [
    'import sys\n',
    'print("Running baseline YOLOv4 training (custom=False)...")\n',
    '!python -u train.py\n',
]

# Fix cell 15 - same unbuffered flag
import re
src15 = ''.join(nb['cells'][15]['source'])
src15 = src15.replace('!python train.py', '!python -u train.py')
nb['cells'][15]['source'] = [src15]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done - both training cells now use python -u for live output')
