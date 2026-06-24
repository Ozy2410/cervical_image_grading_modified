import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 13 - baseline: use %run instead of !python
nb['cells'][13]['source'] = [
    'print("Running baseline YOLOv4 training (custom=False)...")\n',
    '%run train.py\n',
]

# Cell 15 - custom: use %run instead of !python
import re
src15 = ''.join(nb['cells'][15]['source'])
src15 = src15.replace('!python -u train.py', '%run train.py').replace('!python train.py', '%run train.py')
nb['cells'][15]['source'] = [src15]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done - switched to %run for live in-kernel output')
