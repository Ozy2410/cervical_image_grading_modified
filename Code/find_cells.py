import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'lstm' in src.lower() or 'evaluation' in src.lower() or 'eval' in src.lower():
        ct = cell['cell_type']
        print(f'=== Cell {i} ({ct}) ===')
        print(src[:600])
        print()
