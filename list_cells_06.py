import json
with open('06_Colab_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print(source[:200] + ("..." if len(source) > 200 else ""))
