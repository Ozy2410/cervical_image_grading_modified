import json

with open('06_Colab_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    # Exclude cells related to raw dataset download and augmentation
    if '## 2. Dataset Preparation' in source:
        continue
    if 'import kagglehub' in source and 'prahladmehandiratta' in source:
        continue
    if '## 3. Data Augmentation' in source:
        continue
    if 'SKIP_AUGMENTATION = True' in source and 'albumentations' in source:
        continue
    
    new_cells.append(cell)

nb['cells'] = new_cells

with open('06_Colab_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cleaned up raw dataset and augmentation cells from 06_Colab_Training.ipynb")
