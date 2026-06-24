import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find augmentation cell (index 9)
cell = nb['cells'][9]
source = cell['source']

# Wrap entire cell with skip guard
new_source = [
    '# Set to True to skip augmentation (already have augmented data)\n',
    'SKIP_AUGMENTATION = True\n',
    '\n',
    'if not SKIP_AUGMENTATION:\n',
]
for line in source:
    new_source.append('    ' + line if line.strip() else '\n')

new_source.append('\nelse:\n')
new_source.append('    print("Skipping augmentation - 10028 augmented images already exist.")\n')

cell['source'] = new_source

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done')
