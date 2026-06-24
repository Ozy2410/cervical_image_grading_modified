import json
import os

filename = '00_master_cervical_grading_local.ipynb'
if os.path.exists(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                # Fix dataset generation absolute path bug
                if "os.path.abspath(img_path)" in line:
                    line = line.replace("os.path.abspath(img_path)", "os.path.relpath(img_path, '.')")
                # Fix augmentation absolute path bug
                if "os.path.abspath(aug_img_path)" in line:
                    line = line.replace("os.path.abspath(aug_img_path)", "os.path.relpath(aug_img_path, '.')")
                new_source.append(line)
            cell['source'] = new_source
            
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully patched {filename} to use relative paths instead of absolute paths with spaces.")
else:
    print(f"{filename} not found.")
