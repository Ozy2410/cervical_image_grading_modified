import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('05_Master_Notebook_FIXED.ipynb', encoding='utf-8'))
for i in [17, 21, 23]:
    src = ''.join(nb['cells'][i]['source'])
    print(f"===== Cell {i} =====")
    print(src[:3000])
    print()
