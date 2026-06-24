import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][13]['source'] = [
    'import subprocess, sys, os\n',
    'os.chdir(os.path.dirname(os.path.abspath("train.py")) if os.path.exists("train.py") else os.getcwd())\n',
    'print("Running baseline YOLOv4 training (custom=False)...")\n',
    'proc = subprocess.Popen(\n',
    '    [sys.executable, "-u", "train.py"],\n',
    '    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n',
    '    text=True, bufsize=1\n',
    ')\n',
    'for line in proc.stdout:\n',
    '    print(line, end="", flush=True)\n',
    'proc.wait()\n',
]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done.')
