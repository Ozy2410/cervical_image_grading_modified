import json, sys, os

NOTEBOOK = '05_Master_Notebook_FIXED.ipynb'

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_streaming_cell(label, custom_flag):
    """Create a training cell that streams output live using Popen."""
    toggle = []
    if custom_flag is not None:
        flag_val = 'True' if custom_flag else 'False'
        toggle = [
            'import re\n',
            f'# Set custom = {flag_val}\n',
            'with open("train.py", "r", encoding="utf-8") as _f:\n',
            '    _content = _f.read()\n',
            f'_content = re.sub(r"custom\\s*=\\s*(True|False)", "custom = {flag_val}", _content)\n',
            'with open("train.py", "w", encoding="utf-8") as _f:\n',
            '    _f.write(_content)\n',
        ]

    return toggle + [
        'import subprocess, sys, os\n',
        f'print("{label}")\n',
        'proc = subprocess.Popen(\n',
        '    [sys.executable, "-u", "train.py"],\n',
        '    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n',
        '    text=True, bufsize=1\n',
        ')\n',
        'for _line in proc.stdout:\n',
        '    print(_line, end="", flush=True)\n',
        'proc.wait()\n',
    ]

# Scan ALL cells and fix any that contain the old broken patterns
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))

    # Fix cell 13 - baseline training
    if i == 13:
        nb['cells'][i]['source'] = make_streaming_cell(
            'Running baseline YOLOv4 training (custom=False)...', False)
        print(f'Fixed cell {i} (baseline training)')

    # Fix cell 15 - custom MSA-YOLO training
    elif i == 15:
        nb['cells'][i]['source'] = make_streaming_cell(
            'Running MSA-YOLO training (custom=True)...', True)
        print(f'Fixed cell {i} (custom training)')

    # Fix any remaining cells that still have !python or old rewrite pattern
    elif '!python' in src or ('train.py updated' in src):
        new_src = src.replace('!python -u train.py', 
                              'import subprocess,sys\nproc=subprocess.Popen([sys.executable,"-u","train.py"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\nfor l in proc.stdout: print(l,end="",flush=True)\nproc.wait()')
        new_src = new_src.replace('!python train.py',
                                  'import subprocess,sys\nproc=subprocess.Popen([sys.executable,"-u","train.py"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\nfor l in proc.stdout: print(l,end="",flush=True)\nproc.wait()')
        nb['cells'][i]['source'] = [new_src]
        print(f'Fixed stray !python in cell {i}')

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('\nAll done. Notebook fully patched.')
