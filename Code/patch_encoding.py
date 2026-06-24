import json

NOTEBOOK = '05_Master_Notebook_FIXED.ipynb'

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_streaming_cell(label, custom_flag):
    toggle = []
    if custom_flag is not None:
        flag_val = 'True' if custom_flag else 'False'
        toggle = [
            'import re\n',
            f'with open("train.py", "r", encoding="utf-8") as _f:\n',
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
        '    encoding="utf-8", errors="replace", bufsize=1\n',
        ')\n',
        'for _line in proc.stdout:\n',
        '    print(_line, end="", flush=True)\n',
        'proc.wait()\n',
    ]

nb['cells'][13]['source'] = make_streaming_cell(
    'Running baseline YOLOv4 training (custom=False)...', False)
nb['cells'][15]['source'] = make_streaming_cell(
    'Running MSA-YOLO training (custom=True)...', True)

# Fix cell 19 too if it has the old pattern
for i, cell in enumerate(nb['cells']):
    if i in (13, 15): continue
    src = ''.join(cell.get('source', []))
    if 'bufsize=1' in src and 'encoding' not in src:
        nb['cells'][i]['source'] = [src.replace(
            'text=True, bufsize=1',
            'encoding="utf-8", errors="replace", bufsize=1'
        )]
        print(f'Fixed encoding in cell {i}')

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done.')
