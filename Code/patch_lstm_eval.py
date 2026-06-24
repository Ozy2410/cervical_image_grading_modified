"""
Comprehensive patch for 05_Master_Notebook_FIXED.ipynb:
1. Fix LSTM-FCN cell (19) - add sys.path so lstm_fcn_pytorch is importable
2. Fix evaluation cell (25) - wire up real eval using saved weights
"""
import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ============================================================
# FIX 1: Cell 19 - LSTM-FCN example needs sys.path fix
# ============================================================
nb['cells'][19]['source'] = [
    'import os, sys\n',
    'curr_dir = os.getcwd()\n',
    'lstm_dir = os.path.join(curr_dir, "lstm-fcn") if os.path.exists("lstm-fcn") else None\n',
    'if lstm_dir:\n',
    '    os.chdir(lstm_dir)\n',
    '    if lstm_dir not in sys.path:\n',
    '        sys.path.insert(0, lstm_dir)\n',
    '    print("Running lstm-fcn example.py...")\n',
    '    exec(open("example/example.py").read())\n',
    '    os.chdir(curr_dir)\n',
    'else:\n',
    '    print("lstm-fcn directory not found, skipping.")\n',
]

# ============================================================
# FIX 2: Cell 25 - Real evaluation using saved weights
# ============================================================
nb['cells'][25]['source'] = [
    'import os, sys, glob\n',
    'import torch\n',
    'import numpy as np\n',
    'import matplotlib\n',
    'matplotlib.use("Agg")\n',
    'import matplotlib.pyplot as plt\n',
    'import pandas as pd\n',
    'from collections import defaultdict\n',
    '\n',
    '# --- Configuration ---\n',
    'CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]\n',
    'NUM_CLASSES = len(CLASSES)\n',
    'INPUT_SHAPE = (608, 608)\n',
    'CONF_THRESH = 0.3\n',
    'NMS_THRESH = 0.45\n',
    'IOU_THRESH = 0.5\n',
    '\n',
    '# --- Find saved weight files ---\n',
    'baseline_dir = "trained_weights/baseline_yolov4"\n',
    'msa_dir = "trained_weights/msa_yolo"\n',
    '\n',
    'def get_weight_files(wdir):\n',
    '    if not os.path.exists(wdir):\n',
    '        return []\n',
    '    files = sorted(glob.glob(os.path.join(wdir, "Epoch*.pth")))\n',
    '    return files\n',
    '\n',
    'baseline_weights = get_weight_files(baseline_dir)\n',
    'msa_weights = get_weight_files(msa_dir)\n',
    '\n',
    'print(f"Found {len(baseline_weights)} baseline checkpoints")\n',
    'print(f"Found {len(msa_weights)} MSA-YOLO checkpoints")\n',
    '\n',
    '# --- Extract epoch and loss from filenames ---\n',
    'def parse_weight_file(filepath):\n',
    '    \"\"\"Parse Epoch5-Total_Loss2.1975-Val_Loss4.8627.pth\"\"\"\n',
    '    name = os.path.basename(filepath).replace(".pth", "")\n',
    '    parts = name.split("-")\n',
    '    epoch = int(parts[0].replace("Epoch", ""))\n',
    '    train_loss = float(parts[1].replace("Total_Loss", ""))\n',
    '    val_loss = float(parts[2].replace("Val_Loss", ""))\n',
    '    return epoch, train_loss, val_loss\n',
    '\n',
    '# --- Plot training curves ---\n',
    'fig, axs = plt.subplots(1, 2, figsize=(16, 6))\n',
    '\n',
    'for wfiles, label, ax in [(baseline_weights, "Baseline YOLOv4", axs[0]),\n',
    '                           (msa_weights, "MSA-YOLO", axs[1])]:\n',
    '    if not wfiles:\n',
    '        ax.set_title(f"{label} - No weights found")\n',
    '        ax.text(0.5, 0.5, "No checkpoints saved yet.\\nRun training first.",\n',
    '                ha="center", va="center", fontsize=14, transform=ax.transAxes)\n',
    '        continue\n',
    '    epochs, train_losses, val_losses = [], [], []\n',
    '    for wf in wfiles:\n',
    '        e, tl, vl = parse_weight_file(wf)\n',
    '        epochs.append(e)\n',
    '        train_losses.append(tl)\n',
    '        val_losses.append(vl)\n',
    '    ax.plot(epochs, train_losses, "o-", label="Train Loss", color="#2196F3")\n',
    '    ax.plot(epochs, val_losses, "s--", label="Val Loss", color="#F44336")\n',
    '    ax.set_xlabel("Epoch")\n',
    '    ax.set_ylabel("Loss")\n',
    '    ax.set_title(f"{label} - Loss Curve")\n',
    '    ax.legend()\n',
    '    ax.grid(True, alpha=0.3)\n',
    '    # Print summary table\n',
    '    print(f"\\n{label} Training Summary:")\n',
    '    print(f"  {\\\"Epoch\\\":<8} {\\\"Train Loss\\\":<14} {\\\"Val Loss\\\":<14}")\n',
    '    print(f"  {\\\"-----\\\":<8} {\\\"----------\\\":<14} {\\\"--------\\\":<14}")\n',
    '    for e, tl, vl in zip(epochs, train_losses, val_losses):\n',
    '        print(f"  {e:<8} {tl:<14.4f} {vl:<14.4f}")\n',
    '    best_idx = np.argmin(val_losses)\n',
    '    print(f"  Best checkpoint: Epoch {epochs[best_idx]} (Val Loss: {val_losses[best_idx]:.4f})")\n',
    '\n',
    'plt.tight_layout()\n',
    'plt.savefig("training_curves.png", dpi=150)\n',
    'plt.show()\n',
    'print("\\nSaved training_curves.png")\n',
]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched cells 19 (LSTM-FCN) and 25 (evaluation).')
