import json

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

eval_code = '''import os, sys, glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# --- Configuration ---
CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
NUM_CLASSES = len(CLASSES)
INPUT_SHAPE = (608, 608)

# --- Find saved weight files ---
baseline_dir = "trained_weights/baseline_yolov4"
msa_dir = "trained_weights/msa_yolo"

def get_weight_files(wdir):
    if not os.path.exists(wdir):
        return []
    files = sorted(glob.glob(os.path.join(wdir, "Epoch*.pth")),
                   key=lambda x: int(os.path.basename(x).split("-")[0].replace("Epoch","")))
    return files

baseline_weights = get_weight_files(baseline_dir)
msa_weights = get_weight_files(msa_dir)

print(f"Found {len(baseline_weights)} baseline checkpoints")
print(f"Found {len(msa_weights)} MSA-YOLO checkpoints")

# --- Extract epoch and loss from filenames ---
def parse_weight_file(filepath):
    name = os.path.basename(filepath).replace(".pth", "")
    parts = name.split("-")
    epoch = int(parts[0].replace("Epoch", ""))
    train_loss = float(parts[1].replace("Total_Loss", ""))
    val_loss = float(parts[2].replace("Val_Loss", ""))
    return epoch, train_loss, val_loss

# --- Plot training curves ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for wfiles, label, ax in [(baseline_weights, "Baseline YOLOv4", axs[0]),
                           (msa_weights, "MSA-YOLO", axs[1])]:
    if not wfiles:
        ax.set_title(f"{label} - No weights found")
        ax.text(0.5, 0.5, "No checkpoints saved yet.\\nRun training first.",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        continue
    epochs, train_losses, val_losses = [], [], []
    for wf in wfiles:
        e, tl, vl = parse_weight_file(wf)
        epochs.append(e)
        train_losses.append(tl)
        val_losses.append(vl)
    ax.plot(epochs, train_losses, "o-", label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses, "s--", label="Val Loss", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label} - Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"\\n{label} Training Summary:")
    header = f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14}"
    print(header)
    print(f"  {'-----':<8} {'----------':<14} {'--------':<14}")
    for e, tl, vl in zip(epochs, train_losses, val_losses):
        print(f"  {e:<8} {tl:<14.4f} {vl:<14.4f}")
    best_idx = int(np.argmin(val_losses))
    print(f"  Best checkpoint: Epoch {epochs[best_idx]} (Val Loss: {val_losses[best_idx]:.4f})")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("\\nSaved training_curves.png")
'''

# Convert to notebook source format (list of lines)
nb['cells'][25]['source'] = [line + '\n' for line in eval_code.split('\n')]

with open('05_Master_Notebook_FIXED.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed evaluation cell 25')
