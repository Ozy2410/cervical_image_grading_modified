import sys
sys.path.insert(0, '.')
import torch
from nets.yolo4 import YoloBody

print("Testing baseline YOLOv4 (custom=False)...")
model_base = YoloBody(num_anchors=3, num_classes=5, custom=False)
x = torch.randn(1, 3, 608, 608)
with torch.no_grad():
    outs = model_base(x)
print(f"  Outputs: {len(outs)} heads")
for i, o in enumerate(outs):
    print(f"  Head {i}: {o.shape}")

print("\nTesting MSA-YOLO (custom=True)...")
model_msa = YoloBody(num_anchors=3, num_classes=5, custom=True)
with torch.no_grad():
    outs = model_msa(x)
print(f"  Outputs: {len(outs)} heads")
for i, o in enumerate(outs):
    print(f"  Head {i}: {o.shape}")

print("\nBoth models pass forward test!")
