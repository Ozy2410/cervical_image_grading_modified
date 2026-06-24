"""
Offline augmentation for SIPaKMeD YOLO training.

Reads:
  - lzsp_train20201202.txt  (list of image paths, one per line)
  - <image_path>.txt        (YOLO sidecar label file per image)

Writes:
  - sipakmed_data/aug_images/<aug_name>.bmp   (augmented image)
  - sipakmed_data/aug_labels/<aug_name>.txt   (matching YOLO label)

Then appends augmented image paths to lzsp_train20201202.txt.
"""

import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

AUG_IMG_DIR = "sipakmed_data/aug_images"
AUG_LBL_DIR = "sipakmed_data/aug_labels"
ANNOTATION_FILE = "lzsp_train20201202.txt"
NUM_COPIES = 2  # how many augmented versions per image

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)

transform = A.Compose([
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3,
    clip=True,
))

# Load existing image paths (skip anything already augmented)
with open(ANNOTATION_FILE, 'r') as f:
    all_lines = [l.strip() for l in f if l.strip()]

original_lines = [l for l in all_lines if 'aug_' not in os.path.basename(l)]
print(f"Found {len(original_lines)} original images to augment ({NUM_COPIES} copies each).")

aug_image_paths = []
skipped = 0

for img_path in tqdm(original_lines, desc="Augmenting"):
    # Find matching YOLO label file
    txt_path = os.path.splitext(img_path)[0] + '.txt'
    if not os.path.exists(txt_path):
        skipped += 1
        continue

    # Read YOLO labels: class_id cx cy w h (all normalized 0-1)
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls_id, cx, cy, bw, bh))

    if not labels:
        skipped += 1
        continue

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        skipped += 1
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [(cx, cy, bw, bh) for (_, cx, cy, bw, bh) in labels]
    class_labels = [cls_id for (cls_id, _, _, _, _) in labels]

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(NUM_COPIES):
        try:
            result = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            if len(result['bboxes']) == 0:
                continue

            aug_img_name = f"aug_{i}_{base_name}.bmp"
            aug_img_path = os.path.join(AUG_IMG_DIR, aug_img_name)
            aug_lbl_path = os.path.join(AUG_LBL_DIR, f"aug_{i}_{base_name}.txt")

            # Save augmented image
            cv2.imwrite(aug_img_path, cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))

            # Save augmented YOLO label
            with open(aug_lbl_path, 'w') as f:
                for box, cls in zip(result['bboxes'], result['class_labels']):
                    cx, cy, bw, bh = box
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            aug_image_paths.append(os.path.relpath(aug_img_path, '.'))

        except Exception as e:
            pass

# Append new augmented image paths to annotation file
if aug_image_paths:
    # Remove any stale aug entries from the annotation file first
    clean_lines = [l for l in all_lines if 'aug_' not in os.path.basename(l)]
    with open(ANNOTATION_FILE, 'w') as f:
        for line in clean_lines:
            f.write(line + '\n')
        for aug_path in aug_image_paths:
            f.write(aug_path + '\n')

    print(f"\nDone!")
    print(f"  Augmented images created : {len(aug_image_paths)}")
    print(f"  Skipped (no label/image) : {skipped}")
    print(f"  Total in annotation file : {len(clean_lines) + len(aug_image_paths)}")
else:
    print("No augmented images were created. Check that label files exist alongside images.")
