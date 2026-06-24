import os
import glob

SIPAKMED_ROOT = 'sipakmed_data'
AUG_ROOT = 'sipakmed_data/aug_images'

CLASSES = [
    'im_Dyskeratotic',
    'im_Koilocytotic',
    'im_Metaplastic',
    'im_Parabasal',
    'im_Superficial-Intermediate',
]

lines = []
missing_labels = 0

print("Generating dataset list...")

# 1. Add Original SIPaKMeD images
for cls in CLASSES:
    # Full slide images
    class_dir = os.path.join(SIPAKMED_ROOT, cls, cls)
    if os.path.exists(class_dir):
        for img_file in sorted(glob.glob(os.path.join(class_dir, '*.[bj][mp][pg]*'))):
            txt = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(txt):
                lines.append(os.path.relpath(img_file, '.').replace('\\', '/') + '\n')
            else:
                missing_labels += 1

    # Cropped images
    cropped_dir = os.path.join(SIPAKMED_ROOT, cls, cls, 'CROPPED')
    if os.path.exists(cropped_dir):
        for img_file in sorted(glob.glob(os.path.join(cropped_dir, '*.[bj][mp][pg]*'))):
            txt = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(txt):
                lines.append(os.path.relpath(img_file, '.').replace('\\', '/') + '\n')
            else:
                missing_labels += 1

# 2. Add Augmented Images (if any exist)
if os.path.exists(AUG_ROOT):
    # Augmented images were saved as .bmp or .jpg!
    for img_file in sorted(glob.glob(os.path.join(AUG_ROOT, '*.[bj][mp][pg]*'))):
        basename = os.path.basename(img_file)
        txt = os.path.join(SIPAKMED_ROOT, 'aug_labels', os.path.splitext(basename)[0] + '.txt')
        if os.path.exists(txt):
            lines.append(os.path.relpath(img_file, '.').replace('\\', '/') + '\n')
        else:
            missing_labels += 1

with open('lzsp_train20201202.txt', 'w') as f:
    f.writelines(lines)

print(f"Generated lzsp_train20201202.txt with {len(lines)} labeled images.")
print(f"Skipped {missing_labels} images with no .txt label file.")
