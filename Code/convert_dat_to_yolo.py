"""
Convert ALL SIPaKMeD .dat polygon files to YOLO format .txt label files.
Handles both:
  - Full slide images: sipakmed_data/im_<class>/<class>/*.bmp  (with _cytNN.dat)
  - Cropped cell images: sipakmed_data/im_<class>/<class>/CROPPED/*.bmp  (with _cyt.dat)
"""

import os
import glob
from PIL import Image

CLASSES = {
    'im_Dyskeratotic': 0,
    'im_Koilocytotic': 1,
    'im_Metaplastic': 2,
    'im_Parabasal': 3,
    'im_Superficial-Intermediate': 4,
}

SIPAKMED_ROOT = 'sipakmed_data'
converted_images = 0
total_cells = 0
skipped = 0


def parse_dat(dat_path):
    """Read polygon x,y points from a .dat file. Returns list of (x,y) tuples."""
    points = []
    try:
        with open(dat_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(' ', ',').split(',')
                if len(parts) >= 2:
                    points.append((float(parts[0]), float(parts[1])))
    except Exception as e:
        pass
    return points


def bbox_from_polygon(points, img_w, img_h):
    """Compute normalized YOLO bbox from polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
    y_min, y_max = max(0, min(ys)), min(img_h, max(ys))
    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    return cx, cy, bw / img_w, bh / img_h


def convert_directory(img_dir, class_id, cyt_pattern):
    """Convert all images in img_dir using cyt_pattern to find annotation dats."""
    global converted_images, total_cells, skipped

    bmp_files = glob.glob(os.path.join(img_dir, '*.bmp'))
    for bmp_path in bmp_files:
        try:
            with Image.open(bmp_path) as img:
                img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        base = os.path.splitext(os.path.basename(bmp_path))[0]

        # Find matching cyt dat files
        cyt_dats = sorted(glob.glob(os.path.join(img_dir, cyt_pattern.format(base=base))))

        if not cyt_dats:
            skipped += 1
            continue

        yolo_lines = []
        for dat_path in cyt_dats:
            points = parse_dat(dat_path)
            if not points:
                continue
            result = bbox_from_polygon(points, img_w, img_h)
            if result is None:
                continue
            cx, cy, bw, bh = result
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_cells += 1

        if yolo_lines:
            txt_path = os.path.splitext(bmp_path)[0] + '.txt'
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            converted_images += 1
        else:
            skipped += 1


for class_name, class_id in CLASSES.items():
    class_dir = os.path.join(SIPAKMED_ROOT, class_name, class_name)
    if not os.path.exists(class_dir):
        print(f"Skipping {class_name} - not found")
        continue

    # 1. Full slide images: 001.bmp -> 001_cyt01.dat, 001_cyt02.dat ...
    convert_directory(class_dir, class_id, '{base}_cyt*.dat')

    # 2. Cropped cell images: CROPPED/001_01.bmp -> CROPPED/001_01_cyt.dat
    cropped_dir = os.path.join(class_dir, 'CROPPED')
    if os.path.exists(cropped_dir):
        convert_directory(cropped_dir, class_id, '{base}_cyt.dat')

    print(f"  {class_name}: done")

print(f"\n=== CONVERSION COMPLETE ===")
print(f"  Converted images : {converted_images}")
print(f"  Total cell boxes : {total_cells}")
print(f"  Skipped (no dat) : {skipped}")
print(f"  Avg cells/image  : {total_cells / max(1, converted_images):.2f}")
