import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, mosaic=False):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.mosaic = mosaic

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip()
        image_path = line.split()[0]

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.fromarray(np.uint8(np.random.rand(self.input_shape[0], self.input_shape[1], 3) * 255))

        iw, ih = image.size

        # Load YOLO .txt label file.
        # Augmented images store labels in aug_labels/ instead of next to the image.
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if 'aug_images' in image_path.replace('\\', '/'):
            aug_labels_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'aug_labels')
            txt_path = os.path.join(aug_labels_dir, base_name + '.txt')
        else:
            txt_path = os.path.splitext(image_path)[0] + '.txt'
        box = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for lbl_line in f:
                    parts = lbl_line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert normalized YOLO -> pixel x_min,y_min,x_max,y_max
                    x_min = int((cx - bw / 2) * iw)
                    y_min = int((cy - bh / 2) * ih)
                    x_max = int((cx + bw / 2) * iw)
                    y_max = int((cy + bh / 2) * ih)
                    box.append([x_min, y_min, x_max, y_max, cls_id])
        box = np.array(box, dtype=np.float32) if box else np.zeros((0, 5), dtype=np.float32)

        # Resize image with letterboxing
        h, w = self.input_shape
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, dtype=np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))

        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
