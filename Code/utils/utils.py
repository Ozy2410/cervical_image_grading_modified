import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate([((top + bottom) / 2)[..., None], ((left + right) / 2)[..., None]], axis=-1)
    box_hw = np.concatenate([(bottom - top)[..., None], (right - left)[..., None]], axis=-1)

    box_yx = (box_yx / input_shape - offset) * scale * image_shape
    box_hw = box_hw / input_shape * scale * image_shape

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    return np.concatenate([
        box_mins[..., 1:2], # x_min
        box_mins[..., 0:1], # y_min
        box_maxes[..., 1:2], # x_max
        box_maxes[..., 0:1]  # y_max
    ], axis=-1)

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

class DecodeBox(object):
    def __init__(self, anchors, num_classes, input_shape):
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape

    def decode_box(self, inputs):
        # Dummy box decoding logic returning clean tensors
        return torch.zeros((1, 100, 4)), torch.zeros((1, 100, self.num_classes)), torch.zeros((1, 100))

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # Dummy NMS logic returning clean boxes
    return [None]
