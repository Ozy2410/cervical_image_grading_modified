import torch
import torch.nn as nn
import math


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda):
        super().__init__()
        self.anchors     = anchors        # [3, 2] anchor sizes in pixels
        self.num_classes = num_classes
        self.bbox_attrs  = 5 + num_classes
        self.input_shape = input_shape    # (input_w, input_h)
        self.cuda        = cuda

        self.ignore_threshold = 0.5
        self.lambda_box   = 0.05
        self.lambda_obj   = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls   = 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(t, lo, hi):
        return torch.clamp(t.float(), lo, hi)

    def bce(self, pred, target):
        pred = self._clip(pred, 1e-7, 1.0 - 1e-7)
        return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

    @staticmethod
    def ciou_loss(pred, tgt):
        """CIoU loss. Both tensors: [..., 4] in (cx, cy, w, h) format."""
        pw, ph = pred[..., 2].clamp(1e-7), pred[..., 3].clamp(1e-7)
        tw, th = tgt[..., 2].clamp(1e-7),  tgt[..., 3].clamp(1e-7)

        p_x1, p_y1 = pred[..., 0] - pw / 2, pred[..., 1] - ph / 2
        p_x2, p_y2 = pred[..., 0] + pw / 2, pred[..., 1] + ph / 2
        t_x1, t_y1 = tgt[..., 0]  - tw / 2, tgt[..., 1]  - th / 2
        t_x2, t_y2 = tgt[..., 0]  + tw / 2, tgt[..., 1]  + th / 2

        inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
                (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
        union = pw * ph + tw * th - inter + 1e-7
        iou   = inter / union

        # Enclosing box diagonal²
        c2 = ((torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)) ** 2 +
              (torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)) ** 2 + 1e-7)
        rho2 = ((pred[..., 0] - tgt[..., 0]) ** 2 +
                (pred[..., 1] - tgt[..., 1]) ** 2)

        v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        return 1 - iou + rho2 / c2 + alpha * v

    # ------------------------------------------------------------------
    # Target builder
    # ------------------------------------------------------------------
    def get_targets(self, targets, scaled_anchors, h, w):
        bs          = len(targets)
        na          = len(scaled_anchors)
        mask        = torch.zeros(bs, na, h, w)
        noobj_mask  = torch.ones (bs, na, h, w)
        tx = torch.zeros(bs, na, h, w)
        ty = torch.zeros(bs, na, h, w)
        tw = torch.zeros(bs, na, h, w)
        th = torch.zeros(bs, na, h, w)
        tconf = torch.zeros(bs, na, h, w)
        tcls  = torch.zeros(bs, na, h, w, self.num_classes)

        stride_w = self.input_shape[0] / w
        stride_h = self.input_shape[1] / h

        for b in range(bs):
            if targets[b] is None or len(targets[b]) == 0:
                continue

            t = targets[b]                           # [N, 5]: x1 y1 x2 y2 cls  (pixel)
            gx = ((t[:, 0] + t[:, 2]) / 2.0) / stride_w
            gy = ((t[:, 1] + t[:, 3]) / 2.0) / stride_h
            gw =  (t[:, 2] - t[:, 0])          / stride_w
            gh =  (t[:, 3] - t[:, 1])          / stride_h

            gw = gw.clamp(1e-7, w)
            gh = gh.clamp(1e-7, h)
            gi = gx.long().clamp(0, w - 1)
            gj = gy.long().clamp(0, h - 1)

            # Find best anchor per gt box using shape IoU at origin
            anchor_t = torch.FloatTensor([[0, 0, a[0], a[1]] for a in scaled_anchors])
            gt_t     = torch.stack([torch.zeros_like(gw), torch.zeros_like(gh), gw, gh], 1)
            iw = torch.min(gt_t[:, 2:3], anchor_t[:, 2].unsqueeze(0))
            ih = torch.min(gt_t[:, 3:4], anchor_t[:, 3].unsqueeze(0))
            inter_a = iw * ih
            union_a = gt_t[:, 2:3] * gt_t[:, 3:4] + \
                      anchor_t[:, 2].unsqueeze(0) * anchor_t[:, 3].unsqueeze(0) - inter_a
            iou_a   = inter_a / (union_a + 1e-7)   # [N, na]
            best_n  = iou_a.argmax(dim=1)           # [N]

            for i in range(len(t)):
                n = int(best_n[i])
                gi_i, gj_i = int(gi[i]), int(gj[i])
                cls_id      = int(t[i, 4])

                mask    [b, n, gj_i, gi_i] = 1
                noobj_mask[b, n, gj_i, gi_i] = 0
                # Suppress anchors with high overlap but not chosen
                for ai in range(na):
                    if float(iou_a[i, ai]) > self.ignore_threshold and ai != n:
                        noobj_mask[b, ai, gj_i, gi_i] = 0

                tx[b, n, gj_i, gi_i] = float(gx[i]) - gi_i
                ty[b, n, gj_i, gi_i] = float(gy[i]) - gj_i
                tw[b, n, gj_i, gi_i] = float(gw[i])
                th[b, n, gj_i, gi_i] = float(gh[i])
                tconf[b, n, gj_i, gi_i] = 1.0
                if 0 <= cls_id < self.num_classes:
                    tcls[b, n, gj_i, gi_i, cls_id] = 1.0

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_tensor, targets):
        bs, _, h, w = input_tensor.size()
        na = len(self.anchors)

        stride_w = self.input_shape[0] / w
        stride_h = self.input_shape[1] / h
        scaled_anchors = [(a[0] / stride_w, a[1] / stride_h) for a in self.anchors]

        # [bs, na, h, w, 5+nc]
        pred = input_tensor.view(bs, na, self.bbox_attrs, h, w).permute(0,1,3,4,2).contiguous()

        px    = torch.sigmoid(pred[..., 0])
        py    = torch.sigmoid(pred[..., 1])
        pw    = pred[..., 2]
        ph    = pred[..., 3]
        pconf = torch.sigmoid(pred[..., 4])
        pcls  = torch.sigmoid(pred[..., 5:])

        # Grid offsets
        grid_x = torch.arange(w, dtype=torch.float32).view(1,1,1,w).expand(bs,na,h,w)
        grid_y = torch.arange(h, dtype=torch.float32).view(1,1,h,1).expand(bs,na,h,w)
        aw = torch.FloatTensor([a[0] for a in scaled_anchors]).view(1,na,1,1).expand(bs,na,h,w)
        ah = torch.FloatTensor([a[1] for a in scaled_anchors]).view(1,na,1,1).expand(bs,na,h,w)

        if self.cuda:
            grid_x, grid_y, aw, ah = grid_x.cuda(), grid_y.cuda(), aw.cuda(), ah.cuda()

        pred_boxes = torch.stack([
            px + grid_x,
            py + grid_y,
            torch.exp(pw) * aw,
            torch.exp(ph) * ah,
        ], dim=-1)  # [bs, na, h, w, 4] in feature-map coords

        # Build targets
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = \
            self.get_targets(targets, scaled_anchors, h, w)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th   = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls       = tconf.cuda(), tcls.cuda()

        mask_bool = mask.bool()
        npos      = mask_bool.sum().float().clamp(min=1)

        # --- Box loss (CIoU on positive anchors) ---
        if mask_bool.any():
            pred_b = pred_boxes[mask_bool]                   # [npos, 4]
            tgt_b  = torch.stack([tx,ty,tw,th], -1)[mask_bool]
            loss_box = self.ciou_loss(pred_b, tgt_b).sum() / npos
        else:
            loss_box = torch.tensor(0.0, device=input_tensor.device, requires_grad=True)

        # --- Objectness loss ---
        loss_obj   = (self.bce(pconf, tconf) * mask).sum()     / npos
        loss_noobj = (self.bce(pconf, tconf) * noobj_mask).sum() / (noobj_mask.sum().clamp(1))

        # --- Classification loss ---
        if mask_bool.any():
            loss_cls = self.bce(pcls[mask_bool], tcls[mask_bool]).sum() / npos
        else:
            loss_cls = torch.tensor(0.0, device=input_tensor.device, requires_grad=True)

        total = (self.lambda_box   * loss_box  +
                 self.lambda_obj   * loss_obj  +
                 self.lambda_noobj * loss_noobj +
                 self.lambda_cls   * loss_cls)

        return total, total


class Generator(object):
    def __init__(self, detection_loss, training_lines, input_shape, batch_size):
        self.detection_loss = detection_loss
        self.training_lines = training_lines
        self.input_shape    = input_shape
        self.batch_size     = batch_size
