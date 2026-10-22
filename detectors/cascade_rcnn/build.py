"""
Cascade R-CNN builder (ResNet-50-FPN, 3 stages), Image task.

Paper: Cai & Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection" (CVPR 2018, arXiv 1712.00726).
Code : detectron2 modeling/roi_heads/cascade_rcnn.py (CascadeROIHeads; configs Misc/cascade_mask_rcnn_R_50_FPN_*:
       IOUS 0.5 / 0.6 / 0.7, BBOX_REG_WEIGHTS (10, 10, 5, 5) / (20, 20, 10, 10) / (30, 30, 15, 15), class-agnostic
       box regression, gradient of the RoI features scaled by 1 / 3, stage 1 = sampled RoIs (512, 25 %, GT added),
       stages 2-3 = all boxes of the previous stage re-labelled with the stage IoU (no sampling); inference: scores
       averaged over the 3 stages, boxes of the last stage). The official zhaoweicai/cascade-rcnn is Caffe;
       detectron2 is the approved official-equivalent. Backbone, RPN, RoIAlign, 2-FC heads and losses are the
       torchvision Faster R-CNN pieces of detectors/faster_rcnn (one 2-FC head + predictor per stage).

Raw contract: one row per proposal: last-stage box and the stage-averaged class probabilities (objectness 1).

[Unified] as detectors/faster_rcnn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.ops import box_iou, clip_boxes_to_image

from detectors.anchor_free.common import HeadStats, split_targets
from detectors.faster_rcnn.build import FasterRCNNDet

STAGE_IOUS = (0.5, 0.6, 0.7)
STAGE_WEIGHTS = ((10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0))


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class CascadeRCNN(FasterRCNNDet):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        rh = self.det.roi_heads
        rep = rh.box_head.fc7.out_features
        in_feat = rh.box_head.fc6.in_features
        self.box_heads = nn.ModuleList([rh.box_head] + [TwoMLPHead(in_feat, rep) for _ in range(2)])
        self.cls_scores = nn.ModuleList([nn.Linear(rep, num_classes + 1) for _ in STAGE_IOUS])
        self.bbox_preds = nn.ModuleList([nn.Linear(rep, 4) for _ in STAGE_IOUS])      # class-agnostic
        rh.box_predictor = None
        self.coders = [det_utils.BoxCoder(w) for w in STAGE_WEIGHTS]
        self.matchers = [det_utils.Matcher(t, t, allow_low_quality_matches=False) for t in STAGE_IOUS]
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _stage(self, k, features, proposals, image_sizes):
        rh = self.det.roi_heads
        f = rh.box_roi_pool(features, proposals, image_sizes)
        f = _ScaleGradient.apply(f, 1.0 / len(STAGE_IOUS))
        f = self.box_heads[k](f)
        return self.cls_scores[k](f), self.bbox_preds[k](f)

    def _decode(self, k, deltas, proposals, image_sizes):
        boxes = self.coders[k].decode(deltas, proposals).squeeze(1)
        out, i = [], 0
        for p, size in zip(proposals, image_sizes):
            out.append(clip_boxes_to_image(boxes[i:i + len(p)], size).detach())
            i += len(p)
        return out

    @torch.no_grad()
    def _label(self, k, proposals, tgt):
        labels, reg = [], []
        for p, t in zip(proposals, tgt):
            if len(t["boxes"]):
                m = self.matchers[k](box_iou(t["boxes"], p))
                lab = t["labels"][m.clamp(min=0)].clone()
                lab[m < 0] = 0
                gt = t["boxes"][m.clamp(min=0)]
            else:
                lab = torch.zeros(len(p), dtype=torch.long, device=p.device)
                gt = torch.zeros_like(p)
            labels.append(lab)
            reg.append(self.coders[k].encode([gt], [p])[0])
        return labels, reg

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None and self.training:                           # PreciseBN (see detectors/faster_rcnn)
            self._features(x)
            return x.new_zeros(B, 0, 5 + C)
        if targets is None:
            images, features, _ = self._features(x)
            proposals, _ = self.det.rpn(images, features)
            n = min(len(p) for p in proposals)
            proposals = [p[:n] for p in proposals]
            probs, deltas = [], None
            for k in range(len(STAGE_IOUS)):
                if k > 0:
                    proposals = self._decode(k - 1, deltas, proposals, images.image_sizes)
                logits, deltas = self._stage(k, features, proposals, images.image_sizes)
                probs.append(F.softmax(logits, -1))
            boxes = torch.cat(self._decode(len(STAGE_IOUS) - 1, deltas, proposals, images.image_sizes))
            scores = (sum(probs) / len(probs))[:, 1:]
            boxes, scores = boxes.view(B, n, 4), scores.view(B, n, C)
            cxcy = (boxes[..., :2] + boxes[..., 2:]) / 2
            wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
            return torch.cat([cxcy, wh, torch.ones_like(scores[..., :1]), scores], -1)
        tgt = [{"boxes": t["boxes"], "labels": t["labels"] + 1} for t in split_targets(targets, B, x.shape[2])]
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        proposals, labels, reg = self.sample(proposals, tgt)            # stage 1: torchvision sampling (coder 10,10,5,5)
        loss_dict, deltas, stat = dict(rpn_losses), None, None
        for k in range(len(STAGE_IOUS)):
            if k > 0:
                proposals = self._decode(k - 1, deltas, proposals, images.image_sizes)
                labels, reg = self._label(k, proposals, tgt)
            logits, deltas = self._stage(k, features, proposals, images.image_sizes)
            lab, rt = torch.cat(labels), torch.cat(reg)
            pos = lab > 0
            loss_dict[f"loss_cls_s{k + 1}"] = F.cross_entropy(logits, lab)
            loss_dict[f"loss_box_s{k + 1}"] = F.smooth_l1_loss(deltas[pos], rt[pos], beta=1 / 9,
                                                                reduction="sum") / lab.numel()
            if k == 0:
                stat = (logits.detach(), lab)
        loss = sum(loss_dict.values())
        logits, lab = stat
        fg = lab > 0
        self.stats.update({**loss_dict, "losses": loss, "num_pos": fg.float().sum() / B},
                          F.softmax(logits, -1)[:, 1:].max(-1).values, logits[:, 1:], fg, lab[fg] - 1)
        return loss, x.new_zeros(B, 0, 5 + C)
