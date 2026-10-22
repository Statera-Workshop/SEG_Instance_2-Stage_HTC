"""
Mask R-CNN (ResNet-50-FPN) builder - the 2-Stage instance segmentation reference. Seg task.

Paper: He, Gkioxari, Dollar & Girshick, "Mask R-CNN" (ICCV 2017, arXiv 1703.06870).
Code : torchvision 0.20.1 models/detection/mask_rcnn.py (MultiScaleRoIAlign 14x14, MaskRCNNHeads 4 x conv 256,
       MaskRCNNPredictor deconv 2x + 1x1 conv) and roi_heads.maskrcnn_loss, used directly; torchvision is the
       approved official-equivalent source (the authors' code is Detectron / Caffe2). The box part is the
       detection stage's Faster R-CNN (detectors/faster_rcnn), unchanged.

    mask head  MultiScaleRoIAlign 14x14 (P2-P5, sampling 2) -> 4 x (conv 3x3 256 + ReLU) -> deconv 2x 256 + ReLU
               -> 1x1 conv (C + 1) class-specific 28x28 masks, per-pixel BCE on positive RoIs of the matched
               class, targets = ground-truth masks RoI-aligned to the proposal (torchvision maskrcnn_loss)
    inference  mask head on the final (post-NMS) boxes, channel of the predicted class, sigmoid, pasted
               into the box and thresholded at 0.5 (tasks/seg_task.py)

Raw contract: the Faster R-CNN rows ([B, P * C, 5 + C], one row per (proposal, class)); aux = FPN features.

[Unified] same as Faster R-CNN (no ImageNet weights, trainable BN, common init, no resize / mean-std, Adam 1e-4,
common post-processing, PreciseBN); the loss is the plain sum of RPN, box and mask losses (as torchvision).
"""


import torch


def split_seg_targets(targets, batch_size, img_size):
    """{"boxes": [N, 6], "masks": [N, S, S]} -> per-image dicts (boxes xyxy px, labels 1..C, masks uint8);
    same degenerate-box filter as detectors.anchor_free.common.split_targets."""
    t_all, m_all = targets["boxes"], targets["masks"]
    out = []
    for b in range(batch_size):
        sel = t_all[:, 0] == b
        t, m = t_all[sel], m_all[sel]
        cxcywh = t[:, 2:6] * img_size
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
        out.append({"boxes": boxes[keep], "labels": t[keep, 1].long() + 1, "masks": m[keep]})
    return out
