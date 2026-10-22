"""
Cascade Mask R-CNN (ResNet-50-FPN, 3 stages) builder. Seg task.

Paper: Cai & Vasconcelos, "Cascade R-CNN: High Quality Object Detection and Instance Segmentation" (TPAMI 2019,
arXiv 1906.09756), Sec. 6 (Cascade Mask R-CNN).
Code : detectron2 modeling/roi_heads/cascade_rcnn.py CascadeROIHeads with MASK_ON (configs
       Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml): the box cascade is detectors/cascade_rcnn (unchanged); ONE mask head
       (the Mask R-CNN head of detectors/mask_rcnn: RoIAlign 14x14, 4 x conv 256, deconv 2x, class-specific 28x28)
       trained on the positive stage-1 sampled RoIs (detectron2 calls _forward_mask on the proposals returned by
       label_and_sample_proposals) with the torchvision maskrcnn_loss; at inference the mask head runs on the final
       detections (last-stage boxes, stage-averaged class scores).

Raw contract: as Cascade R-CNN (one row per proposal); aux = FPN features; decode_masks kind "roi".

[Unified] as detectors/cascade_rcnn / detectors/mask_rcnn (no ImageNet weights, PreciseBN, common init, Adam 1e-4,
common post-processing); total loss = RPN + 3 stage losses + mask loss (plain sum).
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.ops import MultiScaleRoIAlign, box_iou

from detectors.cascade_rcnn.build import STAGE_IOUS, CascadeRCNN
from detectors.mask_rcnn.build import split_seg_targets


class CascadeMaskRCNN(CascadeRCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        self.mask_head = MaskRCNNHeads(256, (256, 256, 256, 256), 1)
        self.mask_predictor = MaskRCNNPredictor(256, 256, num_classes + 1)

    def mask_logits(self, features, boxes, image_sizes):
        return self.mask_predictor(self.mask_head(self.mask_roi_pool(features, boxes, image_sizes)))

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None:
            if self.training:                                    # PreciseBN forward
                return super().forward(x)
            images, features, _ = self._features(x)
            raw = self._cascade_eval(images, features, B, C)
            return raw, {"features": features, "image_sizes": images.image_sizes}

        tgt = split_seg_targets(targets, B, x.shape[2])          # boxes xyxy px, labels 1..C, masks
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        proposals, labels, reg = self.sample(proposals, tgt)     # stage 1 sampling (GT added)
        s1_props, s1_labels = proposals, labels
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

        # mask head on the positive stage-1 RoIs; matched gt = highest IoU (= the stage-1 Matcher for positives)
        pos_props, pos_idx = [], []
        for p, l, t in zip(s1_props, s1_labels, tgt):
            keep = l > 0
            pos_props.append(p[keep])
            pos_idx.append(box_iou(p[keep], t["boxes"]).argmax(1) if keep.any() and len(t["boxes"])
                           else torch.zeros(0, dtype=torch.long, device=p.device))
        if sum(len(p) for p in pos_props):
            ml = self.mask_logits(features, pos_props, images.image_sizes)
            loss_dict["loss_mask"] = maskrcnn_loss(ml, pos_props, [t["masks"] for t in tgt],
                                                   [t["labels"] for t in tgt], pos_idx)
        else:
            loss_dict["loss_mask"] = sum(p.sum() for p in self.mask_predictor.parameters()) * 0.0

        loss = sum(loss_dict.values())
        logits, lab = stat
        fg = lab > 0
        self.stats.update({**loss_dict, "losses": loss, "num_pos": fg.float().sum() / B},
                          F.softmax(logits, -1)[:, 1:].max(-1).values, logits[:, 1:], fg, lab[fg] - 1)
        return loss, x.new_zeros(B, 0, 5 + C)

    def _cascade_eval(self, images, features, B, C):
        """Inference part of CascadeRCNN.forward (unchanged), on already computed features."""
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

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        feats = OrderedDict((k, v[b:b + 1]) for k, v in aux["features"].items())
        logits = self.mask_logits(feats, [boxes], [aux["image_sizes"][b]])
        idx = torch.arange(len(labels), device=logits.device)
        return {"kind": "roi", "prob": logits[idx, labels + 1].sigmoid()}
