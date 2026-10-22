"""
HTC - Hybrid Task Cascade (ResNet-50-FPN, 3 stages, without the semantic branch) builder. Seg task.

Paper: Chen et al., "Hybrid Task Cascade for Instance Segmentation" (CVPR 2019, arXiv 1901.07518), Sec. 3.1-3.2.
Code : open-mmlab/mmdetection configs/htc/htc-without-semantic_r50_fpn_1x_coco.py, models/roi_heads/htc_roi_head.py
       (HybridTaskCascadeRoIHead, interleaved=True, mask_info_flow=True, with_semantic=False),
       models/roi_heads/mask_heads/htc_mask_head.py (HTCMaskHead). Apache-2.0. Reimplemented here.

    box cascade      detectors/cascade_rcnn unchanged (= detectors/cascade_mask_rcnn): stage IoUs 0.5 / 0.6 / 0.7,
                     stage-1 sampling, stages 2-3 re-labelled, class-agnostic regression, scores averaged at test
    interleaved      the mask branch of stage t is trained on the boxes REGRESSED by the box branch of stage t:
                     refined boxes (+ gt boxes) re-assigned with the stage IoU, up to 128 positives per image
                     (mmdet: re-sample 512 RoIs, 25 % positives, gt added)
    mask info flow   one mask head per stage (RoIAlign 14x14 -> 4 x conv3x3 256 -> deconv 2x -> class-specific 28x28,
                     the Mask R-CNN head); head t (t > 0) adds conv1x1+ReLU(feature of head t-1) to its RoI feature
                     before the convs; the earlier heads are re-run on the stage-t RoIs (gradient flows through them)
    loss             mask loss of stage t x stage weight (1, 0.5, 0.25) (mmdet stage_loss_weights), BCE on the
                     class channel (torchvision maskrcnn_loss)
    inference        final boxes as Cascade (last-stage boxes, stage-averaged scores); masks = mean of the sigmoid
                     mask predictions of the 3 heads run with the info flow on the final boxes (mmdet merge_aug_masks)

[Unified]
  - box cascade and its losses are the framework's Cascade R-CNN (detectron2 equivalent: box losses unweighted, RoI
    feature gradient x 1/3), so HTC stays comparable with Cascade Mask R-CNN; only the mask losses carry mmdet's
    stage weights
  - the semantic segmentation branch (HTC's third part) needs stuff annotations (COCO-stuff) that FL36-Seg does not
    have; the official htc-without-semantic config is the reference
  - no ImageNet weights, PreciseBN, common init, Adam 1e-4, common post-processing (as detectors/cascade_mask_rcnn)

Raw contract: as Cascade R-CNN; aux = FPN features; decode_masks kind "roi".
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.ops import box_iou

from detectors.cascade_mask_rcnn.build import CascadeMaskRCNN
from detectors.cascade_rcnn.build import STAGE_IOUS
from detectors.mask_rcnn.build import split_seg_targets

MASK_STAGE_WEIGHTS = (1.0, 0.5, 0.25)
MASK_POS_PER_IMAGE = 128                                   # 512 sampled RoIs x 0.25 positive fraction


class HTC(CascadeMaskRCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.mask_head, self.mask_predictor                # replaced by one head per stage
        S = len(STAGE_IOUS)
        self.mask_heads = nn.ModuleList(MaskRCNNHeads(256, (256, 256, 256, 256), 1) for _ in range(S))
        self.mask_preds = nn.ModuleList(MaskRCNNPredictor(256, 256, num_classes + 1) for _ in range(S))
        self.conv_res = nn.ModuleList(nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU(inplace=True))
                                      for _ in range(S - 1))    # HTCMaskHead.conv_res of heads 2 and 3

    def _mask_feats(self, roi_feats, upto):
        """Mask info flow: features of heads 0..upto on the same RoIs."""
        feats, last = [], None
        for i in range(upto + 1):
            x = roi_feats if last is None else roi_feats + self.conv_res[i - 1](last)
            last = self.mask_heads[i](x)
            feats.append(last)
        return feats

    def _mask_loss(self, k, features, boxes, tgt, image_sizes):
        rois, idx = [], []
        for p, t in zip(boxes, tgt):
            if len(t["boxes"]) == 0:
                rois.append(p[:0])
                idx.append(torch.zeros(0, dtype=torch.long, device=p.device))
                continue
            cand = torch.cat([p, t["boxes"]])                      # sampler adds the gt boxes
            m = self.matchers[k](box_iou(t["boxes"], cand))       # stage-k IoU, -1 below
            pos = torch.nonzero(m >= 0).squeeze(1)
            if len(pos) > MASK_POS_PER_IMAGE:
                pos = pos[torch.randperm(len(pos), device=pos.device)[:MASK_POS_PER_IMAGE]]
            rois.append(cand[pos])
            idx.append(m[pos])
        if sum(len(r) for r in rois) == 0:
            return sum(p.sum() for p in self.mask_preds[k].parameters()) * 0.0
        roi_feats = self.mask_roi_pool(features, rois, image_sizes)
        logits = self.mask_preds[k](self._mask_feats(roi_feats, k)[-1])
        return maskrcnn_loss(logits, rois, [t["masks"] for t in tgt], [t["labels"] for t in tgt], idx)

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None:
            return super().forward(x)                              # PreciseBN / inference: Cascade Mask R-CNN path

        tgt = split_seg_targets(targets, B, x.shape[2])            # boxes xyxy px, labels 1..C, masks
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        proposals, labels, reg = self.sample(proposals, tgt)       # stage 1 sampling (GT added)
        loss_dict, stat = dict(rpn_losses), None
        for k in range(len(STAGE_IOUS)):
            if k > 0:
                labels, reg = self._label(k, proposals, tgt)
            logits, deltas = self._stage(k, features, proposals, images.image_sizes)
            lab, rt = torch.cat(labels), torch.cat(reg)
            pos = lab > 0
            loss_dict[f"loss_cls_s{k + 1}"] = F.cross_entropy(logits, lab)
            loss_dict[f"loss_box_s{k + 1}"] = F.smooth_l1_loss(deltas[pos], rt[pos], beta=1 / 9,
                                                                reduction="sum") / lab.numel()
            if k == 0:
                stat = (logits.detach(), lab)
            refined = self._decode(k, deltas, proposals, images.image_sizes)    # detached stage-k boxes
            loss_dict[f"loss_mask_s{k + 1}"] = MASK_STAGE_WEIGHTS[k] * self._mask_loss(
                k, features, refined, tgt, images.image_sizes)              # interleaved execution
            proposals = refined                                              # = Cascade's next-stage proposals

        loss = sum(loss_dict.values())
        logits, lab = stat
        fg = lab > 0
        self.stats.update({**loss_dict, "losses": loss, "num_pos": fg.float().sum() / B},
                          F.softmax(logits, -1)[:, 1:].max(-1).values, logits[:, 1:], fg, lab[fg] - 1)
        return loss, x.new_zeros(B, 0, 5 + C)

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        feats = OrderedDict((k, v[b:b + 1]) for k, v in aux["features"].items())
        roi_feats = self.mask_roi_pool(feats, [boxes], [aux["image_sizes"][b]])
        hf = self._mask_feats(roi_feats, len(STAGE_IOUS) - 1)
        prob = sum(self.mask_preds[i](f).sigmoid() for i, f in enumerate(hf)) / len(hf)
        idx = torch.arange(len(labels), device=prob.device)
        return {"kind": "roi", "prob": prob[idx, labels + 1]}


def build_model(task, num_classes):
    return HTC(num_classes)
