"""Appearance embeddings of person boxes from a frozen detector's box head: no extra download.

torchvision's Faster R-CNN (ResNet-50 FPN, COCO) pools each box's FPN features and runs its
two-layer box head to a 1024-d vector; that vector is what its classifier reads.  Here it is
L2-normalised and used as an appearance descriptor of a box.  A learned ReID network would be
better; this is the training-free, download-free baseline, and DINOv2 via timm is the
alternative if a separability check (same-id vs other-id cosine AUC on ground-truth crops)
says it is worth the 88 MB.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class BoxHeadEmbedder:
    name = "frcnn-boxhead"

    def __init__(self, device: str | None = None):
        import torch
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.m.eval().to(device)
        self.torch = torch
        self._feat_cache: dict = {}

    def features(self, frame_rgb: np.ndarray, key=None):
        """FPN features of a frame plus the transformed image size; cached by key when given."""
        if key is not None and key in self._feat_cache:
            return self._feat_cache[key]
        img = self.torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            images, _ = self.m.transform([img])
            feats = self.m.backbone(images.tensors)
        out = (feats, images.image_sizes[0], frame_rgb.shape[:2])
        if key is not None:
            self._feat_cache = {key: out}      # keep one frame
        return out

    def __call__(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray, key=None) -> np.ndarray:
        """(n, 1024) L2-normalised box-head features for boxes in frame pixels."""
        boxes = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
        if boxes.shape[0] == 0:
            return np.zeros((0, 1024), np.float32)
        feats, (h_t, w_t), (H, W) = self.features(frame_rgb, key)
        s = np.array([w_t / W, h_t / H, w_t / W, h_t / H], np.float32)
        b = self.torch.from_numpy(boxes * s).to(self.device)
        with self.torch.inference_mode():
            pooled = self.m.roi_heads.box_roi_pool(feats, [b], [(h_t, w_t)])
            x = self.m.roi_heads.box_head(pooled)
            x = self.torch.nn.functional.normalize(x, dim=1)
        return x.float().cpu().numpy()


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine for L2-normalised rows; (n, m)."""
    return 1.0 - np.asarray(a) @ np.asarray(b).T
