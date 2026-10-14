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


class DinoEmbedder:
    """DINOv2 ViT-S/14 (timm, 88 MB from the Hugging Face hub) on the box crop: CLS token, L2-normalised.

    The crop is the box with a 10 % margin resized to 224 x 224.  Self-supervised features of this kind
    are the usual training-free stand-in for a ReID network; DINOv2 is not trained for identity either,
    but its features are far less class-bound than a COCO detector's box head.
    """
    name = "dinov2-s14"

    def __init__(self, device: str | None = None, model: str = "vit_small_patch14_dinov2.lvd142m", size: int = 224):
        import timm
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.m = timm.create_model(model, pretrained=True, num_classes=0, img_size=size).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.mean = np.array(cfg["mean"], np.float32)
        self.std = np.array(cfg["std"], np.float32)
        self.size = size
        self.torch = torch

    def crops(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray, margin: float = 0.1) -> np.ndarray:
        from PIL import Image
        H, W = frame_rgb.shape[:2]
        out = []
        for x1, y1, x2, y2 in np.asarray(boxes_xyxy, np.float32).reshape(-1, 4):
            w, h = x2 - x1, y2 - y1
            x1, x2 = max(int(x1 - margin * w), 0), min(int(x2 + margin * w), W)
            y1, y2 = max(int(y1 - margin * h), 0), min(int(y2 + margin * h), H)
            if x2 <= x1 or y2 <= y1:
                out.append(np.zeros((self.size, self.size, 3), np.float32))
                continue
            im = Image.fromarray(frame_rgb[y1:y2, x1:x2]).resize((self.size, self.size), Image.BILINEAR)
            out.append((np.asarray(im, np.float32) / 255.0 - self.mean) / self.std)
        return np.stack(out) if out else np.zeros((0, self.size, self.size, 3), np.float32)

    def __call__(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray, key=None, batch: int = 64) -> np.ndarray:
        c = self.crops(frame_rgb, boxes_xyxy)
        if c.shape[0] == 0:
            return np.zeros((0, self.m.num_features), np.float32)
        outs = []
        with self.torch.inference_mode():
            for i in range(0, c.shape[0], batch):
                x = self.torch.from_numpy(np.ascontiguousarray(c[i:i + batch].transpose(0, 3, 1, 2))).to(self.device)
                f = self.m(x)
                outs.append(self.torch.nn.functional.normalize(f, dim=1).float().cpu().numpy())
        return np.concatenate(outs)


def make_embedder(kind: str = "frcnn", device: str | None = None):
    return DinoEmbedder(device) if kind == "dino" else BoxHeadEmbedder(device)


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine for L2-normalised rows; (n, m)."""
    return 1.0 - np.asarray(a) @ np.asarray(b).T
