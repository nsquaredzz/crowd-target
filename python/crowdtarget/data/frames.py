"""Frames of one episode window, and the object SAM 2 indexes in place of its image tensor.

SAM 2's video predictor only ever does `state["images"][i]` and `len(state["images"])`, so a
lazy object that resizes a frame to 1024 x 1024 and normalises it on demand stands in for the
(T, 3, 1024, 1024) float tensor `init_state` would build from a JPEG folder (1.2 GB for 100
frames).  Resizing is a plain squash to 1024 x 1024, as SAM 2 does (utils/misc.py), from the
source resolution so no vertical detail is thrown away before it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class FrameSource:
    """Frames t_start .. t_end (1-based MOT frame numbers) of a JPEG folder, cached in RAM as uint8."""

    def __init__(self, paths: list[Path], scale: tuple[int, int] | None = None):
        self.paths = [Path(p) for p in paths]
        self.scale = scale
        self._cache: dict[int, np.ndarray] = {}
        im = Image.open(self.paths[0])
        self.W, self.H = im.size
        if scale:
            self.W, self.H = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        if i not in self._cache:
            im = Image.open(self.paths[i]).convert("RGB")
            if self.scale:
                im = im.resize(self.scale, Image.BILINEAR)
            self._cache[i] = np.asarray(im)
        return self._cache[i]

    def clear(self) -> None:
        self._cache.clear()


class LazyFrames:
    """Duck-types the tensor SAM 2 indexes: item i is a (3, 1024, 1024) float32 CPU tensor."""

    def __init__(self, src: FrameSource, image_size: int = 1024):
        self.src = src
        self.image_size = int(image_size)
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int):
        import torch
        if i not in self._cache:
            im = Image.fromarray(self.src[i]).resize((self.image_size, self.image_size), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
            self._cache[i] = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return self._cache[i]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def video_height(self) -> int:
        return self.src.H

    @property
    def video_width(self) -> int:
        return self.src.W
