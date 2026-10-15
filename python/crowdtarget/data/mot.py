"""MOTChallenge (MOT17/MOT20) sequences: gt.txt, det/det.txt, seqinfo.ini, JPEG frames.

gt.txt rows: frame, id, bb_left, bb_top, bb_width, bb_height, conf (1 = consider), class, visibility.
MOT20 classes: 1 pedestrian, 2 person on vehicle, 7 static person, 8 distractor, 12 reflection,
13 crowd; only class 1 with conf 1 is a tracking target.  Frames are 1-based; pixels are
top-left origin.  det.txt rows: frame, -1, x, y, w, h, score, -1, -1, -1.
"""
from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np

GT_COLS = ("frame", "id", "x", "y", "w", "h", "conf", "cls", "vis")


def read_gt(seq_dir: Path) -> np.ndarray:
    """(N, 9) float64 rows of gt/gt.txt in GT_COLS order."""
    p = Path(seq_dir) / "gt" / "gt.txt"
    a = np.loadtxt(p, delimiter=",", ndmin=2, dtype=np.float64)
    if a.shape[1] < 9:
        pad = np.ones((a.shape[0], 9 - a.shape[1]))
        a = np.concatenate([a, pad], 1)
    return a[:, :9]


def read_det(seq_dir: Path) -> np.ndarray:
    """(N, 6) rows frame, x, y, w, h, score of det/det.txt."""
    p = Path(seq_dir) / "det" / "det.txt"
    a = np.loadtxt(p, delimiter=",", ndmin=2, dtype=np.float64)
    return a[:, [0, 2, 3, 4, 5, 6]]


def read_det(seq_dir: Path) -> np.ndarray:
    """(N, >=7) rows of det/det.txt: frame, -1, x, y, w, h, conf, ..."""
    return np.loadtxt(Path(seq_dir) / "det" / "det.txt", delimiter=",", ndmin=2, dtype=np.float64)


def dets_by_frame(det: np.ndarray, min_conf: float = 0.0) -> dict[int, np.ndarray]:
    """frame -> (n, 4) xyxy boxes."""
    det = det[det[:, 6] >= min_conf]
    out: dict[int, np.ndarray] = {}
    for t in np.unique(det[:, 0]).astype(int):
        r = det[det[:, 0] == t]
        out[t] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], 1)
    return out


def seq_info(seq_dir: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(Path(seq_dir) / "seqinfo.ini")
    s = cp["Sequence"]
    return {"name": s.get("name", Path(seq_dir).name), "W": int(s["imWidth"]), "H": int(s["imHeight"]),
            "fps": float(s["frameRate"]), "n_frames": int(s["seqLength"]), "imDir": s.get("imDir", "img1"),
            "imExt": s.get("imExt", ".jpg")}


def targets(gt: np.ndarray) -> np.ndarray:
    """Rows that are tracking targets: pedestrians (class 1) flagged to consider."""
    return gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]


def frame_path(seq_dir: Path, t: int, info: dict | None = None) -> Path:
    info = info or seq_info(seq_dir)
    return Path(seq_dir) / info["imDir"] / f"{t:06d}{info['imExt']}"


def rows_at(gt: np.ndarray, t: int) -> np.ndarray:
    return gt[gt[:, 0] == t]


def track(gt: np.ndarray, id_: int) -> np.ndarray:
    """Rows of one identity sorted by frame."""
    r = gt[gt[:, 1] == id_]
    return r[np.argsort(r[:, 0])]


def xyxy(rows: np.ndarray) -> np.ndarray:
    """(n, 4) x1, y1, x2, y2 from gt rows."""
    return np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], -1)


def sequences(root: Path, split: str = "train") -> list[Path]:
    d = Path(root) / split
    return sorted(p for p in d.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists())
