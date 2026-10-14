#!/usr/bin/env python3
"""Person detections with YOLO11x (Ultralytics, COCO weights) cached per sequence, and their recall on MOT20 GT.

    python scripts/detect.py --seq MOT20-03 --every 20 --imgsz 1280 1920     # recall at two input sizes on a frame subsample
    python scripts/detect.py --set dev --imgsz 1280                           # every frame of the dev episode windows, cached

Cache: data/cache/det_yolo11x_<imgsz>_<seq>.npz with arrays frame (N,), box (N, 4) xyxy, conf (N,), merged across
runs.  Recall is measured like neighbours.detector_recall: GT boxes with visibility >= 0.5 and height >= 40 px that
have a detection at IoU >= 0.5.  Weights: data/detector/yolo11x.pt (SHA-256 beside it), fetched from the
Ultralytics GitHub release, never committed.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import neighbours as NB  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import write_out  # noqa: E402

CACHE = ROOT / "data" / "cache"
WEIGHTS = ROOT / "data" / "detector" / "yolo11x.pt"


def cache_path(seq, imgsz):
    return CACHE / f"det_yolo11x_{imgsz}_{seq}.npz"


def load_cache(seq, imgsz):
    p = cache_path(seq, imgsz)
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for t in np.unique(z["frame"]):
        m = z["frame"] == t
        out[int(t)] = (z["box"][m], z["conf"][m])
    return out


def save_cache(seq, imgsz, dets):
    frames, boxes, confs = [], [], []
    for t, (b, c) in dets.items():
        frames.append(np.full(len(b), t, np.int32))
        boxes.append(b.reshape(-1, 4))
        confs.append(c)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(seq, imgsz), frame=np.concatenate(frames) if frames else np.zeros(0, np.int32),
                        box=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
                        conf=np.concatenate(confs) if confs else np.zeros(0, np.float32))


class Detector:
    def __init__(self, imgsz=1280, conf=0.1, device=None):
        import torch
        from ultralytics import YOLO
        self.m = YOLO(str(WEIGHTS))
        self.imgsz, self.conf = imgsz, conf
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def __call__(self, frame_rgb):
        r = self.m.predict(frame_rgb[..., ::-1].copy(), imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False, device=self.device)[0]
        return r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)


def ensure(seq, frames, imgsz, det=None, log_every=200):
    """Detections for the given frames of a sequence, computing and caching what is missing."""
    seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
    info = seq_info(seq_dir)
    dets = load_cache(seq, imgsz)
    missing = [int(t) for t in frames if int(t) not in dets]
    if missing:
        from PIL import Image
        det = det or Detector(imgsz)
        t0 = time.time()
        for k, t in enumerate(missing):
            dets[t] = det(np.asarray(Image.open(frame_path(seq_dir, t, info)).convert("RGB")))
            if (k + 1) % log_every == 0:
                print(f"  {seq}: {k + 1}/{len(missing)} frames, {(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
        save_cache(seq, imgsz, dets)
    return dets


def recall(seq, frames, imgsz, dets, min_conf=0.25):
    gt = targets(read_gt(ROOT / "data" / "mot20" / "MOT20" / "train" / seq))
    dbf = {t: dets[t][0][dets[t][1] >= min_conf] for t in frames if t in dets}
    return NB.detector_recall(dbf, gt, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=None)
    ap.add_argument("--set", default=None, choices=[None, "dev", "held", "all"])
    ap.add_argument("--every", type=int, default=None, help="frame subsample stride for a recall measurement")
    ap.add_argument("--imgsz", nargs="*", type=int, default=[1280])
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold used for the recall numbers")
    a = ap.parse_args()
    lines = [f"YOLO11x (COCO) person detections on MOT20, {time.strftime('%Y-%m-%d')}", ""]
    results = {}
    if a.set:
        eps = E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json")
        by_seq = {}
        for ep in eps:
            by_seq.setdefault(ep.seq, set()).update(range(ep.t_start, ep.t_end + 1))
        for imgsz in a.imgsz:
            det = Detector(imgsz)
            for seq, frames in sorted(by_seq.items()):
                t0 = time.time()
                dets = ensure(seq, sorted(frames), imgsz, det)
                r = recall(seq, sorted(frames), imgsz, dets, a.conf)
                results[f"{seq}_{imgsz}_episodes"] = r
                lines.append(f"{seq} imgsz {imgsz} episode windows ({len(frames)} frames): recall {r['recall']:.3f} ({r['detected']}/{r['gt_boxes']})  {time.time() - t0:.0f} s")
                print(lines[-1], flush=True)
    if a.seq and a.every:
        for imgsz in a.imgsz:
            det = Detector(imgsz)
            for seq in a.seq:
                info = seq_info(ROOT / "data" / "mot20" / "MOT20" / "train" / seq)
                frames = list(range(1, info["n_frames"] + 1, a.every))
                t0 = time.time()
                dets = ensure(seq, frames, imgsz, det)
                r = recall(seq, frames, imgsz, dets, a.conf)
                results[f"{seq}_{imgsz}_every{a.every}"] = r
                lines.append(f"{seq} imgsz {imgsz} every {a.every}th frame ({len(frames)} frames): recall {r['recall']:.3f} ({r['detected']}/{r['gt_boxes']})  "
                             f"{(time.time() - t0) / max(len(frames), 1):.2f} s/frame")
                print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    write_out("detector_recall", {"results": results, "conf": a.conf, "weights_sha256": (ROOT / "data" / "detector" / "yolo11x.sha256").read_text().split()[0]
                                   if (ROOT / "data" / "detector" / "yolo11x.sha256").exists() else None}, text)
    print(text)


if __name__ == "__main__":
    main()
