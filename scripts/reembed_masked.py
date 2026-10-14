#!/usr/bin/env python3
"""Recompute the per-frame embeddings of a P0 run with the background inside the box painted grey
(using SAM 2's own mask), so that neighbours and background do not enter the appearance score.

    python scripts/reembed_masked.py --run p0_sam2_bp_dev            # writes data/runs/p0_sam2_bp_dev_masked/*.npz + out/<run>_masked.json

The masks saved by run_episodes.py are at 1/4 resolution; they are upsampled by nearest neighbour.
Everything else in the npz is copied, so p1_martingale.py runs on the new stem unchanged.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget.data.mot import frame_path, seq_info  # noqa: E402
from crowdtarget.embed import BoxHeadEmbedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p0_sam2_bp_dev")
    ap.add_argument("--fill", type=int, default=114, help="grey level painted outside the mask")
    ap.add_argument("--dilate", type=int, default=2, help="mask dilation in low-res pixels before painting")
    a = ap.parse_args()
    src = ROOT / "data" / "runs" / a.run
    dst = ROOT / "data" / "runs" / f"{a.run}_masked"
    dst.mkdir(parents=True, exist_ok=True)
    payload = json.loads((ROOT / "out" / f"{a.run}.json").read_text())
    emb = BoxHeadEmbedder()
    infos = {}
    t0 = time.time()
    done = 0
    for rec in payload["records"]:
        name = f"{rec['seq']}_{rec['id']}_{rec['gap_start']}"
        if not (src / f"{name}.npz").exists():
            continue
        z = dict(np.load(src / f"{name}.npz"))
        seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / rec["seq"]
        if rec["seq"] not in infos:
            infos[rec["seq"]] = seq_info(seq_dir)
        info = infos[rec["seq"]]
        T, h, w = z["mask_shape"]
        masks = np.unpackbits(z["masks_small"], axis=-1)[:, :, :w].astype(bool)
        down = info["H"] // h
        embs = np.zeros_like(z["embs"])
        for i, t in enumerate(z["frames"]):
            box = z["boxes"][i]
            if not np.isfinite(box[0]) or masks[i].sum() == 0:
                continue
            m = masks[i]
            if a.dilate:
                from numpy.lib.stride_tricks import sliding_window_view
                pad = np.pad(m, a.dilate)
                m = sliding_window_view(pad, (2 * a.dilate + 1, 2 * a.dilate + 1)).any((-2, -1))
            big = np.repeat(np.repeat(m, down, 0), down, 1)[: info["H"], : info["W"]]
            frame = np.asarray(Image.open(frame_path(seq_dir, int(t), info)).convert("RGB")).copy()
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            region = frame[y1:y2, x1:x2]
            keep = big[y1:y2, x1:x2]
            region[~keep] = a.fill
            embs[i] = emb(frame, box[None], key=(name, int(t)))[0]
        z["embs"] = embs
        np.savez_compressed(dst / f"{name}.npz", **z)
        done += 1
        print(f"{name}: re-embedded {int(np.isfinite(z['boxes'][:, 0]).sum())} frames", flush=True)
    shutil.copy(ROOT / "out" / f"{a.run}.json", ROOT / "out" / f"{a.run}_masked.json")
    print(f"{done} episodes in {time.time() - t0:.0f} s -> {dst}")


if __name__ == "__main__":
    main()
