#!/usr/bin/env python3
"""Does the appearance embedding separate identities?  Same-person vs other-person distances on
ground-truth crops of MOT20, as an AUC, plus the effect of partial occlusion on same-person distance.

    python scripts/embed_check.py --seqs MOT20-02 MOT20-03 --frames 30

Pairs: same identity across frames up to 60 apart (visibility >= 0.7 both), other identity in the
same frame (visibility >= 0.7 both).  AUC = P(distance same < distance other).  Also reported: the
distance of a half-occluded crop (visibility 0.3-0.5) of a person to a fully visible crop of the
same person, against the same-person fully-visible distances.  Writes out/embed_check.{json,txt}.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.embed import BoxHeadEmbedder  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos < neg) by ranks (smaller distance = same person)."""
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1
    rp = ranks[: pos.size].sum()
    p_pos_greater = (rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(1.0 - p_pos_greater)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=["MOT20-02", "MOT20-03"])
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--min-h", type=float, default=60)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    emb = BoxHeadEmbedder(a.device)
    t0 = time.time()
    rows = []   # (seq, frame, id, vis, embedding)
    for s in a.seqs:
        seq = ROOT / "data" / "mot20" / "MOT20" / "train" / s
        info = seq_info(seq)
        gt = targets(read_gt(seq))
        anchors = np.linspace(1, info["n_frames"] - 50, a.frames).astype(int)
        frames = sorted({int(t + d) for t in anchors for d in (0, 15, 45)})   # same-person pairs 15-45 frames apart
        for t in frames:
            r = gt[(gt[:, 0] == t) & (gt[:, 5] >= a.min_h) & (gt[:, 8] >= 0.3)]
            if r.shape[0] == 0:
                continue
            img = np.asarray(Image.open(frame_path(seq, int(t), info)).convert("RGB"))
            boxes = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], -1)
            e = emb(img, boxes, key=(s, int(t)))
            for j in range(r.shape[0]):
                rows.append((s, int(t), int(r[j, 1]), float(r[j, 8]), e[j]))
    E = np.stack([x[4] for x in rows])
    same, other, occl = [], [], []
    for i in range(len(rows)):
        si, ti, idi, vi = rows[i][:4]
        for j in range(i + 1, len(rows)):
            sj, tj, idj, vj = rows[j][:4]
            if si != sj:
                continue
            d = 1.0 - float(E[i] @ E[j])
            if idi == idj and ti != tj and abs(ti - tj) <= 60:
                if vi >= 0.7 and vj >= 0.7:
                    same.append(d)
                elif (vi >= 0.7 and 0.3 <= vj <= 0.5) or (vj >= 0.7 and 0.3 <= vi <= 0.5):
                    occl.append(d)
            elif idi != idj and ti == tj and vi >= 0.7 and vj >= 0.7:
                other.append(d)
    same, other, occl = map(np.asarray, (same, other, occl))
    res = {"embedding": emb.name, "n_crops": len(rows), "n_same": int(same.size), "n_other": int(other.size),
           "n_occluded_pairs": int(occl.size),
           "auc_same_vs_other": auc(same, other) if same.size and other.size else float("nan"),
           "auc_same_vs_occluded": auc(same, occl) if same.size and occl.size else float("nan"),
           "median_same": float(np.median(same)) if same.size else float("nan"),
           "median_other": float(np.median(other)) if other.size else float("nan"),
           "median_occluded": float(np.median(occl)) if occl.size else float("nan"),
           "machine": machine_info({"device": emb.device}), "seconds": round(time.time() - t0, 1)}
    text = (f"Embedding separability on MOT20 ground-truth crops ({', '.join(a.seqs)}, {a.frames} frames each), {time.strftime('%Y-%m-%d')}\n\n"
            f"embedding {emb.name}: {len(rows)} crops; pairs same {same.size}, other {other.size}, half-occluded same {occl.size}\n"
            f"AUC same-vs-other (same person closer): {res['auc_same_vs_other']:.3f}\n"
            f"AUC same-vs-half-occluded-same:          {res['auc_same_vs_occluded']:.3f}\n"
            f"median 1-cos: same {res['median_same']:.3f}  other {res['median_other']:.3f}  half-occluded same {res['median_occluded']:.3f}\n\n"
            "Reading: an AUC near 0.5 means the feature does not tell people apart; a half-occluded crop of the same person farther\n"
            "than another person means occlusion moves the score more than a switch does, and a martingale on it detects occlusion.\n")
    pj, pt = write_out("embed_check", res, text)
    print(text)
    print(f"wrote {pj}")


if __name__ == "__main__":
    main()
