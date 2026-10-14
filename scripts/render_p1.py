#!/usr/bin/env python3
"""Proof video for one episode: the frame with SAM 2's mask and the ground-truth boxes, and the
martingale's wealth over time with the alarm line.

    python scripts/render_p1.py --run p0_sam2_bp_dev --episode MOT20-05_123_456 --key centroid_k4 --out out/proof_<name>.mp4

Colours: target ground truth white ring, other identities thin grey, SAM 2's mask green while
the per-frame outcome is correct, amber when absent, red when on another identity.  The lower
strip is log10 wealth with the 1/alpha lines for alpha 0.1, 0.05, 0.01.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack.render import Writer, caption, font  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402

GREEN, AMBER, RED, WHITE, GREY, INK = (74, 222, 128), (251, 191, 36), (248, 113, 113), (240, 240, 240), (120, 130, 140), (18, 18, 18)


def strip(frames, wealth, t, width, height=120, alphas=(0.1, 0.05, 0.01), T0=None, T1=None, switch=None):
    im = Image.new("RGB", (width, height), INK)
    d = ImageDraw.Draw(im)
    f = font(12)
    L, R, Tp, B = 40, 8, 14, 16
    lo, hi = -1.0, max(3.0, float(np.log10(max(wealth) if len(wealth) else 1)) + 0.5)
    def X(fr): return L + (width - L - R) * (fr - T0) / max(T1 - T0, 1)
    def Y(v): return Tp + (height - Tp - B) * (1 - (v - lo) / (hi - lo))
    for a in alphas:
        y = Y(np.log10(1 / a))
        d.line((L, y, width - R, y), fill=(70, 70, 70), width=1)
        d.text((2, y - 7), f"1/{a:g}", fill=GREY, font=f)
    if switch is not None:
        d.line((X(switch), Tp, X(switch), height - B), fill=RED, width=1)
    pts = [(X(fr), Y(np.log10(max(w, 1e-3)))) for fr, w in zip(frames, wealth) if fr <= t]
    if len(pts) > 1:
        d.line(pts, fill=GREEN, width=2)
    d.line((X(t), Tp, X(t), height - B), fill=WHITE, width=1)
    d.text((L, 0), "log10 wealth of the conformal martingale; red line = ground-truth switch", fill=WHITE, font=f)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", required=True, help="<seq>_<id>_<gap_start>")
    ap.add_argument("--key", default="centroid_k4")
    ap.add_argument("--martingale", default="mixture", choices=["mixture", "jumper"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--down", type=int, default=2)
    a = ap.parse_args()
    seq, id_, gap_start = a.episode.rsplit("_", 2)
    id_, gap_start = int(id_), int(gap_start)
    z = np.load(ROOT / "data" / "runs" / a.run / f"{a.episode}.npz")
    p1 = np.load(ROOT / "data" / "runs" / f"p1_{a.run}" / f"{a.episode}_{a.key}.npz")
    seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
    info = seq_info(seq_dir)
    gt = targets(read_gt(seq_dir))
    frames = z["frames"]
    masks = np.unpackbits(z["masks_small"], axis=-1)[:, :, : z["mask_shape"][2]].astype(bool)
    outcomes = z["outcomes"]
    wf, ww = p1["frames_tested"], p1[f"wealth_{a.martingale}"]
    import json
    rec = next(r for r in json.loads((ROOT / "out" / f"{a.run}.json").read_text())["records"]
               if r["seq"] == seq and r["id"] == id_ and r["gap_start"] == gap_start)
    writer = None
    for i, t in enumerate(frames):
        img = Image.open(frame_path(seq_dir, int(t), info)).convert("RGB")
        W, H = img.size
        img = img.resize((W // a.down, H // a.down))
        d = ImageDraw.Draw(img)
        s = 1.0 / a.down
        for r in gt[gt[:, 0] == t]:
            x1, y1, x2, y2 = r[2] * s, r[3] * s, (r[2] + r[4]) * s, (r[3] + r[5]) * s
            if int(r[1]) == id_:
                d.rectangle((x1, y1, x2, y2), outline=WHITE, width=2)
            else:
                d.rectangle((x1, y1, x2, y2), outline=GREY, width=1)
        m = masks[i]
        mi = Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
        colour = GREEN if outcomes[i] == "correct" else AMBER if outcomes[i] == "absent" else RED
        overlay = Image.new("RGB", img.size, colour)
        img = Image.composite(Image.blend(img, overlay, 0.55), img, mi)
        d = ImageDraw.Draw(img)
        phase = "gap" if rec["gap_start"] <= t <= rec["gap_end"] else ("before" if t < rec["gap_start"] else "after")
        fr = np.asarray(img)
        st = strip(wf, ww, t, fr.shape[1], T0=int(frames[0]), T1=int(frames[-1]), switch=rec.get("switch_frame"))
        w_now = [w for f_, w in zip(wf, ww) if f_ <= t]
        text = f"{a.episode}  frame {t}  {phase}  outcome {outcomes[i]}  score {z['scores'][i]:.1f}  wealth {w_now[-1] if w_now else 1:.2g}"
        frame = np.concatenate([fr, st, caption(fr.shape[1], text, ((GREEN, "mask on target"), (RED, "mask on another id"), (AMBER, "absent"), (WHITE, "target GT")))], 0)
        if writer is None:
            writer = Writer(a.out, (frame.shape[1], frame.shape[0]), a.fps)
        writer.add(frame)
    writer.close()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
