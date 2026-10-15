#!/usr/bin/env python3
"""Side-by-side proof video: SAM 2.1 alone against the published fixes on one episode.

    python scripts/render_compare.py --episode MOT20-05_611_2418 \
        --runs p0_sam2_bp_dev:SAM2.1,fix_samurai_dev:+SAMURAI,fix_dam_dev:+DAM4SAM-style --out out/compare_MOT20-05_611.mp4

Each panel: ground truth of the target as a white box, other identities thin grey, the run's
mask green while on the target, amber when SAM 2 reports it absent, red when on another
identity.  For SAMURAI the Kalman prediction is drawn dashed amber; for the DAM-style run a
frame promoted to an anchor flashes a cyan border.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from certtrack.render import Writer, caption, font  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402

GREEN, AMBER, RED, WHITE, GREY, CYAN, INK = (74, 222, 128), (251, 191, 36), (248, 113, 113), (240, 240, 240), (120, 130, 140), (56, 189, 248), (18, 18, 18)


def load_run(run: str, episode: str):
    z = np.load(ROOT / "data" / "runs" / run / f"{episode}.npz")
    masks = np.unpackbits(z["masks_small"], axis=-1)[:, :, : z["mask_shape"][2]].astype(bool)
    diag = json.loads(str(z["diag"])) if "diag" in z.files else []
    rec = next(r for r in json.loads((ROOT / "out" / f"{run}.json").read_text())["records"]
               if f"{r['seq']}_{r['id']}_{r['gap_start']}" == episode)
    return {"frames": z["frames"], "masks": masks, "outcomes": z["outcomes"], "scores": z["scores"], "diag": diag, "rec": rec}


def dashed(d: ImageDraw.ImageDraw, box, colour, step=6):
    x1, y1, x2, y2 = box
    for x in np.arange(x1, x2, 2 * step):
        d.line((x, y1, min(x + step, x2), y1), fill=colour, width=2)
        d.line((x, y2, min(x + step, x2), y2), fill=colour, width=2)
    for y in np.arange(y1, y2, 2 * step):
        d.line((x1, y, x1, min(y + step, y2)), fill=colour, width=2)
        d.line((x2, y, x2, min(y + step, y2)), fill=colour, width=2)


def panel(img: Image.Image, gt_rows, id_, run, i, label, s):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for r in gt_rows:
        x1, y1, x2, y2 = r[2] * s, r[3] * s, (r[2] + r[4]) * s, (r[3] + r[5]) * s
        d.rectangle((x1, y1, x2, y2), outline=WHITE if int(r[1]) == id_ else GREY, width=2 if int(r[1]) == id_ else 1)
    m = run["masks"][i]
    o = str(run["outcomes"][i])
    colour = GREEN if o == "correct" else AMBER if o == "absent" else RED
    mi = Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
    img = Image.composite(Image.blend(img, Image.new("RGB", img.size, colour), 0.55), img, mi)
    d = ImageDraw.Draw(img)
    dg = run["diag"][i - 1] if run["diag"] and 1 <= i <= len(run["diag"]) else {}
    if "pred_box" in dg:
        dashed(d, [v * s for v in dg["pred_box"]], AMBER)
    if dg.get("anchored"):
        d.rectangle((0, 0, img.size[0] - 1, img.size[1] - 1), outline=CYAN, width=6)
    if dg.get("commit") == "det":
        d.rectangle((0, 0, img.size[0] - 1, img.size[1] - 1), outline=AMBER, width=6)
    f = font(15, bold=True)
    extra = ""
    if "kf_iou" in dg:
        extra = f"  KF IoU {dg['kf_iou']:.2f}  mem {'+' if dg['in_memory'] else '-'}"
    elif "distractor" in dg:
        extra = f"  distractor {dg['distractor']:.2f}"
    elif "state_after" in dg:
        st = dg["state_after"] + ("" if dg.get("firm_after") else " (provisional)" if dg["state_after"] == "tracking" else "")
        extra = f"  {st}" + (f"  E {dg['E']:.2f}" if dg.get("E") == dg.get("E") and dg.get("E") is not None else "") + \
                (f"  set {dg['n_cands']}" if dg.get("n_cands", 0) >= 2 else "") + ("  VETO" if dg.get("veto") else "")
        for c in dg.get("cands", []) if dg["state_after"] == "ambiguous" else []:
            d.rectangle([v * s for v in c], outline=CYAN, width=2)
    text = f"{label}: {o}{extra}"
    d.rectangle((0, 0, 8 + d.textlength(text, font=f), 22), fill=INK)
    d.text((4, 3), text, fill=colour if o != "correct" else GREEN, font=f)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--runs", required=True, help="comma list of run:label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--down", type=float, default=3)
    ap.add_argument("--crop", default=None, help="x1,y1,x2,y2 in frame pixels: show only this region (zoom)")
    a = ap.parse_args()
    seq, id_, gap_start = a.episode.rsplit("_", 2)
    id_, gap_start = int(id_), int(gap_start)
    runs = [(r.split(":")[0], r.split(":")[1]) for r in a.runs.split(",")]
    data = [(load_run(r, a.episode), lab) for r, lab in runs]
    seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
    info = seq_info(seq_dir)
    gt = targets(read_gt(seq_dir))
    frames = data[0][0]["frames"]
    rec = data[0][0]["rec"]
    writer = None
    s = 1.0 / a.down
    for i, t in enumerate(frames):
        img = Image.open(frame_path(seq_dir, int(t), info)).convert("RGB")
        W, H = img.size
        img = img.resize((int(W * s), int(H * s)))
        rows = gt[gt[:, 0] == t]
        panels = [panel(img, rows, id_, run, i, lab, s) for run, lab in data]
        if a.crop:
            cx1, cy1, cx2, cy2 = [int(round(float(v) * s)) for v in a.crop.split(",")]
            panels = [p_[cy1:cy2, cx1:cx2] for p_ in panels]
            # put the label back on top of the crop
            for j, (run, lab) in enumerate(data):
                full = panel(img, rows, id_, run, i, lab, s)
                panels[j] = np.concatenate([full[:24, cx1:cx2] * 0 + full[:24, :cx2 - cx1], panels[j][24:]], 0) if cx2 - cx1 <= full.shape[1] else panels[j]
        row = np.concatenate([np.pad(p, ((0, 0), (0, 4), (0, 0)), constant_values=18) for p in panels], 1)
        phase = "gap" if rec["gap_start"] <= t <= rec["gap_end"] else ("before" if t < rec["gap_start"] else "after")
        summary = "   ".join(f"{lab}: {run['rec']['outcome']}" for run, lab in data)
        text = f"{a.episode}  frame {t}  {phase}   |   episode: {summary}"
        frame = np.concatenate([row, caption(row.shape[1], text, ((GREEN, "mask on target"), (RED, "mask on another id"), (AMBER, "absent / prediction / re-prompt"), (WHITE, "target GT"), (CYAN, "candidate set / anchor")))], 0)
        if writer is None:
            writer = Writer(a.out, (frame.shape[1], frame.shape[0]), a.fps)
        writer.add(frame)
    writer.close()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
