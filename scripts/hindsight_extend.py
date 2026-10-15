#!/usr/bin/env python3
"""After a hindsight commit, continue forward: SAM 2.1 prompted with the chosen candidate's box at the
decision frame, stepped to the clip end.  This is what a deployed tracker does after deciding, and it
gives full-length proof videos.

    python scripts/hindsight_extend.py --run hindsight_oracle_fix_sound_yolo_dev --episode MOT20-03_169_136

Writes data/runs/<run>_ext/<episode>.npz and out/<run>_ext.json (merged per episode).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget import metrics as M  # noqa: E402
from crowdtarget import sam2io  # noqa: E402
from crowdtarget.data.frames import FrameSource, LazyFrames  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import write_out  # noqa: E402

DOWN = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hindsight_oracle_fix_sound_yolo_dev")
    ap.add_argument("--episode", required=True)
    a = ap.parse_args()
    seq, id_, gs = a.episode.rsplit("_", 2)
    id_, gs = int(id_), int(gs)
    rec = next(r for r in json.loads((ROOT / "out" / f"{a.run}.json").read_text())["records"] if r["seq"] == seq and r["id"] == id_ and r["gap_start"] == gs)
    if rec.get("decision") != "commit":
        sys.exit(f"{a.episode}: decision was {rec.get('decision')}, nothing to continue")
    z = np.load(ROOT / "data" / "runs" / a.run / f"{a.episode}.npz")
    c = np.load(ROOT / "data" / "runs" / a.run / f"{a.episode}_cands.npz")
    frames = z["frames"]
    out = np.unpackbits(z["masks_small"], axis=-1)[:, :, : z["mask_shape"][2]].astype(bool)
    k = int(np.argmax(c["meet"]))
    box = c["boxes"][k]
    t_b = int(rec["t_b"])
    t_end = int(frames[-1])
    seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
    info = seq_info(seq_dir)
    gt = targets(read_gt(seq_dir))
    if t_end > t_b:
        pred = sam2io.build("b+")
        src = FrameSource([frame_path(seq_dir, t, info) for t in range(t_b, t_end + 1)])
        lazy = LazyFrames(src)
        state = sam2io.make_state(pred, lazy)
        sam2io.prompt_box(pred, state, 0, 1, box)
        h4, w4 = out.shape[1:]
        for j in range(len(src)):
            res = sam2io.step(pred, state, j)
            m, s = res[1]
            i = int(np.nonzero(frames == t_b + j)[0][0])
            out[i] = m[::DOWN, ::DOWN][:h4, :w4] if s > 0 else False
            src._cache.pop(j - 2, None)
            lazy._cache.pop(j - 2, None)
    outcomes = [M.frame_outcome(out[i], gt, int(frames[i]), id_, scale=DOWN) for i in range(len(frames))]
    tg = gt[gt[:, 1] == id_]
    vis_post = {int(r[0]): bool(r[8] >= 0.5) for r in tg[tg[:, 0] > rec["gap_end"]]}
    res = M.episode_outcome(outcomes, [int(f) for f in frames], rec["gap_start"], rec["gap_end"], vis_post)
    new = dict(rec)
    new.update(res)
    new["extended_from"] = t_b
    stem = f"{a.run}_ext"
    runs = ROOT / "data" / "runs" / stem
    runs.mkdir(parents=True, exist_ok=True)
    diag = [{"state_after": "forward" if int(f) <= rec["t_a"] else ("hindsight (backward)" if int(f) <= t_b else "after commit")} for f in frames]
    np.savez_compressed(runs / f"{a.episode}.npz", masks_small=np.packbits(out, axis=-1), mask_shape=np.array(out.shape),
                        outcomes=np.array(outcomes), frames=frames, diag=np.array(json.dumps(diag)), scores=z["scores"])
    pj = ROOT / "out" / f"{stem}.json"
    old = json.loads(pj.read_text())["records"] if pj.exists() else []
    old = [r for r in old if not (r["seq"] == seq and r["id"] == id_ and r["gap_start"] == gs)] + [new]
    write_out(stem, {"records": old, "summary": M.summarise(old)}, M.format_summary(M.summarise(old), stem))
    print(f"{a.episode}: {res['outcome']}, post-correct {100 * res['post_correct_frac']:.0f} %")


if __name__ == "__main__":
    main()
