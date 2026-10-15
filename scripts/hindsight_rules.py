#!/usr/bin/env python3
"""Re-score saved hindsight candidates under alternative meet rules, each labelled with when it was written.

    python scripts/hindsight_rules.py --run hindsight_oracle_fix_sound_yolo_dev

pre-registered   median IoU over the meet window >= 0.5, second < 0.25 (python/crowdtarget/hindsight.py)
rank (post hoc)  written after the first smoke episode (611: true 0.14, every other 0.00): commit to the
                 argmax if m1 >= 0.1 and m1 >= 3 x m2; it is reported beside the pre-registered rule, never
                 instead of it.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from crowdtarget import hindsight as HS  # noqa: E402


def rank_rule(m):
    if m.size == 0:
        return {"kind": "absent", "index": None}
    o = np.argsort(-m)
    m1, m2 = float(m[o[0]]), float(m[o[1]]) if m.size > 1 else 0.0
    return {"kind": "commit", "index": int(o[0])} if (m1 >= 0.1 and m1 >= 3 * m2) else {"kind": "absent" if m1 < 0.1 else "ambiguous", "index": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hindsight_oracle_fix_sound_yolo_dev")
    a = ap.parse_args()
    recs = {f"{r['seq']}_{r['id']}_{r['gap_start']}": r for r in json.loads((ROOT / "out" / f"{a.run}.json").read_text())["records"]}
    tally = {"pre": {"right": 0, "wrong": 0, "none": 0}, "rank": {"right": 0, "wrong": 0, "none": 0}}
    lines = [f"Hindsight decisions under two meet rules, run {a.run}", "", f"{'episode':22s} {'true':>5s} {'best other':>10s} {'pre-registered':>15s} {'rank (post hoc)':>16s}"]
    for f in sorted(glob.glob(str(ROOT / "data" / "runs" / a.run / "*_cands.npz"))):
        key = Path(f).name[:-len("_cands.npz")]
        c = np.load(f)
        m, ids = c["meet"], list(c["gt_ids"])
        tid = recs[key]["id"]
        mt = float(m[ids.index(tid)]) if tid in ids else float("nan")
        mo = float(max([m[k] for k in range(len(m)) if ids[k] != tid], default=0.0))
        out = []
        for name, d in (("pre", HS.decide(m)), ("rank", rank_rule(m))):
            if d["kind"] == "commit":
                ok = ids[d["index"]] == tid
                tally[name]["right" if ok else "wrong"] += 1
                out.append(f"commit {'RIGHT' if ok else 'WRONG ' + str(ids[d['index']])}")
            else:
                tally[name]["none"] += 1
                out.append(d["kind"])
        lines.append(f"{key:22s} {mt:5.2f} {mo:10.2f} {out[0]:>15s} {out[1]:>16s}")
    lines += ["", f"pre-registered: {tally['pre']}", f"rank (post hoc): {tally['rank']}"]
    text = "\n".join(lines) + "\n"
    (ROOT / "out" / f"{a.run}_rules.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
