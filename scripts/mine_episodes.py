#!/usr/bin/env python3
"""Mine single-target episodes from MOT20 train and fix the dev / held-out split.

    python scripts/mine_episodes.py                # data/mot20/MOT20/train -> data/episodes/mot20_{all,dev,held}.json, out/episodes_mot20.txt

The rules are in python/crowdtarget/episodes.py and are not tuned after this runs.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget.data.mot import read_gt, seq_info, sequences  # noqa: E402
from crowdtarget.episodes import dev_split, mine, summary, write  # noqa: E402


def main():
    root = ROOT / "data" / "mot20" / "MOT20"
    eps = []
    t0 = time.time()
    for seq in sequences(root, "train"):
        info = seq_info(seq)
        gt = read_gt(seq)
        e = mine(gt, info["name"], info["n_frames"], info["W"], info["H"], info["fps"])
        print(f"{info['name']}: {info['n_frames']} frames, {len(e)} episodes {summary(e)}")
        eps += e
    dev, held = dev_split(eps, per_bin=8)
    out = ROOT / "data" / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    write(eps, out / "mot20_all.json")
    write(dev, out / "mot20_dev.json")
    write(held, out / "mot20_held.json")
    lines = [f"MOT20 train single-target episodes, mined {time.strftime('%Y-%m-%d')} by the rules in episodes.py", "",
             f"all:  {summary(eps)}", f"dev:  {summary(dev)}", f"held: {summary(held)}", "",
             f"{'seq':10s} {'id':>5s} {'t_start':>7s} {'gap':>11s} {'g':>4s} {'bin':>6s} {'h0':>5s} {'set':>4s}"]
    devkeys = {(e.seq, e.id, e.gap_start) for e in dev}
    for e in sorted(eps, key=lambda e: (e.seq, e.id, e.gap_start)):
        lines.append(f"{e.seq:10s} {e.id:5d} {e.t_start:7d} {e.gap_start:5d}-{e.gap_end:<5d} {e.g:4d} {e.bin:>6s} "
                     f"{e.start_box[3] - e.start_box[1]:5.0f} {'dev' if (e.seq, e.id, e.gap_start) in devkeys else 'held':>4s}")
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "episodes_mot20.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))
    print(f"wrote {out} and out/episodes_mot20.txt in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
