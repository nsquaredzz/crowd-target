#!/usr/bin/env python3
"""Side-by-side outcomes of several runs on the same episodes.

    python scripts/compare_runs.py p0_sam2_bp_dev fix_samurai_dev fix_sound_public_dev fix_sound_yolo_dev

Prints one row per episode (outcome per run, with the firm-only outcome for sound runs) and a summary
per run over the episodes the first run has.  Writes out/compare_<runs>.txt.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget.harness import write_out  # noqa: E402


def load(run):
    p = ROOT / "out" / f"{run}.json"
    if not p.exists():
        return {}
    return {(r["seq"], r["id"], r["gap_start"]): r for r in json.loads(p.read_text())["records"]}


def short(r):
    if r is None:
        return "-"
    o = r["outcome"]
    if o == "switched":
        o += f"->{r['switch_id']}({r['switch_phase'][0]})"
    if "firm" in r and r["firm"]["outcome"] != r["outcome"]:
        o += f" [firm: {r['firm']['outcome']}]"
    return o


def main():
    runs = sys.argv[1:] or ["p0_sam2_bp_dev", "fix_samurai_dev", "fix_sound_public_dev", "fix_sound_yolo_dev"]
    data = {run: load(run) for run in runs}
    keys = sorted(data[runs[0]], key=lambda k: (k[0], k[2]))
    w = max(len(r) for r in runs) + 2
    lines = [f"{'episode':22s} {'g':>4s} " + " ".join(f"{r:{w}s}" for r in runs)]
    for k in keys:
        base = data[runs[0]][k]
        lines.append(f"{k[0] + '_' + str(k[1]) + '_' + str(k[2]):22s} {base['g']:4d} " + " ".join(f"{short(data[r].get(k)):{w}s}" for r in runs))
    lines.append("")
    for r in runs:
        rs = [data[r][k] for k in keys if k in data[r]]
        if not rs:
            continue
        n = len(rs)
        c = {o: sum(x["outcome"] == o for x in rs) for o in ("recovered", "switched", "lost", "ambiguous")}
        f = {o: sum(x.get("firm", {}).get("outcome") == o for x in rs) for o in ("recovered", "switched", "lost")} if any("firm" in x for x in rs) else None
        lines.append(f"{r:{w}s} n {n:2d}  recovered {c['recovered']:2d}  switched {c['switched']:2d}  lost {c['lost']:2d}  ambiguous {c['ambiguous']:2d}"
                     + (f"   | firm-only: recovered {f['recovered']} switched {f['switched']} lost {f['lost']}" if f else ""))
    text = "\n".join(lines) + "\n"
    write_out("compare_" + "_".join(r.replace("_dev", "").replace("p0_sam2_bp", "sam2") for r in runs), {"runs": runs, "episodes": [list(k) for k in keys]}, text)
    print(text)


if __name__ == "__main__":
    main()
