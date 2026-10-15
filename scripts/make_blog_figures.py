#!/usr/bin/env python3
"""Static figures for the blog note, drawn from the result files in out/.

    python scripts/make_blog_figures.py                  # every figure whose data exists
    python scripts/make_blog_figures.py hidden clues     # by name: hidden | clues | outcomes | hindsight

Light surface only: the site shows figures on white.  Palette from the dataviz reference instance,
validated with its script: outcomes use blue / yellow / violet / red in that stack order (all checks
pass; yellow is below 3:1 on the surface, so every segment is labelled and the post carries the table).
Writes out/fig_<name>.webp at 1800 px wide.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TH = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
          s1="#2a78d6", recovered="#2a78d6", ambiguous="#eda100", lost="#4a3aa7", switched="#e34948")
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def style(ax, grid="y"):
    ax.set_facecolor(TH["surface"])
    for s in ax.spines.values():
        s.set_color(TH["axis"])
        s.set_linewidth(0.8)
    ax.tick_params(colors=TH["muted"], length=0, labelsize=8.5)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=TH["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def header(fig, title, sub):
    fig.text(0.03, 0.95, title, color=TH["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 0.885, sub, color=TH["ink2"], fontsize=9.2, va="center")


def save(fig, name):
    png = OUT / f"fig_{name}.png"
    fig.savefig(png, dpi=1800 / (fig.get_size_inches()[0]), facecolor=TH["surface"])
    plt.close(fig)
    webp = OUT / f"fig_{name}.webp"
    Image.open(png).save(webp, "WEBP", quality=90, method=6)
    png.unlink()
    print(f"  {webp.relative_to(ROOT)}  {webp.stat().st_size / 1e3:.0f} kB")


def rounded_barh(ax, y, left, width, height, color):
    ax.barh(y, width, left=left, height=height, color=color, linewidth=0, zorder=3)


# ----------------------------------------------------------------------------------------------
def fig_hidden():
    d = json.loads((OUT / "census2.json").read_text())
    R = d["records"]
    bo = np.array([r["blackout_100"] for r in R])
    edges = [(0, 0, "0"), (1, 5, "1–5"), (6, 10, "6–10"), (11, 20, "11–20"), (21, 40, "21–40"), (41, 10 ** 6, "41+")]
    counts = [int(((bo >= lo) & (bo <= hi)).sum()) for lo, hi, _ in edges]
    vis = np.mean([r["sliver_ge100"] for r in R])
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    fig.patch.set_facecolor(TH["surface"])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.18)
    x = np.arange(len(edges))
    ax.bar(x, counts, width=0.55, color=TH["s1"], linewidth=0, zorder=3)
    for xi, c in zip(x, counts):
        ax.text(xi, c + 1.2, str(c), ha="center", va="bottom", color=TH["ink2"], fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels([e[2] for e in edges], color=TH["ink2"])
    ax.set_xlabel("longest run of frames in which less than 100 pixels of the person show", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("episodes", color=TH["ink2"], fontsize=9)
    ax.set_ylim(0, max(counts) * 1.18)
    style(ax)
    header(fig, "A hidden person is rarely invisible",
           f"{len(R)} MOT20 occlusion episodes (visibility under 0.3 for 20–200 frames). On {100 * vis:.0f} % of the occluded frames, 100+ pixels of the person show.")
    save(fig, "hidden")


def fig_clues():
    d = json.loads((OUT / "census.json").read_text())
    R = d["records"]
    agg = {"tube": [0, 0], "neighbour": [0, 0], "height": [0, 0], "budget": [0, 0]}
    for r in R:
        thr = r["thr"]
        for c in r["cands"]:
            if c["id"] == r["id"] or not c["in_disc"]:
                continue
            for k in agg:
                agg[k][1] += 1
            agg["tube"][0] += not c["in_tube"]
            agg["neighbour"][0] += c["vis_gap"] >= 0.8
            agg["height"][0] += c["d_rho"] > thr["kappa"]
            agg["budget"][0] += bool(np.isfinite(c["E_min"]) and (c["E_min"] > thr["tau_min"] or c["cons_min"] > thr["q_cons"]))
    drop = d["summary"]["all"]["true_dropped_at"]
    n = len(R)
    rows = [("seen the whole time, so not the hidden one", "neighbour", 0),
            ("colours the person never showed", "budget", drop.get("budget", 0)),
            ("wrong height for this person", "height", drop.get("height", 0)),
            ("could not have walked there in time", "tube", drop.get("tube", 0))]
    vals = [100 * agg[k][0] / agg[k][1] for _, k, _ in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    fig.patch.set_facecolor(TH["surface"])
    fig.subplots_adjust(left=0.36, right=0.95, top=0.76, bottom=0.15)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.42, color=TH["s1"], linewidth=0, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 1.0, yi, f"{v:.0f} %", va="center", ha="left", color=TH["ink"], fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=TH["ink2"], fontsize=9)
    for yi, (_, _, wrong) in zip(y, rows):
        ax.annotate(f"right person wrongly ruled out: {wrong} of {n}", xy=(0, yi), xycoords=("axes fraction", "data"), xytext=(-8, -11),
                    textcoords="offset points", ha="right", va="center", color=TH["muted"], fontsize=7.8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("share of the other people within walking reach that the clue rules out, on its own", color=TH["ink2"], fontsize=9)
    style(ax, grid="x")
    header(fig, "What each clue rules out",
           f"{sum(a[1] for a in agg.values()) // 4:,} people within reach of the hidden person at re-emergence, {n} episodes, ground truth only.")
    save(fig, "clues")


def outcome_counts(run):
    p = OUT / f"{run}.json"
    if not p.exists():
        return None
    R = json.loads(p.read_text())["records"]
    return {k: sum(r["outcome"] == k for r in R) for k in ("recovered", "ambiguous", "lost", "switched")}, len(R)


def fig_outcomes(runs, name, title, sub):
    rows = []
    for label, run in runs:
        oc = outcome_counts(run)
        if oc:
            rows.append((label, oc[0], oc[1]))
    if not rows:
        print(f"  {name}: no data yet")
        return
    keys = ["recovered", "ambiguous", "lost", "switched"]
    nice = {"recovered": "recovered", "ambiguous": "says “one of these”", "lost": "lost", "switched": "on the wrong person"}
    fig, ax = plt.subplots(figsize=(8.4, 1.55 + 0.55 * len(rows)))
    fig.patch.set_facecolor(TH["surface"])
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.2, right=0.92, top=1 - 1.0 / H, bottom=0.45 / H)
    y = np.arange(len(rows))[::-1]
    gap = 0.004
    for yi, (label, oc, n) in zip(y, rows):
        left = 0.0
        for k in keys:
            w = oc[k] / n
            if w > 0:
                ax.barh(yi, w - gap, left=left, height=0.44, color=TH[k], linewidth=0, zorder=3)
                txt = str(oc[k])
                if w >= 0.05:
                    ax.text(left + (w - gap) / 2, yi, txt, ha="center", va="center", fontsize=8.6,
                            color="#ffffff" if k in ("recovered", "lost", "switched") else TH["ink"], zorder=4)
                left += w
        ax.text(1.005, yi, f"n = {n}", va="center", ha="left", color=TH["muted"], fontsize=8.3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=TH["ink2"], fontsize=9.2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    style(ax, grid=None)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TH[k]) for k in keys]
    leg = fig.legend(handles, [nice[k] for k in keys], loc="upper left", bbox_to_anchor=(0.19, 1 - 0.6 / H),
                     ncol=4, frameon=False, fontsize=8.8, handlelength=1.0, handleheight=0.8, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(TH["ink2"])
    fig.text(0.03, 1 - 0.2 / H, title, color=TH["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 1 - 0.42 / H, sub, color=TH["ink2"], fontsize=9.0, va="center")
    save(fig, name)


def fig_hindsight():
    p = OUT / "hindsight_oracle_fix_sound_yolo_dev.json"
    if not p.exists():
        print("  hindsight: no data yet")
        return
    R = [r for r in json.loads(p.read_text())["records"] if r.get("triggered") and "meet" in r]
    R.sort(key=lambda r: -r.get("meet_true", 0))
    fig, ax = plt.subplots(figsize=(8.4, 1.4 + 0.3 * len(R)))
    fig.patch.set_facecolor(TH["surface"])
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.2, right=0.96, top=1 - 1.0 / H, bottom=0.5 / H)
    y = np.arange(len(R))[::-1]
    for yi, r in zip(y, R):
        t, o = r.get("meet_true", 0.0), r.get("meet_best_other", 0.0)
        ax.hlines(yi, min(t, o), max(t, o), color=TH["axis"], linewidth=1.4, zorder=2)
        ax.plot(o, yi, "o", color=TH["switched"], markersize=7, markeredgecolor=TH["surface"], markeredgewidth=1.3, zorder=3)
        ax.plot(t, yi, "o", color=TH["recovered"], markersize=7, markeredgecolor=TH["surface"], markeredgewidth=1.3, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['seq'][-2:]} · id {r['id']}" for r in R], color=TH["ink2"], fontsize=8.6)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("how well the backward path lands on where the person was last seen for certain (median IoU)", color=TH["ink2"], fontsize=9)
    style(ax, grid="x")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=TH["recovered"], linestyle="none", markersize=7, label="the true person"),
         Line2D([], [], marker="o", color=TH["switched"], linestyle="none", markersize=7, label="best of the 7 others")]
    leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.2, 1 - 0.62 / H), ncol=2, frameon=False, fontsize=8.8)
    for t in leg.get_texts():
        t.set_color(TH["ink2"])
    fig.text(0.03, 1 - 0.2 / H, "Hindsight: following everyone backward", color=TH["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 1 - 0.42 / H, f"{len(R)} dev episodes where our tracker lost firm hold; candidates from the ground truth (the person and the 7 nearest).",
             color=TH["ink2"], fontsize=9.0, va="center")
    save(fig, "hindsight")


def main():
    which = set(sys.argv[1:]) or {"hidden", "clues", "outcomes", "hindsight"}
    if "hidden" in which:
        fig_hidden()
    if "clues" in which:
        fig_clues()
    if "outcomes" in which:
        fig_outcomes([("SAM 2.1 alone", "p0_sam2_bp_dev"), ("our tracker", "fix_sound_yolo_dev")], "outcomes_dev",
                     "Where the tracker ends up after the occlusion (development set)",
                     "16 MOT20 episodes, 21–70 frames hidden. The rules were designed while looking at these.")
        fig_outcomes([("SAM 2.1 alone", "p0_sam2_bp_held"), ("+ SAMURAI", "fix_samurai_held"), ("our tracker", "fix_sound_yolo_held")], "outcomes_held",
                     "Where the tracker ends up after the occlusion (held-out set)",
                     "96 MOT20 episodes never looked at while the rules were written.")
    if "hindsight" in which:
        fig_hindsight()


if __name__ == "__main__":
    main()
