# Results

Every number is in a file under `out/` with the machine and timings. Mac M4, 16 GB, MPS. MOT20
train, single-target episodes mined by the rules in `python/crowdtarget/episodes.py`.

## Part I — the data regime (2026-10-15)

Census of occlusion gaps in MOT20 train (`out/census_mot20.txt`): 2,064 maximal runs of 20
frames or more at visibility < 0.2, but under the v1 episode rules only 8 episodes. The rule
that removes almost everything is the clean run before and after the gap: the median box
visibility in these crowds is 0.35-0.52, so a full occlusion is rarely bracketed by half-visible
frames. v2 rules (gap at visibility < 0.3, 15 of 30 pre-gap and 40 of 60 post-gap frames at
>= 0.5, height >= 40 px), set before any tracker ran: 120 episodes, 77 / 29 / 14 in the gap
bins 20-40 / 40-80 / 80+, dev set 24 (8 per bin by sorted stride), 96 held out
(`out/episodes_mot20.txt`).

## Part II — Piece 1, the conformal switch detector, closed (2026-10-15)

The mathematics holds: on simulated exchangeable streams the online conformal p-values are
uniform (KS distance < 0.02 over 8,000 draws) and both martingales keep the false-alarm rate
under Ville's bound over 2,000 streams at alpha 0.1 and 0.05, while a shift of 2.5 standard
deviations is caught in a median of under 25 steps (`tests/test_martingale.py`).

What fails is the premise that a tracked person's appearance after an occlusion gap is
exchangeable with its appearance before. Measured on ground-truth crops of the 24 dev
episodes, no tracker involved (`out/embed_gap_check_dev*.txt`): the pool is the target's crops
before the gap (the 30-frame pre window, or 300 frames of history), scored by 1 - cosine to the
pool's mean embedding; "same" is the target's crops after the gap, "other" every other
identity in those frames.

| embedding | pool | same-person post-gap crops above the whole pool | AUC same-across-gap vs other, pooled | median per episode |
|---|---|---|---|---|
| Faster R-CNN box head | 30 pre-gap frames | 96 % | 0.69 | 0.69 |
| Faster R-CNN box head | 300 frames | 61 % | 0.68 | 0.74 |
| DINOv2 ViT-S/14 | 30 pre-gap frames | 89 % | 0.75 | 0.81 |
| DINOv2 ViT-S/14 | 300 frames | 45 % | 0.71 | 0.81 |

Without a gap the same box-head feature separates identities at AUC 0.90 (frames 15-45 apart,
`out/embed_check.txt`); a half-occluded crop of the same person sits at distance 0.21, between
same (0.12) and other (0.27). Across a gap, per episode, the DINOv2 share above the pool is
bimodal: 0 % on ten episodes, 90-100 % on nine. On the latter the person has turned, moved
into other light or changed what is visible of them; their new appearance is an outlier to
their own history, and a conformal p-value against that history is ~1/(n+1) whatever the
identity. A test martingale on such a stream detects appearance change, not identity change,
and alarms on about half the episodes in which SAM 2 recovers the right person. The first
tracked episode shows it (`out/p1_p0_sam2_bp_dev*.txt`, `MOT20-01_42_256`): recovered, every
frame on the right person, wealth 500 after the gap under every gate tried.

Verdict against the plan's kill criterion (false alarms above 2 alpha at k = 12): failed, by
construction of the score rather than by the test, on both embeddings available without
training. Not pursued further: a learned ReID network (would narrow the across-gap gap but not
close it on the 90-100 % episodes, where the appearance did change), masking the background
with SAM 2's mask (changed nothing consistently on two episodes), testing the post-gap stream
against itself (answers a different question).

What survives: the census (the clean-occlusion regime is rare in MOT20), the across-gap
separability tables (a number people who build ReID-based re-association on crowds may want),
and a validated implementation of conformal test martingales (`python/crowdtarget/martingale.py`).

## Part III — SAM 2.1 base+ alone, and the martingale on its stream (16 of 24 dev episodes)

The run was stopped by decision after the 20-40 and 40-80 bins, once Part II had settled the
question; the 80+ bin did not run. `scripts/run_episodes.py --set dev`, 1.5-2.6 s per frame at
1080p on MPS, prompted with the ground-truth box 30 frames before the gap, memory as shipped.

| bin | n | recovered | switched (in gap / after) | lost |
|---|---|---|---|---|
| 20-40 | 8 | 7 | 1 (0 / 1) | 0 |
| 40-80 | 8 | 6 | 1 (1 / 0) | 1 |
| all | 16 | 13 | 2 | 1 |

`out/p0_sam2_bp_dev.txt`. SAM 2.1 alone recovers 13 of 16: the failure the project set out to
attack is rare on these episodes (the plan's own P0 threshold was switched + lost below 15 %,
here 19 % on a small sample). One switch began inside the gap, a drift onto the occluder.

The martingale on the tracked stream, gated to frames where SAM 2 claims a whole person
(object score > 2, mask area >= half the pre-gap median), `out/p1_p0_sam2_bp_dev_gs2_ga0.5.txt`:

| detector, alpha 0.05 | stream | switches detected within 30 frames | recovered episodes alarmed |
|---|---|---|---|
| mixture martingale | every 4th frame | 0 of 2 | 13 of 13 (one more alarmed before its switch) |
| simple jumper | every 4th frame | 1 of 2 | 13 of 13 |
| learned-null CUSUM, h = 6 | every 4th frame | 0 of 2 | 13 of 13 |
| mixture martingale | every 12th frame | 0 of 2 | 12 of 13 |
| simple jumper | every 12th frame | 0 of 2 | 0 of 13 |
| SAM 2 object score < 2 | every frame | 0 of 2 | 3 of 13 |

As Part II predicted from the crops: on the tracked stream the martingales alarm on every
recovered episode at every alpha, and the one setting that does not alarm (the jumper at every
twelfth frame) detects nothing. Proof videos: `out/proof_recovered_MOT20-01_42.mp4` (recovered,
wealth rising through the gap and after it) and `out/proof_switch_MOT20-05_611.mp4` (the switch
inside the gap).

## Part IV — the sound tracker: census first (2026-10-14, evening)

Plan `~/.claude/plans/streamed-mixing-shell.md`, theorems in THEORY.md Part IV. The design outputs
a set of identity hypotheses and commits only when one-sided certificates leave one or confirm
one. Before building the tracker, a ground-truth-only census measured what each certificate buys
on all 120 MOT20 episodes (`scripts/census.py`, `out/census.{json,txt}`, 5 s per episode, two-fold
cross-conformal thresholds at alpha 0.05 per layer).

**Verdict: killed on the plan's tube criterion, and far from the pass line.** The true person's
foot point is outside the tube at the end of the gap on 18 % of episodes (criterion: 10 %); the
set is resolved to the right singleton on 21 % (pass: 60 %); median set size 3 (pass: 1).

| | n | coverage | set median | singleton correct | disc → tube → neighbour → height → budget (median candidates) | E(true) q90 | true left tube | hand-overs | occluders |
|---|---|---|---|---|---|---|---|---|---|
| all | 120 | 0.82 | 3 | 21 % | 12 → 11 → 4 → 4 → 3 | 0.06 | 17 % | 20.7 | 36.8 |
| gap 20-40 | 77 | 0.92 | 2 | 30 % | 9 → 9 → 3 → 3 → 2 | 0.04 | 14 % | 14.0 | 25.0 |
| gap 40-80 | 29 | 0.72 | 4 | 7 % | 18 → 17 → 8 → 7 → 4 | 0.06 | 21 % | 27.3 | 48.4 |
| gap 80+ | 14 | 0.50 | 4.5 | 0 % | 31 → 24 → 14 → 10 → 4 | 0.16 | 29 % | 44.0 | 77.2 |

Coverage: the true person was dropped by the budget on 8 episodes, the tube on 7, height on 5,
the disc on 1 (21 of 120; the three conformal layers at alpha 0.05 each account for about 15 of
those, as the theorem allows; the tube's 7 are hypothesis failures).

**What each layer buys, measured on the 1,746 impostors inside the reach disc, each layer alone:**

| layer | impostors excluded | hypothesis violation (true person wrongly excluded) |
|---|---|---|
| neighbour continuity (visible >= 80 % of the gap) | 61 % | 0 of 120 (ground-truth oracle); detector rule violated on 3.3-4.5 % of certain runs |
| colour budget + conservation | 36 % | 8 of 120 (6.7 %, alpha 0.05 x 2 layers) |
| metric height | 11 % | 5 of 120 |
| tube (reachable set through the other people's boxes) | 10 % | 21 of 120 outside at gap end; 7 excluded at exit |

Bookkeeping does the work. Colour is second and is sound (median certified excess of the true
person 0.00, 90th percentile 0.06) but weak: MOT20 is dark coats, and the two cross-fold
thresholds differ by a factor of three (0.19 / 0.07) because seven outdoor episodes in MOT20-05
show excess 0.14-0.25 on the true person: shadows and colour temperature, outside the diagonal
model, exactly where THEORY.md said it would fail. Height removes one impostor in nine: people
are the same height. The tube, in a crowd this dense, covers the disc: it touched a median of 37
other people over a gap and removes one impostor in ten; its hypotheses fail most in MOT20-03
(9 of 25 episodes, mean time inside 0.12: the person is hidden by something that is not a person).

**Confirmation (the human "red hat").** Rare colours exist on 11 % of episodes; where they exist
the true person still shows one at exit on 77 %. Distinctive anchors: 5.6 kept per episode, the
true person certified at exit on 32 %, a surviving impostor certified on 0.09 per episode. The
commit rule with the crowd-measured bound (`|S| p̄ <= 0.1`) never fired: the reference crowd is
25-50 people, so the Clopper-Pearson bound on a 0-count is 0.06-0.11 and no set of two or more
qualifies. Without the bound, "exactly one survivor passes" gives anchors 13 right / 1 wrong,
rare colours 4 / 0: a useful cue with 93 % measured precision that this crowd is too small to
certify.

**Detector recall** of the public detections at visibility >= 0.5: 0.96 (MOT20-01, 02), 0.67
(03), 0.77 (05). In a tracker the candidate set would miss the true person on up to a third of
exits in MOT20-03; Theorem 6's `P(D)` term is the largest in the bound there.

**P1 redone with the budget score** (`scripts/p1_sound_alarm.py`, `out/p1s_p0_sam2_bp_dev.txt`):
the embedding distance alarmed on 13 of 13 recovered episodes; the certified excess alarms on
5 of 13 (mixture, alpha 0.05, k = 4) and 3 of 13 (jumper), and detects 0 of 2 switches, because
both impostors wear the target's palette. The jumper at k = 12 alarms on none and detects none.
Kill criterion (false alarms > 2 alpha) still failed; the score is sound but has no power here.

**What is kept.** Theorems 1-8 with their hypotheses; the layer-by-layer identifiability of
MOT20 (bookkeeping 61 %, colour 36 %, height 11 %, reach 10 %); the confirmation precision of
certified anchors (13/1) and rare colours (4/0); the per-sequence hypothesis checks (tube
containment, detector recall, continuity-rule violation 3-5 %). What is not built: the tracker
(P2), by the plan's own rule.
