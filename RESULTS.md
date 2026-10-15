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

## Part V — steps 1 and 2: a better detector, and SAM 2 vetoed by the certificates (2026-10-15, small hours)

Niyath asked for the two cheapest fixes from Part IV's list: a real detector for the bookkeeping,
and SAM 2's commits vetoed by the certificates. Both built; the 16 dev episodes with a baseline
were run. The held-out 96 were not.

**Step 1, detector.** YOLO11x (Ultralytics, COCO weights, 110 MB, SHA-256 in `data/detector/`),
input size 1920, every frame of the dev episode windows (`scripts/detect.py`, `out/detector_recall.txt`).
Recall on ground-truth boxes with visibility >= 0.5, IoU >= 0.5:

| sequence | public 2019 detections | YOLO11x conf 0.25 | YOLO11x conf 0.1 |
|---|---|---|---|
| MOT20-01 | 0.96 | 0.97 | — |
| MOT20-02 | 0.95 | 0.98 | — |
| MOT20-03 (night) | 0.67 | 0.84 | 0.90 |
| MOT20-05 (outdoor, small people) | 0.77 | 0.68 | 0.81 |

A COCO detector at native resolution beats the public detections on three sequences and ties on
the fourth only at a low threshold; a crowd-trained detector would do better on MOT20-05 and was
not fetched.

**Step 2, the tracker** (`python/crowdtarget/controller.py`, `run_episodes.py --fix sound --det yolo`).
SAM 2.1's mask is accepted into memory and output only if it passes its own gates (score > 0, area
>= half the running median), the colour budget (certified excess <= 0.2, conservation <= 1.1), the
known-neighbour veto (its box does not coincide at IoU >= 0.5 with a detection track that stood
beside the person while tracked), and, while absent, the tube's reach. A refused frame is dropped
from memory and the last accepted frames re-aliased (SAMURAI's trick, used for a different reason).
While absent, one admissible detection re-prompts SAM 2 provisionally; firm after 5 accepted frames,
revoked before that. Re-acquisition needs score > 2 and conservation >= 0.4. Two rules were changed
after seeing dev failures and are named as such in the module docstring (the containment veto
refused id 85 seen through an occluder; the re-acquisition gate after id 505 accepted a fragment).

**Result on the 16 dev episodes** (`out/compare_sam2_fix_samurai_fix_sound_public_fix_sound_yolo.txt`):

| | recovered | switched | lost | ambiguous (true person in the set) |
|---|---|---|---|---|
| SAM 2.1 alone | 13 | 2 | 1 | — |
| + SAMURAI (4 episodes run) | 1 | 3 | 0 | — |
| sound tracker, public detections, 6 episodes, before the two rule changes | 3 | 2 | 1 | 0 |
| **sound tracker, YOLO11x, 16 episodes** | **13** | **0** | 2 | 1 (100 % of post frames) |

Per episode, what changed:
- **MOT20-05 id 611** (SAM 2 and SAMURAI switch): recovered, 100 % of post frames on the person.
  Two frames refused (2442 by area, 2443 by the neighbour veto in the public run); the baseline
  kept those two frames in memory and slid onto 631, then 638. Video `out/compare_MOT20-05_611_three.mp4`.
- **MOT20-03 id 505** (SAM 2 lost, SAMURAI switched): ambiguous. The tracker never committed; its
  candidate set contained the right person on every post-gap frame. Video
  `out/compare_MOT20-03_505_three.mp4`. The firm-only view scores it lost: no claim was made.
- **MOT20-05 id 480** (SAM 2 switched to 482): lost, 18 % of post frames correct. Forty provisional
  commits, none held: SAM 2's mask alternated between the person and a merged blob that
  conservation refused. Not switched any more, not recovered either.
- **MOT20-03 id 169** (SAM 2 recovered): lost. A regression. At night a false detection inside the
  reach passed the budget (everything dark lands in the dark bin), was re-prompted 54 times, and
  the real person, who re-emerged beside the tube and walked out of its reach while the tracker
  was busy, was never a candidate (true person in the set on 0 % of frames).
- The other 12: unchanged, recovered, 100 % of post frames.

**What the numbers say.** Silent switches 2 → 0 on the episodes the design was built from, with
one new loss. The certificates do what they were meant to do, refuse the occluder, and the cost is
exactly the one predicted: abstention (505) and churn where SAM 2's own mask is unstable (480).
The id 169 loss exposed that the tube's reach must be the gate of *entry* for a hypothesis, not a
condition it has to keep satisfying: a person who re-emerges and walks away leaves the reach in
two frames. The second cut (track-carried hypotheses, candidate confidence >= 0.25 and track age >= 3,
a 10-frame cooldown on a track whose prompt was refused) is running as `fix_sound2_yolo_dev`.

**Honesty.** Sixteen episodes, three of which the rules were designed on. The held-out 96 (about
11 hours including their SAM 2.1 baseline) are the test of the claim "no silent switch"; nothing in
this part is that test.

## Part VI — how hidden is the hidden person? (2026-10-15, ground truth only)

`scripts/census2.py`, `scripts/census3.py`, `out/census2.txt`, `out/census3.txt`, 120 episodes.

| | all | gap 20-40 | gap 40-80 | gap 80+ |
|---|---|---|---|---|
| gap frames (visibility < 0.3) with >= 100 px of the person showing outside nearer boxes | 88 % | 93 % | 78 % | 81 % |
| median showing pixels on a gap frame | 772 | 826 | 454 | 1314 |
| longest run with < 100 px showing (median / 90th pct) | 0 / 20 frames | 0 / 6 | 1 / 33 | 2 / 59 |
| episodes whose longest blackout is <= 5 frames | 77 % | 87 % | 59 % | 57 % |
| gap frames with the head region (top 15 %) less than half covered | 79 % | 84 % | 69 % | 71 % |

"Hidden" in MOT20 means a sixth of the body still shows, most of the time. The re-identification
problem is mostly a problem of holding on to a sliver, plus bridging blackouts of a few frames.

Two ways to use that, measured:
- *Unexplained foreground near the last position* (frame minus background minus every other person's
  box) has a >= 300 px blob on 98 % of gap frames, but its centroid is on the target on 5 %. The
  residual is dominated by unannotated people, loose boxes and background error. Dead as measured.
- *Certified sliver steps* (the raw-pixel box test of certified-track, exact brightness offset, 21 px
  template at the sliver's centroid, S = {0}, searched +-10 px on the next frame): a usable template on
  33 % of consecutive sliver pairs; when usable it certifies on the target 82 % of the time and on
  the nearest other person 0.0 % (0 of 1,288). Chains are short (median 2 steps) because one template
  per frame is sparse, not because it is wrong.

Depth-order exclusion at exit was measured vacuous with the speed-bound margin used (0 % excluded).

### Part VI, continued — certified sliver chains, built and measured (2026-10-15, 01:00–03:30)

`python/crowdtarget/sliver.py`, `scripts/census_chain.py`, `out/census_chain.txt` (templates fully
inside the sliver), `out/census_chain_d128.txt`; tracker integration `controller.py` with
`use_chain` (`run_episodes.py --fix sliver`).

Several 21 px templates per sliver instead of one, carried frame to frame by the certified step.
Two mining rules were measured on all 120 episodes, consecutive gap frames with >= 100 sliver px:

| mining rule | episodes seeded | pairs with templates | certified positions on / off the target's box | off-box share | gap frames joined by a certified step (median) | longest chain / gap |
|---|---|---|---|---|---|---|
| template centre on the sliver, 2 px erosion | 88 % | 93 % | 9,872 / 1,150 | 10.4 % (10 episodes hold 721 of the 1,150) | 59 % | 0.40 |
| whole 21 px patch inside the sliver, templates that leave the mask dropped | 46 % | 69 % | 2,065 / 35 | 1.7 % | 0 % (mean 16 %) | 0.00 |

The first rule's false steps are templates cut at the sliver's edge that take occluder pixels with
them and then certify on the occluder for the rest of the gap. The second rule removes them and with
them most of the chain: a sliver thinner than 21 px holds no template. Blackouts were almost never
bridged by searching the old templates (k = 6: 4 of 50, then 0 of 45). The certificate is
identity-safe and far too sparse to carry identity through a gap. Plan criteria (templates on
>= 70 % of pairs, false < 1 %, chains covering the gap): failed on coverage either way.

In the tracker the chain made things worse. On id 611 the first version certified one or two
templates inside a mask that had slid onto id 631 and, because a chained mask waived the area gate
and the neighbour veto, accepted it: recovered became switched. Requiring three certified
positions did not restore it; the rule changes made alongside (every simultaneous detection a
known other; veto active while tracking; a containment veto; a continuity gate) moved other
episodes too (505 ambiguous became switched). All of that is kept behind `variant="v2"` and
`use_chain`; `--fix sound` is again exactly the rule set of the 16-episode table above
(`variant="v1"`), verified by re-running id 611 (recovered, 1 commit, 0 vetoes, as before).

Also found while tracing id 480: ids 480 and 482 stand at the top edge of MOT20-05, 57 to 75 px
tall, truncated by the frame; YOLO11x has no detection within 120 px of either. No bookkeeping, no
budget and no 21 px template can act there. That episode is outside what these tools can reach.

**Kept from the sliver work:** the measurement that "hidden" people show a sixth of their body on
88 % of their occluded frames; the certified step's 0.0 % false rate against neighbours (census 3)
and 1.7 % off-box rate when templates lie wholly on the person (census 4); `sliver.py` and its
tests. **Not kept:** any claim that chains carry identity through a gap in this footage.

## Part VII — hindsight tracking (2026-10-15)

Written before the first run. Protocol in `python/crowdtarget/hindsight.py`: the sound tracker (v1)
runs forward; at its first loss of firmness (5 non-firm frames) the anchor t_a is its last firm frame;
at the end of the clip t_b every detection within reach becomes a candidate and is tracked backward
by SAM 2.1, jointly, to t_a - 10; the identity is the candidate whose backward track meets the
forward track (median IoU >= 0.5 over the last 11 firm frames, second best < 0.25), otherwise the
set or absent. Kill criteria: any switched episode, or fewer than 13 recovered of the 16 dev
episodes; pass: 0 switched and >= 14 recovered. The decision frame is the clip end, which the
episode rules place 60 frames after re-emergence: the one dependence on the episode definition.

*Amendment, before any result, after measuring cost.* A reverse SAM 2.1 step costs about 0.45 s per
object per frame on the M4 (1, 4, 16 objects: 1.4, 3.5, 10.4 s per frame), and at the clip end the
reach holds 12-47 detections with the true person ranked 17th-28th by distance on half the episodes,
or missing (3 of 13: not detected at that frame). The deployment protocol would cost 30-40 minutes per
episode and is mostly a candidate-generation problem. The first test is therefore an oracle census of
the matching step alone: decision 15 frames after the ground-truth re-emergence, candidates = the
target and the 7 people nearest it (ground-truth boxes), same meet rule. Kill: the true person
committed on fewer than 10 of the 13 triggered episodes, or any wrong commit. Only if it passes is
candidate generation worth building.

## Part VIII — held-out benchmark (launched 2026-10-15, written before any held-out result)

The 96 held-out MOT20 episodes (69 / 21 / 6 by gap bin; MOT20-05 60, -03 19, -02 15, -01 2;
12,313 frames), never looked at while any rule in this repository was written. Runs, all on the M4:
SAM 2.1 base+ alone (`p0_sam2_bp_held`); our tracker exactly as in Part V (`--fix sound`, variant v1,
YOLO11x detections at 1920, `fix_sound_yolo_held`); SAM 2.1 + SAMURAI (`fix_samurai_held`), last,
if time allows. Hindsight is not part of this benchmark: its candidate generation is not built.

Primary metric: episodes ending on the wrong person (switched). Secondary: recovered; lost; ambiguous
with the true person in the set. Predictions written now: our tracker has at most half of SAM 2.1's
switched episodes, and recovers within 3 episodes of SAM 2.1. If switched is not halved, the claim
"it stops identity switches" is dropped from the write-up.
