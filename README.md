# crowd-target

One target person through a crowd, on top of a frozen SAM 2.1, with one new piece: a
statement that **the tracker has switched person**, with a controlled false-alarm rate, made
without training and without a noise model.

Status 2026-10-15, end of day: **closed as a negative result.** The martingale is valid on
exchangeable streams, but a tracked person's appearance after an occlusion is an outlier to
their own earlier appearance on about half of real crowd episodes, with either training-free
embedding tried, so the detector fires on appearance change rather than identity change.
Numbers in RESULTS.md. No claim is made.

## The piece

A tracked person gives a stream of nonconformity scores: the distance of the current mask's
appearance embedding to the person's own earlier frames. Online conformal p-values turn that
stream into independent uniforms under exchangeability; a test martingale bets on them; by
Ville's inequality the alarm `W_t >= 1/alpha` has `P(false alarm, ever) <= alpha` under the
null. The mathematics is Vovk's (conformal test martingales, 2003-2021) and Ramdas's
(anytime-valid testing); it has not been used in visual tracking, where switch detection is a
threshold on the model's own score, a cycle-consistency check (SENTRY, 2026) or a learned
verifier. `python/crowdtarget/martingale.py`.

What was measured (RESULTS.md Part II): on ground-truth crops of 24 MOT20 episodes, the share
of the same person's post-gap crops that score outside the whole pre-gap pool is 96 % for the
Faster R-CNN box-head feature with a 30-frame pool, 61 % with 300 frames of history, 45 % for
DINOv2 ViT-S/14 with 300 frames; across-gap same-vs-other AUC 0.69-0.75. Per episode it is
bimodal: on about half, the person's appearance genuinely changed. A conformal p-value against
the person's own history is then ~1/(n+1) whatever the identity, and the martingale alarms on
half the episodes in which SAM 2 recovers the right person. That fails the kill criterion
written before the runs (false alarms above 2 alpha), by the score, not by the test.

## Protocol

Single-target episodes mined from MOT20 train by fixed rules (`python/crowdtarget/episodes.py`):
an identity with an occlusion gap of 20 to 200 frames, a clean run before and after, a box at
least 48 px tall; bins 20-40 / 40-80 / 80+; a dev set of 8 per bin by sorted stride, the rest
held out. SAM 2.1 base+ is prompted with the ground-truth box 30 frames before the gap and
stepped to 60 frames after it; the outcome is recovered / switched (to whom, when: during the
gap or after) / lost, by mask-in-box containment.

## Reproduce

```
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision numpy pillow scipy pytest hydra-core iopath tqdm
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2@$(cat sam2_commit.txt | cut -d' ' -f3)"
.venv/bin/pip install -e ../certified-skip -e ../certified-track -e .
.venv/bin/python scripts/fetch_data.py                 # MOT20.zip, 5.03 GB, SHA-256 checked
.venv/bin/python scripts/mine_episodes.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_episodes.py --set dev --model b+
```

The SAM 2.1 base+ checkpoint is read from `data/sam2/` or from `../certified-skip/data/sam2/`.
SAM 2 is Meta's code from GitHub at the commit in `sam2_commit.txt` (the `sam2` wheel on PyPI
is a third-party repackaging). `certified-skip` and `certified-track` sit beside this repo for
the renderer, the ffmpeg reader and the output harness.

## What is new, what is not, what is missing

**Not new.** SAM 2.1 and its memory (Meta 2024); training-free memory selection in SAM2Long,
DAM4SAM, SAMURAI, HiM2SAM, SENTRY; single-target episodes as an analysis; mask-in-box
containment; Faster R-CNN box-head features as an appearance descriptor; CUSUM with a learned
null (certified-skip §6); conformal test martingales and anytime-valid testing themselves.

**New, to my knowledge.** The application of a conformal test martingale to a tracker's own
appearance stream as an identity-switch detector, and the measurement that says why it cannot
work with an appearance score: the across-gap separability tables, which anyone building
ReID-based re-association on crowds may want. The census of clean occlusion episodes in MOT20.

**Missing / not claimed.** Any guarantee when the scores are not exchangeable (consecutive
frames are correlated; the measured false-alarm rate is the honest number); look-alikes whose
appearance does not move at a switch (the martingale is silent; MOT20 has few); a benchmark
number on LaSOT, VOT-LT, LVOS or MOSE (not run); a learned ReID embedding (not used);
everything that would follow the alarm (re-detection, identity sets) is planned, not built.

## Part IV (2026-10-14, evening): the sound tracker, census first

Status: **census done; a first-cut tracker built and run on the 16 dev episodes (RESULTS.md Part V):
SAM 2.1 alone 13 recovered / 2 switched / 1 lost; the sound tracker with YOLO11x detections
13 / 0 / 2 + 1 ambiguous with the right person in its set throughout.** Silent switches to zero on
the episodes the rules were designed from, one new loss (a night-time false detection); the
held-out 96 are not run. The ground-truth census (Part IV) killed the full plan on its tube
criterion and left resolution at 21 % against a 60 % pass line: on MOT20, neighbour bookkeeping
excludes 61 % of impostors, the colour budget 36 %, height 11 %, the tube 10 %; certified anchors
confirm with 13 right / 1 wrong but the surrounding crowd is too small to bound their false rate.
After Piece 1 closed, two published fixes were re-implemented on top of
frozen SAM 2.1 for comparison (`python/crowdtarget/fixes.py`: SAMURAI's motion-aware memory, and
a DAM4SAM-style distractor-anchor rule, deviations listed in the docstring; `run_episodes.py
--fix`). SAMURAI still switches on MOT20-05 id 611: the impostor walks where the Kalman filter
predicts and is accepted into memory. A point prediction is the wrong object for a hidden person.

The new design outputs a **set** of identity hypotheses and commits only when certificates leave
one, or confirm one. Every layer is one-sided with a stated hypothesis: the colour budget
(bin-wise maximum over views of an illumination-corrected, height-normalised chromaticity
histogram; excess mass certifies "not a sub-view of anything seen", worst case over the gains
estimated from the static background), conservation, the tube (reachable set through the other
people's boxes, with a failure signal), neighbour continuity on the public detections, metric
height from the ground plane, and two crowd-relative confirmations (rare colours, distinctive
certified texture anchors) whose false rate is the share of the surrounding crowd that passes
them. Commits are revocable (backtracking from a checkpoint). Theorems 1–8 in THEORY.md Part IV.

**Not new.** Chromaticity and the diagonal illumination model; histogram intersection (Swain,
Ballard 1991, the opposite containment direction); colour-constant indexing (Funt, Finlayson
1995); background medians; occlusion geodesics (Possegger et al. 2014); reachable sets; multiple
hypothesis tracking; network-flow association; height as a soft biometric; SAM2Long's memory
branching; SAM2MOT; split conformal; the certified patch test of certified-track.

**New, to my knowledge.** Identity after occlusion as a sound set-valued estimate with a coverage
theorem in which every layer is a one-sided certificate with its hypothesis stated and its
violation rate measured; the height-normalised bin-wise-maximum budget and its certified excess;
the tube as a theorem with a failure signal; certificate-pruned SAM 2 memory forks; crowd-relative
confirmation; and the identifiability census of MOT20 occlusion episodes layer by layer
(`scripts/census.py`, `out/census.txt`), which bounds what any method built from colour, motion,
height and bookkeeping can do on this data.

**Missing.** The tracker itself (P2 of the plan) is built only if the census passes its kill
criteria; shadows and mixed colour temperature; night footage; detector recall in dense crowds;
anything learned.

## Licence

MIT for this code. MOT20 and the SAM 2.1 weights stay under their own terms and are fetched
from their owners' servers, never redistributed here.
