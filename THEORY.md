# Theory

Part IV is the sound tracker (plan of 2026-10-14, evening). Parts I–III are the conformal switch
detector of the first day and are in `python/crowdtarget/martingale.py`'s docstring and RESULTS.md.

## Part IV. A sound set-valued identity through occlusion

### Setting

A fixed camera films a crowd. Frame `I_t(p) ∈ R^3` (sRGB). One target is designated before an
occlusion gap `[t_0, t_1]`; after the gap, candidates `C` are visible regions (ground-truth boxes
in the census, detections and SAM 2 masks in the tracker). The output per frame is a set `S_t` of
candidates, or a singleton (a commit). **Sound** means the true person is in `S_t`; the theorems
bound the probability that it is not.

Every layer below is a function `s_i(C, t)` of a candidate and an exclusion rule `s_i > q_i`. The
thresholds are calibrated on the *true person's* scores over calibration episodes; power against
impostors is measured, not proven.

### Theorem 1 (containment: the colour budget)

*Model.* Diagonal illumination: on frame `t` every pixel of a static surface satisfies
`I_t(p) = diag(g_t) I_0(p) + n_t(p)`, `g_t ∈ R^3_+` the frame's gain, `|n| ≤ σ`. Per-channel power
laws commute with diagonal scaling, so the model holds in gamma-encoded sRGB with other gains.
*Coordinates.* `c(J) = (J_R, J_G) / (J_R + J_G + J_B)` for corrected `J = diag(ĝ_t)^{-1} I_t(p)`,
quantised to a `K × K` grid, plus a dark bin for `max J < D`. Chromaticity is invariant to a
*uniform* gain; the division by `ĝ_t` is what makes it invariant to a diagonal one.
*Histogram.* For a region `R` of a person of pixel height `h`: `ν_R[b] = #{p ∈ R : c(p) ∈ b} / h²`.
Under a fixed camera the pixel area of a person scales as `h²` (foreshortening aside), so `ν` is a
count per unit of the person.
*Budget.* `μ[b] = max_{t ∈ pre-gap} ν_{R_t}[b]`, the bin-wise maximum over the views shown before
the gap. *Dilation* `μ̃` is the maximum over each bin's 3 × 3 neighbourhood; the dark bin is
untouched.
*Excess.* `E(C) = Σ_b max(ν_C[b] − (1+ε) μ̃[b], 0) / Σ_b ν_C[b]`, and the certified excess is its
minimum over the gain box `{ĝ_t (1 + η) : |η_k| ≤ δ_t}` evaluated on the 27-point grid
`η_k ∈ {−δ_t, 0, δ_t}`.

**Theorem 1.** Let `C` be a region of the target whose surfaces were all shown during the pre-gap
window, viewed at height `h_C`, under a gain `g` inside the box, with at most a fraction `η` of its
pixels moved across a bin edge by noise and quantisation. Then the certified excess satisfies
`E(C) ≤ η`.

*Proof.* Fix the pre-gap view `t*` that shows the surface each pixel `p ∈ C` belongs to (there is
one by hypothesis; the same surface may be shown in several views). Scale: a surface of area `a`
in world units occupies `a·h²·k` pixels for a constant `k` of the camera, so per unit `h²` its
count is the same in `C` and in `R_{t*}`. Colour: the corrected chromaticity of `p` equals the
corrected chromaticity of its pixel in `R_{t*}` up to the gain error and noise. On the grid point
of the gain box nearest `g`, the gain error is at most `δ_t` per channel in relative terms, which
moves `c` by less than one bin for the bin widths used (`1/K` on each axis) when `δ_t ≤ 1/(2K)`
relative to a chromaticity of order one; the one-bin dilation absorbs it. So every pixel of `C`
not disturbed by noise lands in a bin `b` whose dilated budget `μ̃[b] ≥ ν_{R_{t*}}[b'] ≥` its own
contribution, because `μ` is the maximum over views and dominates `ν_{R_{t*}}`. Summing, the
counts of `C` outside `(1+ε) μ̃` are at most the `η` fraction of disturbed pixels. The minimum
over the grid is at most the value at the nearest grid point. ∎

*Contrapositive.* `E(C) > η` certifies that `C` is **not** a sub-view of any surface the person
showed before the gap, under any illumination in the box: an impostor, or a surface never shown
(a turn that reveals a backpack). The test cannot tell these apart, and does not claim to.

*Remarks.* (i) The maximum over views, not the mean, is what makes a later partial view a
sub-measure: the mean would be exceeded by any view that shows one part in a higher proportion.
(ii) The test is one-sided: absence of a colour is never evidence. (iii) Shadows and mixed
colour temperature are local gains, outside the model; the measured violation rate on real
episodes is what the calibration absorbs.

### Theorem 2 (conservation)

Under the hypotheses of Theorem 1, `Σ_b ν_C[b] ≤ (1+ε) Σ_b μ[b]`: a view of the person has at
most the person's mass per unit height². *Proof.* Sum the per-pixel argument. ∎ A region that
merges two people fails it.

### Theorem 3 (the tube)

*Objects.* `O_t` = union of every other person's box on a grid of `CELL` px; `X_{t_0}` = a
one-cell disc about the target's last visible foot cell; `X_t = dilate(X_{t−1}, r) ∩ O_t` for
`t_0 < t ≤ t_1`, with `r = ⌈v_max h / CELL⌉` cells; once the person may be visible again
(`t > t_1`) `X_t = dilate(X_{t−1}, r)`. The reach at `t` is `dilate(X_{t−1}, r)`.

*Hypotheses.* H1: the target's foot point moves at most `v_max h` px per frame. H2: a person
whose foot point lies in free space (outside `O_t`) during the gap is detected there with
probability ≥ `ρ` on each such frame, independently of the past (census: visibility defines the
gap, so `ρ = 1`).

**Theorem 3.** If no admissible candidate passed the certificates during the gap, the target's
foot point at `t ≤ t_1` lies in `X_t` with probability at least `1 − (1−ρ)^k`, `k` the number of
gap frames on which it would have stood in free space; with `ρ = 1` it lies in `X_t` surely.
After the gap it lies in the reach. *Proof.* Induction. At `t_0` the foot is in `X_{t_0}`. If it is
in `X_{t−1}`, by H1 it is in `dilate(X_{t−1}, r)`; if it were outside `O_t` it would be in free
space and, by H2, detected with probability `ρ`, contradicting the premise; so with probability
≥ `ρ` it is in `O_t`, hence in `X_t`. The failure events across frames multiply by the
independence in H2. After `t_1` no intersection is taken, so only H1 is used. ∎

*Failure signal.* `X_t = ∅` means H1 or H2 failed at this frame; the reach is kept and the event
is counted. *Hand-over.* An identity whose box starts to hold part of `X_t` is a new occluder the
target may be behind; the count of these is the hand-over count. *Cost.* One dilation and one
intersection per frame on a `⌈H/CELL⌉ × ⌈W/CELL⌉` grid.

### Theorem 4 (neighbour continuity)

*Rule.* On the public detections, a previous box and a detection are a *certain* match when their
IoU is ≥ `τ_m` and neither has another partner at IoU ≥ `τ_a`. A track ends at its first frame
without a match. A neighbour is excluded as the hidden target when one track covers ≥ 80 % of the
gap frames with certain matches.

*Hypothesis H3.* Two people whose boxes have IoU below `τ_a` on consecutive frames are not
swapped by the rule, i.e. a certain match is the same person on both frames.

**Theorem 4.** Under H3 a neighbour excluded by the rule was visible, as the same person, on ≥ 80 %
of the gap frames, hence is not the target, which by definition of the gap was visible on none.
*Proof.* Chain the certain matches. ∎ H3's violation rate is measured on ground truth (certain
runs of ≥ 10 frames containing two identities) and reported with the census; it is the α of this
layer, with a binomial confidence bound, not a theorem.

### Theorem 5 (metric height)

*Model.* Fixed camera, planar ground, upright people: pixel height is affine in the foot's image
row to first order, `h = a y + b`, fitted per sequence on fully visible boxes.
`ρ = h / (a y + b)` is a person's height in units of the scene's mean height.

**Theorem 5.** For a given person `ρ` does not depend on depth, up to the fit's residual and the
box's own error. *Proof.* Similar triangles: `h ∝ H_world / Z`, `y − y_horizon ∝ 1/Z` for feet on
the plane, so `h` is affine in `y` with slope proportional to `H_world`; dividing by the
population fit gives the person's height relative to the population's. ∎ The exclusion
`|ρ_C − ρ_target| > κ` wrongly excludes the true person with probability ≤ α when `κ` is the
conformal quantile of the true person's `|ρ_exit − ρ_pre|` (Theorem 6).

### Theorem 6 (soundness)

*Split conformal.* For a layer `i` with per-episode statistic `s_i` of the true person (a single
number per episode: the minimum over exit frames for `E` and conservation, the median difference
for `ρ`), calibrated on `n` episodes exchangeable with the test episode, let `q_i` be the
`⌈(n+1)(1−α_i)⌉`-th smallest calibration value. Then `P(s_i(test) > q_i) ≤ α_i` (Vovk, Gammerman,
Shafer 2005; the rank of the test value among `n+1` exchangeable values is uniform).

**Theorem 6.** Let `D` be the event that the true person, though visible in the exit window, is
not among the candidates; `N` the event that the continuity rule excludes the true person (H3
fails); `A` the event that a confirmation certifies an impostor (Theorem 7). Then
`P(true ∉ S_t for some t in the episode, or a wrong commit) ≤ Σ_i α_i + P(D) + P(N) + P(A)`,
the sum over the conformal layers (budget, conservation, height, and the tube when its speed bound
is itself the conformal quantile of the per-episode speed needed to keep the true person inside).
*Proof.* A wrong commit requires the true person to be absent from `S_t`, which requires one of
the listed events, or a false confirmation; union bound. ∎

The three measured terms are where the proof leans on data: `P(D)` is one minus the detector's
recall on half-visible people (census), `P(N)` the continuity rule's violation rate, `P(A)` the
crowd pass rate of Theorem 7. Completeness, how often `S_t` is a singleton, is not a theorem; it
is the census number.

*Two conditions.* Calibrate on the kind of region you deploy on (ground-truth boxes in the census,
the tracker's own masks in the tracker); one statistic per episode, not one test per frame, so
there is no multiple-testing inflation across frames.

### Theorem 7 (crowd-relative confirmation)

*Objects.* A confirmation `f` is a predicate on candidates built from the target's pre-gap
appearance: shows ≥ k of the target's *rare* colours (bins with ≥ 2 % of the budget's mass whose
one-bin neighbourhood at most `p_max` of the other people present show), or is certified by ≥ 1
*distinctive* anchor (a certified texture patch of the target that certifies on none of the other
people present). Its crowd pass rate `p̂_f` is the share of the other `n` people in the pre-gap
frames that pass `f`; `p̄_f` its Clopper–Pearson upper confidence bound.

*Rule.* Commit to the unique member of `S_t` passing `f` when `|S_t| · p̄_f ≤ α_conf`.

**Theorem 7.** If the impostors in `S_t` are exchangeable with the other people in the pre-gap
frames with respect to `f`, the committed member is wrong with probability at most
`(|S_t| − 1) p_f ≤ |S_t| p̄_f ≤ α_conf`, the first inequality with the confidence of the bound.
*Proof.* A wrong commit requires some impostor in `S_t` to pass `f` (and the true person not to);
union bound over the impostors, each passing with probability `p_f`. ∎

The reference population is the node's own crowd: "red hat" means something among black coats and
nothing at a match of the team in red. The hypothesis that exit candidates are exchangeable with
the pre-gap crowd is checked in the census by comparing `p̂_f` with the actual pass rate of
impostors at exit.

### Theorem 8 (revocation)

*Rule.* Every commit is a checkpoint (frame, `X_t`, the candidate set, the frozen pre-gap identity
memory). The committed hypothesis keeps being tested by every exclusion layer with the same
per-episode statistics. A failure revokes the commit, kills the hypothesis, and re-opens the set
from the checkpoint: alternatives are carried forward by their detection tracks, the tube is
re-propagated from the checkpoint through the stored `O_t` grids for anyone who stayed hidden,
with the revoked identity removed.

**Theorem 8.** Revocation applies only exclusions, so the bound of Theorem 6 is unchanged; and on
the event that the true person's path satisfies every certificate (probability ≥ 1 − the bound),
the search never discards it permanently, so it is in the output set at the end of the episode.
*Proof.* Each revocation removes one hypothesis that failed a certificate; on the good event the
true hypothesis never fails one, so it is never removed, and re-opening from a checkpoint restores
every alternative that was present there. This is depth-first search with the certificates as
constraints: exclusions are forward checking, revocation is backtracking. ∎

What revocation changes is not soundness but the *duration* of a wrong commit, which becomes a
measured quantity (frames on the wrong person before revocation), and the unrevoked switch rate at
episode end beside the standard switch count.

### What is not provable, said plainly

Two people in the same wardrobe, of the same height, hidden together behind the same group, carry
no information in colour, texture, height or motion that tells them apart until they separate.
`|S_t| = 2` is the correct output then, and it stays until one of them walks into free space and
fails a certificate. Night footage puts most pixels in the dark bin and the budget cannot
discriminate. The exchangeability behind the conformal guarantees holds within the sequences the
calibration came from; a new scene needs recalibration, not a new method.
