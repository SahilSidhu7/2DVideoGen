# critic/ — the objective judge

Every other lane in this project measures itself against its own loss, its own
grammar, or its own eyeball impression of a GIF. This directory does not trust
any of that. It answers one question about a finished clip: **is this a real,
usable animation video, or not** — and it says so even when the answer is no.

```
critic/
  frames_io.py   load a video / gif / frame-directory into RGB numpy frames
  metrics.py     the five metrics — numbers only, no pass/fail
  verdict.py     the gate — thresholds, USABLE / PROMISING / NOT-AN-ANIMATION
  evaluate.py    CLI: run every metric on one clip, write report.json
  out/           report.json per clip scored so far (this README's source data)
```

```bash
USE_TF=0 python critic/evaluate.py <video|gif|frame-dir> \
    [--guide <matching input-guide clip>] \
    [--pose-guides <dir of svg/guides.py pose/ PNGs>] \
    [--yolo yolo11n-pose.pt] [--max-frames 200] [--identity-max 60]
```

Writes `<out>/report.json` (raw numbers) and prints the same verdict
`verdict.py` would print on its own:

```bash
USE_TF=0 python critic/verdict.py critic/out/<stem>/report.json
```

## Relationship to the published metrics (paper/RESEARCH.md §6)

The research lane's evaluation survey (FVD / content-debiased FVD / JEDi /
CLIPSIM / VQAScore / warping error / FVMD / EvalCrafter) is the reference this
critic was built against, not ignored in favour of home-grown numbers:

| this critic | published metric | why this one and not that one |
|---|---|---|
| `temporal_stability.warp_error` | **warping error** (§6, "compute optical flow between consecutive frame pairs, warp frame t, take pixel-wise diff from t+1") | same definition, computed with Farneback flow (CPU) instead of RAFT (GPU, heavier) — the hardware note rules out an extra deep net just for the judge. Comparable in spirit, not numerically identical to a RAFT-based figure in another paper. |
| `temporal_stability.flicker_ratio` | the **raw inter-frame-diff ratio** §6 explicitly calls out as "already computed once, in Attempt 18" (1.29 guide / 30.44 output = 23.6x) | reproduced verbatim as a metric here (see the calibration run below) rather than reinvented. |
| — | **FVD / content-debiased FVD / JEDi** | need a pretrained I3D-family feature net and, per §6, "hundreds to thousands of clips per condition" for a stable Fréchet estimate. This project has at most a few dozen clips per condition and an 8 GB card at 50% duty — not a defensible sample size, so it is not implemented here rather than reported on a sample too small to mean anything. |
| — | **CLIPSIM / VQAScore** | both need a downloadable CLIP/VQA checkpoint; none is cached on this machine (`identity_preservation` below explains the same constraint) and fetching one was out of scope for this pass. This is a real gap, not a stylistic choice — see "Most damning finding" below. |
| — | **FVMD** | motion-specific Fréchet distance, same sample-size problem as FVD. |
| `motion_presence`, `identity_preservation`, `pose_fidelity`, `scene_complexity` | not in the published set | added because none of FVD/CLIPSIM/warping-error catches this project's specific known failure modes: a frozen clip that scores perfectly on stability (§ below), a character that changes identity mid-clip (Attempt 18), or a render that ignores its pose guide entirely (Attempt 18). |

## The eight metrics

### 1. Temporal stability

Three numbers, because raw pixel diff alone cannot tell fast legitimate
motion from flicker:

- **`mean_abs_diff`** — mean absolute grayscale change between consecutive
  frames, normalised to [0, 1]. Informational; feeds the other two.
- **`warp_error`** — Farneback optical flow from frame t to t+1, warp t
  forward, mean absolute residual against t+1, normalised to [0, 1]. This is
  the real signal: low residual means the frame-to-frame change *is* motion;
  high residual means content is popping in and out that no motion field can
  explain.
- **`flicker_ratio`** — `mean_abs_diff(output) / mean_abs_diff(guide)`, only
  computed when `--guide` is given. This is the exact Attempt 18 quantity.

**Thresholds** (`verdict.py`):

| | pass | hard-fail | reason |
|---|---|---|---|
| `warp_error` | ≤ 0.06 | ≥ 0.15 | Video-prediction/generation work reporting a warping-error metric typically treats residuals under ~5% of dynamic range as well-aligned (§6). 0.06 gives this project's own rasteriser aliasing a little room; 0.15 (2.5x that) is the point past which optical flow explains almost none of the frame-to-frame change. |
| `flicker_ratio` | ≤ 3x | ≥ 8x | Attempt 18 measured 23.6x and every frame was visibly a different character — an order-of-magnitude blowup, not embellishment. Genuine secondary motion a stylizer legitimately adds on top of a guide (hair sway, shading) plausibly doubles or triples the guide's own inter-frame change, hence the 3x pass bar; 8x (≈1/3 of Attempt 18's measured failure) is set as the point no plausible secondary motion explains it. |

### 2. Identity preservation

Crops the character (YOLO11-pose's largest detected person box, or a 70%
center-crop if none is found — the pipeline never leaves the region
undefined), embeds each crop as an **HSV joint colour histogram**
(12×8×8 bins, L2-normalised), and reports mean and worst-case pairwise cosine
similarity across the clip (subsampled to `--identity-max` frames, default
60, to keep the O(n²) comparison cheap).

This is the fallback explicitly permitted by the brief ("a colour+histogram
fallback if VRAM is tight"), used by default rather than a CLIP embedding —
no CLIP checkpoint is cached on this machine
(`~/.cache/huggingface/hub` has Qwen/T5/SD/ControlNet/MiniLM but no
`clip-vit-*`), and fetching one was out of scope for this pass. **It is
deliberately crude**: it measures palette/composition stability, not "is this
the same drawn character" in any structural sense — a costume swap in the
same colour scheme would slip past it entirely. See "Most damning finding."

**Thresholds:**

| | pass | hard-fail | reason |
|---|---|---|---|
| `mean_similarity` | ≥ 0.75 | — (soft only) | below 0.75 the *average* frame pair already reads as a different colour scheme. |
| `worst_similarity` | ≥ 0.45 | < 0.30 | worst-case below 0.30 means at least one point in the clip swapped almost completely — Attempt 18 ("hair, face and background all reorganise" every frame) is expected to land under this. |

#### Bug found and fixed: mixed crop regimes (Attempt 21, scene lane)

The crop this metric compares frame-to-frame used to be chosen **per frame**:
a YOLO detection box when one fired, the 70% centre-crop fallback otherwise —
independently, frame by frame. This is wrong, and it cost a real clip its
verdict. On `out/a21_street.mp4` (four hand-scripted characters,
`scenescript.py`, Attempt 21) YOLO fired on exactly **1 of 24** sampled
frames — a 48×135 box around one small figure — while the other 23 used the
identical centre-crop box. The metric then compared that one geometrically
unrelated crop against the rest and reported `worst_similarity: 0.4368`,
under the 0.45 pass bar, capping the clip at PROMISING. **The clip never
changed. Only the measurement basis changed, once, mid-clip**, and the metric
reported that as an identity change. A colour-based independent audit of the
same clip (`scenescript.py --audit`) found all 4 characters visible in 84 of
84 sampled frames — nothing about this clip's actual content supports a
near-failing identity score. Forcing the fallback crop for every frame gives
**0.9972** on the same clip.

**Fix:** `character_boxes()` now picks exactly one crop regime for the whole
clip and never mixes them, using the same "refuse to score on noise" bar
`pose_fidelity` already applies (`MEASURABLE_DETECTION_RATE = 0.50`,
`DETECTION_CONF_FLOOR = 0.40`, metric 4):

- Confident-detection rate < 50% of sampled frames → **every** frame uses the
  centre-crop fallback, regardless of the handful that did detect something.
- Confident-detection rate ≥ 50% → **every** box is YOLO-derived: a frame
  without its own confident detection reuses the nearest confident detection
  in time (held forward, or backward for leading frames) rather than falling
  through to the centre crop.

Either way, a "detection regime" clip contains zero fallback-derived boxes
and a "fallback regime" clip contains zero detection-derived ones — the two
are never compared against each other within one clip. The regime actually
used, why, and the confident-detection rate behind that choice are always in
the report (`crop_regime`, `crop_regime_reason`,
`confident_detection_rate`), so a result is never presented without the
context that explains it. `svg/out/demo/anisvg_8b_reel.mp4` sits exactly at
the 50% boundary in this project's own baseline table and correctly lands in
`detection` regime with no mixing (see the table below).

**This is worth stating as a failure mode in its own right, because it will
recur wherever a metric quietly switches its own measurement basis mid-
series:** the failure signature is a near-perfect score on almost the whole
clip dragged through a pass bar by one or a few outlier samples measured a
different way. It is a strong, easy check for this general class of bug: if
removing a single frame or sample from a "worst-case" statistic changes the
verdict, and that sample was measured on a different basis than its
neighbours, the metric — not the clip — is what changed. **A second, real
instance of the same bug** surfaced retroactively once this fix was applied:
`release/corpus-anime/corpus-anime.mp4` had been scoring `NOT-AN-ANIMATION`
on `identity_preservation.worst_similarity: 0.29` — attributed at the time
(see "Most damning finding" below, corrected here) to genuine identity drift
across a multi-clip reel. Re-run with the fix, the same clip scores
`worst_similarity: 0.99995`, `crop_regime: fallback` throughout (confident
detection rate 20%, below the 50% bar) — the 0.29 was the same crop-mixing
artefact, not identity drift. `corpus-anime.mp4` is now `PROMISING`, capped
only by `warp_error` (0.0711 vs a 0.06 bar), which was never in question.

### 3. Motion presence

The metric that exists specifically to stop a frozen clip from scoring
perfectly. Farneback flow magnitude between consecutive frames, split into:

- **`motion_coverage`** — fraction of pixels per frame pair whose flow
  clears a 0.5 px noise floor (below that, Farneback's own quantisation noise
  on a flat background is indistinguishable from motion).
- **`motion_energy`** — mean flow magnitude *among only those pixels*,
  normalised by the frame diagonal.

Averaging flow over every pixel including background would dilute a thin
line-art figure moving against a large flat background to near zero even
when the figure's own motion is large — exactly what this project's
stick-figure and vector renders look like. Restricting to moving pixels
avoids under-scoring genuinely animated but visually sparse content.

**Threshold:** `motion_coverage` ≥ **0.2%** of the frame per pair to pass;
below it, hard-fail as "stable because nothing moved" regardless of how good
`warp_error`/`flicker_ratio` look — a frozen frame trivially has zero of
both. Verified directly: a 30-frame clip built by repeating one real frame
30x scores `warp_error 0.00001`, `flicker N/A`, perfect identity — and
`motion_coverage 0.0025%`, which the gate correctly reports as
`NOT-AN-ANIMATION: motion_presence: coverage 0.0025% < floor 0.20% — stable
because nothing moved`. Command: see the `static_test` row below.

### 4. Pose fidelity

Only computed when `--pose-guides` (a directory of `svg/guides.py`
`pose/*.png` frames — exact skeletons recovered from the fixed rig, not
detected) is supplied. Guide joints are recovered by locating each
OpenPose-palette limb colour in the guide PNG (reusing
`svg/guides.py:POSE_COLOURS`), taking the two extreme points along the
limb's principal axis, and resolving proximal/distal by distance to the
body centroid. Output joints come from `yolo11n-pose.pt` (COCO-17) on the
matching output frame. Per-joint error is Euclidean distance normalised by
frame diagonal; **PCK@0.10** (fraction of joints within 10% of the diagonal)
is the standard "percentage of correct keypoints" convention from
pose-estimation benchmarks (Yang & Ramanan 2011 / MPII use a comparable
fraction of a reference body segment as "correct").

**Threshold:** PCK ≥ 50% to pass, < 20% is hard-fail — **but only when the
detector actually found the character.** Below 20% correct, the render is
doing a materially different pose than the guide asked for (the exact
Attempt 18 failure: guide = raised-arm wave, render = hands on hips).

**Correction, cross-lane (Attempt 19, builder lane):** the first version of
this metric silently trusted whatever YOLO returned, including near-zero
sample counts. The builder's Attempt 19 control measured the actual scope of
the problem: CMU OpenPose (`controlnet_aux`) returns **no person at all** on
our styled output, our flat cel renders, and our raw line-art skeletons —
while reading a real photograph (`zidane.jpg`) fine, 12/12 keypoints. YOLO11-
pose does somewhat better but found the cel figure in only **2 of 6** tested
clips, at confidence **0.41 and 0.22**. The root cause the builder traced it
to is the same one that made ControlNet ignore our pose guides in Attempt 18:
our rig's proportions are non-human by construction (ear span **1.65x**
shoulder width, vs **~0.52** for a real human per the builder's measurement),
and every detector here — ours and ControlNet's — was trained on photographs
of human proportions. **One cause, two symptoms.**

So `pose_fidelity` now tracks detection explicitly and refuses to report a
score built on noise:

- `detection_rate` — fraction of sampled guide frames where *any* person box
  was found, at any confidence.
- `mean_confidence` — mean box confidence where a person was found at all.
- **`DETECTION_CONF_FLOOR = 0.40`** — matches the builder's own split (0.41
  read as a real, if marginal, detection; 0.22 read as noise).
- **`MEASURABLE_DETECTION_RATE = 0.50`** — at least half the sampled guide
  frames must clear the confidence floor before a PCK number is trusted at
  all.

Below that bar the result is `available: false, measurable: false`, with
`reason` spelling out the detection rate and mean confidence, and it is
**never hard-gated** (see the comment above `POSE_PCK_PASS` in `verdict.py`)
— an undetectable clip inherits neither a pass nor a fail.

Only one asset in this project ships both a rendered output and matching
pose guides for the same clip: `svg/out/styled/frames` against
`svg/out/styled/guides/pose`. **Corrected result:** a confident detection in
only 2/8 frames (mean confidence 0.53 where found at all, in 3/8) — reported
as `pose_fidelity: UNMEASURABLE`, not the PCK 77% the uncorrected first pass
of this metric printed (that number was computed from 3 frames, one of which
was below the confidence floor, and should never have been treated as a
result). The clip's overall verdict is unaffected — it already hard-fails on
`flicker_ratio` — but the earlier PCK figure was wrong to report as a score
and is retracted here.

### 5. Scene complexity

Counts `person` detections per frame with `yolo11n-pose.pt`, reports the
modal count and what fraction of frames match it (`cast_stability`). Never
gates the verdict — see the comment in `verdict.py`: no output anywhere in
this project achieves a stable multi-character cast with background and
props (ATTEMPTS.md's stated end goal), so gating on it would mark every clip
in the repo NOT-AN-ANIMATION regardless of animation quality. It is reported
because the gap between "temporally usable" and "the project's actual goal"
should stay visible, not because it decides usability here.

**Honest limitation, stated rather than hidden:** `yolo11n-pose.pt` only
detects the `person` class. It cannot count props or background elements, so
this metric measures at most half of "multiple characters, a background, and
props" — the other half is not measured by any tool used in this pass.

**Correction (Attempt 21, scene lane): a bare cast count of zero was being
reported as if it were a measurement, and it was not one.** `yolo11n-pose.pt`
detects **no person at all** in either of this project's own hand-written
single-figure controls, `control_wave` and `control_kick`
(`modal_count: 0` on both) — and the same is true of a hand-scripted scene
containing three or four visible, correctly-drawn characters
(`out/a21_park.mp4`, `out/a21_street.mp4`): a photo-trained detector does not
read flat stick-figure line art, at any cast size. Reporting `modal_count: 0`
in that situation is indistinguishable from "this clip genuinely has no
characters," which is false every time it has happened in this project.

`scene_complexity` now applies the same measurability bar `pose_fidelity`
already uses (`MEASURABLE_DETECTION_RATE = 0.50`,
`DETECTION_CONF_FLOOR = 0.40`): below a 50% confident-detection rate across
the sampled frames, the result is `available: false, measurable: false`,
with `reason` naming the detection rate and mean confidence, e.g.

```
scene_complexity: UNMEASURABLE: confident (>= 0.40 conf) person detection in
only 0/40 frames (mean conf 0.00 where detected at all in 0/40) - this
reports detector failure on the art style, never a cast count of zero
```

Still never hard-gated, exactly as before — only what "no measurement" looks
like changed, from a silent zero to an explicit refusal to answer.

### 6. Content correctness / prompt alignment (CLIPSIM)

Added specifically to close a gap the first five metrics leave wide open:
none of them check whether the clip is *of* the thing it claims to be — see
"Most damning finding" below, which is exactly what motivated this metric.

**CLIPSIM** = mean cosine similarity between the clip's caption and each
frame's image embedding, both from **CLIP ViT-B/32**
(`openai/clip-vit-base-patch32`, transformers, **CPU-only** — ~600 MB,
downloaded fresh for this pass, never touches the GPU since the fan is
damaged and this metric doesn't need it). This is the CLIPSIM/CLIPScore
definition from paper/RESEARCH.md §6 (Hessel et al. 2021, EMNLP; first
applied to video in GODIVA, arXiv 2021). Reports `mean`, `min` (worst single
frame) and `std` across frames — mean catches a clip that is uniformly
off-topic, `std` catches one that starts right and drifts, a distinct
failure mode from being wrong throughout.

Only computed when `--caption` is given. **Deliberately not computed on the
project's own multi-clip demo reels** (`v1-8b-icons`, `v2-1.7b-icons`,
`corpus-stickman`, `corpus-anime`, both `anisvg_*_reel` videos) — each
concatenates six clips with six different captions, so no single caption
describes the file and scoring one against it would be measuring nothing.
Where a single caption genuinely applies (the hand-written controls, the
single-action Era 2/3 renders, the Attempt 18 hybrid, and one isolated
generated clip pulled out of its reel with its real training caption) it was
computed directly.

**Threshold — calibrated, and the calibration is a negative result, stated
plainly rather than tuned away.** The brief: score the hand-written controls
(known-good content) against a known-bad clip from `release/v2-1.7b-icons`
and pick a threshold that separates them. `release/v2-1.7b-icons` itself is
a multi-clip reel with no single caption, so the actual test used the one
clip from that model's training/eval set with a caption ATTEMPTS.md already
calls out by name — **`svg/out/gen17/g002`**, captioned "two four-pointed
star-shaped figures, a larger purple one and a smaller pink one..." which
ATTEMPTS.md #15 states draws "a large pink form and a small purple one... and
neither is a star."

Measured (CLIP ViT-B/32, CPU, mean over 20 frames):

| clip | caption used | CLIPSIM mean |
|---|---|--:|
| `control_wave.mp4` (hand-written, correct content) | its own correct caption | **0.329** |
| `control_kick.mp4` (hand-written, correct content) | its own correct caption | **0.233** |
| `g002` (model-generated, known-bad: blobs, not stars) | its own correct caption | **0.306** |
| `control_kick.mp4` (correct clip) | the **wrong** caption (wave's) | 0.235 |
| `control_wave.mp4` (correct clip) | an unrelated caption ("a car...") | 0.163 |

**The populations do not separate.** The known-bad blobs clip (0.306) scores
*higher* than the known-good kick control (0.233) — a threshold that passes
both controls cannot also fail the blobs, and a threshold that fails the
blobs also fails the kick control. Worse: giving the kick control the
*wrong* caption (wave's) barely moves its score (0.233 → 0.235) — CLIP
ViT-B/32 cannot reliably tell "kicking" from "waving" in this domain at all.
It *can* separate a wholly different category (an unrelated "car" caption
drops to 0.163), so there is a floor worth gating on, just not the one this
metric was built to provide.

**`CONTENT_CORRECTNESS_FLOOR = 0.18`** — the midpoint between the lowest
in-domain-but-plausible score measured across this project's own clips
(0.203, a different `gen17` clip against its own caption) and the one
clearly-wrong-category score measured (0.163, the car caption). This is
defensible only as a "did CLIP land on a completely different concept"
tripwire, hard-gated as such. **It does not, and at this model/budget cannot,
catch the failure it was added to catch** — g002 clears it at 0.306. See
"Most damning finding" for what this means for the paper.

Command used for the calibration numbers above:

```bash
USE_TF=0 python -c "
import sys; sys.path.insert(0, 'critic')
import metrics
from frames_io import load_frames
frames, _, _ = load_frames('critic/out/control_wave.mp4', max_frames=30)
print(metrics.content_correctness(frames, 'a stick figure waving its hand', max_frames=20))
"
```

### 7. Structural correctness (AniSVG source, not pixels)

CLIPSIM measured that perceptual embedding similarity cannot separate "drew
what was asked" from "drew a colour-matched blob" in this domain. But the
AniSVG generation path has something a normal video-generation project does
not: **the output is the source.** A generated clip is text — a declared cast
of shapes with a fixed paint order — not a raster image with content that has
to be inferred by another model. ATTEMPTS.md states corpus captions are
emitted from the generating parameters, true by construction rather than
inferred, so content correctness can be checked *exactly*, by parsing,
instead of *approximately*, by embedding.

**Scope, stated up front and enforced in code:** this metric applies **only**
to the symbolic AniSVG path — `svg/generate.py` output and the procedural
`chars.py`/`anime_chars.py` corpus. It reports `available: false` for any
clip with no `--anisvg` source given, which in this project means every
raster/diffusion output (`svg/out/styled`). **There is no declarative source
to parse for a diffusion image — this metric has, and can have, no opinion on
it.** That is a real asymmetry in what this project can currently prove about
its two output paths, not a gap to paper over; see "Most damning finding."

Built on `svg/generate.py`'s own `extract()`/`score()` rather than
re-implementing parsing — `structural_correctness()` calls `score()` first
and only proceeds if it says the document parses. Checks, all on the parsed
`anisvg.Anim`, not the rendered pixels:

- **`parses`** — reuses `svg/generate.py:score()`.
- **`ids_contiguous`** — for this format, shape index *is* the paint-order/
  part-role index, so "declared parts in the required paint order" reduces to
  "shape ids are exactly `0..n-1`, no gaps or duplicates."
- **`schema_shape_count_ok`**, **`points_per_shape_ok`**, **`cast_stable`** —
  only checked when the caller declares an expected schema (`--schema stick`
  or `--schema anime`), **never auto-inferred from shape count**. Measured
  reason this matters: `release/v1-8b-icons/clip00.anisvg` (an ordinary
  Lottie-icon-domain generation, nothing to do with the character rig)
  happens to declare exactly 10 shapes — the same count as the 10-part stick
  schema. Auto-classifying by count alone would have flagged that icon as a
  broken stick figure. `evaluate.py --schema` is only ever passed for clips
  known to come from `chars.py`/`anime_chars.py` or a model trained on them.
- **`points_per_shape_ok`** — every shape in a schema-matched clip is
  resampled by this project's own converter to the **same** point count.
  Measured directly, not assumed: **12/12 across all 12 sampled
  `corpus-stickman` and `corpus-anime` clips (both schemas, 6 each), zero
  exceptions.**
  A caught-and-fixed mistake, left visible rather than quietly corrected: the
  first version of this check assumed "a limb is a straight 2-point line,"
  read off `chars.py`'s pre-encoding joint pairs (`shapes.append(np.array([joints[a],
  joints[b]]))`) without checking what the *encoder* actually emits. It would
  have failed the project's own known-good procedural corpus — exactly the
  kind of self-inflicted false failure this pass is supposed to catch, not
  cause. Verifying against the real `.anisvg` files before shipping (below)
  caught it: every declared shape, head or limb, has exactly 12 points, not
  2.
- **`cast_stable`** — `chars.py`/`anime_chars.py` never hide a declared body
  part, so on a schema-matched clip any frame with an invisible part is a
  defect the pixel metrics cannot see: to optical flow, a limb popping out of
  existence looks like ordinary motion, not a missing part.
- **`star_score` / `geometry_verified`** — the flagship check, built for
  exactly the caption pattern ATTEMPTS.md #15 names ("two four-pointed
  star-shaped figures..."). A star has a determinable vertex count and radius
  alternation — but this corpus's converter resamples *every* outline to a
  roughly fixed vertex count regardless of primitive (measured: a real
  ground-truth star and a generated blob both have 16 vertices — vertex count
  alone carries no signal). So `star_score(pts, k)` instead measures the
  discrete Fourier coefficient of the shape's (centroid-normalised) radius
  sequence at frequency `k`: a true k-pointed star has k radius peaks and k
  troughs; a circle or blob has none, at any vertex count. `geometry_verified`
  is `True` when at least as many declared shapes clear `STAR_SCORE_THRESHOLD
  = 0.05` as the caption's object count requires.

**Calibration — the apples-to-apples pair.** The exact same caption exists
twice in this project: as real ground-truth AniSVG
(`svg/data/train/val.jsonl`, id `10887279`) and as a trained model's
generation of it (`svg/out/gen17/g002` — ATTEMPTS.md #15's own "given 'two
four-pointed stars,' it emits a large pink form and a small purple one... and
neither is a star"). Measured `star_score(k=4)`, best over all frames, per
declared shape:

| clip | shapes scoring ≥0.05 (star-like) | shape count | `geometry_verified` |
|---|---|--:|---|
| ground truth (10887279) | 3, at 0.135 / 0.154 / 0.140 | 11 | **True** |
| g002 (blobs) | 0, all four at 0.0048-0.0052 | 4 | **False** |

**This is the separation CLIPSIM could not produce**, on the same pair, at
zero GPU cost. Exact command:

```bash
USE_TF=0 python -c "
import sys; sys.path.insert(0, 'critic')
import metrics
text = open('svg/out/gen17/g002.anisvg', encoding='utf-8').read()
cap = 'Two four-pointed star-shaped figures, a larger purple one and a smaller pink one, pulse and shift in size and position before shrinking and fading out against a plain white background.'
print(metrics.structural_correctness(text, cap))
"
```

**A second, honest null this pass also found and did not gate on:** raw
`declared_shapes` vs the caption's object count. "Two star figures" reads
like it should mean 2 declared shapes; real ground truth declares **11**
(main star bodies plus shading/decoration shapes). A naive
`declared_shapes == expected_object_count` gate would fail the project's own
correct ground truth, so it is reported as
`declared_vs_expected_count_note` and **never hard-gated** — two honest nulls
in this pass (CLIPSIM's population overlap, and this count mismatch) beat
tuning either one until it looked right.

**Threshold defence for `STAR_SCORE_THRESHOLD = 0.05`:** it sits in the ~27x
gap between the weakest genuine star shape measured (0.135) and the highest
score any blob shape reached (0.0052) — there is no tuning pressure on this
number; the two clusters do not come close to touching.

### 8. Scene presence audit (colour + connected components, detector-free)

The honest working replacement for `scene_complexity` on this project's own
material, added once metric 5's correction (above) made explicit that a
photo-trained detector cannot answer "is this multi-character scene actually
multi-character" here at all. The scene lane's own workaround
(`scenescript.py --audit`) already answered it without any detector: a
declared character has a declared colour, so presence is "is enough of that
colour on screen, in a connected shape" — not detected, verified against a
caller-supplied ground truth, the same principle `svg/guides.py` already uses
for exact pose guides (declare the truth, don't infer it).

`scenescript.py --audit` is coupled to `.scene` script parsing and lives
outside `critic/`, so — per the brief — **the technique is reimplemented
here, independently, not imported**, keeping `critic/` free of any dependency
on `scenescript.py`'s file format while using the same tolerance (colour
distance < 45), the same minimum-ink floor (40 px) and the same 8-connectivity
(`cv2.connectedComponents`), so numbers from the two tools are comparable.

`metrics.scene_presence_audit(frames, colors)` takes a list of `(r,g,b)`
tuples, one per declared cast member, via `evaluate.py --character-colors
"AAFF5A,A064FF,FFBE46,F0F5FF"` (comma-separated hex). Reports, per character,
frames visible / presence rate / mean ink px / mean blob count, and overall
the fraction of sampled frames where **every** declared character is
visible.

**Scope limit, stated rather than hidden: this is a verification tool, not a
detector.** It has nothing to measure without being told what to look for, so
it can never replace `scene_complexity` on arbitrary/unlabelled input — only
on exactly this project's own scripted, colour-cast material, where the cast
colours are already known by construction. It is **never gated** (see
`verdict.py`): a wrong or incomplete colour list would silently under-report
presence, so the result is reported for the record, the same "measures
presence and occlusion, not personhood" framing `scenescript.py --audit`
itself uses, not treated as an infallible substitute for a real detector.

Measured on this project's own multi-character scenes:

```bash
USE_TF=0 python critic/evaluate.py out/a21_park.mp4 --max-frames 200 --identity-max 24     --caption "three stick figures in a park, two of them passing a ball" --clip-max 20     --character-colors "50F0F0,FF963C,FF6EB4" -o critic/out/a21_park
USE_TF=0 python critic/evaluate.py out/a21_street.mp4 --max-frames 200 --identity-max 24     --caption "four stick figures on a night street kicking a ball to each other" --clip-max 20     --character-colors "AAFF5A,A064FF,FFBE46,F0F5FF" -o critic/out/a21_street
```

| clip | characters | all visible (this tool, 90-frame sample) | all visible (`scenescript.py --audit`, full clip) |
|---|--:|---|---|
| `a21_park.mp4` | 3 | 66/90 (73%) | 75/75 sampled (100%) |
| `a21_street.mp4` | 4 | 61/90 (68%) | 84/84 sampled (100%) |

The two tools disagree on the exact rate because they sample differently
(this tool subsamples up to 90 frames via `np.linspace` over whatever
`--max-frames` already loaded; `scenescript.py --audit` steps every 5th frame
of the full clip) — not because either mistracks presence outright. Both
agree on the finding that matters: **every declared character is visible in
the large majority of sampled frames, at every sampling density tried**,
which is the thing `scene_complexity` cannot see on this art at all.

## Baseline table

22 clips, all from `critic/out/*/report.json`, produced by:

```bash
USE_TF=0 python critic/evaluate.py <input> --max-frames <N> --identity-max <K> \
    [--caption "<caption>"] [--guide <dir>] [--pose-guides <dir>] \
    [--anisvg <path.anisvg>] [--schema stick|anime] [--character-colors <hex,hex,...>]
```

Default GPU-guard duty cycling (`tools.gpuguard.Guard`, 50%) was active for
every row that used YOLO; CLIPSIM and structural correctness never touch the
GPU (metrics 6 and 7 — three structural-only rows used `--no-gpu-guard --yolo
''`, so their `id worst`/`crop_regime` are N/A by construction, not omission).

**Every `id worst` value below is post-fix** — the crop-regime bug (metric 2)
is retroactively fixed for every row that used YOLO; see "Before / after the
crop-regime fix" below for exactly what changed.

| clip | n | warp_error | id worst (`crop_regime`) | motion_cov | CLIPSIM | structural (`geometry_verified`) | **verdict** |
|---|--:|--:|--:|--:|--:|---|---|
| `control_wave.mp4` **(positive control)** | 30 | 0.0011 | 1.00 (fallback) | 1.01% | 0.329 | N/A (not AniSVG) | **USABLE** |
| `control_kick.mp4` **(positive control)** | 30 | 0.0048 | 1.00 (fallback) | 3.29% | 0.233 | N/A (not AniSVG) | **USABLE** |
| `out/a21_park.mp4` **(scene lane, Attempt 21 — 3 characters, background, props)** | 200 | 0.0055 | 1.00 (fallback) | 5.32% | 0.311 | N/A (not AniSVG) | **USABLE** |
| `out/a21_street.mp4` **(scene lane, Attempt 21 — 4 characters, fresh script)** | 200 | 0.0065 | **1.00** (fallback) | 4.89% | 0.336 | N/A (not AniSVG) | **USABLE** ⟵ *flipped by the crop-regime fix (was PROMISING at 0.4368)* |
| `corpus-stickman.mp4` (procedural reel, not model output) | 40 | 0.0063 | 1.00 (fallback) | 5.18% | N/A (reel) | N/A (reel-level) | USABLE |
| `release/corpus-stickman/clip00` **(schema positive control, isolated)** | 25 | 0.0072 | 1.00 (no YOLO) | 6.47% | N/A (no caption) | schema=stick, all checks pass | **USABLE** |
| `corpus-anime.mp4` (procedural reel, not model output) | 40 | 0.0711 | **1.00** (fallback) | 26.9% | N/A (reel) | N/A (reel-level) | **PROMISING** ⟵ *flipped by the crop-regime fix (was NOT-AN-ANIMATION at 0.29)* |
| `release/corpus-anime/clip00` **(schema positive control, isolated)** | 29 | 0.0058 | 1.00 (no YOLO) | 9.01% | N/A (no caption) | schema=anime, all checks pass | **USABLE** |
| `v1-8b-icons.mp4` (Qwen3-8B LoRA) | 40 | 0.0106 | 1.00 (fallback) | 4.78% | N/A (reel) | N/A (reel-level) | USABLE |
| `v2-1.7b-icons.mp4` (Qwen3-1.7B LoRA) | 40 | 0.0022 | 1.00 (fallback) | 2.02% | N/A (reel) | N/A (reel-level) | USABLE |
| `svg/data/train/val.jsonl#10887279` **(ground truth for g002's caption, isolated)** | 18 | 0.0088 | 0.84 (no YOLO) | 10.4% | 0.337 | schema=None, `geometry_verified=True` (3 shapes ≥0.05) | **USABLE** |
| `svg/out/gen17/g002` **(the SAME caption's model generation — ATTEMPTS.md #15's "neither is a star")** | 28 | 0.0089 | 0.99 (fallback) | 21.1% | 0.306 | `geometry_verified=False` (0 shapes ≥0.05) | **NOT-AN-ANIMATION** ⟵ *flipped by metric 7* |
| `anisvg_8b_reel.mp4` | 40 | 0.0660 | 0.98 (**detection** — sits exactly at the 50% regime boundary, correctly not mixed) | 18.6% | N/A (reel) | N/A (reel-level) | PROMISING (warp_error at bar) |
| `anisvg_17b_reel.mp4` | 40 | 0.0054 | 1.00 (fallback) | 1.70% | N/A (reel) | N/A (reel-level) | USABLE |
| `svg/out/styled/frames` (Attempt 18 hybrid, SD1.5+ControlNet) | 8 | **0.1408** | 0.63 (fallback) | 60.4% | 0.283 | **N/A — no AniSVG source at all, by construction** | **NOT-AN-ANIMATION** |
| `out/ar_walk.mp4` (Era 3, MotionAR) | 30 | 0.0049 | 1.00 (fallback) | 3.11% | 0.260 | N/A (not AniSVG) | USABLE |
| `out/nn_kick.mp4` (Era 2, distilled net) | 30 | 0.0047 | 1.00 (fallback) | 3.39% | 0.244 | N/A (not AniSVG) | USABLE |
| `out/realmotion_walk.mp4` (Era 3, RealMotionNet) | 30 | 0.0050 | 1.00 (fallback) | 2.90% | 0.281 | N/A (not AniSVG) | USABLE |
| `out/realmotion_dance.mp4` (Era 3, RealMotionNet) | 30 | 0.0050 | 1.00 (fallback) | 4.51% | 0.228 | N/A (not AniSVG) | USABLE |
| `out/rm_route_kick.mp4` | 30 | 0.0069 | 1.00 (fallback) | 4.53% | 0.272 | N/A (not AniSVG) | USABLE |
| `out/route_ar_walk.mp4` | 30 | 0.0059 | 1.00 (fallback) | 4.11% | 0.307 | N/A (not AniSVG) | USABLE |
| `critic/tmp/static_test.mp4` **(synthetic negative control — one real frame repeated 30x)** | 30 | 0.00001 | 1.00 (fallback) | 0.003% | N/A | N/A (not AniSVG) | **NOT-AN-ANIMATION** (motion floor) |

("no YOLO" for the three structural-only rows means `--yolo ''` was passed —
`character_boxes()` never touches a detector for them, so they were never
exposed to the crop-regime bug, before or after the fix.)

Exact per-row commands (the ones with non-default flags):

```bash
# positive controls
USE_TF=0 python stickman.py -o critic/out/control_wave.mp4 --action wave --seconds 3
USE_TF=0 python stickman.py -o critic/out/control_kick.mp4 --action kick --seconds 4
USE_TF=0 python critic/evaluate.py critic/out/control_wave.mp4 --max-frames 30 --identity-max 15 \
    --caption "a stick figure waving its hand" --clip-max 20
USE_TF=0 python critic/evaluate.py critic/out/control_kick.mp4 --max-frames 30 --identity-max 15 \
    --caption "a stick figure running up and kicking a ball" --clip-max 20

# scene lane, Attempt 21 - multi-character scenes, with metric 8's ground-truth colours
USE_TF=0 python scenescript.py scenes/park_meet.scene    -o out/a21_park.mp4   --frames out/a21_park_frames
USE_TF=0 python scenescript.py scenes/street_relay.scene -o out/a21_street.mp4 --frames out/a21_street_frames
USE_TF=0 python critic/evaluate.py out/a21_park.mp4 --max-frames 200 --identity-max 24 \
    --caption "three stick figures in a park, two of them passing a ball" --clip-max 20 \
    --character-colors "50F0F0,FF963C,FF6EB4" -o critic/out/a21_park
USE_TF=0 python critic/evaluate.py out/a21_street.mp4 --max-frames 200 --identity-max 24 \
    --caption "four stick figures on a night street kicking a ball to each other" --clip-max 20 \
    --character-colors "AAFF5A,A064FF,FFBE46,F0F5FF" -o critic/out/a21_street

# Era 2/3 single-action renders (caption matches the action in the filename)
USE_TF=0 python critic/evaluate.py out/ar_walk.mp4 --max-frames 30 --identity-max 15 \
    --caption "a stick figure walking" --clip-max 20
USE_TF=0 python critic/evaluate.py out/nn_kick.mp4 --max-frames 30 --identity-max 15 \
    --caption "a stick figure kicking a ball" --clip-max 20
# (realmotion_walk / route_ar_walk: "a stick figure walking";
#  realmotion_dance: "a stick figure dancing"; rm_route_kick: "a stick figure kicking a ball")

# isolated known-bad generated clip, pulled from its training/eval set with its real
# caption AND its raw .anisvg source (--anisvg is what turns on metric 7)
mkdir -p critic/tmp/g002_frames && cp svg/out/gen17/g002_f*.png critic/tmp/g002_frames/
CAP="Two four-pointed star-shaped figures, a larger purple one and a smaller pink one, pulse and shift in size and position before shrinking and fading out against a plain white background."
USE_TF=0 python critic/evaluate.py critic/tmp/g002_frames --max-frames 30 --identity-max 20 \
    --caption "$CAP" --clip-max 20 --anisvg svg/out/gen17/g002.anisvg -o critic/out/g002_blobs

# the ground-truth match for g002's caption, rendered fresh from its real corpus text
USE_TF=0 python -c "
import sys, json
sys.path.insert(0, 'svg')
from anisvg import Anim
import raster
rows = [json.loads(l) for l in open('svg/data/train/val.jsonl', encoding='utf-8')]
gt = next(r for r in rows if r.get('caption','').startswith('Two four-pointed star'))
open('critic/tmp/gt_star.anisvg', 'w', encoding='utf-8').write(gt['text'])
anim = Anim.from_text(gt['text'])
for i, svg in enumerate(anim.to_svgs()):
    raster.render(svg, width=256).save('critic/tmp/gt_star/f%03d.png' % i)
"
USE_TF=0 python critic/evaluate.py critic/tmp/gt_star --max-frames 30 --identity-max 18 \
    --caption "$CAP" --clip-max 18 --anisvg critic/tmp/gt_star.anisvg \
    --yolo '' --no-gpu-guard -o critic/out/gt_star

# schema positive controls: one clip pulled from each procedural reel, its own
# .anisvg source, --schema declared explicitly (never auto-inferred - see metric 7)
USE_TF=0 python critic/evaluate.py critic/tmp/corpus_stick00 --max-frames 30 --identity-max 20 \
    --anisvg release/corpus-stickman/clip00.anisvg --schema stick \
    --yolo '' --no-gpu-guard -o critic/out/corpus_stick00
USE_TF=0 python critic/evaluate.py critic/tmp/corpus_anime00 --max-frames 30 --identity-max 20 \
    --anisvg release/corpus-anime/clip00.anisvg --schema anime \
    --yolo '' --no-gpu-guard -o critic/out/corpus_anime00

# Attempt 18 hybrid — guide/pose-guides are the first 8 frames of
# svg/out/styled/guides/pose (the frame count render_style.py actually styled)
USE_TF=0 python critic/evaluate.py svg/out/styled/frames \
    --guide critic/tmp/guide8 --pose-guides critic/tmp/guide8 --identity-max 8 \
    --caption "anime style character, cel shaded, clean line art, flat colours, white background" \
    --clip-max 8

# synthetic negative control (motion-floor calibration)
python -c "
import cv2
cap = cv2.VideoCapture('critic/out/control_wave.mp4'); ok, f = cap.read(); cap.release()
w = cv2.VideoWriter('critic/tmp/static_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (f.shape[1], f.shape[0]))
for _ in range(30): w.write(f)
w.release()"
USE_TF=0 python critic/evaluate.py critic/tmp/static_test.mp4 --max-frames 30 --identity-max 20

# multi-clip reels (no --caption: no single caption describes the whole file, see metric 6)
USE_TF=0 python critic/evaluate.py <reel-path> --max-frames 40 --identity-max 20
```

### Before / after the crop-regime fix — two real verdicts changed

Re-running every clip that uses YOLO through the fixed `character_boxes()`
(metric 2) changed exactly two verdicts in this table, both upward, both
caused by the same bug and neither a property of the clip:

| clip | `identity_preservation.worst_similarity` before | after | verdict before | verdict after |
|---|--:|--:|---|---|
| `out/a21_street.mp4` | 0.4368 (mixed: 1 detection crop vs 23 fallback crops) | **0.9972** (fallback throughout) | PROMISING | **USABLE** |
| `release/corpus-anime/corpus-anime.mp4` | 0.2932 (mixed regimes across the 6-clip reel) | **0.99995** (fallback throughout) | NOT-AN-ANIMATION | **PROMISING** (still capped by `warp_error`, unrelated) |

Every other row in the table that used YOLO also has an updated
`worst_similarity` (see the baseline table above) — most moved by a
percentage point or less and none crossed a pass bar, because most clips
either detect confidently enough for the `detection` regime to apply
cleanly (`anisvg_8b_reel.mp4`, sitting exactly at the 50% boundary) or
detect so rarely that they were already effectively all-fallback under the
old per-frame logic too. The two rows above are the ones where the *old*
logic's occasional partial detections actually mattered, and both cases were
this project's own material being penalised for a measurement artefact, not
for anything wrong with the render.

**No positive or negative control changed.** `control_wave`, `control_kick`
(USABLE) and the synthetic frozen-frame negative control (NOT-AN-ANIMATION,
"stable because nothing moved") are unaffected — re-verified directly after
the fix, not assumed.

### Before / after CLIPSIM (metric 6) — a null result

Comparing verdicts with the five original metrics against verdicts with
CLIPSIM added: **zero clips changed verdict.** `corpus-anime`,
`svg/out/styled/frames`, and `critic/tmp/static_test.mp4` were
NOT-AN-ANIMATION before CLIPSIM and still are (identity, flicker, and
motion-floor respectively, unchanged); every USABLE clip, including `g002`
and the `v2-1.7b-icons` blobs it comes from, was still USABLE.
`CONTENT_CORRECTNESS_FLOOR` never fired on a single real clip in this table —
the lowest score anywhere (0.228, `realmotion_dance`) sits well above the
0.18 floor. CLIP ViT-B/32, at CPU budget with no domain fine-tuning, provably
cannot separate "drew what was asked" from "drew a colour-matched blob" at
any threshold that doesn't also fail hand-written ground truth. This null is
kept in the table exactly as measured — see metric 6 above.

### Before / after structural correctness (metric 7) — the metric that worked

Adding metric 7 changes exactly **one** verdict in this table, and it is the
one that matters most: **`svg/out/gen17/g002` flips from USABLE to
NOT-AN-ANIMATION.** Five pixel metrics and CLIPSIM all cleared it (1.00/0.99
identity, 0.0089 warp error, 0.306 CLIPSIM — comfortably above the floor that
also passes the hand-written controls). `structural_correctness` is the first
and only metric in this critic that looks at the file's declared geometry
rather than its rendered pixels, and it catches immediately what nothing else
could: 0 of the caption's 2 requested star shapes verify as stars
(`star_shapes_found: 0`, all four declared shapes score 0.0048-0.0052 on
`star_score(k=4)`, ~27x below the weakest genuine star shape in the matched
ground-truth clip). No other row in the table changes — every clip that was
already NOT-AN-ANIMATION stays that way for its existing reason, and every
clip without an AniSVG source (the hand-written controls, every Era 2/3
render, the Attempt 18 hybrid) is unaffected because the metric reports
`available: false` for them by design, not by omission.

**Read together, metrics 6 and 7 are the actual finding of this whole pass.**
Both were built to answer the same question — does the clip show what its
caption says — on the same calibration pair. One is a null result (CLIPSIM
cannot separate the populations, reported honestly rather than tuned until it
looked right); the other separates them cleanly, at zero GPU cost, because it
does not have to *infer* content from pixels when the content is *declared*
in the source. That is not a general solution to content correctness — it is
specific to the one output path in this project where generation and source
are the same text. See "Most damning finding" for what that means for the
project's other half.

## Most damning finding — unsoftened

**Content correctness was closed for one half of this project and could not
be closed for the other — and that asymmetry, not either metric individually,
is the finding to carry into the paper.**

ATTEMPTS.md #15 is explicit that even the best trained SVG model
("v2-1.7b-icons") produces "recognisable geometry: **still no**" — given "two
four-pointed stars," it draws a blob and a smaller blob, neither a star
(`svg/out/gen17/g002`, isolated and measured directly in this table). CLIPSIM
was built specifically to catch that and could not: CLIP ViT-B/32, trained on
natural photographs, cannot separate "kicking" from "waving" in this domain
(0.233 vs 0.235 for the same clip, right caption vs wrong), let alone
"four-pointed star" from "blob" — g002 scored 0.306, *higher* than the
hand-written kick control's own correct-caption score. That is a real,
measured limit of perceptual embedding on this hardware, reported as a null
result rather than tuned away (see metric 6).

**Structural correctness then closed exactly that gap, for exactly the paths
where it is possible to close it.** Because a generated AniSVG clip's output
*is* its source — a declared cast of shapes, not inferred pixels — g002's
failure could be checked exactly: `star_shapes_found: 0` against the
caption's requirement of 2, scores 0.0048-0.0052 against a matched
ground-truth clip's 0.135-0.154. **This is the one verdict in the entire
table that a content-correctness metric changed** (see "before / after"
above) — g002 flips from USABLE to NOT-AN-ANIMATION, and it is the correct
flip: the file genuinely does not declare the geometry its caption claims.

**The gap that remains is not a metric failure — it is a structural
asymmetry between the project's two output paths, and it must be stated
plainly for the paper: the AniSVG path now has an exact, zero-GPU-cost
content-correctness check; the diffusion/hybrid path (`svg/out/styled`) has
none, and cannot, by construction — there is no declarative source behind a
diffusion image to parse.** `svg/out/styled/frames` is rejected by this
critic on temporal grounds (23.6x flicker) with total confidence, but if a
future version of that pipeline fixed its flicker, this critic would have
**no way to verify it actually painted the anime character its prompt asked
for** — only CLIPSIM, already measured unable to make that call in this
domain. **Any claim in the paper that compares content correctness "before
and after the hybrid" or "AniSVG vs. diffusion" has to say explicitly that
one side of that comparison was checked exactly and the other was not checked
at all** — not "checked and passed." Treating the two paths as evaluated to
the same standard would be exactly the kind of self-flattering result this
critic exists to prevent.

Three secondary findings, all real, none softened:

- **The hybrid diffusion path (Attempt 18) is correctly and unambiguously
  rejected.** `svg/out/styled/frames` reproduces Attempt 18's own number
  almost exactly — **23.59x** measured here versus **23.6x** logged in
  ATTEMPTS.md — and hard-fails on `flicker_ratio` (≥8x bar) and soft-fails
  `warp_error` (0.14 vs a 0.06 pass bar). Verdict: NOT-AN-ANIMATION. This is
  the critic doing its job: the project's own log calls this attempt's style
  quality real ("genuine anime... at a quality the token budget could never
  buy") and its stability a failure, and the gate agrees with the second half
  regardless of the first.
- **RETRACTED, corrected by the crop-regime fix above:** this bullet
  previously reported `release/corpus-anime/corpus-anime.mp4` failing
  identity (worst-case 0.29) and attributed it to genuine palette drift
  across the reel's 6 concatenated clips. That explanation was itself wrong,
  and for the same reason the finding above (metric 2) documents: 0.29 was
  the crop-regime-mixing artefact, not identity drift of any kind — with the
  fix applied, the same clip scores `worst_similarity: 0.99995`. The clip is
  now `PROMISING`, capped only by `warp_error` (0.0711 vs a 0.06 bar), which
  was never in dispute. Left here, struck through in substance rather than
  deleted, because a wrong explanation that was quoted with confidence is
  exactly the kind of thing a paper should not inherit silently — see the
  crop-regime section under metric 2 for the real cause and the corrected
  number.
- **Pose fidelity is unmeasurable on exactly the outputs this project most
  needs to evaluate, and the first version of this metric hid that.** The
  builder lane's Attempt 19 measured that CMU OpenPose finds no person at all
  on our styled frames, our flat cel renders, or our rendered skeletons, and
  that YOLO11-pose (what this critic uses) detects the cel figure confidently
  in only 2 of 6 tested clips. This critic's own `svg/out/styled/frames` row
  is a live instance of the same failure: a confident detection in 2/8
  frames. The metric's first pass computed a PCK score from those 3 frames
  anyway and reported 77% — a number that looked like a pose-accuracy result
  and was actually a measurement of almost nothing. It has been corrected to
  report `UNMEASURABLE` with the detection rate and mean confidence attached,
  and is no longer hard-gated on stylised clips (see metric 4). The root
  cause, per the builder's measurement, is that this project's cel rig has an
  ear span 1.65x its shoulder width against ~0.52 for a real human — every
  pose estimator available here, like ControlNet in Attempt 18, was trained
  on human photographs and the rig's proportions sit outside that
  distribution by construction. **This project currently has no validated way
  to measure whether a stylised frame reproduces its intended pose**, and
  that is a real, unresolved measurement gap, not a metric that happens to
  read zero.
