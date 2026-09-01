---
title: >
  Two Routes to 2D Animation on an 8&nbsp;GB Consumer GPU:
  A Closed Route, an Open One, and What a Small Model Can and Cannot Learn to Write
---

# Two Routes to 2D Animation on an 8 GB Consumer GPU: A Closed Route, an Open One, and What a Small Model Can and Cannot Learn to Write

**Status:** working draft, current through Attempt 23 (`ATTEMPTS.md`, 2929
lines). All numbers in this draft are taken from `ATTEMPTS.md`,
`critic/README.md`, and `paper/RESEARCH2.md` as they stand at time of
writing, with an inline source note on every table; `paper/RESEARCH.md`
and `paper/RESEARCH2.md` are cited for related-work claims only. Every
superseded number from earlier evaluation runs has been replaced with the
corrected figure — see §5 for what changed and why.

## Abstract

We report twenty-three logged attempts (`ATTEMPTS.md`) to build a small,
locally-trainable system for 2D animation generation on a single 8 GB
consumer laptop GPU (RTX 4060 Laptop, thermally constrained to a
documented 50% duty cycle), together with the evaluation harness built to
judge the results honestly. The project's arc has two halves. The first
(Attempts 17–20) pursues styled character animation through Stable
Diffusion 1.5 + ControlNet + AnimateDiff, and closes that route by
measurement: even after correcting a genuine skeleton-topology bug and
isolating the deeper cause (a stylized character's proportions sit outside
a human-pose model's training distribution), and even after adding the
first mechanism in the project's history to reduce flicker at all
(36.2×→25.0×, a real 31% cut), every configuration tried remains a hard
failure (25.0× against an 8× bar) and the fix that helps flicker costs
pose obedience outright (an obeyed pose in 1 of 4 frames drops to 0 of 8).
The second half (Attempts 21–23) changes direction entirely: a CPU-only,
symbolic scene-script compositor produces the project's first
multi-character animation — three and four independently-acted characters
with background, props, occlusion and a persistent ball — and clears the
evaluation harness's temporal gates by roughly 25× the margin the
diffusion route's best configuration needed (warp error 0.0055 against a
0.06 pass bar, versus 0.135 for AnimateDiff). A 60M-parameter model then
learns to *write* these scripts from natural language, training in under
nine minutes on CPU with no GPU at any point: it parses and renders 100%
of held-out prompts, satisfies every stated fact in 90.7% of them
in-distribution, and generalizes to unseen phrasing perfectly at the
format level — but suffers a 65.7-point compositional generalization gap
on cast size specifically. A controlled probe traces this exactly: emitted
cast size equals clause count, not the stated numeral, in 15 of 15 trials,
and the cause is a confound the project introduced into its own training
data (clause count equalled cast size in 99.9% of synthesized examples).
Deconfounding the data and retraining — same model, same hyperparameters —
cuts the gap to 23.5 points at a measured 4-point in-distribution cost.

A third attempt on this half of the arc produces the paper's headline
finding: **the "poorly staged" scenes the project attributed to the model
are mostly a decoding artefact, not a model defect.** Every position
number reported for Attempts 21–22 used beam search, which returns the
approximately most-likely *sequence* — for a slot the prompt never
constrains, that is by definition the marginal mode. Decoding the *same
checkpoint* on the *same prompts* four ways and scoring each with a new
staging metric (independently calibrated: the project's two hand-written
scenes score 0.912 mean with zero overlap against eleven model-written
scenes at 0.352 mean) shows staging rising from 0.188 to 0.862 under
sampling instead of beam search on one checkpoint, and 0.333 to 0.845 on
another — same weights, same prompts, only the decoding argument changed.
Thirty scenes under beam search emit three distinct x-coordinates in
total; the same weights, sampled, emit fifty-five. A companion ablation
shows what would have shipped without this check: under beam search, a
deterministic post-decode spacing pass does essentially all the visible
work (staging 0.178→0.587), which reads as a successful engineering fix
and is, by the project's own later measurement, the wrong scientific
claim — sampling alone, no pass, reaches 0.879. We state this exactly:
*good engineering, bad science claim.* Sample-and-rerank then recovers
what sampling costs in structural validity (60.7%→94.0% in-distribution,
the highest this project has recorded) while leaving staging untouched,
and the resulting pipeline produces two deliverables that clear the
evaluation harness — one that stages above the worse of the two
hand-written reference clips for the first time in this project's
history, and one with six independently-animated characters, the
project's highest measured motion coverage. This is not reported as an
unqualified win: the same attempt's own control shows the model's
apparent obedience to stated spatial instructions (e.g. "leftmost") is
within noise of a mismatched-prompt baseline (+2.7 points in distribution,
a falsified prediction stated as such), sampling costs 13–20 points of
action-fidelity and layout obedience that reranking does not recover, and
cast size now saturates at nine rather than six — the wall relocated by
exactly the width of the training-range extension, not removed, which we
report as confirming a known general limitation of sequence-to-sequence
counting rather than a defect specific to this system.

We report three central findings, not one or two. First, extending
Attempts 17–20: style, motion, identity and pose behave as separable
defects that trade against each other on this hardware and model family
rather than compounding for free. Second, from Attempt 22: a small model
trained on synthesized data can look incapable of a whole class of
reasoning (here, counting) when the truer diagnosis is that its training
data never asked it to do that reasoning at all — checkable by a
controlled probe, rather than assumed. Third, from Attempt 23: an
apparent model defect can itself be a decoding artefact, checkable the
same way — by holding the weights and the prompts fixed and varying only
the decoding strategy — and this is the project's third corrected
diagnosis of this kind, after the COCO-18 control that overturned a
topology explanation (Attempt 19) and the clause-count probe that
overturned "the model cannot count" (Attempt 22); we treat the pattern of
finding, and then re-examining, its own apparent failures as a
methodological contribution in its own right. We also report, as a
contribution in its own right, four cases in which this project's own
evaluation harness or decoding pipeline produced a false or unmeasurable
verdict and was corrected in the open: a CLIP-based content-correctness
metric that cannot separate known-bad geometry from correct geometry in
this domain, an identity metric that silently mixed two measurement bases
mid-clip and cost one real clip its verdict twice, pose/person detectors
that read zero on every stick figure this project has produced, and beam
search collapsing an apparently-fine learned distribution into a single
unrepresentative point. Finally we answer the project's own standing
indictment — "the models never beat the script" — directly and without
spin. Attempt 22's model-written scene is thinner (2 characters, 2 props,
4 events) than the hand-written one (3 characters, 7 props, 8 events) and
both are USABLE; that comparison still holds. What has changed since is
narrower and precise: with a corrected decoder, a model-written scene now
stages *better* than the weaker of the two hand-written reference clips
(0.886 vs. 0.825) — composition, not direction, since the same attempt's
own control shows the model still does not reliably obey a stated
left/right instruction. "100% hand-written" is no longer true of the
project's best output category; "the model takes direction" is not yet
true of it either, and we say both.

## 1. Introduction

Most published work on video and animation generation assumes tens to
hundreds of gigabytes of accelerator memory and training budgets far
beyond a single consumer machine. This paper asks a narrower, more
answerable question: **what is actually reachable in 2D animation
generation on one 8 GB laptop GPU with no cloud budget, and what fails,
generalizes, or partially works, and why?** We treat the answer as worth
publishing in its own right — not as a preamble to a working system, but
as the deliverable.

The project (`2DVideoGen`) logs twenty-three attempts. The first four eras
(equation-based rendering, distillation, real pose/motion data, and a
symbolic vector format) establish the representational groundwork and are
summarized in §3. The paper's results then split into two connected
arcs, and its central claims are drawn from both.

**Arc one (Attempts 17–20) closes a route by measurement.** A hybrid of a
small language model for motion and Stable Diffusion 1.5 + ControlNet +
AnimateDiff for style is built, measured, diagnosed twice (once wrongly,
once correctly), and ultimately shown to fail its own evaluation gate
under every configuration tried — including the one configuration that
delivered this project's first genuine reduction in flicker. The specific
finding that survives this arc, and that we promote to the paper's first
central claim, is that flicker, identity and pose obedience behave as
three separable defects on this hardware and model family: fixing one
measurably costs another, rather than the three compounding toward a
shared solution as more mechanism is added.

**Arc two (Attempts 21–23) opens a different route and gets further on it
than anything earlier in the project.** Abandoning diffusion-rendered
style entirely, a CPU-only symbolic scene compositor produces this
project's first-ever multi-character animation — several independently
acted stick figures, a background, props, correct occlusion, and a
stateful ball — and clears the evaluation harness's temporal-stability
gates by a wide margin. A 60M-parameter language model is then trained,
in under nine minutes on a CPU, to write these scripts from natural
language. It generalizes at the format level essentially perfectly and
fails compositionally in one specific, diagnosable way: a spurious
correlation the project introduced into its own synthesized training data.
Isolating that confound with a controlled probe, fixing it at the data
level, and reporting the resulting trade honestly is the paper's second
central claim, and we hold it up as a general methodological point about
evaluating small models trained on synthesized data — a model can look
incapable of a whole class of reasoning when the truer fault is what its
training data ever asked it to learn.

**A third attempt on arc two (Attempt 23) then reopens the diagnosis of
arc two's own most-cited defect.** Attempt 22 named "staging" — characters
clustering at nearly the same position — as the single biggest remaining
defect and attributed it to unsupervised position slots. Attempt 23 builds
an independently-calibrated staging metric, decodes the *identical
checkpoint* on the *identical prompts* under four decoding strategies, and
finds the defect is mostly a property of beam search, not of what the
model had learned: staging rises from 0.188 to 0.862 on one checkpoint and
0.333 to 0.845 on another purely by sampling instead of beam-searching. A
companion ablation shows a deterministic spacing pass would otherwise have
been mistaken for the fix, when it is doing most of its visible work only
because beam search was suppressing an already-adequate learned
distribution — reported as *good engineering, bad science claim*. This is
the paper's third central finding, and it completes a pattern: this
project has now corrected its own diagnosis of an apparent model failure
three times (the COCO-18 topology control in Attempt 19, the clause-count
probe in Attempt 22, and the decoder ablation in Attempt 23), each time by
building a control that could have shown the opposite result and did not
assume the first plausible explanation. We treat that pattern itself as a
methodological contribution (§5).

Running through all three arcs is a fourth, cross-cutting contribution:
this project's own evaluation harness and decoding pipeline produced at
least four false or unmeasurable verdicts in the course of the work, each
caught, diagnosed, and corrected in the open rather than quietly fixed. We
treat that as a methodological contribution in its own right (§5), because
the failure mode — a metric or decoding default silently mismeasuring or
misrepresenting non-photographic, stylized, symbolic, or unsupervised-slot
content — is not specific to this project's tooling.

### Contributions

1. **A measured demonstration that flicker, identity and pose fidelity are
   three separate defects that trade against each other on this hardware
   and model family, not three sub-problems of one shared deficiency.**
   An explicit temporal-attention module (AnimateDiff) delivers this
   project's first genuine reduction in flicker — 36.2× → 25.0×, a 31% cut
   in the styled term against an unchanged guide — while simultaneously
   erasing pose obedience (an obeyed guide pose in 1 of 4 frames drops to
   0 of 8), and it still leaves flicker roughly 3× over the evaluation
   harness's own 8× bar. An identity anchor (IP-Adapter) that fixes the
   separate identity-preservation failure (worst-case similarity
   0.744 → 0.896) costs a further, small increase in flicker. Neither
   mechanism helps the other's number.
2. **A multi-character symbolic animation system that runs entirely on
   CPU and clears the project's own evaluation gates by a wide margin.**
   `scenescript.py` composes a cast, background, props and a timeline into
   a rendered video with correct occlusion and a persistent, stateful
   prop; the resulting clip passes warp error at 0.0055 against a 0.06
   pass bar — roughly 25× under the hard-fail line where the best
   diffusion configuration (Attempt 20) sat at 0.135 — and a script
   written after the compositor's code was frozen renders correctly on
   the first attempt with zero code changes.
3. **A 60M-parameter model that learns to write these scripts from
   natural language, training in under nine minutes on CPU with no GPU
   used at any point**, generalizing to unseen phrasing at the format
   level perfectly (100% parse, 100% render, both in- and
   out-of-distribution) while showing a specific, diagnosable
   compositional failure on cast size.
4. **A controlled probe that traces a 65.7-point generalization gap to a
   single spurious correlation the project put into its own training
   data**, and a methodological point we generalize beyond this project:
   a model that appears unable to perform a class of reasoning (counting,
   here) should be tested with a probe that holds the reasoning's inputs
   independent before concluding the model lacks the capability — in this
   case the model could produce any cast size, it had simply learned to
   read clause count (99.9% correlated with cast size in the original
   synthesized data) instead of the stated numeral. Deconfounding the data
   and retraining, with no other change, cut the gap to 23.5 points at a
   measured, honestly reported 4-point in-distribution cost.
5. **A decoding-strategy ablation that reattributes the project's own
   most-cited model defect.** Attempt 22 attributed poor staging
   (characters clustering near-identically) to the model. Decoding the
   *same checkpoint* on the *same prompts* under beam search versus
   sampling shows staging rising from 0.188 to 0.862 on one checkpoint and
   0.333 to 0.845 on another — an artefact of beam search reporting the
   mode of an already-adequate learned distribution, not a defect in what
   the model had learned. A companion four-cell ablation shows a
   deterministic post-decode spacing pass, which would otherwise have been
   reported as the fix, is doing almost all of its visible work only
   because beam search was suppressing the same distribution sampling
   reveals (staging 0.178→0.587 via the pass versus 0.178→0.879 via
   decoding alone) — stated exactly as the project states it: *good
   engineering, bad science claim.* This is the paper's third corrected
   diagnosis of an apparent model failure (after the COCO-18 topology
   control, contribution 7, and the clause-count probe, contribution 4),
   and we treat the pattern of self-correction as a contribution distinct
   from any one of the three individual corrections.
6. **A direct, unspun answer to the project's own standing indictment**
   ("the models never beat the script"): Attempt 22's model-written scene
   remains measurably thinner than the hand-written one (2 characters, 2
   props, 4 events, against 3, 7 and 8) though both clear the evaluation
   harness's USABLE bar; with the decoding fix, a model-written scene now
   *stages* above the weaker of the two hand-written reference clips
   (0.886 vs. 0.825) for the first time, while the same attempt's own
   control shows the model still does not reliably obey a stated
   left/right instruction — composition improved measurably, direction
   did not. "100% hand-written" is no longer true of the project's best
   output category; the model does not yet take direction, and we say
   both plainly.
7. **Four corrected evaluation or decoding failures, reported as a
   methodological contribution in their own right**: a CLIP-embedding
   content-correctness metric that cannot separate known-bad geometry from
   correct geometry in this domain; an identity-preservation metric that
   silently compared crops taken under two different measurement regimes
   within one clip, costing one real clip its verdict, discovered a second
   time retroactively once the first fix was applied; pose/person
   detectors (CMU OpenPose, YOLO11-pose) that read zero or near-zero on
   every stick figure this project has produced, hand-written or
   model-generated, independent of cast size; and beam search collapsing
   an adequate learned position distribution into a single
   unrepresentative point, discovered only because a control decoded the
   same weights a second way. We generalize this: both metrics built
   against photographic training data and decoding defaults tuned for
   fluent text need independent verification, not assumption, before being
   trusted on stylized, symbolic, or unsupervised-slot generation.
8. **A corrected two-stage diagnosis of a pose-conditioning failure in the
   diffusion route.** A ControlNet-OpenPose hybrid ignored an exact,
   hand-verified pose guide; the first hypothesis (wrong keypoint
   topology) was fixed completely and the pose was *still* ignored. A
   controlled ablation against a genuine photo annotation and a hand-built
   canonical skeleton isolated the real cause: the stylized character's
   metric proportions, not its topology, put the guide outside the
   conditioning model's training distribution — a finding we state as
   general to human-pose conditioning of non-human-proportioned
   characters, not specific to this implementation.
9. **The first stacked VRAM measurement of a ControlNet + AnimateDiff
   motion-module pipeline on an 8 GB consumer card**, closing an open
   question the project's own literature survey (`paper/RESEARCH.md`)
   could only mark unverified: 4.41 GiB resident weights, 6.15 GiB peak
   reserved at 512×8, fitting without offload; only 512×16 requires it.
10. **A full record across four earlier representational eras** (equations,
    distilled nets, real pose data, symbolic vector graphics), each capped
    for a different, measured reason, kept so the dead ends are not
    silently re-attempted.

We do not claim a fully solved system. Arc one is a closed route, stated
as closed. Arc two produces the project's best output to date — including
a scene that stages better than one of the project's two hand-written
reference clips — and still falls short of a well-staged, prompt-
controllable, natural-language-driven animation system: the model does not
obey stated spatial instructions above the base rate, cast size saturates
at nine, and sampling's staging gain costs 13–20 points of action fidelity
that reranking does not recover. §13 says plainly what remains, including
what this paper does not evaluate at all (no user study, no comparison
against a competing system).

## 2. Related Work

We draw only on citations that appear in this project's own literature
surveys (`paper/RESEARCH.md`, covering Attempts 17–20's diffusion route,
and `paper/RESEARCH2.md`, written in parallel with Attempt 23 to cover the
staging/counting/constrained-decoding literature for the scene-script
route); we do not introduce new references beyond what these two surveys
cite. Attempt 21 is pure systems work internal to the project and cites no
external literature in `ATTEMPTS.md`; we do not invent citations for it.

**Temporal consistency for per-frame diffusion.** AnimateDiff (Guo et al.,
arXiv:2307.04725) inserts a motion module, trained separately, into a
frozen Stable Diffusion 1.5 UNet, and transfers to any SD1.5 checkpoint
without retraining; community-reported VRAM for 512×512×16-frame generation
ranges roughly 8–21 GB depending on settings, i.e. workable on an 8 GB card
only at reduced resolution/frame count. AnimateDiff-Lightning (Lin & Yang,
ByteDance, arXiv:2403.12706) distills the same motion module to 1–8-step
inference for roughly 10× speedup; its VRAM footprint relative to base
AnimateDiff is not separately quantified in the sources surveyed.
Training-free alternatives include Text2Video-Zero (Khachatryan et al.,
ICCV 2023, arXiv:2303.13439), which anchors appearance via cross-frame
attention but is reported at a 12 GB minimum — above this project's budget
as published — and FRESCO (Yang et al., CVPR 2024), a zero-shot spatial-
and temporal-correspondence method superseding Rerender-A-Video (Yang et
al., SIGGRAPH Asia 2023) by the same authors' own account. TokenFlow
(Geyer et al., arXiv:2307.10373) propagates diffusion features via
inter-frame correspondence rather than warping pixels, but is reported at
5091 s for 240 frames on a 24 GB card (PipeFlow, arXiv:2512.24026, which is
9.6× faster at the same task) — a wall-clock cost that rules it out for
iterative work on a laptop already duty-cycled for thermal reasons.
ControlVideo's full cross-frame attention is the most expensive of this
group by construction (quadratic in frame count) and was not adopted
without local measurement. Newer sparse/lightweight temporal-attention
methods (Sparse Forcing, arXiv:2604.21221; LiteAttention, arXiv:2511.11062;
Sparse VideoGen, ICML/NeurIPS 2025; HyperVAttention, arXiv:2607.03012)
target diffusion-transformer video backbones rather than the SD1.5+UNet
stack used here and are not shown composing with ControlNet-SD1.5 in the
sources surveyed.

**Identity preservation.** AnimateAnyone and MagicAnimate (cited via
Animate-X, arXiv:2410.10306, and a MagicAnimate ResearchGate record) add a
ReferenceNet — a duplicated UNet branch injecting appearance features from
one fixed reference image — plus a lightweight Pose Guider, addressing
exactly the "every frame is a different character" failure mode reported
in this project's own Attempt 18 (§7.1). MagicAnimate is reported to still
show fine-grained inconsistency and background drift under its own
guidance signal. IP-Adapter / IP-Adapter-FaceID is a lighter-weight
alternative (frozen base, trained cross-attention adapter only), reported
at roughly 6–8 GB for SD1.5 at 512×512 — the only concrete figure found
that sits inside this project's budget without qualification, though it is
a weaker identity lock than a full ReferenceNet and its full-body/clothing
consistency is untested in the sources surveyed.

**Script-to-multi-shot generation.** VideoDirectorGPT (Lin, Zala, Cho,
Bansal, COLM 2024, arXiv:2309.15091) uses a large hosted LLM purely as a
planner, expanding one prompt into a structured video plan (per-scene
descriptions, entity layouts, cross-scene consistency groupings) that
drives a separate grounded video generator — architecturally the closest
published analogue to this project's own Attempt 22 (a small local model
planning a multi-entity scene for a separate, deterministic renderer),
though built on models far larger than fit this project's card and
without this project's own compositional-generalization diagnosis. ViMax
(github.com/hkuds/vimax) is a broader, more recent agentic script-to-video
pipeline with no VRAM/scale figures found in the survey.

**Small video diffusion backbones.** Wan2.1-T2V-1.3B is reported at
roughly 8.19 GB inference VRAM at FP16 (one source; another states 11 GB
for the Diffusers pipeline specifically), with GGUF-quantized builds down
to 4–6 GB at 480p; LoRA training, however, was only demonstrated on a
24 GB card in the sources surveyed, and whether it fits under 8–12 GB is
explicitly unverified. CogVideoX-2B is reported at roughly 8 GB at FP8 for
480p/49-frame/8fps inference.

**Layered/vector animation generation.** AniClipart (Wu et al., IJCV 2024,
arXiv:2404.12347) animates a static clipart image via Bézier curves over
keypoints with a text-to-video diffusion prior (Score Distillation
Sampling) as the motion signal and an explicit rigidity constraint;
Differentiable Motion Trajectory (Zhu et al., CGF 2025/2026) does the
analogous thing for stroke-based sketches. Neither is shown handling
multiple independent characters plus a background — the survey's own
reading is that multi-character vector-animation composition is an open
gap in the published literature, which is the gap Attempt 21 (§9) closes
for this project's own material, by a route (deterministic symbolic
compositing, not diffusion or SDS) not covered in this survey.

**Datasets.** Sakuga-42M (arXiv:2405.07425) is the largest hand-drawn
animation corpus found (42M keyframes, ~1.4M clips, auto-generated
captions, CC BY-NC-SA 4.0). AnimeRun (arXiv:2211.05709) and CreativeFlow+
supply dense correspondence/optical-flow labels at far smaller scale.
HumanML3D and AMASS supply captioned 3D human motion, not 2D or
stick-figure native; Motion-X/Motion-X++ (arXiv:2501.05098) is reported
larger than HumanML3D with subsets already reformatted to its schema. This
project's own corpus, LottieAnimation-660K (already CC-BY-NC-SA-4.0
licensed and in active use), is confirmed by the survey as the right
choice among these for vector, captioned data with free shape
correspondence, with the known limitation (this project's own Attempt 16)
that it is motion-graphics iconography containing no characters.

**Small text-to-motion models.** MoMask (44.85M params), Light-T2M
(4.48M params, arXiv:2412.11193), and TriC-Motion (13.86M params,
arXiv:2602.08462) are two to three orders of magnitude smaller than even
this project's smallest trained language model (Qwen3-0.6B), suggesting
VRAM is unlikely to be the binding constraint for this class of motion
model; MotionGPT (OpenMotionLab, NeurIPS 2023) is excluded from this size
class because it is built on an LLM backbone.

**Evaluation.** FVD, and its content-bias correction (content-debiased
FVD; JEDi), are the standard distributional video-quality metrics but
require both a pretrained I3D-family feature network and, per the
survey's own reading, "hundreds to thousands of clips per condition" for a
stable estimate — a sample size this project's few dozen clips per
condition does not meet, and the reason FVD is not attempted here (§4).
CLIPSIM (rooted in CLIPScore, Hessel et al., EMNLP 2021, first applied to
video in GODIVA) and its newer alternative VQAScore/t2v_metrics measure
text-frame alignment; warping error (Farneback- or RAFT-based optical
flow, warp-then-diff) is the standard blind temporal-consistency metric
and is exactly the quantity this project's own Attempt 18 first
approximated as a raw inter-frame-diff ratio. Fréchet Video Motion
Distance (FVMD) isolates motion consistency specifically; EvalCrafter
(arXiv:2310.11440) is a broader benchmarking suite correlated against
human judgment. This project's evaluation harness (§4) is built directly
against this survey rather than in ignorance of it, and documents where it
diverges and why.

**Script-to-multi-character animation at small scale (RESEARCH2 §5).**
This project's own second survey searched specifically for a published
system doing prompt-to-multi-character-animation at this project's scale
(sub-100M parameters, CPU-trainable) and found none: AniMaker (SIGGRAPH
Asia 2025, arXiv:2506.10540) and AniME (arXiv:2508.18781), the most
current systems found in this space, are both multi-agent pipelines built
on large hosted LLMs and large video-diffusion backbones, not the
"one small model, one CPU forward pass" class this project occupies; the
2026 commercial storyboard-generator landscape surveyed (M Studio, DomoAI,
TapVid, AnimateAI, Novi AI) is entirely large-hosted-model product
surfaces. **As far as this survey found**, the prompt → symbolic scene
script → deterministic renderer pipeline this project has built, entirely
on CPU, is the smallest-scale system in this specific niche in the
literature to date. We state this as the survey states it, hedge intact —
absence of a counter-example in one search pass is evidence, not proof.

**Compositional generalization and count extrapolation (RESEARCH2 §3).**
The published record on fixing systematic generalization failures like
this project's cast-size wall (§10.4–10.5, §11.10) is dominated by
data-level, not architectural, interventions: GECA (Andreas, ACL 2020,
aclanthology.org/2020.acl-main.676) took a seq2seq model on SCAN's hardest
"jump" split from unable to generalize at all to succeeding most of the
time via rule-based recombination of fragments seen in similar contexts,
and COGS/CFQ remained largely unsolved by naive scaling for years
afterward per the surveys RESEARCH2 checked. "When Can Transformers Count
to n?" (Yehudai, Kaplan, Ghandeharioun, Geva, Globerson, arXiv:2407.15160)
gives a falsifiable, checkable prediction this project used directly
(§11.4): exact counting is reliably learnable when embedding dimension
exceeds the relevant vocabulary size, so a `d_model` of 512 against closed
vocabularies of at most ten words rules out an embedding-width bottleneck
as the cause of this project's own counting/extrapolation limits. A
separate memory-augmented-counting source RESEARCH2 checked (exact
citation unresolved in that survey, reported there as unverified) found
plain transformer baselines generally "incapable of extrapolation beyond
the training range" on counting tasks — independent corroboration that
this project's own cast-size wall is, per RESEARCH2's own framing, "a
documented, general property of plain transformer/seq2seq architectures
... not something specific to t5-small or to this project's data," and we
report it as confirming a known limitation rather than as a novel defect
(§11.10).

**Constrained decoding (RESEARCH2 §4).** Grammar/automata-guided
constrained decoding is a well-studied, sound technique class — ABS
(2025, arXiv:2506.09701) generalizes constrained beam search to any
constraint compilable to a finite automaton, which the project's own
duplicate-colour constraint (state = subset of ten colours used so far)
is trivially an instance of; Outlines and Guidance integrate as logits
processors directly against HuggingFace `transformers` models, this
project's own stack, and RESEARCH2 recommended hand-writing the one
constraint needed first and holding those libraries as the fallback once
more constraints accumulate — the recommendation this project followed
(§11.4) and later found reason to reconsider once sampling replaced beam
search and began emitting tokens outside the DSL vocabulary entirely
(§11.9). The DETR-style Hungarian set-decoder (Carion et al.,
ECCV 2020, github.com/facebookresearch/detr) is, per RESEARCH2 §4, the
one technique surveyed that structurally addresses this project's staging
and duplicate-colour problems together — N cast slots predicted with
mutual attention make a repeated colour visible to the model during
generation in a way strict left-to-right decoding cannot, and a repulsion
term over the predicted positions attacks staging in the same head — but
it is the only surveyed item requiring genuine architecture work (a new
decoder head, a new order-invariant matching loss) against this project's
"minutes of CPU, no architecture change" retraining budget, and it was
deliberately not built in Attempt 23 for that reason (§11.4). We report
it as future work in §13, with the reasoning intact rather than as an
unexplained gap.

## 3. System Description

### 3.1 Four representational eras (`ATTEMPTS.md`)

**Era 1 — equations.** `geovid.py` renders scenes described as parametric/
explicit/polar/field numpy equation layers. A `t5-small` (60M param) model
maps prompts to a compact DSL spec, trained on a synthesized dataset (no
labeled prompt→scene data exists) — 100% per-slot validation accuracy, but
capped to the 9 archetypes its own grammar defines. A hand-scripted gait
engine (`stickman.py`) produces the best-looking output of the project at
that stage, entirely by hand, with no learned component — stated in the
log as the project's own indictment of Era 1–2: the models never beat the
script. (This is the standing indictment §11 answers.)

**Era 2 — distillation.** Two small nets (`PoseNet`, 81k params, <0.5°/
joint error; `ImgPoseNet`, 70k params, ~6°/joint error) are trained to
imitate the hand-written engine as teacher. A distilled student cannot
exceed its teacher by construction; in the shipped web app, the neural
path is opt-in (`EQV_NEURAL=1`) because the hand-scripted engine renders
more reliably.

**Era 3 — real data.** YOLO11-pose over 5000 COCO val2017 images supplies
5051 real pose labels, genuinely useful as *perception* but not generation
— it does not produce new animation. `RealMotionNet`, trained on six
Muybridge motion studies (3746 frames), loops smoothly but blurs from
cross-clip averaging; `MotionAR` (GRU, autoregressive, val loss 0.0006) is
sharper but non-looping, and is the shipped default for learned motion.

**Era 4 — AniSVG.** After a measured failure of per-frame raster tracing
of public-domain cartoons (§3.2), the project adopts a symbolic vector
format that declares a cast of shapes once per shot and encodes every
frame as integer-delta edits to that cast — translate, rotate, scale,
opacity, per-vertex morph, hide/show. Because shape *k* is the same shape
at every timestep by construction, this format gets shape correspondence
for free where per-frame raster tracing could not, and is what makes
low-token-cost delta encoding possible at all. Era 4 is also where the
project's two arcs (§1) diverge: AniSVG-trained language models pursued
styled character generation via a diffusion hybrid (§7–8); the
scene-script system of Attempts 21–23 (§9–11) instead composes the
project's original Era 1 rig (`stickman.py`) directly, without a learned
appearance model, and reaches multi-character animation first.

### 3.2 Why raster tracing was rejected (Attempt 11)

Public-domain Fleischer cartoon frames were traced to per-frame SVG with a
full built pipeline (crop/posterize/palette-fit/trace/rasterize/encode).
Measured on 144 real traced frames: raw SVG costs 6223 tok/frame; the best
compact encoding reaches 3794 tok/frame (1.64× compression). Delta encoding
against the previous frame bought almost nothing (1.61×→1.64×) because
independently-traced frames have no shape correspondence to diff against —
frame-to-frame path correspondence in raster-traced animation is an open
research problem. A background/foreground split made things *worse*
(0.49×) because these cartoons pan and truck constantly (35–80% of pixels
move per shot) and the ragged motion-mask boundary adds more path
complexity than it saves. At ~4000 tok/frame, 15 fps costs ~60,000
tokens/second of video — far beyond what a small model (~2000 coherent
tokens generatable, i.e. ~40–60 tok/frame for a short clip) can produce.
This measured failure is what motivated AniSVG: what full-scene raster
tracing cannot supply — persistent shape identity across frames — is
exactly what a declarative cast-of-shapes format supplies for free.

### 3.3 AniSVG format and corpus

```
H <w> <h> <fps> <nframes>                 header
P <hex> ...                               palette, index = position
S <id> <pal> <sw> <x0> <y0> <dx dy>...    cast member; sw = stroke width x10
@ <frame> <op> ...                        per-frame edits; unlisted shapes hold
```

The corpus source is `LottieGPT/LottieAnimation-660K` (660k text-captioned
Lottie/Bodymovin JSON files, CC-BY-NC-SA-4.0). Sampling a Lottie shape tree
at successive times gives shape correspondence for free — the property
raster tracing could not supply. Only the metadata shard (2.96 GB of the
repository's 132 GB; the video-preview shard is never touched) is needed.
Filtering (cast size 2–20, no unsupported Lottie feature, motion in at
least half the frames, not static) retains 20% of a 1500-record pilot,
projecting to roughly 130k usable clips from 660k. Final corpus: **68/68
shards, 91,387 clips, 96 MB gzipped**; packed to **78,971 train/val clips,
198.3M tokens**, with over-long clips trimmed rather than dropped.

Measured cost on real corpus data (400 clips, Qwen3 tokenizer): median
71 tok/frame, 1768 tok/clip. Compared against Attempt 11's raster tracing:
~19× cheaper on a matched 36-shape, 48-frame Lottie (~4000 tok/frame →
~213 tok/frame).

### 3.4 The training bottleneck was VRAM from the loss, not the weights

`Qwen3ForCausalLM` upcasts logits to fp32 for cross-entropy against a
151,936-token vocabulary; a single ~1850-token sample allocates ~1.05 GB
for the logit tensor alone, plus the same again for its gradient. Scoring
the sequence in chunks (`chunked_ce`) cut peak VRAM from 4.92 GB to
3.16 GB at 0.6B for a 7% speed cost, with builtin, manual, and chunked
loss implementations agreeing to 0.8340 on a fixed batch.

| base | quant | peak VRAM | s/clip |
|---|---|---|---|
| Qwen3-0.6B | bf16 | 3.16 GB | 0.76 |
| Qwen3-1.7B | bf16 | 5.30 GB | 1.32 |
| Qwen3-4B | NF4 | 6.28 GB | 4.00 |
| Qwen3-8B | NF4 + Soup layer streaming | ~3.3 GB | see below |

*(Source: `ATTEMPTS.md` Attempt 14 / `svg/README.md`.)* Qwen3-8B was
trained via `Soup` layer streaming, reported by its authors at 119.6 tok/s
on a 4 GB card, which puts a full 198.3M-token epoch at roughly 19 days on
this hardware — the real run therefore trained on a rate-measured subset
rather than a full epoch.

### 3.5 Two trained AniSVG generators, and what scale did not fix (Attempt 15)

| | v1 Qwen3-8B (streamed, 1024 ctx) | v2 Qwen3-1.7B (resident, 4096 ctx) |
|---|---|---|
| clips seen | 28,034 | 77,397 |
| final train loss | 0.4349 | **0.2709** |
| valid generations | 94% | **100%** |
| recognisable geometry | no | **still no** |

*(Source: `ATTEMPTS.md` Attempt 15.)* v2 saw roughly 7× the animation
signal of v1. The measured finding: palette and op structure are learned;
vertex sequences (long runs of exact integers where one wrong digit
deforms the shape) are not — this is **coordinate regression through
next-token prediction**, not a model-size or corpus-size problem, and it
is what motivates splitting motion (language model) from appearance
(diffusion) in the arc-one hybrid (§3.7, §7–8).

### 3.6 A generated corpus, since no real anime vector-character corpus exists (Attempt 16)

LottieAnimation-660K is motion-graphics iconography with no characters; a
search for real anime vector-animation data at usable scale found none.
The project therefore generates its own character corpus from
`stickman.StickFigure`: `chars.py` (20,000 line-art stick-figure clips) and
`anime_chars.py` (20,000 cel-style clips: oversized head, hair silhouette,
large low-set eyes, tapered limbs). The cel figure's fixed 16-part schema
in fixed paint order is later what lets a pose skeleton be recovered from
text a model wrote (§7.1).

### 3.7 The diffusion hybrid: motion from a language model, style from diffusion (Attempt 17)

The project's stated reasoning: the token budget is the ceiling — a small
model affords ~40–80 tokens/frame, motion is cheap in tokens and
appearance is not. The split: the language model generates motion
(AniSVG deltas), a diffusion model renders appearance, conditioned on
exact guides (`guides.py`) rather than a per-frame detector estimate.
Feasibility was measured: Wan 1.3B and AnimateDiff-Lightning run at
inference on 8 GB; LTX-2 LoRA *training* needs 32–80 GB, ruling out
training a video backbone on this hardware while running a pretrained one
does not. Results and the eventual closure of this route are §7–8.

## 4. Evaluation Harness (`critic/`)

The project built an eight-metric evaluation harness rather than trusting
loss curves or visual inspection, explicitly cross-referenced against
`paper/RESEARCH.md`'s survey of published metrics.

**Table 1 — this critic's metrics vs. the published metrics they stand in for.**

| this critic | published metric (§2) | why substituted |
|---|---|---|
| `temporal_stability.warp_error` | warping error (optical-flow warp-then-diff) | same definition, Farneback flow (CPU) instead of RAFT (GPU) — an extra deep net for the judge was ruled out on this hardware |
| `temporal_stability.flicker_ratio` | the raw inter-frame-diff ratio already computed in Attempt 18 | reproduced verbatim as a metric rather than reinvented |
| — | FVD / content-debiased FVD / JEDi | needs a pretrained I3D-family net and "hundreds to thousands of clips per condition" for a stable estimate; this project has at most a few dozen per condition |
| — | CLIPSIM / VQAScore | implemented (metric 6) but shown to be a null result in this domain (§4.3) |
| — | FVMD | same sample-size problem as FVD |
| `motion_presence`, `identity_preservation`, `pose_fidelity`, `scene_complexity`, `scene_presence_audit` | not in the published set | added because none of FVD/CLIPSIM/warping-error catch this project's specific known failure modes (a frozen clip scoring perfectly, an identity that reorganizes mid-clip, a render that ignores its pose guide, a multi-character scene a photo-trained detector cannot see at all) |

Metrics 1–5 (temporal stability, identity preservation via HSV histogram
cosine similarity, motion presence via Farneback flow coverage/energy,
pose fidelity via YOLO11-pose PCK, scene complexity via person-count
stability) each carry explicit pass/hard-fail thresholds justified in
`critic/README.md`. Metric 8, `scene_presence_audit` (§4.6), was added in
response to Attempt 21's own finding that `scene_complexity` cannot see
this project's material at all. We restate here only the results that
bear on the paper's central claims, and flag every number that has been
corrected since an earlier evaluation pass.

### 4.1 Motion-floor calibration (sanity check)

A synthetic clip built by repeating one real frame 30× verifies the
harness does not reward a frozen output: `warp_error` 0.00001, identity
1.00, but `motion_coverage` 0.0025% — correctly gated
`NOT-AN-ANIMATION: ... stable because nothing moved` against a 0.20%
floor. *(Source: `critic/README.md` §"Motion presence".)*

### 4.2 The baseline table (22 clips, current)

The following is reproduced verbatim from `critic/README.md`'s current
22-clip baseline table — **every `id worst` value below is post-fix**
for the crop-regime bug described in §5.2, and where a prior version of
this table (an earlier evaluation pass) reported a different number, that
earlier number is superseded and not used anywhere else in this paper.

| clip | n | warp_error | id worst (`crop_regime`) | motion_cov | CLIPSIM | structural | verdict |
|---|--:|--:|--:|--:|--:|---|---|
| `control_wave.mp4` (positive control) | 30 | 0.0011 | 1.00 (fallback) | 1.01% | 0.329 | N/A | USABLE |
| `control_kick.mp4` (positive control) | 30 | 0.0048 | 1.00 (fallback) | 3.29% | 0.233 | N/A | USABLE |
| `out/a21_park.mp4` (Attempt 21, 3 characters) | 200 | 0.0055 | 1.00 (fallback) | 5.32% | 0.311 | N/A | **USABLE** |
| `out/a21_street.mp4` (Attempt 21, 4 characters, fresh script) | 200 | 0.0065 | **1.00** (fallback) | 4.89% | 0.336 | N/A | **USABLE** ⟵ *flipped by the crop-regime fix, was PROMISING at 0.4368* |
| `corpus-anime.mp4` (reel) | 40 | 0.0711 | **1.00** (fallback) | 26.9% | N/A | N/A | **PROMISING** ⟵ *flipped by the crop-regime fix, was NOT-AN-ANIMATION at 0.29* |
| ground truth `10887279` (isolated) | 18 | 0.0088 | 0.84 (no YOLO) | 10.4% | 0.337 | `geometry_verified=True` | USABLE |
| `svg/out/gen17/g002` (same caption, model gen.) | 28 | 0.0089 | 0.99 (fallback) | 21.1% | 0.306 | `geometry_verified=False` | **NOT-AN-ANIMATION** ⟵ *flipped by structural metric* |
| `svg/out/styled/frames` (Attempt 18 hybrid) | 8 | **0.1408** | 0.63 (fallback) | 60.4% | 0.283 | N/A — no AniSVG source | **NOT-AN-ANIMATION** |
| `critic/tmp/static_test.mp4` (negative control) | 30 | 0.00001 | 1.00 (fallback) | 0.003% | N/A | N/A | **NOT-AN-ANIMATION** |

*(Source: `critic/README.md`, "Baseline table," full 22-row table; rows
selected here for paper relevance.)*

### 4.3 CLIPSIM cannot separate content-correct from content-wrong generations (null result)

The calibration test: score the hand-written positive controls
(known-correct content) and a known-bad generated clip (`g002`, captioned
"two four-pointed star-shaped figures...", which draws two blobs — neither
a star, per Attempt 15) with CLIP ViT-B/32 (`openai/clip-vit-base-patch32`,
CPU-only), against their true captions.

| clip | caption used | CLIPSIM mean |
|---|---|--:|
| `control_wave.mp4` (correct content) | its own correct caption | **0.329** |
| `control_kick.mp4` (correct content) | its own correct caption | **0.233** |
| `g002` (known-bad: blobs, not stars) | its own correct caption | **0.306** |
| `control_kick.mp4` | wrong caption (wave's) | 0.235 |
| `control_wave.mp4` | unrelated caption ("a car...") | 0.163 |

*(Source: `critic/README.md` §"Content correctness / prompt alignment
(CLIPSIM)".)* The known-bad blob clip scores *higher* (0.306) than the
known-good kick control (0.233): no single threshold passes both controls
and fails the blobs. `CONTENT_CORRECTNESS_FLOOR = 0.18` is kept only as a
"completely different concept" tripwire, not as the content-correctness
check it was built to be; over the full baseline table, adding CLIPSIM
changed **zero** verdicts.

### 4.4 Structural correctness separates the same pair by ~27× (positive result)

Because a generated AniSVG clip's output *is* its source — a declared cast
of shapes in a fixed paint order — content correctness can, on this one
path, be checked exactly. The check (`star_score`) measures the discrete
Fourier coefficient of a shape's centroid-normalized radius sequence at
the frequency the caption implies.

| clip | shapes scoring ≥0.05 (star-like) | shape count | `geometry_verified` |
|---|---|--:|---|
| ground truth (`10887279`) | 3, at 0.135 / 0.154 / 0.140 | 11 | **True** |
| `g002` (blobs) | 0, all four at 0.0048–0.0052 | 4 | **False** |

*(Source: `critic/README.md` §"Structural correctness (metric 7)".)* The
weakest genuine star shape (0.135) and the strongest blob shape (0.0052)
differ by roughly **27×**, with no tuning pressure on the 0.05 threshold.
Adding this metric changes exactly one verdict in the table: `g002` flips
from USABLE (passing five pixel metrics and CLIPSIM) to
**NOT-AN-ANIMATION**, correctly.

### 4.5 The evaluation asymmetry, stated explicitly

Metrics 6 and 7, read together, are one of the harness's central findings:
one is a null result (CLIP embedding similarity cannot separate the
populations in this domain); the other separates the same pair cleanly,
at zero GPU cost, because it parses declared source rather than inferring
content from pixels. This is **not a general solution to content
correctness** — it is specific to the one output path where generation and
source are the same text. The diffusion/hybrid path (`svg/out/styled`) has
no declarative source to check against; it is rejected on temporal
grounds with total confidence, but this harness would have no way to
verify a future, temporally-fixed version of that pipeline actually
painted the character its prompt asked for. Attempt 21's own composed
scenes carry the same asymmetry in a new form (§9.7): a scene script is
itself a declarative source and could in principle be checked exactly the
same way AniSVG is, but nothing does that yet.

### 4.6 A detector-free presence audit, added because `scene_complexity` cannot see this project's own material (metric 8)

`yolo11n-pose.pt`, the person detector `scene_complexity` (metric 5) is
built on, finds **no person at all** in either of this project's own
hand-written single-figure controls (`control_wave`, `control_kick`,
`modal_count: 0` on both) and the same is true of hand-scripted scenes
containing three or four visible, correctly-drawn characters
(`out/a21_park.mp4`, `out/a21_street.mp4`). *(Source: `critic/README.md`
§"Scene complexity".)* A photo-trained detector does not read flat
stick-figure line art, at any cast size — reporting `modal_count: 0` in
that situation is indistinguishable from "this clip genuinely has no
characters," which is false every time it has happened in this project.
`scene_complexity` now applies the same `MEASURABLE_DETECTION_RATE = 0.50`
bar `pose_fidelity` already used, and reports `UNMEASURABLE` with the
detection rate attached rather than a false cast count of zero, below that
bar; it remains, as before, never hard-gated.

The working replacement, `scene_presence_audit` (metric 8), verifies
presence rather than detecting it: given the declared colour of each cast
member (supplied by the caller, not inferred), it checks colour distance
under a tolerance and connected-component count per frame — the same
technique `scenescript.py --audit` (§9) uses, reimplemented independently
in `critic/` so the two tools stay comparable without a code dependency.

| clip | characters | all visible (this tool, 90-frame sample) | all visible (`scenescript.py --audit`, full clip) |
|---|--:|---|---|
| `a21_park.mp4` | 3 | 66/90 (73%) | 75/75 sampled (100%) |
| `a21_street.mp4` | 4 | 61/90 (68%) | 84/84 sampled (100%) |

*(Source: `critic/README.md` §"Scene presence audit".)* The two tools
disagree on the exact rate because they sample differently, not because
either mistracks presence; both agree that every declared character is
visible in the large majority of sampled frames — the thing
`scene_complexity` cannot see on this art at all. This metric is,
explicitly, a verification tool rather than a detector: it has nothing to
measure without being told what to look for, so it cannot replace
`scene_complexity` on arbitrary or unlabelled input, and it is never
gated.

## 5. Evaluation Methodology as a Contribution

This project's evaluation harness and decoding pipeline produced four
distinct false or unmeasurable verdicts in the course of the work. We
report all four as a contribution in their own right, because the
underlying failure mode — a metric trained or calibrated on photographic
content, or a decoding default tuned for fluent free text, silently
mismeasuring or misrepresenting stylized, symbolic, procedurally
generated, or unsupervised-slot content — is not specific to this
project's tooling, and we believe it generalizes to anyone evaluating
generative 2D animation or any generative model with unsupervised output
slots.

### 5.1 Case 1 — CLIPSIM cannot separate correct from incorrect content in this domain (§4.3)

Already described above: a known-wrong "blob" generation (0.306) scores
*higher* than a correctly-drawn hand-written control (0.233), and giving a
correct clip the wrong caption barely moves its score (0.233→0.235). CLIP
ViT-B/32 at CPU budget, with no domain fine-tuning, provably cannot
distinguish "kicking" from "waving," let alone a correct star from a
blob, in this project's stylized 2D domain. This is reported as a null
result rather than silently dropped or its threshold tuned until it
looked like it worked.

### 5.2 Case 2 — a crop-regime artefact that cost one real clip its verdict twice

`identity_preservation` (metric 2) crops "the character" per frame — a
YOLO detection box when one fires, a 70% centre-crop fallback otherwise —
and previously chose between the two **independently, frame by frame**.
On `out/a21_street.mp4` (four hand-scripted characters, Attempt 21), YOLO
fired on exactly 1 of 24 sampled frames (a 48×135 box around one small
figure); the other 23 frames used the identical centre-crop box. The
metric then compared that one geometrically unrelated crop against the
rest and reported `worst_similarity: 0.4368`, under the 0.45 pass bar,
capping the clip at PROMISING. **The clip never changed. Only the
measurement basis changed, once, mid-clip**, and the metric reported that
as an identity change; an independent colour-based audit of the same clip
found all 4 characters visible in 84 of 84 sampled frames. Forcing the
fallback crop for every frame gives **0.9972** on the same clip. *(Source:
`critic/README.md` §"Bug found and fixed: mixed crop regimes".)*

**The fix** (`character_boxes()`) chooses exactly one crop regime for the
whole clip and never mixes them, using the same `MEASURABLE_DETECTION_RATE
= 0.50` / `DETECTION_CONF_FLOOR = 0.40` bar `pose_fidelity` already
applied: below 50% confident detection, every frame uses the fallback
crop; at or above it, every box is YOLO-derived, with gaps held forward or
backward in time rather than falling through to the fallback. The regime
used, why, and the confident-detection rate behind that choice are always
reported (`crop_regime`, `crop_regime_reason`, `confident_detection_rate`).

**A second, previously undiagnosed instance of the same bug was caught
retroactively once the fix was applied.** `release/corpus-anime/corpus-anime.mp4`
had been scoring `NOT-AN-ANIMATION` on `identity_preservation.worst_similarity:
0.29`, attributed at the time to genuine identity drift across a
multi-clip reel — a wrong explanation that had been carried in this
project's own documentation with confidence. Re-run with the fix, the same
clip scores `worst_similarity: 0.99995`, `crop_regime: fallback`
throughout (confident detection rate 20%, below the 50% bar); the 0.29
was the same crop-mixing artefact, not identity drift of any kind, and the
clip is now `PROMISING`, capped only by `warp_error` (0.0711 against a
0.06 bar), which was never in dispute. *(Source: `critic/README.md`,
retraction under "Most damning finding.")* We state the general lesson
explicitly, as the project's own documentation now does: **the failure
signature of this bug class is a near-perfect score on almost a whole clip
dragged through a pass bar by one or a few outlier samples measured a
different way** — if removing a single sample from a "worst-case"
statistic changes the verdict, and that sample was measured on a different
basis than its neighbours, the metric, not the clip, is what changed.

### 5.3 Case 3 — person/pose detectors read zero on this project's stylized art, independent of what is actually there

CMU OpenPose (`controlnet_aux`) returns no person at all on this
project's rendered skeletons, flat cel renders, or SD1.5 anime line-art
output, while reading a real photograph fine (12/12 keypoints on
`zidane.jpg`); YOLO11-pose does somewhat better but detected a cel figure
confidently in only 2 of 6 tested clips (§8.3, Attempt 19). The same
detector finds **no person in any frame** of `out/a21_park.mp4`, a clip
that genuinely contains three characters, and one person in 1 of 200
frames of `out/a21_street.mp4`, which contains four — and, checked
directly as a control before drawing any conclusion, the identical
`modal_count: 0` result also appears on this project's own single-figure
hand-written positive controls. *(Source: `ATTEMPTS.md` Attempt 21,
Result 3; `critic/README.md` §"Scene complexity" correction.)* The root
cause, established independently in Attempt 19 (§7.1), is the same
proportion mismatch that makes ControlNet ignore this project's pose
guides: every detector available here was trained on photographs of human
proportions, and this project's line-art rig sits outside that
distribution regardless of how many correctly-drawn characters are on
screen. `scene_complexity` was corrected to report `UNMEASURABLE` with
the detection rate attached rather than a false cast count of zero (§4.6).

### 5.4 Case 4 — beam search collapsing an adequate learned distribution into a false "the model cannot stage" verdict

Every position number reported in Attempt 22, and initially in Attempt
23's own ablation cells, was decoded with `num_beams=4`. Beam search
returns the approximately most-likely *sequence*; for an output slot the
prompt does not constrain (character x-position, in this project's DSL),
the most likely single value is, by definition, the marginal mode of
whatever the model actually learned — so "the model emits x≈9.4 nearly
every time" is consistent with two different situations that call for
opposite fixes: the model's conditional distribution has genuinely
collapsed (a data or architecture problem), or the distribution is fine
and the search is standing on its peak (a one-argument decoding fix).
Nothing in Attempt 22 distinguished these. *(Source: `ATTEMPTS.md` Attempt
23 §6.)*

Decoding the *same checkpoint* on the *same 30 held-out prompts* four
ways and scoring each with an independently-calibrated staging metric
(§11.2) answered it directly:

| decode | v3 staging | v3 distinct x values | v2 staging | v2 distinct x |
|---|--:|--:|--:|--:|
| `beam4` (what every earlier number in this project used) | **0.188** | **3** | **0.333** | 6 |
| `greedy` | 0.189 | 3 | 0.393 | 7 |
| **`sample`** (top-p 0.95, T=1.0) | **0.862** | **55** | **0.845** | 20 |
| `sample_lo` (top-p 0.90, T=0.7) | 0.732 | 43 | 0.796 | 21 |

Switching only the decoding argument — same weights, same prompts — moves
staging from 0.188 to 0.862 on one checkpoint and 0.333 to 0.845 on
another, both landing *above* the weaker of the project's two
hand-written reference clips (0.825). Thirty scenes decoded by beam
search produce three distinct x-coordinates in total across the whole
set, with 87.8% of all characters landing on one of them; the identical
weights, sampled, produce fifty-five. **A model that had genuinely
collapsed could not produce fifty-five distinct positions; a search that
maximizes sequence likelihood produces three whether or not the model had
collapsed.** This corrects a specific, previously-published claim in this
project's own record (Attempt 22 §10.1: staging is "the single biggest
defect in every generated scene ... the decoder emits modal values and
characters cluster," attributed to absent supervision) — the absent
supervision is real, but the *reported symptom* was mostly an artefact of
how the earlier numbers were decoded, not of what the model had learned.

**A companion ablation shows what would have been reported, and believed,
without this check.** A deterministic post-decode spacing pass — plausible
engineering, and RESEARCH2's own top-ranked recommendation for this
defect — moves beam-search staging from 0.178 to 0.587 and looks, in
isolation, like a successful fix for a model limitation. Reading it
alongside the decoding-only result shows what it actually is:

| source of the staging number | staging (in distribution) |
|---|--:|
| model output, read by beam search | 0.178 |
| ... plus a deterministic spacing pass enforcing it after decoding | 0.587 |
| **model output, read by sampling instead — no pass at all** | **0.862** |
| hand-written reference clips | 0.825–1.000 |

Under beam search, the deterministic pass does essentially all of the
visible work and never reaches the hand-written band (only 17% of scenes
clear the worse reference clip); changing the decoder alone reaches 0.862,
past that same clip, because it is the model's own answer rather than an
imposed one. **We state this exactly as the project states it: good
engineering, bad science claim.** Without the decoding ablation, this
project would have shipped the spacing pass and reported that the model
had learned to stage a scene — a claim the same project's own later
measurement shows to be false. *(Source: `ATTEMPTS.md` Attempt 23 §7,
§11.7.)*

### 5.5 The general lesson

Read together, these four cases motivate a single methodological point we
generalize beyond this project: **a metric's or a decoding default's
provenance — what it was trained, calibrated, or tuned for — bounds what
it can be trusted to measure or produce**, and this bound does not
announce itself; it must be checked against a known-good and a known-bad
case, or a same-weights control decoded a second way, before the result
is trusted as a verdict on the underlying system. CLIP ViT-B/32 was never
going to separate stars from blobs in flat vector art without that check;
a YOLO person detector was never going to count stick figures; a metric
that silently switches its own measurement basis mid-series will produce
a result that looks like a property of the clip and is actually a
property of the metric; and a decoding strategy that reports only the
mode of a learned distribution will produce a result that looks like a
property of the model and is actually a property of the search. All four
were caught in this project only because a control — a known-good clip, a
hand-authored ablation, a re-run after a fix, or the same weights decoded
a second way — was run and compared, never assumed. We consider this
pattern of self-correction (§1; also Attempt 19's topology-to-proportion
correction, §7.1) as important a contribution as any single result it
produced.

## 6. Reproducibility

**Hardware.** RTX 4060 Laptop GPU, 8 GB VRAM, run under a documented
50%-duty-cycle GPU guard (`tools.gpuguard.Guard`) for thermal reasons —
the laptop's fan is damaged and an elevated-shell clock lock was
unavailable, so duty cycling carried thermal management alone for every
GPU loop reported in this paper. `USE_TF=0` is required project-wide or
transformers 4.57 attempts a Keras-3 backend and fails. **Attempts 21–23
use no GPU at any point except the critic's own YOLO pass** — scene
composition and every version of the script-writing model are CPU-only,
and this is reported as a property of the route, not an incidental detail
(§9, §10, §11).

**Model versioning.** Three checkpoints are preserved under
`model/checkpoints/{v1-confounded,v2-deconfounded,v3-staged}/`, each with
its own weights, the exact training data that produced it, and a run
summary, recorded in `model/MODELS.md`. Nothing in Attempt 23 writes into
the v1/v2 directories; v3 trains to a new directory. `model/verify_checkpoints.py`
proves, rather than asserts, that the preserved copies still work:
**v2-deconfounded reproduces Attempt 22's published headline output
byte-for-byte** on the prompt "two friends meet in the park, one waves,
then they kick a ball around" (`bg park dur 91 | cast ana blue 96 100 ;
bo amber 92 90 | prop ball 58 100 ; tree 69 110 | tl 1 26 bo walk 92 ; 8
28 ana wave ; 28 47 bo kick ball ana ; 57 79 bo walk 92`, zero problems,
zero repairs, matching Attempt 22 §8 exactly), and v1-confounded
reproduces its own documented failure mode (a cast of one referencing an
undeclared second character) on the same prompt — a load producing
anything else would have been the alarm. *(Source: `ATTEMPTS.md` Attempt
23 §1.)*

**Key commands** (all `USE_TF=0`; see `svg/README.md`, `critic/README.md`,
and `ATTEMPTS.md` Attempts 19–22 for the complete set):

```bash
# corpus build and training (AniSVG / diffusion hybrid, arc one)
python svg/stream_corpus.py -o svg/data/corpus --max-clips 40000 --progress 2000
python svg/pack_dataset.py --corpus svg/data/corpus -o svg/data/train
python svg/train_lora.py --base Qwen/Qwen3-1.7B-Base --data svg/data/train
python svg/coco18.py
python svg/render_style.py --clip release/corpus-anime/clip00.anisvg \
    --guide coco18 --frames 8 -o svg/out/a19/styled-coco18-final

# AnimateDiff temporal module (Attempt 20)
python critic/evaluate.py svg/out/a20/w512_f8    --guide svg/out/a20/guides_w512/coco18    --max-frames 8 -o svg/out/a20/critic/w512_f8
python critic/evaluate.py svg/out/a20/w384_f8    --guide svg/out/a20/guides_w384/coco18    --max-frames 8 -o svg/out/a20/critic/w384_f8
python critic/evaluate.py svg/out/a20/w256_f8    --guide svg/out/a20/guides_w256/coco18    --max-frames 8 -o svg/out/a20/critic/w256_f8
python critic/evaluate.py svg/out/a20-ip/w512_f8 --guide svg/out/a20-ip/guides_w512/coco18 --max-frames 8 -o svg/out/a20-ip/critic
python critic/evaluate.py svg/out/a19/styled-coco18-final/frames \
    --guide svg/out/a19/styled-coco18-final/guides/coco18 --max-frames 8 -o svg/out/a20/critic/a19_ref
python svg/animate.py --sweep           -o svg/out/a20c-noofl
python svg/animate.py --sweep --offload -o svg/out/a20c-ofl
python svg/animate.py --frames 8 --width 512 -o svg/out/a20
python svg/animate.py --frames 8 --width 512 --ip-adapter --ip-scale 0.6 -o svg/out/a20-ip

# multi-character scene composition (Attempt 21) — no GPU
python scenescript.py --selftest
python scenescript.py scenes/park_meet.scene    -o out/a21_park.mp4   --frames out/a21_park_frames
python scenescript.py scenes/street_relay.scene -o out/a21_street.mp4 --frames out/a21_street_frames
python scenescript.py scenes/park_meet.scene    --audit out/a21_park_frames
python scenescript.py scenes/street_relay.scene --audit out/a21_street_frames
python critic/evaluate.py out/a21_park.mp4   --max-frames 200 --identity-max 24 \
    --caption "three stick figures in a park, two of them passing a ball" --clip-max 20 \
    --character-colors "50F0F0,FF963C,FF6EB4" -o critic/out/a21_park
python critic/evaluate.py out/a21_street.mp4 --max-frames 200 --identity-max 24 \
    --caption "four stick figures on a night street kicking a ball to each other" --clip-max 20 \
    --character-colors "AAFF5A,A064FF,FFBE46,F0F5FF" -o critic/out/a21_street

# the model that writes scene scripts (Attempt 22) — no GPU except the final critic pass
python model/scene_synth.py --n 6000 --val 700 --outdir model/data --prefix scene
python model/scene_synth.py --n 6000 --val 700 --outdir model/data --prefix scene2 --deconfound
python model/train.py --train model/data/scene_train.jsonl --val model/data/scene_val.jsonl \
    --out model/scene_ckpt --epochs 6 --batch 16 --lr 3e-4 --max-in 104 --max-out 176 --val-cap 200
python model/train.py --train model/data/scene2_train.jsonl --val model/data/scene2_val.jsonl \
    --out model/scene_ckpt2 --epochs 6 --batch 16 --lr 3e-4 --max-in 104 --max-out 176 --val-cap 200
python model/scene_eval.py --ckpt model/scene_ckpt  --n 150 --out model/eval_out
python model/scene_eval.py --ckpt model/scene_ckpt2 --val model/data/scene2_val.jsonl --n 150 --out model/eval_out2
python model/scene_infer.py "two friends meet in the park, one waves, then they kick a ball around" \
    --ckpt model/scene_ckpt2 -o scenes/a22_generated.scene
python scenescript.py scenes/a22_generated.scene -o out/a22_generated.mp4 --frames out/a22_generated_frames
python scenescript.py scenes/a22_generated.scene --audit out/a22_generated_frames
python critic/evaluate.py out/a22_generated.mp4 --max-frames 200 --identity-max 24 \
    --caption "two stick figures in a park, one waves, then one kicks a ball to the other" \
    --clip-max 20 -o critic/out/a22_generated

# staging, decoding ablation, constrained decoding and the two deliverables (Attempt 23) — no GPU except critic
python model/scene_synth.py --n 6000 --val 700 --outdir model/data --prefix scene3 --deconfound --stage
python model/scene_data_report.py model/data/scene3_train.jsonl
python model/train.py --train model/data/scene3_train.jsonl --val model/data/scene3_val.jsonl \
    --out model/scene_ckpt3 --epochs 6 --batch 16 --lr 3e-4 --max-in 160 --max-out 256 --val-cap 200
python model/scene_staging.py scenes/park_meet.scene scenes/street_relay.scene --dir scenes --glob "a22_*.scene"
python model/scene_decode_probe.py --ckpt model/scene_ckpt3 --val model/data/scene3_val.jsonl --n 30
python model/scene_decode_probe.py --ckpt model/checkpoints/v2-deconfounded \
    --val model/data/scene2_val.jsonl --n 30 --max-out 176 --max-in 104
python model/scene_eval.py --ckpt model/scene_ckpt3 --val model/data/scene3_val.jsonl --n 150 \
    --out model/eval_out3 --max-in 160 --max-out 256 --sample --unique-colours --rerank 4 --tag _v3sr
python model/scene_probe.py --ckpt model/scene_ckpt3
python model/scene_layout_baseline.py \
    "model/eval_out3/in_distribution_v3.json|model/data/scene3_val.jsonl|v3"
python model/scene_infer.py \
    "two friends meet in the park, one waves, then they kick a ball around" \
    --ckpt model/scene_ckpt3 --max-in 160 --max-out 256 \
    --sample --unique-colours --rerank 4 --seed 23 -o scenes/a23_generated.scene
python scenescript.py scenes/a23_generated.scene -o out/a23_generated.mp4 --frames out/a23_generated_frames
python critic/evaluate.py out/a23_generated.mp4 --max-frames 200 --identity-max 24 --clip-max 20 \
    -o critic/out/a23_generated \
    --caption "two stick figures in a park, one waves, then they kick a ball to each other"
python model/scene_infer.py \
    "six friends spread out on a night street, all of them dancing, then two of them kick a ball around" \
    --ckpt model/scene_ckpt3 --max-in 160 --max-out 320 \
    --sample --unique-colours --rerank 4 --seed 5 -o scenes/a23_six.scene
```

**Frame/artefact locations.** Attempt 20: `svg/out/a20/w512_f8/` (headline
run), `svg/out/a20-ip/` (+IP-Adapter), `svg/out/a20b/` (16-frame sweep),
`svg/out/a20c-noofl/` / `svg/out/a20c-ofl/` (VRAM re-measurement). Attempt
21: `out/a21_park.mp4`, `out/a21_street.mp4`, contact sheets
`out/a21_sheet.png` / `out/a21_sheet_street.png`. Attempt 22: `model/eval_out/`
and `model/eval_out2/` (every generated string, parse error and repair,
per prompt), `model/eval_out/clause_probe.json` / `clause_probe2.json`
(the confound probe), `scenes/a22_generated*.scene`, `out/a22_generated*.mp4`,
`critic/out/a22_*`. Attempt 23: `model/MODELS.md` and
`model/checkpoints/{v1-confounded,v2-deconfounded,v3-staged}/`;
`model/eval_out3/` (every cell, `compare_indist.txt`, `compare_ood.txt`,
`decode_probe_v{2,3}.json`, `clause_probe3.txt`, `layout_baseline.txt`,
`staging_calibration.{json,txt}`, `data_report.txt`); `model/scene_train3.log`;
`scenes/a23_generated.scene`, `scenes/a23_six.scene`; `out/a23_generated.mp4`,
`out/a23_six.mp4`; `critic/out/a23_generated/`, `critic/out/a23_six/`.

**A process failure worth recording as a reproducibility lesson in its own
right:** four Attempt 23 evaluation runs, written to the same output
directory under different flag combinations, silently overwrote each
other before the `--tag` argument existed on `model/scene_eval.py`; the
loss was caught and the runs re-executed with distinct tags. We record
this rather than omit it because it is the same class of failure §5
documents for metrics — an unlabeled measurement basis silently replacing
another — applied to raw experiment logging rather than to a metric's
internal logic.

Every GPU loop in this paper ran through `tools.gpuguard.Guard` at 50%
duty; the clock lock needed an elevated shell and was unavailable, so
duty cycling carried thermal management alone throughout. `USE_TF=0`
before every import; the existing interpreter was reused across attempts,
no new environment except the Python 3.12 `.venv-soup` for Soup layer
streaming (§3.4).

## 7. Experiments and Results I — The Diffusion Route, Closed by Measurement

### 7.1 A diagnosis corrected in the open: skeleton topology was not the cause

Attempt 18 rendered a procedural clip's exact pose guides through SD1.5 +
ControlNet-OpenPose (fixed seed, identical prompt, only the conditioning
image varying between frames):

| | inter-frame change |
|---|---|
| pose guides (input) | 1.29 |
| styled frames (output) | 30.44 |
| **flicker ratio** | **23.6×** |

*(Source: `ATTEMPTS.md` Attempt 18.)* The style half worked — genuine
anime-style output at a quality no token budget could buy directly. Two
failures were logged: (1) the pose was ignored; (2) identity was not
preserved (hair, face, background reorganize every frame despite a fixed
seed). Attempt 18's own hypothesis for failure (1): the hand-built
skeleton was OpenPose-*like* but not true COCO-18 topology — stated as
"fixable... a mapping exercise."

Attempt 19 did that mapping (`svg/coco18.py`) and verified it two ways
before spending any GPU time: the drawing constants were checked against
`controlnet_aux.open_pose.draw_poses` directly, at **max absolute pixel
difference 0 over all 29 frames** of the test clip; and the recovered
joints matched the rig's own ground-truth joints to sub-pixel accuracy
(mean 0.40–0.62 px, max 1.10–1.28 px across four actions, on a 256 px
canvas).

| conditioning | guide | styled | ratio |
|---|--:|--:|--:|
| Attempt 18: limb-axis skeleton, OpenPose palette | 1.29 | 30.44 | **23.6×** |
| Attempt 19: true COCO-18, head from the art | 1.18 | 26.05 | 22.1× |
| Attempt 19 + human-proportion head | 1.14 | 34.99 | 30.8× |
| Attempt 19 + human head + per-clip side convention | 1.09 | 39.58 | **36.2×** |

*(Source: `ATTEMPTS.md` Attempt 19.)* Row 2 is the direct test of Attempt
18's hypothesis: correct COCO-18 topology alone changed **nothing** — the
render still stands hands-on-hips. **The topology diagnosis was right
about what was broken and wrong about what mattered.**

**The control that found the real cause.** Three conditioning images
through the identical pipeline:

| conditioning | result |
|---|---|
| genuine OpenPose annotation of a real photo (`zidane.jpg`) | obeyed |
| hand-built canonical wave skeleton, arm straight up | **obeyed** |
| rig-derived COCO-18, same wave pose | ignored |

*(Source: `ATTEMPTS.md` Attempt 19.)* A correctly-shaped COCO-18 skeleton
*is* obeyed by this model; the fault remained in this project's own
skeleton. Measured against the obeyed reference, in units of torso
length T:

| ratio | ours (from the art) | obeyed reference |
|---|--:|--:|
| neck → nose | 0.103 T | 0.330 T |
| ear span | 0.956 T | 0.400 T |
| shoulder width | 0.579 T | 0.770 T |
| **ear span / shoulder width** | **1.65** | **0.52** |

*(Source: `ATTEMPTS.md` Attempt 19.)* **The real cause: anime proportions
are not human proportions.** The cel figure's head is deliberately drawn
1.65× wider than its own shoulders — the style, by design. Mapped
faithfully to canonical topology, its metric proportions still sit
outside anything a human-photo-trained ControlNet has seen; the model
discards the pose and falls back on its prompt prior. We state this as a
general finding (contribution 7, §1): the property that makes a character
read as anime is the same property that removes it from a human-pose
conditioning manifold. Two targeted corrections followed; obedience is
still only **partial** — a raised arm survives clearly in one of four
inspected frames and is ambiguous in the rest. **Stated as unfixed, not
smoothed over.**

### 7.2 Pose obedience and temporal stability are measured orthogonal, not correlated

The same four-row table above shows flicker rising **monotonically** as
pose conditioning improved: 23.6× → 22.1× → 30.8× → 36.2×. Two
contributing effects: the canonical skeleton is drawn thinner than
Attempt 18's, constraining less of the frame; and pose conditioning is,
by construction, orthogonal to what drives flicker. **A better guide
bought a strictly worse flicker number, on a quantity the guide was never
able to move.**

### 7.3 An off-the-shelf pose estimator cannot score this material

CMU OpenPose returns **no person at all** on the project's rendered
skeletons, flat vector cel renders, or SD1.5 anime line-art output, while
reading a real photograph fine. YOLO11-pose does marginally better,
finding the cel figure in 2 of 6 tested clips at confidence 0.41 and 0.22
with ~100 px mean keypoint error — not ground truth, noise. *(Source:
`ATTEMPTS.md` Attempt 19; corroborated by §5.3's later, broader
finding that the same detectors read zero on Attempt 21's material too,
independent of cast size.)* **One cause, two symptoms**: the same
off-manifold proportions that make ControlNet discard the pose guide also
make general human-pose estimators unable to read this style at all.

## 8. Experiments and Results II — AnimateDiff: A Real Improvement That Still Fails the Gate

Attempt 19 ended on a stated prediction: flicker is not a pose problem, so
fixing it needs an explicit temporal mechanism. `svg/animate.py` is that
mechanism — AnimateDiff's motion module inserted into the same frozen
SD1.5 UNet, inference only, driven by the same COCO-18 guides. Everything
else from Attempt 19's final row is held fixed: `clip00` (the wave), seed
1234, 20 steps, guidance 7.0, conditioning scale 1.0, 8 frames at 512 px.
*(Source for this entire section: `ATTEMPTS.md` Attempt 20.)* The guide's
own inter-frame change at 512 px comes back as 1.09 — identical to
Attempt 19's final row — so any change in the ratio is entirely in the
numerator.

### 8.1 The first genuine reduction in flicker this project has measured, and it is not enough

| method | guide | styled | ratio |
|---|--:|--:|--:|
| A19 final, COCO-18, no temporal mechanism, 512 px | 1.09 | 39.58 | **36.2×** |
| A20 + AnimateDiff motion module, 512 px, 8 frames | 1.09 | 27.32 | **25.0×** |
| A20 + motion module, 384 px, 8 frames | 1.25 | 27.50 | 22.0× |
| A20 + motion module, 256 px, 8 frames | 1.64 | 27.79 | 17.0× |

This is the first intervention in the project's history to move flicker
*down*: 36.2× → 25.0×, a 31% cut in the styled term, with the guide
denominator unchanged. **Both halves of this result matter and are
reported together:** it is a real, measured improvement, and — the
headline failure — **25.0× is still a hard fail** against the critic's 8×
bar; `critic/evaluate.py` returns NOT-AN-ANIMATION at all three
resolutions.

**Trap: the resolution trend above is an artefact.** Styled flicker is
flat across resolution (27.32/27.50/27.79, a 1.7% spread); the apparent
gain at lower resolution is the *guide* moving more on a smaller canvas
(1.09→1.25→1.64), inflating the ratio's denominator. **Only the 512 px
row is comparable to Attempt 19; the 17.0× figure must not be quoted as a
result.**

### 8.2 Warp error crossed a gate the flicker ratio hides

| | flicker ratio | warp error | critic verdict |
|---|---|---|---|
| A19 final, no motion module | 36.2× (hard fail, bar 8×) | **0.185 — hard fail**, bar 0.15 | NOT-AN-ANIMATION |
| A20 + motion module | 25.0× (hard fail) | **0.135 — soft fail**, pass bar 0.06 | NOT-AN-ANIMATION |

The motion module moved warp error out of the hard-fail band and into the
soft-fail band. The verdict does not change, because flicker alone still
disqualifies the clip.

### 8.3 The motion module slightly hurts identity; an IP-Adapter anchor fixes it, and costs a little flicker back

| run | mean similarity | worst similarity |
|---|--:|--:|
| A19 final, no motion module | 0.934 | 0.787 |
| A20 + motion module | 0.897 | 0.744 |
| A20 + motion module + IP-Adapter anchor | **0.960** | **0.896** |

`svg/out/a20-ip/` adds `--ip-adapter --ip-scale 0.6`, anchored on the
clip's own flat-colour vector render of frame 0. Worst-case identity rises
from 0.744 to 0.896, clearing the 0.45 pass bar; mean rises to 0.960.
**Caveat that must survive any reading of this number:** the identity
metric is an HSV colour-histogram embedding, and the anchor run's own
contact sheet shows its palette transferring wholesale. What this result
demonstrably locks is **colour scheme**, not identity in a structural
sense. And the anchor does essentially nothing for flicker (25.0%→25.3%,
warp error 0.135→0.142, both marginally worse). **Two separate defects,
two separate mechanisms, and neither mechanism touches the other's
number** — direct evidence for contribution 1 (§1).

### 8.4 Pose obedience got worse with the motion module, not better

Attempt 19's corrected guide showed a raised wave arm clearly surviving in
1 of 4 inspected frames. In the Attempt 20 sheet, the arm is raised in
**0 of 8**: the figure holds arms folded across its chest in every frame.
Temporal attention makes every frame agree with its neighbours —
including agreeing on a pose that is not the guide's; joint denoising
averages the per-frame conditioning pull into a single clip-wide
compromise. **The motion module trades pose obedience for temporal
agreement** — the second direct instance of contribution 1.

### 8.5 Critic verdict: NOT-AN-ANIMATION throughout; one gate moved

Every configuration tried (8-frame headline run, both resolution
controls, the IP-Adapter run, the 16-frame run) returns
NOT-AN-ANIMATION, disqualified by flicker every time. The one gate that
moved is warp error, 0.185 (hard fail) → 0.135 (soft fail) at 8 frames.

### 8.6 The stacked VRAM measurement the research survey could only mark unverified

Resident weights, fp16, no offload: SD1.5 UNet + ControlNet-OpenPose +
AnimateDiff motion adapter = **4.41 GiB** of the 8 GiB card, before a
single latent exists.

| config | s | peak alloc GiB | peak reserved GiB | guide | styled | ratio |
|---|--:|--:|--:|--:|--:|--:|
| w512_f16 | **OOM** | — | 7.21 | — | — | — |
| w448_f16 | 173 | 6.46 | 7.04 | 1.23 | 99.02 | 80.5× |
| w384_f16 | 137 | 5.92 | 6.39 | 1.31 | 56.53 | 43.3× |
| w320_f16 | 101 | 5.46 | 5.78 | 1.54 | 59.71 | 38.9× |
| w256_f16 | 68 | 5.09 | 5.31 | 1.70 | 95.02 | 55.9× |
| w512_f8 | 129 | 5.74 | **6.15** | 1.09 | **27.32** | **25.0×** |

**Only one configuration requires offload; everything else fits
resident.** `w448_f16` peaks at 7.04 GiB reserved against ~7.5 GiB
addressable. 512 px × 16 frames is the sole exception (OOM at 7.21 GiB
reserved); with `enable_model_cpu_offload()` it reaches 6.60 GiB / 207 s.
**Model-CPU-offload is very nearly free on this hardware** — ~1.2 GiB
saved at under 1.5–5% wall-clock — but this is reported explicitly as
hardware-specific: the 50%-duty guard's mandatory idle window absorbs the
host-device copies, so the figure is not expected to transfer to an
ungoverned card.

### 8.7 16 frames is worse than no motion module at all; the route closes here

At 512 px, flicker goes from 25.0× at 8 frames to **74.0× at 16 frames**,
with the guide denominator essentially unchanged. Every 16-frame
configuration lands between 38.9× and 80.5×, all worse than Attempt 19's
36.2× with no motion module at all. The critic agrees more sharply: warp
error reaches 0.367 at 16 frames (back into the hard-fail band), worst-case
identity falls to 0.584, and `scene_complexity` reports a **modal cast of
0** — no person detected in the plurality of frames. **Below 384 px, the
collapse is the base model's, not AnimateDiff's**: `w256_f16` contains no
figure at all, only coloured blocks, because SD1.5 was trained at 512 px
and 256 px is out of its distribution. This is the finding that closes the
16-frame route as currently built: the VRAM work above establishes 512×16
is technically reachable, and reaching it produces a clip roughly three
times less stable than the half-length one.

### 8.8 Standing on the route

AnimateDiff is the first thing in this project's history to reduce
flicker rather than move it sideways or up, and it is not enough on its
own: a third off, still roughly 3× over the bar, bought at the price of
the pose obedience Attempts 18 and 19 spent their entire budget
recovering, and it does not extend past 8 frames. **Flicker, identity and
pose obedience are three separate defects on this stack, each needs its
own mechanism, and the mechanisms interfere** — the motion module costs
pose, the identity anchor costs a small amount of flicker back. Nothing in
the AnimateDiff route as built reaches USABLE, and the project's own
response to that closure is arc two (§9–10).

## 9. Experiments and Results III — Multi-Character Scene Composition on CPU Alone (Attempt 21)

Attempts 17–20 spent their whole budget trying to get styled anime out of
SD1.5 + ControlNet + AnimateDiff and closed that route by measurement
(§7–8). Attempt 21 is a deliberate change of direction: **several
characters, a background and props, in one shot, driven by a script** —
what the project's own log states has been the end goal since Era 1, and
had never been attempted. Style is explicitly not the objective;
legibility and composition are. The new module, `scenescript.py`, uses no
GPU at any point in generation — the whole pipeline is CPU-only numpy +
Pillow + ffmpeg. *(Source for this entire section: `ATTEMPTS.md` Attempt
21.)*

### 9.1 The format

A scene script is a line-based text file declaring a cast, a background,
props, and a timeline of `<start> <end> <who> <action> [to X] [at TARGET]`
events. Actions are the rig's own (walk/run/jump/kick/wave/dance/idle).
Nothing new is drawn: `stickman.StickFigure` supplies the joints, the same
per-frame equations Era 1 used render them, and `geovid.Scene` rasterises
the result.

### 9.2 Five new problems, and what each turned out to be

None of these exist when there is one figure doing one action.

1. **Root motion has to be taken away from the action.** Every action in
   `stickman.py` bakes its own absolute position; with three characters,
   every one teleports to the middle of the frame the moment it acts. The
   fix: the action supplies the pose, the script supplies the root — each
   frame the compositor reads the action's hip, subtracts it, and
   re-anchors the joint set to where the script says the character is,
   keeping only the action's vertical hip offset so a walk's bob or a
   jump's arc survives while its horizontal sweep is discarded.
2. **Foot slide**, caused by (1): gait rate is derived from the distance
   the script actually asks for rather than assumed. Not solved exactly —
   this matches average stride length, not per-foot contact — and stated
   as approximate.
3. **Depth is not a free parameter, and occlusion is not a draw order.**
   A ground plane is defined once and depth defaults to scale. Because the
   cast is line art, draw order alone does not produce occlusion — each
   character carries a matte, its own silhouette dilated by 4 px. The
   first version of that matte painted the backdrop colour directly into
   the frame, which erased scenery behind a character as well as the
   character behind it — **a matte cannot be a colour.** The compositor
   was rebuilt to paste the original backdrop through the matte first
   (restoring scenery, deleting whatever was behind), then paste the
   character's colour through its ink mask.
4. **Props are stateful; characters are not.** A character is a pure
   function of `(action, phase)`; a ball is not — where it is at time t
   depends on who kicked it and where it rolled. The ball is compiled into
   a segment list over the whole timeline (fly, then roll, then rest) and
   integrated, which is what lets it be kicked at one time and kicked back
   later from wherever it actually landed. The interaction runs the other
   way too: `kick ball at ana` reads the ball's actual current position and
   generates the kicker's approach to it — the script never says where the
   kicker should stand.
5. **A background that is provably still.** The backdrop's layers are
   time-independent by construction and rendered exactly once, then
   copied as the base of every frame — not "stable to within a threshold,"
   the same array.

### 9.3 Result 1 — the videos

| file | script | s | frames | cast | props |
|---|---|--:|--:|--:|--:|
| `out/a21_park.mp4` | `scenes/park_meet.scene` | 12.5 | 375 | 3 | 7 |
| `out/a21_street.mp4` | `scenes/street_relay.scene` | 14.0 | 420 | 4 | 6 |

Render cost is ~1.5 s of one CPU core per second of 854×480 video, and
**no GPU is used to generate any of it** — which after Attempts 17–20
spent their entire budget inside a 7.5 GiB VRAM ceiling is the point, not
a footnote.

### 9.4 Result 2 — the critic

| clip | warp_error (pass ≤0.06) | motion_coverage (floor 0.2%) | identity worst | CLIPSIM | verdict |
|---|--:|--:|--:|--:|---|
| `out/a21_park.mp4` | **0.0055** | **5.32%** | **1.00** | 0.311 | **USABLE** |
| `out/a21_street.mp4` | **0.0065** | 4.89% | **1.00** (post-fix, §5.2) | 0.336 | **USABLE** (post-fix) |
| (reference) `control_wave.mp4`, single figure | 0.0011 | 1.01% | 1.00 | 0.329 | USABLE |

*(Identity values here reflect the corrected crop-regime figures from
`critic/README.md`'s current baseline table, §4.2 and §5.2; the clip's
original evaluation reported `out/a21_street.mp4` at identity worst 0.4368
and verdict PROMISING, since corrected.)* The composed multi-character
scene holds the temporal gates the entire diffusion route failed: warp
error 0.0055 against the 0.06 pass bar and the 0.15 hard-fail bar — **25×
under the hard-fail line** where Attempt 20's best AnimateDiff
configuration sat at 0.135. Motion coverage is roughly 5× the single-figure
control's, consistent with several characters moving independently.
Flicker ratio is N/A — there is no conditioning guide, because nothing
here is conditioned on anything.

### 9.5 Result 3 — measuring the cast without a working detector

`scene_complexity` reports `mean_count 0.00, modal_count 0` on the park
clip and `mean_count 0.05, modal_count 0` on the street clip — a photo-
trained detector does not read flat line art at any cast size, verified
against this project's own single-figure positive controls, which score
the identical zero (§4.6, §5.3). So the cast is measured another way, in
`scenescript.py --audit`: per-frame ink presence per declared character
(colour match, connected components).

| clip | characters | frames with every character visible |
|---|--:|---|
| park | 3 | **75/75 sampled** |
| street | 4 | **84/84 sampled** |

Blob counts run 1.7–4.0 per character (a stick figure is normally two
components; the count rises exactly when another figure's matte cuts a
limb — occlusion working, visible as a number).

### 9.6 Result 4 — the fresh-script test

The log's standing criticism of this project is that its mechanisms only
ever work for the one example hand-authored with them. So
`scenes/street_relay.scene` was written **after the engine was frozen**
and rendered with **no code change of any kind**: a different cast size,
background, props, action mix, and a three-way ball relay instead of a
two-way pass. It rendered first time and is coherent: four distinguishable
characters doing four distinct things in a shared space, correct occlusion
where they cross, the ball changing hands twice. Critic (post-fix): warp
0.0065, motion 4.89%, USABLE.

### 9.7 What is honestly not solved

Foot slide is fitted to average stride, not per-foot ground contact.
There is no author here, only an interpreter — both scripts in this
attempt were written by hand, and Attempt 22 (§10) is the response to
that gap specifically. No collision or contact: two characters can walk
through each other, and the ball's arc is kinematic. Actions cannot blend
across a clip boundary on one body — a later-starting clip wins outright.
Staging is not checked: correct occlusion happily hides a character behind
a foreground prop for a whole clip, and two positions in `park_meet.scene`
had to be moved by eye. CLIPSIM (0.311 park, 0.336 street) still cannot
judge this content, for the same reason established in §4.3/§5.1;
`structural_correctness`, the metric that *did* separate correct from
incorrect content on the AniSVG path, is N/A here because a composed
raster scene has no AniSVG source — a scene script is itself a
declarative source and could in principle be checked exactly the same
way, but nothing does that yet.

### 9.8 Standing

The symbolic path now produces what this project has called the end goal
since Era 1 — several characters, a background and props, in one composed
shot, from a script, as a real video, scoring USABLE from the critic — and
it does it on the CPU. The two hard parts were neither the rig nor the
renderer: they were taking root motion away from the action, and
realising that a matte cannot be a colour. What is still missing is not
composition but authorship: no model writes these scripts, and Attempt 21
by itself is subject to the same sentence as Era 1 — the best-looking
output of the project is still hand-written. Attempt 22 (§10) is the
project's answer to exactly that gap.

## 10. Experiments and Results IV — Learning to Write Scene Scripts (Attempt 22)

Attempt 21 removed half of the project's standing indictment — the
composition machinery generalises to a script written after the engine was
frozen — but left the other half standing: nothing writes the scripts.
This attempt trains a model to. *(Source for this entire section:
`ATTEMPTS.md` Attempt 22.)*

### 10.1 The output space, stated before any model is trained

The failure to avoid is Attempt 2's (Era 1): 100% per-slot validation
accuracy on an output space of nine archetypes and four slots, a number
that "meant almost nothing" because the space was tiny. `model/scene_grammar.py`
defines a genuinely combinatorial space instead: cast size 1–5, character
identity from 10 colours (distinct within a scene), placement on a
21-point grid with continuous scale, 3 backgrounds, 0–3 props from 6
kinds, up to 12 timeline events with continuous timing, 7 actions. A
3-character scene with 8 events already has a cast assignment, 3
positions, 3 scales, a background, a prop list, and 8 events each with two
continuous times, an actor and an action — the sampler samples
*combinations*, not variants of a fixed script.

Two honest reductions stated up front: character names are canonical and
positional (`ana`, `bo`, `cy`, `dee`, `ell` by cast index — a modelling
simplification, not a claim about the rest of the system), and numbers are
emitted as integers in tenths/hundredths rather than decimals — a
tokenizer economy that cut mean target length from 118 tokens to 97.5.

### 10.2 Training: t5-small, CPU, under nine minutes

Same model and harness as Attempt 2 (`t5-small`, 60M params,
`model/train.py`), with three added CLI flags and no change to its logic.

```bash
USE_TF=0 python model/train.py --train data/scene_train.jsonl \
    --val data/scene_val.jsonl --out scene_ckpt \
    --epochs 6 --batch 16 --lr 3e-4 --max-in 104 --max-out 176 --val-cap 200
```

**8 CPU threads, 2250 steps, `train_runtime` 530 s. No GPU touched at any
point.** Final eval_loss 0.9240, against Attempt 2's ~8e-5 — **the
expected result, not a regression**: Attempt 2's target was a near-
deterministic function of the prompt; here the target — every position,
scale, timing, colour and decor choice — is mostly *not* determined by the
prompt at all, so cross-entropy loss is not the metric that matters and is
not quoted as a result. The parse/render/match rates below are.

### 10.3 In-distribution measurement: `model/scene_eval.py`

150 held-out val prompts, disjoint from train, four rates: **parse** (the
string parses under strict `dsl_to_spec`), **valid** (zero structural
problems, before any repair), **render** (`scenescript.py` builds and
rasterises without raising, reported after `repair()`), **prompt match**
(per field, against only the facts the prompt stated).

| | |
|---|--:|
| parse | **150/150 = 100.0%** |
| valid, zero problems, no repair | 103/150 = 68.7% |
| render through `scenescript.py` | **150/150 = 100.0%** |
| byte-identical to the reference DSL | **0/150 = 0.0%** |
| cast count | 145/145 = 100.0% |
| **every stated field correct at once** | **136/150 = 90.7%** |

The 0.0% exact-match line is worth reading twice: the model reproduces the
reference string never — it picks its own positions, scales, timings and
scenery every time — and still satisfies 90.7% of the prompts completely.
The 31.3% not structurally clean are dominated by two faults, neither a
parse failure: two characters sharing a colour (28 cases), overlapping
kicks on one ball (10), an event ending after the clip does (13). Both of
the first two are *global* constraints an autoregressive, left-to-right
decoder with no memory of what it has already emitted has no mechanism to
enforce; `repair()` fixes all of them, which is why render is 100% while
valid is 68.7%, and both numbers are quoted rather than only the flattering
one.

### 10.4 The generalisation test — the actual result

`model/ood_prompts.json` is 24 hand-written prompts, in six groups, all
outside the synthesis distribution: unseen phrasings and verbs, unseen
framings, unseen background wordings, cast sizes never seen in training
(six), action combinations never seen (a single character kicking a ball —
the sampler only ever produced kicks with two or more), five characters
doing the same thing at once, and structurally different prompts.

| | in distribution (150) | out of distribution (24) | gap |
|---|--:|--:|--:|
| parse | 100.0% | **100.0%** | 0.0 |
| render | 100.0% | **100.0%** | 0.0 |
| background | 100.0% | **100.0%** | 0.0 |
| **cast count** | **100.0%** | **33.3%** | **66.7** |
| **every stated field at once** | **90.7%** | **25.0%** | **65.7** |

**The headline number of Attempt 22 is a 65.7-point compositional
generalisation gap, and essentially all of it is one failure.** The
*format* generalises perfectly — 24/24 parse, 24/24 render, 24/24 correct
background. What does not generalise is counting the cast: 16 of 24
failures are `cast_count`, and 15 of those emit a cast of exactly **one**.

### 10.5 The cause, isolated by a controlled probe

The obvious reading — "it cannot count past what it saw" — is wrong, and a
controlled probe says so. `model/eval_out/clause_probe.json` holds the
stated numeral and the number of described characters independent and
asks for the cast size:

| stated numeral | 1 clause | 2 clauses | 3 clauses | 4 clauses | 5 clauses |
|---|--:|--:|--:|--:|--:|
| "one people..." | **1** | | | | |
| "two people..." | 1 | **2** | | | |
| "three people..." | 1 | 2 | **3** | | |
| "four people..." | 1 | 2 | 3 | **4** | |
| "five people..." | 1 | 2 | 3 | 4 | **5** |

*(cell = cast size emitted; bold = numeral and clause count agree.)*
**Emitted cast size equals the number of described characters in 15 of 15
probes, and the stated numeral is ignored in all 10 disagreements.** The
model can produce a cast of five — the capability is there. It never
learned to read the numeral; it learned to count clauses.

**And the cause of *that* is in the data, not the model.** Measured on the
training set: the number of characters the prompt describes equals cast
size in **5681 of 5688 = 99.9%** of synthesized examples, because the
synthesiser gave every character its own clause. "Count the clauses" and
"read the numeral" were the same function on 99.9% of the training
distribution, and gradient descent took the cheaper one. This is a
spurious-correlation failure the project built into its own dataset, and
every out-of-distribution prompt that states a count without describing
each character individually — which is how people actually write — walks
straight into it.

### 10.6 Removing the confound and retraining

`scene_synth.py` gained a `--deconfound` flag: with p=0.45 a multi-member
cast's prompt states the count but describes only a random subset of the
characters; with p=0.22 the whole cast is described in one clause sharing
one action. Nothing else changed — same grammar, model, hyperparameters,
seed. Measured on the new training set, clause count equals cast size in
**68.0%** of examples, down from 99.9%. Retrained: 6 epochs, `train_runtime`
531.7 s, same 8 CPU threads, final eval_loss 0.9470 (v1: 0.9240 — slightly
*worse* loss on a harder, less predictable dataset, again the point about
loss not being the result).

The same probe on v2: **14 of 15 agree with the numeral, against 5 of 15
for v1** (the one miss is an ungrammatical probe string). The diagnosis
was correct and it was a data fault, not a model fault.

| | v1 in-dist | v1 OOD | v2 in-dist | v2 OOD |
|---|--:|--:|--:|--:|
| parse | 100.0% | 100.0% | 100.0% | 100.0% |
| valid, no repair | 68.7% | 62.5% | 40.0% | 54.2% |
| **cast count** | **100.0%** | **33.3%** | **100.0%** | **83.3%** |
| background | 100.0% | 100.0% | 100.0% | 100.0% |
| **all stated fields** | **90.7%** | **25.0%** | **86.0%** | **62.5%** |
| **generalisation gap** | — | **65.7 pts** | — | **23.5 pts** |

The gap is cut from 65.7 points to 23.5 points, at a measured cost of 4
points of in-distribution accuracy (90.7% → 86.0%) — **reported as a cost,
not hidden.** Two things survive deconfounding and are stated plainly:

- **Six characters is a hard capability limit, not a phrasing problem.**
  All three `cast_size` prompts fail on both models; v2 answers "six stick
  figures" with a cast of 4, "half a dozen" with 3, "a crowd of six" with
  4. The training distribution contains 1–5 and the model does not
  extrapolate to 6 — it saturates.
- **The `valid` rate got *worse*, 68.7% → 40.0%, because of duplicate
  colours** (71 of 150 v2 outputs against 28 of 150 for v1). Group scenes
  push mean cast size up, and "no two characters share a colour" is a
  global constraint an autoregressive decoder cannot see. Rendering is
  unaffected (repair recolours), but this is a real defect `repair()` is
  currently carrying, not one that disappeared.
- **Positions still collapse to a handful of modal values** (the deliverable
  below places two characters at 9.6 and 9.2 on a ten-unit-wide frame).
  The prompt says nothing about where anyone stands, so this slot has no
  supervision at all, and the decoder answers an unasked question with its
  prior.

### 10.7 Does it survive the critic? A rate, not an anecdote

Ten generated scenes (five in-distribution, five out-of-distribution)
rendered through the unmodified `scenescript.py` and scored:

| clip | warp (pass ≤0.06) | motion coverage (floor 0.2%) | identity worst | verdict |
|---|--:|--:|--:|---|
| a22_batch_ind0–ind4 (5 clips) | 0.0013–0.0042 | 1.45%–3.25% | 1.00 | USABLE (all 5) |
| a22_batch_ood0, ood2, ood3, ood4 | 0.0005–0.0029 | 0.78%–2.51% | 1.00 | USABLE (all 4) |
| a22_batch_ood1 | 0.0002 | **0.15%** | 1.00 | **NOT-AN-ANIMATION** |

**Critic pass rate: 9/10 USABLE.** The one failure is worth more than the
nine passes. The prompt was *"an empty studio, two figures, no scenery:
they simply stand around doing nothing"* — one of the project's own
out-of-distribution prompts — and the model answered it exactly right: two
characters, studio background, no props, four `idle` events and nothing
else. Motion coverage 0.15% against the critic's 0.2% floor fires the
gate. **The model obeyed the prompt and the judge rejected the
obedience** — the motion floor doing precisely what it was added to do
(§4.1) meeting, for the first time in this project, a clip that is
*supposed* to be nearly still. This is not recorded as a model error.
`identity_preservation` reads 1.00 on every clip because the crop-regime
fix (§5.2) now forces the fallback crop consistently when YOLO cannot
detect the material, and `scene_complexity` reports `UNMEASURABLE` rather
than a false cast of zero (§4.6) — both are Attempt 21 findings the
evaluation harness has since acted on.

### 10.8 The deliverable: an unseen prompt to an .mp4

```bash
python model/scene_infer.py \
  "two friends meet in the park, one waves, then they kick a ball around" \
  --ckpt model/scene_ckpt2 -o scenes/a22_generated.scene
python scenescript.py scenes/a22_generated.scene \
  -o out/a22_generated.mp4 --frames out/a22_generated_frames
```

The model emitted, with **zero structural problems and zero repairs**, a
scene that rendered to `out/a22_generated.mp4`: 9.1 s, 273 frames, 2
characters, 2 props, 4 events. Critic: warp error **0.0022**, motion
coverage 2.03%, identity 1.00, CLIPSIM 0.297, **USABLE**. Cast audit: both
characters visible in 55/55 sampled frames. A second deliverable, from an
out-of-distribution prompt with a three-way group/pair structure
("three friends in the park: all three dance at the same time, then two of
them pass a ball while the third one just watches"), also rendered:
9.2 s, 3 characters, 7 events, warp 0.0046, motion 3.74%, **USABLE**.
**These are the first videos in this project whose script was not written
by a human.**

## 11. Experiments and Results V — Staging Is Mostly a Decoding Artefact (Attempt 23)

Attempt 22 ended by naming staging as "the single biggest defect in every
generated scene" and the thing standing between the pipeline and a scene
that looks composed. Attempt 23 takes that sentence apart, and does so
against `paper/RESEARCH2.md` (§2), whose §7 asked for a falsifiable
prediction per technique stated *before* the measurement — those
predictions are scored honestly below, including the ones that were
wrong. *(Source for this entire section: `ATTEMPTS.md` Attempt 23.)*

### 11.1 Versioning and reproducibility, first

Before any new code ran, v1 and v2 were copied to
`model/checkpoints/{v1-confounded,v2-deconfounded}/` alongside the exact
data that trained them, and `model/verify_checkpoints.py` confirmed both
still load and reproduce their documented behavior — including
v2-deconfounded's byte-for-byte reproduction of Attempt 22's published
output (§6). v3 trains to a new directory; nothing already preserved is
touched.

### 11.2 A staging metric, independently calibrated

`model/scene_staging.py` samples 61 instants across a clip and, at each,
reads every character's actual position and the half-width of its
*current* silhouette — built off the rig by posing `stickman.StickFigure`
through all seven actions (an idle figure is 0.34 units wide about its
hip, a kicking one 0.92, nearly three times as much), not assumed as a
single constant. Overlap only counts as a staging fault when two
silhouettes overlap *and* the characters stand on ground planes within
0.20 units of each other — a near character crossing a far one is depth,
not a defect, and is exactly what Attempt 21's occlusion matte was built
to render correctly. The composite `staging` score is the mean of three
sub-scores (pairwise separation adequacy, stage-span fraction, occlusion
fraction) and is undefined (`NaN`) for a one-character scene, since a
solo cast has no staging to measure.

**Two candidate sub-terms were built, measured against the project's own
reference clips, and thrown out because they disagreed with clips already
known to look good** — reported here because the decision itself is part
of the metric's validation: an edge-clearance term scored 0.00 on both
hand-written scenes (one character's hand crosses the frame edge for a
few frames while walking to a script-specified position) and was demoted
to a diagnostic; a `min()`-over-the-clip separation term scored the
hand-written `street_relay.scene` at 0.016 because two characters pass
within a fraction of a second during a scripted ball hand-off, and was
replaced by a mean-adequacy version. **The honest reading: a metric that
disagrees with the clips it is calibrated against is wrong, and the fix
is to change the metric, not re-describe the clips.**

**Calibration**, against the project's two hand-written reference scenes
(the best-looking output of the whole project, per Attempts 21–22) and
eleven of Attempt 22's model-written scenes:

| | staging |
|---|--:|
| `scenes/park_meet.scene` (hand-written, 3 cast) | **1.000** |
| `scenes/street_relay.scene` (hand-written, 4 cast) | **0.825** |
| **hand-written mean** | **0.912** |
| best model-written (`a22_batch_ood1`) | 0.747 |
| `scenes/a22_generated.scene` (the Attempt 22 deliverable) | 0.332 |
| worst model-written (`a22_batch_ood4`) | 0.000 |
| **model-written mean, n=11** | **0.352** |

**The separation is complete: the worst hand-written scene (0.825) scores
above the best model-written one (0.747), with no overlap.** The Attempt
22 deliverable's 0.332 decomposes as span 0.346 (two characters spanning
15.6% of the usable stage) and occlusion 0.000 (they cover each other in
34.4% of sampled observations) — the 9.6/9.2 stack, as a number, for the
first time.

### 11.3 The root cause is not what was expected: the training data is fine

The obvious next move — the synthesizer must be emitting stacked scenes —
was checked before anything was changed. Scoring 400 rows of the actual
v2 training data with the new metric:

| | staging |
|---|--:|
| hand-written scenes | 0.912 |
| **v2 training targets (n=371 multi-cast of 400)** | **0.865 mean, 0.908 median** |
| v2 model outputs | 0.352 |

**The training data is not badly staged** — it sits 0.05 below the
hand-written clips and 0.51 above what the model trained on it produces —
so the hypothesis this attempt was scheduled to act on is false. The
actual mechanism, named but not quantified in Attempt 22 §10, is
**conditional mode collapse**: the prompt almost never states where
anyone stands, position slots are sampled independently in the original
synthesizer, so the training distribution's conditional structure between
position slots is close to uniform, and a near-uniform conditional has no
informative argmax — beam search then lands on whatever small marginal
bias exists, for every slot at once. This reframes the fix: not "produce
nicer training scenes" but "make position slots statistically depend on
each other," which is a decoding and data-conditioning question, not a
data-quality one.

### 11.4 What was built, and the predictions made before measuring

Per RESEARCH2 §7's own requirement, each change carries a stated
prediction. `model/scene_ckpt2/config.json` settles one item immediately,
without a training run: `d_model` is **512** against closed vocabularies
of at most ten words (10 colours, 8 names, 3 backgrounds, 7 actions),
which "When Can Transformers Count to n?" (arXiv:2407.15160, RESEARCH2
§3) predicts is far above the regime where exact counting becomes
numerically unstable — ruling out "make the model wider" as a fix before
it is tried, and confirmed against the actual checkpoint rather than
t5-small's published defaults. The remaining changes: a
`UniqueColourProcessor` logits processor masking already-used colour
tokens (inference-only, ~60 lines, hand-written against
`transformers.LogitsProcessor` rather than adopting Outlines/Guidance per
RESEARCH2 §4's own recommendation for one small constraint); a
`space_out()` deterministic post-decode spacing pass enforcing a 1.25-unit
minimum separation between emitted x-coordinates while preserving the
model's chosen left-right order; positions sampled *jointly* rather than
independently in the synthesizer, with 12% of multi-cast scenes
deliberately clustered and described as such; spatial-language
paraphrases (seven fact kinds — leftmost, rightmost, middle, back, front,
spread, close) attached to a character's own action clause, used only
when true of the sampled scene; and cast size extended from 1–5 to 1–8.
**RESEARCH2's own top-ranked staging fix — the deterministic spacing
pass — and its own §2 diagnosis of "conditional mode-collapse" were both
followed; the survey did not consider the decoder itself as a candidate
cause**, which §11.6 below shows mattered.

**The DETR-style Hungarian set-decoder (RESEARCH2 §4), the one surveyed
technique that structurally fixes staging and duplicate colours together,
was deliberately not built.** It is the only item requiring genuine
architecture work (a new decoder head, a new order-invariant matching
loss) against a nine-minute CPU retraining budget, and — as §11.8 and
§11.12 below show — a decoding-argument change and a rerank pass closed
most of the gap without it. It remains the principled fix and is future
work (§13), not an unexplained omission.

One more measurement caught a defect that would have quietly corrupted
this attempt: eight-character casts need up to eighteen timeline events,
and at Attempt 22's `--max-out 176`, 11.28% of the new training targets
would have been silently truncated mid-scene. v3 therefore trains at
`--max-out 256` (0% truncated, measured over all 6000 rows) and
`--max-in 160` — a hyperparameter difference from v2 forced by the data,
not chosen, and noted wherever the two are compared.

### 11.5 v3 training: loss rose again, for the third time and the same reason

```bash
python model/train.py --train model/data/scene3_train.jsonl \
  --val model/data/scene3_val.jsonl --out model/scene_ckpt3 \
  --epochs 6 --batch 16 --lr 3e-4 --max-in 160 --max-out 256 --val-cap 200
```

**8 CPU threads, 2250 steps, `train_runtime` 548.5 s. No GPU at any point
except the critic's YOLO pass.**

| | v1 | v2 | **v3** |
|---|--:|--:|--:|
| final eval_loss | 0.9240 | 0.9470 | **1.0042** |
| train_runtime | 530 s | 531.7 s | **548.5 s** |

Loss has risen at every step of this sequence, and every rise bought a
capability: v3's target space has eight possible cast sizes instead of
five and up to eighteen events instead of twelve, raising the entropy of
"the correct answer" without making the model worse. Loss is not the
result and is not quoted as one, for the third time in this project.

### 11.6 The result that reframes the attempt

Reported in full in §5.4: decoding the same checkpoint on the same
prompts under beam search versus sampling moves staging from 0.188 to
0.862 on v3 and 0.333 to 0.845 on v2, with distinct emitted x-values going
from 3 to 55 on v3. **The position collapse Attempt 22 attributed to the
model was mostly the decoder.**

### 11.7 The four-cell ablation: model, decoder, or a pass that fixes it afterwards

All cells: the same 150 in-distribution and 24 out-of-distribution
prompts, `beam4`. `+uc` = the colour processor.

**In distribution (150 prompts):**

| | v2 | v2+uc | v3 | v3+uc |
|---|--:|--:|--:|--:|
| parse | 100.0 | 100.0 | 98.7 | 98.0 |
| **valid, no repair** | 40.0 | **72.0** | 56.0 | **80.7** |
| **duplicate-colour problems** | **71** | **0** | **46** | **0** |
| actions per character | 88.7 | 90.0 | 85.8 | 85.7 |
| layout (stated spatial fact obeyed) | n/a | n/a | 44.3 | 45.8 |
| **ALL, layout excluded (like-for-like)** | **86.0** | **88.0** | **85.8** | **85.7** |
| **staging** | 0.360 | 0.363 | **0.178** | 0.182 |
| **staging, + spacing pass** | **0.670** | 0.670 | 0.587 | 0.585 |
| declared min separation | 0.157 | 0.180 | 0.018 | 0.018 |
| declared min separation, + pass | **1.217** | 1.218 | 1.227 | 1.228 |
| min separation over the whole clip | 0.069 | 0.074 | 0.010 | 0.010 |
| min separation over the clip, + pass | 0.348 | 0.343 | 0.238 | 0.245 |
| staging ≥ 0.825, + pass | 19.3% | 17.9% | 17.0% | 15.7% |

*(Source: `ATTEMPTS.md` Attempt 23 §7; `model/eval_out3/compare_indist.txt`.)*
**The honest decomposition of staging, in distribution:** what the model
learned, read by beam search — 0.178; plus a deterministic pass enforcing
it afterward — 0.587; what the model learned, read by sampling instead —
**0.862**; hand-written reference clips — 0.825–1.000. **Under beam
search the deterministic pass does essentially all of the work, and that
is a good engineering result and a bad science claim** — it raises
declared minimum separation from 0.018 to 1.227 by construction (it
cannot fail to) and moves staging to 0.587, but never reaches the
hand-written band, and only 17% of scenes clear the worse reference
clip. Changing the decoder alone reaches 0.862, past that clip, because
it is the model's own answer rather than an imposed one — **the pass is
a floor, not a fix.**

**Two smaller findings inside the ablation, both falsifying part of
RESEARCH2's own stated prediction 2.** The spacing pass fixes *declared*
minimum separation exactly as designed (0.157→1.217) but *over-clip*
minimum separation only reaches 0.069→0.348, because the pass rewrites
the cast section and never touches `walk`/`run` destinations in the
timeline — a character correctly spaced at t=0 can still walk into
somebody. RESEARCH2 §7 prediction 2 said minimum pairwise distance rises
to an enforced floor with no further qualification; **as stated, that is
falsified** — true of the declared positions, false of the clip, because
nothing had measured over-clip separation before this attempt built the
metric to do so. Separately, v3's 98.7% parse rate (in distribution) is a
length cap, not a format failure: both parse failures end mid-token at
seven- and eight-character casts whose timelines exceed `--max-out 256`
at inference — the same class of bug as the training truncation caught in
§11.4, one layer further out.

### 11.8 Constrained decoding: the cleanest result in the attempt

| | v1 (Attempt 22) | v2 | **v2 + processor** | v3 | **v3 + processor** |
|---|--:|--:|--:|--:|--:|
| in-dist duplicate-colour problems | 28/150 | 71/150 | **0/150** | 46/150 | **0/150** |
| in-dist valid, no repair | 68.7% | 40.0% | **72.0%** | 56.0% | **80.7%** |
| OOD duplicate-colour problems | — | 7/24 | **0/24** | 6/24 | **0/24** |
| OOD valid, no repair | 62.5% | 54.2% | **58.3%** | 50.0% | **70.8%** |

*(Source: `ATTEMPTS.md` Attempt 23 §8.)* **The constraint is enforced
exactly: 0 duplicate colours in 348 generated scenes across four cells,
by construction rather than by rate.** The valid-without-repair rate
Attempt 22 reported as its worst regression (68.7%→40.0%, v1→v2) is not
only recovered but beaten: **80.7%** on v3 with the processor, the
highest raw validity rate this project has recorded at that point.
RESEARCH2's own prediction — duplicates go to 0 with zero change to any
other category — is **half right**: the count goes to exactly 0/150 and
0/24 as predicted, but masking a token shifts the beam path, moving v2's
`ALL` up (86.0→88.0) and v3's parse rate down (98.7→98.0, one extra
length-cap truncation). "Orthogonal" was the right intuition and the
wrong word: the constraint is exact, its side effects are real but
second-order.

### 11.9 What sampling costs, on all 174 prompts

Both models were re-evaluated end to end with `do_sample=True, top_p=0.95,
temperature=1.0`, colour processor on, same 150 + 24 prompts as every
other cell.

| in distribution, v3 | beam4 | sampled | delta |
|---|--:|--:|--:|
| valid, no repair | **80.7%** | 60.7% | **−20.0** |
| actions per character | **85.7%** | 72.5% | **−13.2** |
| layout | **45.8%** | 25.3% | **−20.5** |
| **staging** | 0.182 | **0.875** | **+0.693** |
| **scenes clearing 0.825** | **0.0%** | **74.3%** | **+74.3** |

*(Source: `ATTEMPTS.md` Attempt 23 §9.)* **In distribution, sampling buys
0.693 of staging for 13–20 points of obedience**, concentrated exactly in
the fields the prompt *does* specify (`actions_per_character`, `layout`);
`cast_count`, read from a numeral rather than a spatial judgement, does
not move (100.0% either way). **Out of distribution it is not a trade at
all** — `ALL` stated fields rises 70.8%→75.0% at the same time staging
rises 0.099→0.804, because on unfamiliar input the beam-search mode was
never a better answer, only a more confident one. **`valid` is where
sampling genuinely costs something real**: 80.7%→60.7% in distribution,
because unconstrained sampling occasionally emits a token outside the DSL
entirely (`yellow`, not one of the ten trained colours) that the strict
parser correctly rejects — the strongest argument in this attempt for
extending the colour processor's approach to the whole grammar, per
RESEARCH2 §4's own fallback recommendation once sampling replaces beam
search.

### 11.10 Cast extrapolation: the wall moved from six to nine, exactly as predicted

`model/scene_probe.py`, the Attempt 22 clause probe extended to numeral 9:

| range | v2 (trained 1–5) | **v3 (trained 1–8)** |
|---|--:|--:|
| 1–5, the Attempt 22 range | 14/15 | **14/15** |
| 6–8, new in the v3 range | **0/15** | **15/15** |
| 9, one step past the trained ceiling | 0/5 | **0/5** |

*(Source: `ATTEMPTS.md` Attempt 23 §10.)* v3 answers "nine people" with a
cast of **6, every single time, at every clause count** — a clean
ceiling, not random failure, exactly the pattern v2 showed at 4–5 when
asked for six. On the 24 Attempt 22 out-of-distribution prompts, the
`cast_size` group goes 0/3 (v1) → 0/3 (v2) → **3/3 (v3)**, and OOD
`cast_count` goes 33.3% → 83.3% → **95.8%**. **RESEARCH2's prediction 3 is
confirmed in both halves** — sizes inside the new range reach full
parity, and the size one step past the new ceiling reproduces the same
saturation pattern rather than failing randomly — and per RESEARCH2 §3's
own reading of the compositional-generalization literature (SCAN/GECA:
what fixes systematic generalization is recombination inside or near the
trained distribution, not extrapolation past it), **we report this as
confirming a known general seq2seq limitation, not as a defect specific
to this project.** Training on 1–12 would, by the same pattern, move the
wall to 13.

### 11.11 The spatial supervision did not work, and the control is what shows it

v3 obeys the spatial fact its own prompt states in 44.3% of in-distribution
scenes — a number that reads like a capability until it is compared
against a control. `model/scene_layout_baseline.py` scores each prompt's
spatial facts against a **different** prompt's emitted scene, averaged
over every mismatched pair: with a cast of two, "leftmost" is true about
half the time whatever the model does.

| cell | n | layout obeyed | mismatched-pair chance | **lift** |
|---|--:|--:|--:|--:|
| v3 beam4, in distribution | 97 | 44.3% | 41.7% | **+2.7** |
| v3 sampled, in distribution | 99 | 25.3% | 27.3% | **−2.0** |
| v3 beam4, hand-written spatial OOD | 12 | 50.0% | 49.2% | +0.8 |
| v3 sampled, hand-written spatial OOD | 12 | 58.3% | 34.8% | +23.5 |
| v2 beam4, hand-written spatial OOD (control — never trained on spatial language) | 12 | 58.3% | 45.5% | **+12.9** |

*(Source: `ATTEMPTS.md` Attempt 23 §11.)* **In distribution the lift is
+2.7 points and, under sampling, −2.0: the model is not obeying the
spatial instruction, it is producing geometry that happens to satisfy it
at the base rate**, despite 61.2% of v3's training prompts stating a true
spatial fact. The one apparently positive cell (v3 sampled on
hand-written spatial prompts, +23.5) does not survive its own control:
v2, which never saw a spatial phrase in training, scores +12.9 on the
identical prompts. With n=12 and a never-trained model reaching half the
lift, that cell is noise. **This falsifies half of the attempt's own
prediction 3** (RESEARCH2 §7): the joint position sampling half
demonstrably worked (§11.6, §11.9), the spatial-language half bought no
measurable obedience — likely because binding an ordinal phrase to a
specific cast member is a harder problem than reading a numeral, and
3,673 training examples on a 60M model trained for nine minutes was not
enough of it. **Every staging result in this attempt comes from the
decoder and the position sampler, not from the prompt following spatial
instructions.**

### 11.12 Sample-and-rerank: the configuration that keeps both

`SceneWriter(rerank=4)` draws four samples and keeps the first with the
fewest structural problems, tie-breaking on draw order (not staging
score) so the staging number stays an unbiased read rather than a
selected one.

| in distribution | beam4 | sampled | **sampled + rerank(4)** |
|---|--:|--:|--:|
| **valid, no repair** | 80.7% | 60.7% | **94.0%** |
| actions per character | **85.7%** | 72.5% | 72.7% |
| **staging** | 0.182 | 0.875 | **0.879** |
| **scenes ≥ 0.825** | 0.0% | 74.3% | **75.2%** |

| out of distribution | beam4 | sampled | **sampled + rerank(4)** |
|---|--:|--:|--:|
| **valid, no repair** | 70.8% | 41.7% | **83.3%** |
| **staging** | 0.099 | 0.804 | **0.890** |
| **scenes ≥ 0.825** | 0.0% | 65.0% | **80.0%** |

*(Source: `ATTEMPTS.md` Attempt 23 §12.)* **Reranking recovers everything
sampling broke except action fidelity, at the cost of 4× inference** (a
forward pass measured at 2.7 s): `valid` reaches 94.0% in distribution and
83.3% out of it, both the highest this project has recorded, against v1's
68.7% baseline. Staging is essentially untouched (0.875→0.879), which is
the point of tie-breaking on draw order rather than staging score — the
rerank selects structurally clean scenes, and they happen to stage just
as well, rather than selecting for staging directly. **What rerank does
not recover is `actions_per_character`** (85.7% under beam4 against
72.7% here) — expected, since the rerank criterion is structural
validity, orthogonal to whether the right character waved. **There is no
single best configuration**: beam4+colour-processor maximizes
in-distribution obedience (ALL 57.1%, actions 85.7%, staging 0.182);
sampled+rerank+colour-processor maximizes composed appearance (staging
0.879, valid 94.0%, actions 72.7%) — the two differ by 0.70 of the
staging metric on identical weights, and the paper states which is used,
every time.

### 11.13 The deliverables

```bash
python model/scene_infer.py \
  "two friends meet in the park, one waves, then they kick a ball around" \
  --ckpt model/scene_ckpt3 --max-in 160 --max-out 256 \
  --sample --unique-colours --rerank 4 --seed 23 -o scenes/a23_generated.scene
```

The same prompt Attempt 22 used, directly comparable:

| | Attempt 22 `a22_generated.mp4` | **Attempt 23 `a23_generated.mp4`** |
|---|---|---|
| cast positions | 9.6 and 9.2 | **7.9 and 3.1** |
| minimum separation | 0.4 | **1.385** |
| span of the usable stage | 15.6% | **29.6%** |
| occlusion fraction | 0.344 | **0.000** |
| **staging** | **0.332** | **0.886** |
| critic warp error (pass ≤0.06) | 0.0022 | 0.0014 |
| critic motion coverage (floor 0.2%) | 2.03% | 1.49% |
| identity worst | 1.00 | 0.9999 |
| CLIPSIM | 0.297 | 0.3177 |
| **critic verdict** | USABLE | **USABLE** |
| cast audit | 2/2 visible, 55/55 frames | **2/2 visible, 82/82 frames** |

*(Source: `ATTEMPTS.md` Attempt 23 §13.)* **Staging 0.886 puts a
model-written scene above `street_relay.scene` (0.825) for the first time
in this project.** Motion coverage is lower (1.49% vs. 2.03%) and this is
not a hidden regression: the clip is longer and the characters stand
further apart, so less of the frame changes per unit time, and it still
clears the critic's 0.2% floor seven-fold.

**Second deliverable: a cast of six**, unreachable by any earlier model in
this project. From "six friends spread out on a night street, all of
them dancing, then two of them kick a ball around" →
`scenes/a23_six.scene` → `out/a23_six.mp4`: 11.3 s, 339 frames, **6
characters**, 2 props, 16 timeline events, zero structural problems, zero
repairs.

| | value |
|---|--:|
| staging | **0.903** |
| span of the usable stage | **89.8%** |
| critic warp error | 0.0077 |
| **critic motion coverage** | **6.82%** |
| identity worst | 0.9993 |
| **critic verdict** | **USABLE** |
| cast audit | **6/6 visible in 68/68 sampled frames** |

**6.82% motion coverage is the highest of any clip in this project**
(`out/a21_park.mp4`, the hand-written reference, is 5.32%), from six
independently animated characters composed in one shot from one English
sentence. *(Source: `ATTEMPTS.md` Attempt 23 §13.)* Both critic runs used
`tools.gpuguard.Guard` at 50% duty (0 cooldowns); generation, rendering
and every model in this attempt are CPU-only — the card is touched only
by the critic's YOLO pass.

### 11.14 Scoring every prediction, including the ones that were wrong

| # | prediction (RESEARCH2 §7) | verdict |
|---|---|---|
| 1 | colour processor: duplicates 71/150→0, nothing else changes | **half right** — 0/150 and 0/24 exactly, but masking shifts the beam path (v2 `ALL` 86.0→88.0, v3 `parse` 98.7→98.0) |
| 2 | spacing pass: minimum pairwise distance rises to an enforced floor | **falsified as stated** — true of declared positions (0.157→1.217), false of the clip (0.069→0.348), because the pass never touches walk destinations |
| 3a | staged position sampling: v3 stages well above v2 without any pass | **right, invisible under beam search** — 55 distinct x under sampling vs. v2's 20; *worse* under beam4 (0.182 vs 0.363) because a wider distribution flattens the decoding peak further |
| 3b | spatial language: a measurable rate of obeying a stated instruction | **falsified** — +2.7 points over a mismatched-pair control in distribution, −2.0 under sampling; the 44.3% headline is the base rate |
| 4 | cast range 1–8: 6–8 read correctly, 9 saturates | **confirmed in both halves** — 15/15 inside range, cast of 6 every time at 9 |
| 5 | `d_model` 512 rules out an embedding-width bottleneck | **confirmed** against the actual `config.json` |
| — | *(not predicted, found)* | **the position collapse is mostly a decoding artefact** — 0.188→0.862 on identical weights |

Two of five stated predictions were falsified, in whole or in part, and
are reported that way rather than reframed as successes after the fact.

### 11.15 Where this attempt and RESEARCH2 disagree

RESEARCH2 ranks the deterministic spacing pass second among its
recommendations and does not consider the decoding strategy as a
candidate cause at all; its §2 diagnosis of "conditional mode-collapse"
was the right read of the *distribution* and, absent a decoding-strategy
control, the wrong read of the *symptom* — the measured cause is that
beam search reports the mode of a distribution that turned out to be
adequate, and the fix was one argument to `generate()`, not a factor
graph. We state this as a limitation of the diagnostic tools available
when the survey was written, not a criticism of the survey: nothing in
the project's log at that point had measured staging at all, so nothing
distinguished "the model collapsed" from "the search collapsed it."
Separately, RESEARCH2 §4's recommendation to hand-write the one
constraint needed and hold Outlines/Guidance as a fallback was followed
and, per §11.9, the fallback is now worth reconsidering — once sampling
replaces beam search, unconstrained decoding emits tokens outside the DSL
grammar entirely, which a single hand-rolled processor does not cover.

### 11.16 What is honestly not solved

The model does not obey spatial instructions above the base rate (§11.11).
Cast size saturates at 6 when asked for 9 — a relocated wall, not a
removed one, consistent with a known general seq2seq limitation (§11.10).
Beam search and sampling are 0.70 apart on staging and 13 points apart on
action fidelity on identical weights, and no single decoding configuration
is best across every metric (§11.9, §11.12) — the pipeline currently makes
the user choose. `valid` under sampling is carried by rerank (4 forward
passes to launder one), not by the model directly. The spacing pass does
not touch the timeline, so a scripted `walk to` can still undo it. The
staging metric itself is calibrated on n=13 clips (2 hand-written, 11
model-written) and is measurably weaker above 4 characters, where a
single bad pair is diluted across many pairwise terms — `a23_six` has
three characters within 0.2 units of each other and still scores an
occlusion fraction of only 0.067. Inference truncates large casts (v3
parse 98.7% is entirely 7–8-character scenes exceeding `--max-out 256`, a
cap, not a capability limit). This is one seed on one architecture; the
decoder result is far too large to be seed noise, but the 24- and
12-prompt out-of-distribution sets make per-group numbers indicative, not
tight. Every limitation Attempts 21 and 22 listed that this attempt did
not touch still stands: foot slide, no collision, actions cannot blend
across a clip boundary, canonical character names, unmeasured pacing, and
CLIPSIM's inability to judge this domain at all.

## 12. Answering the Project's Own Indictment

The project's standing self-criticism, stated identically at the end of
Era 1 and again at the end of Attempt 21, is: **"the models never beat the
script."** Attempt 22 answers it directly, and both halves of the answer
are reported, not just the favorable one.

| | hand-written (Attempt 21) | model-written (Attempt 22) |
|---|---|---|
| clip | `out/a21_park.mp4` | `out/a22_generated.mp4` |
| author | a human, by eye, over several passes | t5-small, 60M params, one forward pass |
| verdict | **USABLE** | **USABLE** |
| warp error | 0.0055 | **0.0022** |
| motion coverage | **5.32%** | 2.03% |
| cast | 3 | 2 |
| props | 7 | 2 |
| timeline events | 8 | 4 |
| duration | 12.5 s | 9.1 s |
| staging | positions chosen by hand, two moved after looking at the frames | two characters at x=9.6 and x=9.2, stacked at the right edge of a 10-unit frame |

*(Source: `ATTEMPTS.md` Attempt 22, §9 "Attempt 4's indictment, answered
with a number.")* **The straight answer: no. The model did not beat the
hand-written script — it reached the same verdict with a thinner scene.**
Both clips are USABLE and the generated one has a *lower* warp error, but
warp error rewards stillness, and the honest reading of 0.0022 against
0.0055 is "less is happening," not "better." The row that matters is
motion coverage (2.03% against 5.32%) and the row under it: the human's
scene has three characters, seven props and eight events against the
model's two, two and four, and the human's characters are spread across
the frame because a human looked at a frame and moved them, while the
model's are stacked at one edge because the prompt never said where anyone
should stand.

**What has changed, and it is not nothing:** the sentence "100%
hand-written" is now false — a natural-language prompt the model had never
seen produced a valid, unrepaired scene script that rendered to a USABLE
video with no human in the loop. 90.7% of held-out prompts are satisfied
completely (v1, in-distribution; 86.0% for v2), with 100% parse and 100%
render, and 0% exact-match — the model is not reproducing scripts, it is
writing new ones. 9 of 10 generated clips clear the critic, and the tenth
fails a motion floor for correctly doing what its prompt asked.

**What has not changed:** the best-*looking* output of this project is
still `out/a21_park.mp4`, and it is still hand-written. The model now
writes scripts that are structurally correct and semantically obedient; it
does not yet write scripts that are well *staged*, and staging is what
made the Attempt 21 clip look like a scene rather than merely be a correct
one. The project's indictment is no longer true as originally written, and
its spirit still stands.

**Attempt 23 updates this comparison on exactly one axis, and it is worth
stating precisely rather than folding it back into a cleaner story.** With
a corrected decoder (sampling plus rerank, not the beam search this
section's table used), the same prompt now produces `out/a23_generated.mp4`
at staging **0.886** — above `street_relay.scene` (0.825), the weaker of
the project's two hand-written reference clips, for the first time in this
project's history (§11.13). This is a real change to "the model does not
stage well," which Attempt 22 treated as settled. It is not a change to
the rest of the table above: the Attempt 23 deliverable still has 2
characters, 2 props and 4 events against the hand-written park scene's 3,
7 and 8, and — the sharper point — the same attempt's own control shows
the model's apparent obedience to stated spatial instructions is within
noise of a mismatched-prompt baseline (+2.7 points, §11.11). **The model
now composes a well-spaced scene; it does not yet take direction about
where to place anyone.** Attempt 23's own closing sentence states this
distinction as precisely as we can: "this model composes, and does not
yet take direction."

## 13. Limitations

**Arc one (diffusion route) is closed, not merely unfinished.** No
configuration of SD1.5 + ControlNet + AnimateDiff tried in this project
reaches the evaluation harness's USABLE verdict; the mechanism that helps
flicker measurably costs pose obedience, and the mechanism that helps
identity measurably costs a little flicker back. We do not expect further
tuning of this exact stack to resolve this on the reported hardware — the
finding is that the objectives trade against each other, not merely that
none was tuned enough.

**Arc two (symbolic composition) does not yet compose style or take
spatial direction.** `scenescript.py` produces line-art stick figures, not
styled anime characters — this project has not combined arc two's
composition machinery with arc one's diffusion styling, and doing so is
unattempted. Cast size saturates at 9 for the current model and training
range (1–8) — a wall relocated by exactly the width of the range
extension, not removed, and consistent with a documented general limit of
sequence-to-sequence counting/extrapolation (RESEARCH2 §3) rather than a
defect specific to this system (§11.10). Duplicate character colours are
fully eliminated by a constrained-decoding logits processor (0/150, 0/24,
0 across 348 generated scenes, §11.8) — this one wall is closed, not
merely patched. **The model does not obey stated spatial instructions
above the rate a mismatched control achieves** (+2.7 points in
distribution, §11.11) — every staging improvement in this paper comes from
the decoding strategy and the position sampler, not from the model reading
"leftmost" or "close together."

**Getting a well-staged scene costs something, and the cost is stated
plainly.** Sampling instead of beam search buys 0.693 of the staging
metric at a cost of 13–20 points of in-distribution obedience
(`actions_per_character` 85.7%→72.5%, `layout` 45.8%→25.3%), concentrated
exactly in the fields the prompt specifies; sample-and-rerank recovers
structural validity (60.7%→94.0%) but not action fidelity (72.7%, still
13 points below beam search's 85.7%, §11.9, §11.12). There is no single
decoding configuration that is best on every axis, and the pipeline
currently requires choosing one per use case rather than offering one that
dominates.

**The staging metric itself is a project-defined instrument, calibrated
on a small sample.** Its separation between hand-written and model-written
scenes is complete (n=13: 2 hand-written, 11 model-written, no overlap),
but n=13 is small, two of its five candidate sub-terms were discarded
during calibration for disagreeing with the reference clips (§11.2), and
it is measurably weaker above four characters, where a single close pair
is diluted across many pairwise terms (§11.16). It has not been validated
against human judgment of what "well staged" means, which is a real gap
(see the user-study limitation below).

**The deconfounding fix (Attempt 22) trades 4 points of in-distribution
accuracy for 19.5 points of out-of-distribution improvement, and this
trade is reported, not minimized.** The remaining 23.5-point gap after
deconfounding is not fully understood — the `valid` rate's own regression
(68.7%→40.0%) shows the fix is not free even on the axis it targeted.

**This paper reports no user study, and no comparison against a
competing system.** All quality judgments in this paper are made by this
project's own automated critic (§4) and staging metric (§11.2), both
built and calibrated internally, and by close reading of individual clips
and generated strings — never by an independent human rater blind to
which clip came from which condition, and never against another
published script-to-animation system run on the same prompts. RESEARCH2
§5 found no other system in this specific niche (sub-100M parameters,
CPU-trainable, script-to-multi-character-animation) to compare against as
far as that survey's search found, which explains the absence of a
system-level baseline but does not substitute for one; a human preference
study comparing hand-written, beam-search, sampled, and reranked outputs
is the most direct way to close this gap and was not run here.

**Every rate in Attempt 23 beyond the headline decode probe is measured
on a fixed set of 150 in-distribution and 24 out-of-distribution prompts,
on one seed, one architecture (t5-small), one training run per
checkpoint.** The decoder result itself (0.188 vs. 0.862 staging on
identical weights) is far too large a gap to be seed noise, but the
24-prompt (and, for the spatial-language control, 12-prompt) out-of-
distribution sets make per-group numbers indicative rather than tight,
and no result in this paper has been replicated across multiple training
seeds.

**Future work, stated with the reasoning rather than as an unexplained
gap:** a DETR-style Hungarian set-decoder head (Carion et al., ECCV 2020)
for the cast section is, per RESEARCH2 §4, the one surveyed technique
that structurally addresses staging and duplicate colours together — N
cast slots predicted with mutual attention would make a repeated colour
visible to the model during generation itself, and a repulsion term over
the predicted positions would attack staging in the same head. It is the
only technique surveyed requiring genuine architecture work (a new
decoder head, a new order-invariant matching loss) against this project's
"minutes of CPU, no architecture change" retraining budget, and it was
deliberately not built: a one-argument decoding change (sampling) and a
cheap rerank pass closed most of the measured gap without it (§11.6,
§11.12), so it is no longer the obvious next move, but it remains the
principled fix if extending the constrained-decoding approach to the full
DSL grammar (also future work, per §11.9) does not close what remains of
the `valid` and action-fidelity gaps under sampling.

**Pose conditioning obedience remains partial after Attempt 19's
corrections and then gets strictly worse under the Attempt 20 motion
module** (§7.1, §8.4) — an obeyed guide pose falls from 1 of 4 inspected
frames to 0 of 8 once temporal attention is added.

**Identity preservation via IP-Adapter (§8.3) is measurably improved but
only on a narrow, stated axis** — colour scheme, not structural identity,
per the critic's own HSV-histogram embedding and the anchor run's contact
sheet.

**No off-the-shelf pose or person detector can currently score this
project's styled, symbolic, or multi-character output** (§5.3, §7.3,
§9.5) — a real, unresolved measurement gap in the available tooling for
2D line art of any kind, not specific to any one clip.

**Content-correctness evaluation remains asymmetric across the project's
output paths** (§4.5) — exact for the symbolic AniSVG path, absent by
construction for the diffusion-rendered path, and not yet extended to
Attempt 21/22's composed scene scripts even though they are, like AniSVG,
a declarative source that could in principle be checked exactly.

**FVD, content-debiased FVD, JEDi, and FVMD are not computed anywhere in
this project** because the clip counts available (dozens per condition) do
not meet the sample sizes the published literature uses for a stable
Fréchet estimate; this paper reports warping-error-style and flicker-ratio
metrics instead, per the survey's own recommendation.

**Several numbers cited in Related Work (§2) are marked unverified in the
source survey itself** and are reported only as context, not as claims
this project has confirmed locally.

**Reproducibility caveat:** all GPU experiments ran under a 50% duty-cycle
guard for thermal reasons; numbers measured on undamaged or unthrottled
hardware are not expected to be directly comparable on wall-clock terms,
though the dimensionless ratios (flicker, PCK, prompt-match rates,
staging) should reproduce. Attempts 21–23 use no GPU except the critic's
own YOLO pass and are therefore not subject to this caveat for their own
generation and training — a property we consider part of their result,
not a footnote (§9, §10.2, §11.5).

## 14. Conclusion

Twenty-three logged attempts split this project's account into two arcs,
and both are reported honestly rather than as one continuous march toward
a working system. The first arc closes: Stable Diffusion 1.5, ControlNet
and AnimateDiff, on this hardware and with the corrections this project
applied, reduce flicker for the first time in the project's history
(36.2×→25.0×) and still cannot clear the evaluation harness's own bar, and
the mechanism that helps flicker measurably costs pose obedience while the
mechanism that helps identity measurably costs flicker back. That
interference — **flicker, identity and pose fidelity behave as separable
defects on this hardware and model family, not sub-problems of one shared
deficiency** — is this paper's first central finding, and it rests on
evidence spanning three attempts: Attempt 19 showed pose obedience and
flicker orthogonal even with no temporal mechanism present; Attempt 20
supplied the temporal mechanism and traded it directly for pose; and
underlying both, Attempt 19's proportion finding shows the very choice
that makes a character read as anime is the choice that removes its pose
guide from a human-pose model's training distribution.

The second arc opens, and gets further than the first, in three steps. A
CPU-only symbolic compositor (Attempt 21) produces this project's first
multi-character animation and clears the evaluation harness's temporal
gates by a wide margin. A 60M-parameter model (Attempt 22) trained in
under nine minutes on a CPU then learns to write scripts for it —
generalizing at the format level essentially perfectly and failing
compositionally in one diagnosable way. That diagnosis — tracing a
65.7-point generalization gap to a spurious correlation the project put
into its own synthesized training data, confirming it with a controlled
probe, fixing it at the data level, and reporting the resulting trade
honestly — is this paper's second central finding, and we hold it up as a
general methodological point: **a small model trained on synthesized data
can look incapable of a whole class of reasoning when the truer fault is
what its training data ever asked it to learn**, and this is
distinguishable, by a controlled probe, from a genuine capability limit.

**A third attempt (Attempt 23) then does the same thing to the project's
own diagnosis of itself, and this is the paper's third central finding.**
Attempt 22 named staging as the model's biggest remaining defect;
decoding the identical checkpoint on identical prompts under sampling
instead of beam search moves staging from 0.188 to 0.862 on one
checkpoint and 0.333 to 0.845 on another, and the number of distinct
positions emitted across thirty scenes goes from 3 to 55. The defect the
project had attributed to the model was mostly a property of the search.
A companion ablation shows precisely what would have been believed
without this check: a deterministic spacing pass looks, on its own, like
a successful fix (staging 0.178→0.587), and only comparison against
decoding alone (0.178→0.862, no pass) reveals that beam search was
suppressing an already-adequate distribution the whole time — *good
engineering, bad science claim*, stated as the project states it. This
completes a pattern rather than standing alone: this project has now
overturned its own first explanation of an apparent model failure three
times — a topology bug that was fixed and changed nothing until a
proportion mismatch was found by controlled ablation (Attempt 19), a
"cannot count" verdict that a clause-count probe traced to a 99.9%
data confound (Attempt 22), and now a "cannot stage" verdict traced to a
decoding default (Attempt 23) — and we report the pattern of
self-correction itself as a contribution, alongside the individual
findings it produced.

Running through all three arcs is a fourth contribution we take as
seriously as any of them: this project's own evaluation harness and
decoding pipeline produced four false or unmeasurable verdicts along the
way — a content-correctness metric that could not separate correct
geometry from known-bad geometry, an identity metric that silently mixed
two measurement bases and cost one real clip its verdict twice, person
detectors blind to every stick figure this project has drawn, and a
decoder that reported the mode of a fine distribution as if it were the
distribution itself — each caught and corrected in the open. The
underlying lesson generalizes beyond any one of them: a metric's or a
decoding default's provenance bounds what it can be trusted to measure or
produce, and this bound must be checked against a control, never assumed.

We close by answering the project's own standing indictment precisely,
not by resolving it in either direction. The model still does not beat
the hand-written script on richness: Attempt 22's generated scene is
thinner than the human-authored one (2 characters, 2 props, 4 events
against 3, 7, 8), and that comparison is unchanged by anything in this
paper. What has changed, twice, is real and stated exactly. First: the
claim that this project's best output is "100% hand-written" is false,
because an unseen sentence produces a valid, unrepaired script that
renders to a video the evaluation harness calls USABLE, with no human in
the loop (Attempt 22). Second: with a corrected decoder, a model-written
scene now *stages* better than the weaker of the project's two
hand-written reference clips (0.886 vs. 0.825) — composition, specifically
— while the same attempt's own control shows the model still does not
reliably obey a stated spatial instruction above the rate a mismatched
control achieves (Attempt 23). The model composes; it does not yet take
direction. We take the position stated in the introduction: on hardware
this constrained, a correctly diagnosed and measured result — a closed
route, an open one, a defect traced to its actual cause, or a defect
traced away from where the project first placed it — is a more useful
contribution than an unmeasured or overclaimed one, and we have tried
throughout, including about our own project's prior conclusions, to write
this paper that way.

## References

Full citation strings are as recorded in `paper/RESEARCH.md` and
`paper/RESEARCH2.md`; reproduced here without alteration or addition.
Attempt 21 introduces no new external citations in `ATTEMPTS.md` and none
are added here.

- Guo, Y., et al. "AnimateDiff: Animate Your Personalized Text-to-Image
  Diffusion Models without Specific Tuning." arXiv:2307.04725, 2023.
- Lin, S. & Yang, X. (ByteDance). "AnimateDiff-Lightning: Cross-Model
  Diffusion Distillation." arXiv:2403.12706, 2024.
- Khachatryan, L., et al. "Text2Video-Zero." ICCV 2023 Oral,
  arXiv:2303.13439.
- Yang, S., et al. "FRESCO: Spatial-Temporal Correspondence for Zero-Shot
  Video Translation." CVPR 2024.
- Yang, S., et al. "Rerender-A-Video." SIGGRAPH Asia 2023.
- Geyer, M., et al. "TokenFlow." arXiv:2307.10373, 2023.
- "PipeFlow." arXiv:2512.24026.
- "Sparse Forcing." arXiv:2604.21221.
- "LiteAttention." arXiv:2511.11062.
- "Sparse VideoGen." ICML/NeurIPS 2025.
- "HyperVAttention." arXiv:2607.03012.
- "Animate-X" (cites AnimateAnyone, MagicAnimate). arXiv:2410.10306.
- "MagicAnimate: Temporally Consistent Human Image Animation using
  Diffusion Model."
- Lin, H., Zala, A., Cho, J., Bansal, M. "VideoDirectorGPT." COLM 2024,
  arXiv:2309.15091.
- "ViMax." github.com/hkuds/vimax.
- Wan-AI. "Wan2.1-T2V-1.3B." Hugging Face: Wan-AI/Wan2.1-T2V-1.3B-Diffusers.
- "CogVideoX-2B."
- Wu, R., et al. "AniClipart." IJCV 2024, arXiv:2404.12347.
- Zhu, Yang, Zheng, Zhang, Gao, Huang, Chen. "Vector Sketch Animation
  Generation with Differentiable Motion Trajectories." CGF.
- "Sakuga-42M." arXiv:2405.07425.
- "AnimeRun." arXiv:2211.05709.
- "ATD-12K."
- "CreativeFlow+."
- "LottieAnimation-660K." LottieGPT, Hugging Face.
- "HumanML3D." github.com/EricGuo5513/HumanML3D.
- "AMASS."
- "Motion-X / Motion-X++." arXiv:2501.05098.
- "MoMask" (cited via Light-T2M, arXiv:2412.11193).
- "Light-T2M." arXiv:2412.11193.
- "TriC-Motion." arXiv:2602.08462.
- Tevet, G., et al. "MDM: Human Motion Diffusion Model."
- "MotionGPT." OpenMotionLab, NeurIPS 2023.
- "On the Content Bias in Fréchet Video Distance." CVPR 2024.
- "Content-debiased FVD." github.com/songweige/content-debiased-fvd.
- "JEDi: Beyond FVD." github.com/oooolga/JEDi.
- Hessel, J., et al. "CLIPScore." EMNLP 2021.
- "GODIVA." arXiv, 2021.
- "VQAScore / t2v_metrics." github.com/linzhiqiu/t2v_metrics.
- "FVMD: Fréchet Video Motion Distance."
  github.com/DSL-Lab/FVMD-frechet-video-motion-distance.
- "EvalCrafter." arXiv:2310.11440.

**From `paper/RESEARCH2.md` (staging, counting, and constrained-decoding survey):**

- Andreas, J. "Good-Enough Compositional Data Augmentation (GECA)." ACL
  2020, aclanthology.org/2020.acl-main.676.
- Yehudai, A., Kaplan, G., Ghandeharioun, A., Geva, M., Globerson, A.
  "When Can Transformers Count to n?" arXiv:2407.15160, 2024.
- Cho, H., Cha, J., Awasthi, P., Kalyanaraman, S., Gollakota, A., Yun, C.
  "Position Coupling." NeurIPS 2024, arXiv:2405.20671.
- "Constrained Layout Generation with Factor Graphs." arXiv:2404.00385.
- Ivgi, M., Benny, Y., Ben-David, A., Berant, J., Wolf, L. "Scene Graph to
  Image Generation with Contextualized Object Layout Refinement."
  arXiv:2009.10939.
- Elfeki, M., et al. "GDPP: Learning Diverse Generations Using
  Determinantal Point Processes." ICML 2019, arXiv:1812.00068.
- Jiang, Z., et al. "LayoutFormer++." CVPR 2023, arXiv:2208.08037.
- Kong, X., et al. "BLT: Bidirectional Layout Transformer." ECCV 2022,
  arXiv:2112.05112.
- "Grammar-Aligned Decoding (ASAp)." Park, Wang, Berg-Kirkpatrick, et al.,
  NeurIPS 2024, arXiv:2405.21047.
- "ABS: Enforcing Constraint Satisfaction on Generated Sequences via
  Automata-Guided Beam Search." arXiv:2506.09701, 2025.
- Carion, N., et al. "End-to-End Object Detection with Transformers
  (DETR)." ECCV 2020, github.com/facebookresearch/detr.
- "GBNF / llama.cpp grammars." github.com/ggml-org/llama.cpp.
- "Outlines." github.com/dottxt-ai/outlines.
- "Guidance." github.com/guidance-ai/guidance.
- "AniMaker." SIGGRAPH Asia 2025, arXiv:2506.10540.
- "AniME." arXiv:2508.18781.
- "Plan-and-Write: Structure-Guided Length Control for LLMs without Model
  Retraining." arXiv:2511.01807.
- "PositionID: LLMs can Control Lengths, Copy and Paste with Explicit
  Positional Awareness." arXiv:2410.07035.
- "How Panel Layouts Define Manga." arXiv:2412.19141.
- Comics-understanding survey. arXiv:2409.09502.
