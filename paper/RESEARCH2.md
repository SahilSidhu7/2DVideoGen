# Research survey 2 — closing the staging, counting, and global-constraint walls

Written to unblock Attempt 22 (t5-small scene-script model; parse/render 100%/100% in and
out of distribution; staging collapse, cast-size saturation at 5-6, and 71/150 duplicate-colour
casts are the three measured, standing defects). Scope: web research only, no code changes.
This survey does not repeat `paper/RESEARCH.md` (which covers the now-closed diffusion route,
temporal-consistency metrics, and motion datasets) — it extends the project into the symbolic
scene-script direction that RESEARCH.md predates. Every claim is sourced; anything not pinned to
a source is marked **unverified**.

---

## 1. Executive summary — ranked, adopt in this order

**1. A custom logits processor that tracks "colours used so far" and masks them out when the
decoder is about to emit the next cast member's colour token — wall 3, duplicate colours.**
Expected gain: the measured 71/150 (47%) duplicate-colour rate on v2 goes to 0/150, by
construction, because the constraint (10 colours, ≤5 cast members) compiles to a trivial
finite automaton (state = subset of {0..9} used, ≤2^10 states) — exactly the class of constraint
that automata-guided constrained decoding (grammar-constrained decoding / GBNF / DFA-guided beam
search) is built to enforce exactly, not probabilistically. **Cost: near zero.** No retraining;
implemented as a `LogitsProcessor` in the existing HuggingFace `generate()` call in
`model/scene_infer.py` — the DSL's cast section is fully positional
(`id color x scale ; id color x scale ; ...`), so the processor only needs to know it is at a
colour-token position (parseable from tokens emitted since the last `;`) and which colour tokens
have already appeared. This is smaller in scope than adopting a general library (Outlines/LMQL/
Guidance) and does not require one — write ~40 lines against
`transformers.LogitsProcessor` directly. **Inference-only.**

**2. A deterministic post-decode spacing pass on emitted x-coordinates — wall 1, staging.**
Expected gain: eliminates the measured stacking (two cast members at x=9.6, x=9.2) by solving a
1-D non-overlap/min-spacing problem over the *already-emitted* x values before `spec_to_scene()`
renders them — the same problem "Constrained Layout Generation with Factor Graphs" (arXiv
2404.00385) formalizes as a factor graph with overlap-penalty potentials solved by message
passing, simplified here to 1 dimension (this project's x-axis only; scale is not spatial extent
in the current renderer). **Cost: near zero** — a few dozen lines, closed-form or a handful of
constraint-relaxation iterations, no learned model needed. **Inference-only**, and it composes
with item 1 (both are post-decode repair-style passes, exactly the role `SG.repair()` already
plays in the codebase — this is a natural extension of a function that already exists).

**3. Deconfound cast-size *range*, not just phrasing — extend `scene_synth.py`'s sampler to
occasionally emit cast size 6-8 in training, stated numerically, sometimes with partial
description — wall 2, counting.** This does not "solve" extrapolation in the strict sense (a
model trained on 1-8 that is asked for 12 still faces the identical wall one step further out) —
what it tests is whether the *specific* v1→v2 fix (break the clause-count/cast-size correlation)
continues to generalize once the numeral itself is inside the trained range versus asked to
jump past it. The Attempt 22 log is explicit that 1-5 in training does not extrapolate to 6; the
compositional-generalization literature (SCAN's "jump" split, GECA, meta-seq2seq) backs this up:
what fixed those benchmarks was overwhelmingly **data recombination inside/near the trained
distribution**, not naive scaling, and *true* extrapolation past a trained numeric range remains
an open, largely unsolved problem for seq2seq semantic parsing (see §2.2). Recommended action:
extend the range so the paper's ceiling number is honestly "N-1 in training generalizes to N,
N+1 does not" rather than implying 5 is architecturally special. **Cost: minutes of CPU**
(same 530 s training budget, larger sampled range). **Needs retraining, no architecture change.**

**4. A count-scratchpad prefix in the DSL header — wall 2, counting, second-order fix.**
Add an explicit countdown to the target string itself (e.g. `bg park dur 91 cast5 4 3 2 1 |
cast ana ...`), following the "external memory beats internal counter" finding used by CAPEL
(countdown-aided prompting) and Plan-and-Write (arXiv 2511.01807) for LLM length control, and by
Position Coupling (Cho et al., NeurIPS 2024, arXiv 2405.20671) for arithmetic length
generalization (coupled/task-structured position IDs let a model trained on 1-30-digit addition
generalize to 200-digit addition, 6.67x the trained length — a materially bigger extrapolation
factor than this project needs). The mechanism this project can realistically borrow without a
positional-embedding change is the *scratchpad* half, not the position-ID half (t5-small's
relative-position scheme is a fixed HF component, not something worth hand-modifying on a
530-second training budget). Expected gain: unverified — this is the one item in this ranking
without a directly-transferable published number for *this exact task* (cast enumeration vs.
digit addition), flagged honestly as the riskiest bet of the four. **Needs retraining** (new DSL
target format), **no architecture change**.

**Ruled out of the top ranking, discussed in §2:** DETR-style Hungarian set-prediction cast
head (real fix for both staging and duplicate-colours structurally, but a genuine architecture
change — new decoder head, new loss — bigger than this project's "minutes on CPU" retraining
budget justifies as the *first* move; revisit only if items 1-2 prove insufficient); DPP/GDPP
diversity loss (same category: a training-loss change, not inference-time, and less directly
targeted at this project's exact 1-D x-collapse than a deterministic spacing pass); BLT-style
non-autoregressive layout transformer (architecture replacement, not a patch to the existing
t5-small decoder).

---

## 2. Staging (wall 1)

**The problem, restated from the log:** position (`x`) and `scale` have no supervision signal
from the prompt — the synthesiser's prompts almost never state where anyone stands — so out of
distribution the decoder falls back to the marginal mode of the training distribution and emits
near-identical x values for every cast member (9.6, 9.2 in the shipped deliverable). This is a
**conditional mode-collapse** problem, not a capacity problem: the model has clearly learned to
spread positions in distribution (Attempt 22 §5 shows in-distribution spread is fine), it simply
has nothing to condition on once the input is unfamiliar.

### Candidate techniques

- **Constrained Layout Generation with Factor Graphs** (2024, arXiv 2404.00385, code not
  confirmed public in this pass — **unverified**). Formulates layout constraints (including
  overlap minimization) as factor-graph potentials solved by message passing at inference time,
  decoupled from the generative model itself. Directly transferable: the project's x-axis is
  1-D, so the equivalent potential is just "minimum pairwise spacing + frame margins," solvable
  without belief propagation machinery — a closed-form or few-iteration relaxation suffices.
  **Hardware cost: negligible** (no GPU, no model). **Inference-only.**

- **Scene Graph to Image Generation with Contextualized Object Layout Refinement** (Zhao et al.,
  arXiv 2009.10939). Generates layout **gradually** with each new box refined in the context of
  previously placed ones, reported to raise layout coverage by ~20 points and drop object overlap
  to near zero versus one-shot layout prediction. Relevant as a *principle* — refine positions
  sequentially with visibility of what's already placed — rather than as adoptable code (it is a
  scene-graph-to-image system, not a DSL-to-DSL one). The project's decoder already emits cast
  members sequentially and left-to-right, so in principle it *could* condition on earlier x
  values already emitted in the same sequence (it's autoregressive); the reason it doesn't in
  practice is that OOD it has fallen onto a fixed near-constant output regardless of context —
  this paper's finding argues the fix has to make "already placed here" cheaper for the model to
  attend to than the marginal mode, which post-decode repair sidesteps entirely rather than
  solving in-model.

- **GDPP — Determinantal Point Process diversity loss** (Elfeki et al., ICML 2019, arXiv
  1812.00068, code public: the paper's own repo). An unsupervised penalty added to a
  generator's training loss that explicitly penalizes similarity between generated samples in a
  batch, shown to resist mode collapse on MNIST/CIFAR10/CelebA. Transferable as a **training-time**
  fix: add a repulsion term over the batch of emitted x-coordinates within one scene (pairwise
  distance penalty, DPP-kernel or simply a soft-min-distance hinge loss) to the seq2seq training
  objective. **Cost: retraining required** (same CPU budget, extra loss term), **not
  inference-only** — ranked below the deterministic post-decode pass in §1 because a closed-form
  repair achieves the same measured outcome (no more stacking) without touching the training
  loop or risking degrading the loss that already trains cleanly in 530 s.

- **LayoutFormer++** (Jiang et al., CVPR 2023, arXiv 2208.08037, code:
  github.com/microsoft/LayoutGeneration). Two ideas worth stealing independently of adopting the
  whole system: (1) **constraint serialization** — express constraints as tokens in the input
  sequence, i.e. the DSL itself could gain an optional "no-overlap" directive token, though this
  project's constraint is *unconditional* (always true), so serialization adds nothing here; (2)
  **decoding space restriction** — prune/mask the output distribution at generation time to
  options that don't violate a constraint. This is architecturally the same mechanism as item 1
  in the executive summary (a logits processor), generalized: the same masking mechanism that
  kills duplicate colours can, in principle, also reject an x-token that would land within
  (say) 0.3 units of an already-emitted cast x — cheaper to implement as a hard reject-and-resample
  at generation time than as a full factor-graph solve. Trained on PubLayNet (300K layouts);
  **not directly reusable as weights**, only as a decoding-time technique. **Inference-only**
  for the masking idea; the underlying model itself is not adopted.

- **BLT — Bidirectional Layout Transformer** (Kong et al., ECCV 2022, arXiv 2112.05112).
  Non-autoregressive: predicts a draft layout then iteratively refines low-confidence attributes
  by masking and re-predicting (BERT-style), reported up to 10x faster inference than
  autoregressive layout transformers. Architecturally the cleanest fix for "positions need to see
  each other," since refinement is explicitly bidirectional rather than left-to-right — but this
  is a **different decoder architecture** from t5-small's autoregressive seq2seq decoder, so
  adopting it means training a second, separate model for the cast/position sub-problem and
  splicing its output into the DSL, not a drop-in change. Flagged as the "if the cheap fixes
  don't hold, this is the next tier" option, not adopted now. Param count and VRAM: not stated
  in sources checked (**unverified**), but the model class (transformer over a few dozen layout
  tokens) is comparable in scale to LayoutTransformer/GPT-2-small-class decoders — plausibly
  trainable within this project's CPU budget, but unmeasured.

- **Comics/manga panel-layout research** (surveyed per the task's instruction): the published
  work found (binary-tree panel layouts, template-matching layouts, "How Panel Layouts Define
  Manga," arXiv 2412.19141; the comics-understanding survey, arXiv 2409.09502) is about
  **arranging panels on a page**, not **staging actors within one panel/shot** — i.e. it answers
  a different composition question than this project has. No published system was found that
  addresses within-panel character blocking as a learned or constrained-optimization problem
  specifically for comics/animation; this looks like a genuine gap the project would be filling,
  not a technique to borrow. Noted so the paper's related-work section does not overclaim
  coverage here.

---

## 3. Counting and cast extrapolation (wall 2)

**The problem, restated from the log:** deconfounding clause-count-vs-numeral cut the
generalisation gap from 65.7 to 23.5 points, but cast size 6 (never seen in 1-5 training) fails
0/3 for both models — described in the log as "a hard capability limit, not a phrasing problem."

### Candidate techniques

- **SCAN / COGS / CFQ literature, what actually fixed each.** SCAN's hardest split ("jump",
  primitive generalization) was not fixed by scaling seq2seq models straight — **GECA**
  (Andreas, ACL 2020, aclanthology.org/2020.acl-main.676, code public) — a rule-based data
  augmentation that recombines fragments seen in similar contexts — took a model from "unable to
  make any correct generalizations at all" to "succeeds most of the time" on SCAN's jump/around-
  right splits. COGS and CFQ remained largely unsolved by naive scaling for years afterward
  (multiple surveys found in this pass note "no satisfactory solution" pre-2021/2022); later
  progress came from **learned recombination** ("Learning Algebraic Recombination for
  Compositional Generalization," arXiv 2107.06516) and **meta-learning over synthesized
  demonstrations** ("Improved Compositional Generalization by Generating Demonstrations for
  Meta-Learning," arXiv 2305.13092) — all data/training-procedure interventions, not
  architecture swaps or scale. **Reading for this project:** the class of fix that has ever
  worked for this exact failure mode (systematic generalization to an unseen combination/count)
  is recombination-style data augmentation close to the trained distribution, which is exactly
  the family the project's own v1→v2 deconfounding already belongs to and already validated
  works (25.0%→62.5% OOD prompt-match in the earlier confound-removal pass cited in the log).
  This is direct support for item 3 in the executive summary and a caution against expecting a
  purely architectural trick to close the *true* extrapolation gap (6 unseen, trained on 1-5)
  without also touching the data.

- **"When Can Transformers Count to n?"** (Yehudai, Kaplan, Ghandeharioun, Geva, Globerson,
  2024, arXiv 2407.15160, code: **unverified**). Theoretical result: exact counting is reliably
  learnable when the transformer's embedding dimension is at least as large as the relevant
  vocabulary size; when vocabulary exceeds embedding dimension, weights must scale polynomially
  and the counting mechanism becomes numerically unstable. **Application here:** t5-small's
  hidden dimension is 512 (**unverified for this exact checkpoint — confirm against the actual
  `config.json` in `model/scene_ckpt2`**, but 512 is t5-small's published default), which is far
  larger than the project's tiny closed vocabularies (10 colours, 6 names, 3 backgrounds, 8
  actions) — so this paper's finding predicts the counting wall is **not** an embedding-width
  bottleneck, which rules out "make the model wider" as a fix and points back toward data/decoding
  fixes instead. Useful as a falsifiable prediction (§5), not as a technique to adopt directly.

- **Position Coupling** (Cho, Cha, Awasthi, Kalyanaraman, Gollakota, Yun, NeurIPS 2024, arXiv
  2405.20671, code: **check github link in paper — not directly confirmed public in this pass,
  unverified**). Assigns shared position IDs to task-structurally-equivalent tokens (e.g. same
  digit-significance across operands) instead of raw sequential positions; reported to take
  models trained on 1-30-digit addition to 200-digit addition (6.67x trained length), strongly
  beating index-hinting baselines (which "completely fail starting from length 70"). The
  transferable idea for cast enumeration: give every cast-member's colour/x/scale token triple
  the *same relative offset pattern* regardless of which slot (1st, 2nd, ... 5th) it is in,
  rather than raw absolute decoder-step position — in principle decouples "how far into the
  sequence am I" from "which cast member am I describing." **Cost: requires a custom position-ID
  scheme inside T5's decoder, which is a real code change to the model, not just the data** —
  more invasive than item 3/4 in the executive summary, which is why it is discussed but not
  ranked in the top four; worth a follow-up attempt if items 3-4 do not move the needle on cast
  size 6+.

- **Explicit counter/scratchpad decoding** — CAPEL (countdown-aided prompting, found via this
  pass's search, exact citation/arXiv id **unverified** — described in secondary sources as an
  inference-time prompt-suffix technique for LLMs, not confirmed as a peer-reviewed paper in this
  search) and **Plan-and-Write** (arXiv 2511.01807, "Structure-Guided Length Control for LLMs
  without Model Retraining") — both make the model emit or consume an explicit running counter
  as an external scratchpad rather than relying on an internal, uninspectable count. For a
  seq2seq model with a *fixed, known* target grammar (this project's DSL), the equivalent and
  cheaper move is: put the countdown **in the training targets themselves** (item 4 above) so the
  model learns to copy-and-decrement a token sequence, which is a much easier function to learn
  and extrapolate than "silently track an internal count across 150+ decoder steps." **Cost:
  retraining with a modified DSL**, same order of CPU time as the existing 530 s runs.

- **PositionID** (arXiv 2410.07035, "LLMs can Control Lengths, Copy and Paste with Explicit
  Positional Awareness") — same family as the scratchpad idea, generates an explicit position
  index after every emitted token to make length/position self-tracking learnable; cited here as
  a second, independent confirmation that *explicit* position/count tokens outperform relying on
  the model's implicit sense of "how many have I emitted," which is exactly the project's own
  diagnosis (clause-count confound = the model reading an implicit signal instead of the
  explicit numeral).

- **Memory-augmented counting (ESBN-style) vs. plain transformers** — a PMC-indexed paper on
  numerosity representations in transformers (found via search, exact citation not fully
  resolved in this pass — **unverified**) reports memory-augmented architectures achieving
  "significant extrapolation" on counting tasks where plain transformer baselines "struggled as
  numerosity increased and were incapable of extrapolation beyond the training range." This is
  independent corroboration that the cast-size-6 wall is a documented, general property of plain
  transformer/seq2seq architectures on counting, not something specific to t5-small or to this
  project's data — softens how much the paper should promise on this axis; recommend the paper
  state the wall as "consistent with the published difficulty of transformer count extrapolation"
  rather than as a bug to be fully solved.

---

## 4. Global constraints an autoregressive decoder cannot see (wall 3)

**The problem, restated from the log:** duplicate colours across a cast (71/150 = 47.3% on v2),
overlapping kicks on one ball, events ending after the clip duration — all are whole-output
consistency conditions, currently caught and silently fixed by `SG.repair()`, which means the
raw `valid` rate (40-69%) understates what the model itself gets right.

### Candidate techniques

- **GBNF / llama.cpp grammars** (github.com/ggml-org/llama.cpp/blob/master/grammars/README.md,
  code public, actively maintained). Extended-BNF grammars that mask the vocabulary at every
  decoding step so only grammatical continuations are possible, with automatic JSON-Schema→GBNF
  conversion. **Not the right tool for this project specifically**: it is a `llama.cpp`-ecosystem
  feature (GGUF-format models, C++ runtime), and this project's model is a HuggingFace
  `transformers` T5 checkpoint run directly in Python — adopting GBNF would mean exporting the
  model to GGUF and switching the whole inference stack, which is disproportionate to the
  narrow problem (one duplicate-colour constraint). Ruled out as infrastructure, kept as the
  reference implementation for what a production version of this idea looks like.

- **Outlines** (github.com/dottxt-ai/outlines, code public) and **Guidance**
  (github.com/guidance-ai/guidance, code public) — both integrate as logits processors directly
  against HuggingFace `transformers` models (per this pass's search: "For Transformers-based
  pipelines, Guidance or Outlines integrate as logit processors"), which *is* this project's
  stack. Either could express "no duplicate colour token within the cast section" as a regex/CFG
  constraint. **Cost of adopting either library vs. hand-writing the ~40-line processor described
  in executive-summary item 1: a new dependency for one constraint.** Recommendation: hand-write
  it first (the constraint is small and DSL-specific enough that a general library is overkill),
  keep Outlines/Guidance as the fallback if more constraints accumulate (kick-target validity,
  duration bounds) to the point where hand-writing each one stops being worth it.

- **Automata-guided constrained decoding, general theory** — **ABS: Enforcing Constraint
  Satisfaction on Generated Sequences via Automata-Guided Beam Search** (2025, arXiv 2506.09701,
  code: **unverified**). Generalizes constrained beam search to any constraint compilable to a
  DFA (which includes LTLf and regular expressions), masking transitions that would violate the
  automaton and re-ranking by both model probability and automaton acceptance structure. The
  duplicate-colour constraint ("set of colours used so far has no repeats") is trivially DFA-
  expressible (state = subset of colours seen, transition = colour token, reject transition into
  an already-true bit) — this paper is the formal justification for treating item 1's hand-rolled
  processor as a special case of a well-studied, sound technique rather than an ad hoc hack.

- **Grammar-Aligned Decoding / ASAp** (Park, Wang, Berg-Kirkpatrick et al., NeurIPS 2024, arXiv
  2405.21047, code: check paper — **unverified** public status in this pass). Addresses a real
  subtlety worth flagging to the builder: naive grammar-constrained decoding (mask-and-renormalize)
  **distorts the model's learned distribution** — outputs stay grammatical but their relative
  likelihoods no longer match what the model actually believes, i.e. "grammatical but low
  quality" per the paper's own framing. ASAp instead samples so the output distribution stays
  proportional to the model's true conditional distribution given the constraint. **Relevance
  here:** for a hard, low-cardinality constraint like "don't repeat 1 of 10 colours across ≤5
  slots," simple masking is almost certainly fine (there's no meaningful quality loss from
  refusing an already-used colour — the alternative colours are drawn from the same distribution
  the model would have used anyway) — ASAp is worth knowing about as the more principled fallback
  if a future constraint is added where the *choice among valid options* matters more (e.g. a
  constraint over which action verbs are semantically appropriate), not something this project
  needs for the colour fix specifically.

- **Set prediction / Hungarian matching (DETR-style)** — **End-to-End Object Detection with
  Transformers** (Carion et al., ECCV 2020, code public: github.com/facebookresearch/detr).
  Predicts a fixed number of slots in parallel (no autoregressive ordering among them) and
  matches them to ground truth via the Hungarian algorithm at training time, with the loss
  invariant to prediction order. **This is the structural alternative to sequential emission
  that the task brief is asking about, and it is worth taking seriously as a joint fix for
  walls 1 and 3 together**: replace the cast section's sequential `id color x scale ; ...`
  emission with a small fixed-N (N=5) set-decoder head — N learned query vectors, each producing
  a (colour, x, scale) tuple, matched by the Hungarian algorithm against the ground-truth cast
  (order-free) during training. Two structural benefits fall out for free: (a) since the N slots
  are predicted with mutual attention (all queries attend to the same encoder output and, if the
  head is a small transformer itself, to each other), a duplicate-colour repeat is visible to the
  model *during generation* the way it never is in strict left-to-right decoding — the
  architecture makes the global constraint locally checkable; (b) a repulsion/DPP term (§2) over
  the N predicted x-values is a natural, small addition to the matching loss, attacking staging
  in the same head. **Cost: a genuinely new component** — a set-decoder head grafted onto (or
  replacing) T5's decoder for the cast section specifically, trained with Hungarian matching —
  more engineering than any other item in this document, and the first thing in this survey that
  would need real architecture work rather than a decoding-time or data-level change. Positioned
  in §1 as the "second-tier" item: try the cheap fixes (logits processor + spacing pass) first,
  since they are individually near-free and directly measured against the exact defects logged;
  revisit the set-decoder only if those two do not clear the wall, since it is the only item here
  that trades away the "minutes on CPU, no architecture change" property this project has
  otherwise preserved end to end.

---

## 5. Also surveyed — related work currency and small-model feasibility

**Does anything published do script-to-multi-character-animation with a small model?** No.
Everything found in this pass that does end-to-end script/prompt → multi-character animated
output routes through a large hosted LLM as planner and/or a large video-diffusion backbone as
renderer:
- **AniMaker** (SIGGRAPH Asia 2025, arXiv 2506.10540, code: check paper) — multi-agent
  (Director/Photography/Reviewer/Post-Production agents), MCTS-driven clip candidate selection,
  a new evaluation suite (AniEval) for multi-shot animation — the most current system found in
  this space, and it is agentic-LLM-plus-video-diffusion in scale, not small-model. Model size
  not stated in sources checked (**unverified**), but the architecture (multiple cooperating
  agents, each presumably an LLM call, driving a video generator per clip) is not the "one
  60M-parameter model, one CPU forward pass" class this project occupies.
- **AniME** (arXiv 2508.18781, adaptive multi-agent planning for long animation) — same category,
  agentic/large-model, found alongside AniMaker in this pass, not examined further since it does
  not change the "no small-model precedent found" conclusion.
- The 2026 commercial storyboard-generator landscape (M Studio, DomoAI, TapVid, AnimateAI, Novi
  AI, per this pass's search) is entirely large hosted-model product surfaces (image-conditioned
  character consistency, LLM screenplay parsing) — useful only as evidence that the *commercial*
  version of "script in, animation out" is being built at large scale, which sharpens rather than
  weakens the paper's claim that a sub-100M-parameter, CPU-trainable model doing the same task
  is the contribution.
- VideoDirectorGPT and ViMax (already covered in `paper/RESEARCH.md` §3) remain the closest
  *architectural* analogues (planner-then-renderer split) — not repeated here beyond noting
  neither has a small-model variant either. **Net finding for the paper's related-work section:**
  the "prompt → symbolic scene script → deterministic renderer, entirely on CPU" pipeline this
  project has built appears to be the smallest-scale system in this specific niche found in the
  literature to date — worth stating as a contribution, not just a limitation.

**Any text-to-motion or layout model small enough to train on CPU or 8 GB at 50% duty?**
For **motion** specifically, `paper/RESEARCH.md` §5 already covers this ground (MoMask 44.85M,
Light-T2M 4.48M, TriC-Motion 13.86M parameters — all plausibly CPU/8GB-trainable, not re-derived
here). For **layout** specifically (the more relevant class for wall 1), no source in this pass
stated an exact parameter count for LayoutDM, LayoutFormer++, or BLT (**unverified** across all
three) — these are transformer-class models over short token sequences (tens of layout elements,
not tens of thousands), which puts them in the same rough size class as LayoutTransformer's
GPT-2-small-scale decoder by construction, but this is an inference from architecture class, not
a measured number, and should not be treated as a green light without a direct check of one of
these repos' config files. This project's own scene-grammar cast/position problem is far smaller
than PubLayNet-scale layout generation (≤5 elements vs. PubLayNet's dozens of document elements
per page), which is part of why the deterministic post-decode spacing pass in §1/§2 is favored
over importing one of these systems wholesale — the problem this project has is smaller than the
problem these systems were built to solve.

---

## 6. Ruled out

- **BLT / LayoutDM / LayoutFormer++ as adopted systems (not as borrowed ideas)** — all are
  built and evaluated for graphic-design layout (PubLayNet, RICO — dozens of elements, rich
  element-type vocabularies), a materially bigger and differently-shaped problem than this
  project's ≤5-character, single-axis staging question; adopting the full system means a second
  training pipeline and a second model class for a problem this project's data does not need that
  much machinery to solve. Ideas (decoding-space restriction, gradual/contextual refinement) are
  kept; the systems themselves are not.
- **GBNF / llama.cpp grammar-constrained decoding as infrastructure** — the right long-run answer
  if the project ever exports to GGUF, but switching the whole inference stack (HuggingFace
  `transformers` → `llama.cpp`) to fix one 10-colour uniqueness constraint is disproportionate;
  ruled out for now, not forever.
- **Position Coupling's full mechanism (custom position IDs inside T5)** — real code surgery
  inside the model's attention/position machinery, higher engineering cost and higher risk of
  silently breaking something that currently trains cleanly in 530 s, versus the scratchpad-only
  half of the same idea (executive-summary item 4), which needs no model code changes at all.
  Kept as a documented follow-up, not adopted first.
- **DETR-style Hungarian set-decoder for the cast section** — the single most structurally
  correct fix for both staging and duplicate colours (§4), ruled out only as the *first* move,
  not permanently: it is a genuine architecture change (new head, new matching loss) and this
  project's whole approach through Attempt 22 has been to prefer inference-time/data-level fixes
  given a CPU-minutes retraining budget; promote this item if §1's items 1-2 measurably fail to
  close the wall.
- **GDPP/DPP diversity loss as the first staging fix** — same reasoning as above: a training-loss
  change is strictly more expensive to validate (requires a full retrain-and-reeval cycle, ~530 s
  plus eval) than a deterministic post-decode pass that can be tested against the exact ten
  logged clips in minutes; kept as the fallback if the deterministic pass under-performs on cases
  the log has not yet seen (e.g. 5-character scenes where 1-D spacing alone may not be enough).
- **Sakuga-42M / comics-panel-layout datasets** — already covered by RESEARCH.md's licensing
  discussion (non-commercial) and, per §2 here, comics panel-layout research targets a different
  problem (page composition, not in-panel actor blocking) than this project needs; not a source
  of adoptable technique or data for walls 1-3.
- **AniMaker / AniME / commercial storyboard tools as architecture references** — surveyed per
  the task's instruction (§5) but not adopted: all are large-model-scale systems; nothing in
  them is trainable or runnable within this project's CPU/8GB-at-50%-duty budget, and none
  addresses the specific staging/counting/global-constraint measurements this project has made.

---

## 7. What each technique would predict, so the builder can falsify it

1. **Colour logits processor (item 1):** re-run `model/scene_eval.py`'s validity check on the
   same 150 in-distribution and 24 OOD prompts used in Attempt 22. Prediction: duplicate-colour
   count drops from 71/150 to 0/150 on v2 outputs, with **zero change** to parse rate, render
   rate, or any other validity-problem category (this constraint is orthogonal to everything
   else the decoder emits). If any other metric moves, the processor is masking more than
   intended — a bug, not a research surprise.
2. **1-D spacing pass (item 2):** measure the minimum pairwise x-distance between cast members
   across the same 150+24-prompt set, before and after the pass. Prediction: minimum pairwise
   distance rises from the currently-modal near-zero (9.6 vs 9.2 = 0.4 units on a 10-unit frame)
   to at least some enforced floor (e.g. 1.0 unit), with **no change** to which prompts parse,
   render, or pass the critic's motion-coverage/warp checks (spacing is orthogonal to timing and
   identity). If critic USABLE rate *drops* after adding this pass, the spacing solver is
   pushing characters off-frame or into props — a bug in the solver, not evidence against the
   idea.
3. **Extended cast-size range in training data (item 3):** retrain on 1-8 (or similar), then test
   on 9 (previously-untested, one step past the new trained ceiling) using the same clause-probe
   methodology from Attempt 22 §5-6. Prediction, per the SCAN/GECA reading in §3: prompt-match
   for sizes *within* the new trained range (6-8) should reach parity with today's 1-5 in-distribution
   number; prompt-match for size 9 (one past the new ceiling) should reproduce the same collapse
   pattern seen today at 6 (saturating at a lower count, not failing randomly) — if 9 instead
   generalizes cleanly, that would contradict this survey's reading of the compositional-
   generalization literature and is worth its own write-up.
4. **Count-scratchpad DSL (item 4):** same OOD cast-size-6 prompts, new model trained with the
   countdown token sequence. Prediction: if the scratchpad idea transfers from digit-addition-
   style tasks to cast enumeration, 6-cast prompt-match should rise measurably above the current
   0/3; if it does not move at all, that is evidence the failure is closer to "no representation
   for 'six' as a concept beyond what clause-counting already encoded" than to "no explicit
   counter," and argues for prioritizing item 3 (range extension) over item 4 in any future
   attempt.
5. **"When Can Transformers Count to n?" embedding-width prediction (§3):** this is a
   *falsifiable side-check*, not a technique to build: confirm `model/scene_ckpt2/config.json`'s
   `d_model` against the project's total closed vocabulary size. Prediction: `d_model` (512 for
   t5-small) exceeds vocabulary size by a wide margin, so if items 3-4 *still* fail to move the
   cast-size-6 result, embedding width is not the explanation and the search for a cause should
   stay in the data/decoding space this survey focuses on, not move toward "train a bigger model."
