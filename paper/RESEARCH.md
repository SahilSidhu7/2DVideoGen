# Research survey — temporal stability, small-model motion, and comparable evaluation

Written to unblock Attempt 18 (23.6x flicker amplification; OpenPose-*like* skeleton ignored
by ControlNet; ~40-80 tok/frame budget). Scope: web research only, no code changes. Every
number below is sourced; anything I could not pin to a source is marked **unverified**.

---

## 1. Executive summary — 3 routes, ranked

**1. Fix the skeleton, then add a real temporal mechanism (AnimateDiff motion module) — highest
expected payoff, lowest new infrastructure.** Failure 1 in Attempt 18 is self-diagnosed as fixable
("the rig has exact joints, so emitting true COCO-18 is a mapping exercise") — do that first,
independent of anything else, since a ControlNet given the wrong topology can silently ignore
conditioning. Then replace bare per-frame SD1.5+ControlNet with **AnimateDiff** (motion module
inserted into SD1.5's UNet, trained separately from the base model) run at inference only — no
training required to use it. Hardware verdict: **fits.** Community reports converge on ~8 GB
minimum / ~12-13 GB comfortable for 512x512x16-frame generation, i.e. tight but plausible on an
8 GB 4060 at reduced frame count/resolution; VRAM scales down with fewer frames and smaller tiles.
AnimateDiff-Lightning (ByteDance, arXiv 2403.12706, Mar 2024) distills the same motion module to
1-8 step inference, cutting compute (not stated to cut VRAM) and is inference-only, open weights.
Both are designed to compose with ControlNet in the standard SD1.5 ecosystem. This is the direct,
measured next step from where Attempt 18 stopped.

**2. Route identity preservation through a ReferenceNet-style pose animator (AnimateAnyone /
MagicAnimate) instead of hoping IP-Adapter or a fixed seed fixes it — addresses Failure 2 directly.**
Attempt 18 found identity (hair, face, background) reorganizes every frame; a fixed seed does not
help because it fixes noise, not trajectory. AnimateAnyone and MagicAnimate were built exactly for
this: a **ReferenceNet** branch (a duplicated UNet that extracts appearance features from one
reference image) plus a **Pose Guider** (lightweight conv encoder for the pose map) plus temporal
attention, so the character is spatially anchored to one reference frame instead of resampled per
frame. Both are SD1.5-based, open-source, and reported to still have residual fine-grained
inconsistency and background drift — so it is an improvement, not a solved problem (unverified:
exact VRAM figures for either; not found in this pass). This is complementary to route 1 — a
motion module gives temporal smoothness, a ReferenceNet gives identity anchoring — and both are
inference-time architectures, not full retraining.

**3. Keep motion generation in the small LM (AniSVG) but stop asking a token stream to also carry
appearance — feed the small model's exact pose stream into the fixed skeleton, and treat "style" as
a one-time reference-image problem solved once per character, not once per frame.** This is really
a restatement of the project's own Attempt 17 hybrid, refined by (1) and (2): the LM's job stays
motion (its only defensible budget, per the ATTEMPTS.md token math: ~40-80 tok/frame is a rig, not
a scene); the diffusion side's job becomes "animate this one reference image along this pose
sequence with a motion module," which is a materially different problem than "restyle this pose
image from scratch every frame" and is the problem AnimateAnyone/MagicAnimate/AnimateDiff were
built to solve. Hardware verdict: **fits**, same budget as routes 1+2 combined, because it reuses
the same SD1.5-class models — no new training required for the animator; only the existing AniSVG
LoRA training (already measured, 3.16-6.28 GB, see ATTEMPTS.md) needs to continue.

Ranking rationale: routes 1 and 2 are both direct fixes to the two failure modes Attempt 18 already
isolated, are inference-only (no new training budget needed), and are both standard, maintained,
open-source SD1.5 components. Route 3 is the correct long-term architecture but depends on 1 and 2
already working.

---

## 2. Task A — Temporal consistency for per-frame diffusion on 8 GB

**AnimateDiff** (Guo et al., "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models
without Specific Tuning," arXiv 2307.04725, 2023; code+weights public,
github.com/guoyww/AnimateDiff). Inserts a motion module trained on video data into a frozen SD1.5
UNet; the motion module transfers to any SD1.5 personalized checkpoint without retraining it.
**VRAM to run:** sources disagree by resolution/frame count — one report states 8 GB minimum for
SD1.5 variants, another observed ~12 GB for 512x512x16 frames and up to 21 GB at 512x768x24 frames;
original-repo inference is cited elsewhere at ~13 GB. Net: 8 GB is workable only at reduced
resolution/frame count, which is what this project would need anyway (24-80 tok/frame budget
implies short clips). **VRAM to train the motion module itself:** not found (unverified) — but this
project only needs inference, since the motion module ships pretrained. **Composes with
ControlNet-OpenPose on SD1.5:** yes, this is the standard combination in the community tooling
(sd-webui-animatediff-for-ControlNet, ComfyUI workflows); requires the skeleton be true COCO-18 to
actually take effect (see Task A ControlNet note below).
Sources: https://github.com/guoyww/AnimateDiff , https://github.com/DavideAlidosi/sd-webui-animatediff-for-ControlNet , https://education.civitai.com/beginners-guide-to-animatediff/

**AnimateDiff-Lightning** (Lin & Yang, ByteDance, "AnimateDiff-Lightning: Cross-Model Diffusion
Distillation," arXiv 2403.12706, 2024; weights public on Hugging Face). Progressive adversarial
distillation of the motion module to 1/2/4/8-step inference, ~10x faster than AnimateDiff, broader
base-model style compatibility via cross-model distillation. **VRAM:** not separately quantified in
sources found (unverified) — distillation reduces step count/wall-clock, not necessarily peak VRAM,
since the UNet+ControlNet+motion-module footprint is unchanged; treat as same ballpark as
AnimateDiff until measured locally.
Sources: https://arxiv.org/abs/2403.12706 , https://syncedreview.com/2024/03/20/bytedances-animatediff-lightning-shines-in-state-of-the-art-video-creation-in-lightning-speed/

**Text2Video-Zero** (Khachatryan et al., ICCV 2023 Oral, arXiv 2303.13439; code+weights public,
github.com/Picsart-AI-Research/Text2Video-Zero). Training-free: motion is injected by warping
latents across frames, and a **cross-frame attention** (every frame attends to the first frame's
keys/values instead of self-attention) anchors appearance. **VRAM to run:** the repo states a
**12 GB minimum** even in its low-memory mode — this is *above* the project's 8 GB budget as
published; would need further quantization/tiling to fit, unverified whether that closes the gap.
**Training:** zero — it is a pure inference-time reprogramming of an existing SD1.5 checkpoint, no
motion-module weights to acquire. **Composes with ControlNet on SD1.5:** yes, explicitly —
dedicated methods `process_controlnet_pose()`, `process_controlnet_canny()`,
`process_controlnet_depth()` ship in the repo.
Sources: https://arxiv.org/abs/2303.13439 , https://github.com/Picsart-AI-Research/Text2Video-Zero

**FRESCO** (Yang et al., CVPR 2024, "FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video
Translation," code+weights public, github.com/williamyang1991/FRESCO, project page
mmlab-ntu.com/project/fresco). Zero-shot, training-free: combines **intra-frame** (spatial) and
**inter-frame** (temporal, optical-flow-based) correspondence constraints, explicitly built to be
more robust to large/fast motion than its predecessor Rerender-A-Video. Composes with "off-the-shelf
ControlNet, LoRA" per the repo, though OpenPose specifically is not called out (HED/Canny/Depth are
the examples given in the WebUI). **VRAM:** not published in the sources checked (unverified).

**Rerender-A-Video** (Yang et al., SIGGRAPH Asia 2023). Predecessor to FRESCO; cited by FRESCO's own
authors as "less robust to large and quick motion." Given FRESCO supersedes it directly and is from
the same authors, prefer FRESCO. **VRAM:** unverified.

**TokenFlow** (Geyer et al., arXiv 2307.10373, 2023; code public). Training-free; enforces
consistency by propagating diffusion features according to inter-frame **correspondences** (nearest-
neighbor feature matching) rather than warping pixels. **Speed, not VRAM, is the documented
bottleneck:** a later paper (PipeFlow, arXiv 2512.24026) reports TokenFlow takes 5091 s to process
240 frames on the hardware it was benchmarked on (RTX 3090, 512x512), i.e. impractical for iterative
work at that frame count; PipeFlow itself is 9.6x faster at the same task. VRAM figure not found
(unverified) but a 3090 is a 24 GB card, so headroom on this project's 8 GB card is unverified and
should be measured, not assumed.
Sources: https://arxiv.org/abs/2307.10373 , https://arxiv.org/pdf/2512.24026

**ControlVideo** (extends ControlNet to video with full cross-frame attention). VRAM: unverified
in this pass; full cross-frame attention (every frame attends to every other frame) is the most
expensive of the group by construction (O(n^2) in frame count) and is the first one to rule out on
an 8 GB card for anything beyond a handful of frames — flagged for local measurement before
adoption.

**Newer (2025-2026) — sparse/lightweight temporal attention, not yet composed with ControlNet-SD1.5
in what was found:** Sparse Forcing (arXiv 2604.21221), LiteAttention (arXiv 2511.11062), Sparse
VideoGen 1/2 (ICML/NeurIPS 2025, github.com/svg-project/Sparse-VideoGen), HyperVAttention (arXiv
2607.03012). These target DiT-based video models (video diffusion transformers, not SD1.5+UNet) and
are inference accelerators for large video backbones — relevant to Task B's small-video-model route,
not directly to the SD1.5+ControlNet stack this project has already built. Treat as **not composable
with the current pipeline without a backbone change** (unverified whether any target SD1.5).
Sources: https://arxiv.org/html/2604.21221v1 , https://arxiv.org/pdf/2511.11062 , https://github.com/svg-project/Sparse-VideoGen

**Identity-preservation architectures (directly answer Attempt 18 Failure 2), not in the original A
list but decisive for it:**
- **AnimateAnyone** and **MagicAnimate** (MagicAnimate: "Temporally Consistent Human Image Animation
  using Diffusion Model"): both add a **ReferenceNet** (a second UNet copy that injects appearance
  features from one fixed reference image via cross-attention/feature concatenation) plus a
  lightweight **Pose Guider** for the conditioning map, plus temporal attention layers. This is
  architecturally the fix for "every frame is a different character" — the character's appearance is
  drawn from one anchor image, not resynthesized from the prompt each frame. Reported limitation:
  MagicAnimate still shows fine-grained inconsistency and background drift under DensePose guidance.
  Both are SD1.5-based and open-source. VRAM: unverified in this pass — flagged for local
  measurement; the ReferenceNet doubles UNet parameter count in the graph (two UNet-shaped branches)
  which is a real added cost over plain SD1.5+ControlNet and should be budgeted against the 8 GB
  ceiling explicitly.
  Sources: https://arxiv.org/html/2410.10306 (Animate-X, cites both), https://www.researchgate.net/publication/384132413_MagicAnimate_Temporally_Consistent_Human_Image_Animation_using_Diffusion_Model
- **IP-Adapter / IP-Adapter-FaceID(-Plus/-PlusV2)**: lightweight adapter (frozen base, trains only
  new cross-attention image-projection layers) for image-prompted generation; FaceID variants
  preserve facial identity across drastic style changes and can be applied per-frame with a
  companion LoRA (weight 0.5-0.7) for consistency. **VRAM: SD1.5 + adapter runs at roughly 6-8 GB**
  for 512x512 — the only concrete number found that sits inside this project's budget without
  qualification. Lighter-weight than a full ReferenceNet; worth trying before ReferenceNet-class
  methods given the VRAM headroom argument, though it is a weaker identity lock (embedding-based, not
  a full appearance branch) and untested for full-body/clothing consistency (its strength is faces).
  Sources: https://stable-diffusion-art.com/ip-adapter/ , https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter

---

## 3. Task B — Script-to-multi-shot-animation systems

**VideoDirectorGPT** (Lin, Zala, Cho, Bansal, COLM 2024, arXiv 2309.15091; code public,
github.com/HL-hanlin/VideoDirectorGPT). An LLM (GPT-4) is used purely as a **planner**: it expands
one text prompt into a structured "video plan" — per-scene descriptions, entity layouts (bounding
boxes), backgrounds, and cross-scene entity-consistency groupings — which then drives a downstream
grounded video generator. This is the closest published analogue to the project's own "small LM
emits motion, diffusion renders appearance" split, except VideoDirectorGPT uses a large hosted LLM
for planning (GPT-4, not locally trainable) and a separate large video generator for rendering.
Directly useful as an **architecture reference** for multi-scene consistency grouping, not as a
component to run locally as-is. Reports state-of-the-art layout/movement control and competitive
single-scene generation; multi-scene consistency is its explicit contribution.
Source: https://arxiv.org/abs/2309.15091 , https://videodirectorgpt.github.io/

**ViMax** (2025-2026, github.com/hkuds/vimax). Agentic pipeline chaining narrative planning →
character/script/storyboard/shot generation → image generation → video generation → assembly, i.e.
a full script-to-video agent stack. Newer than VideoDirectorGPT and broader in scope (screenwriter +
producer + director roles as separate agent stages). No VRAM/scale figures found in this pass
(unverified) — flagged as worth a deeper look given it is the most directly on-target "script to
multi-shot video" system found, but likely also assumes large hosted models at each stage.
Source: https://github.com/hkuds/vimax

**Small video diffusion backbones (the "run one, don't train one" tier):**
- **Wan2.1-T2V-1.3B** (Wan-AI, Hugging Face: Wan-AI/Wan2.1-T2V-1.3B-Diffusers). **Inference VRAM:
  ~8.19 GB at FP16**, fits an RTX 4060 (one source), another states 11 GB for the Diffusers
  pipeline specifically — treat 8-11 GB as the realistic run-time range depending on
  pipeline/optimizations; GGUF-quantized builds bring it to **4-6 GB at 480p**. Output: 480p,
  16 fps, up to 81 frames (~5 s). **Training/LoRA VRAM:** a 24 GB card (RTX 3090) is reported
  as "more than sufficient, didn't even use half," 512-res LoRA training at ~2.5 h for 3500 steps —
  i.e. training was demonstrated on 24 GB, not 8 GB; whether it fits under 12 GB is unverified and
  should be measured, not assumed, before committing to training this backbone on the project's card.
  Sources: https://www.spheron.network/blog/ai-video-generation-gpu-guide/ , https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers , https://willitrunai.com/blog/wan-2-2-vram-requirements , https://www.mimicpc.com/learn/how-to-train-wan21-lora-with-diffusion-pipe
- **CogVideoX-2B**: **~8 GB VRAM at FP8** for 480p/49 frames/8fps (inference); one source instead
  classifies it as a 12 GB-tier model (RTX 3060 12GB / RTX 4070) at default precision — the FP8
  number is the one that would fit this project's card, and should be verified locally since FP8
  support depends on GPU generation (4060 is Ada, FP8 tensor cores are present, so plausible but
  unverified end-to-end here).
  Source: https://www.spheron.network/blog/ai-video-generation-gpu-guide/
- These are **standalone T2V models**, not yet shown composing multiple independent characters +
  a separate background layer in what was found — they generate one coherent scene per clip. Using
  one as the "renderer" stage of a VideoDirectorGPT-style pipeline (one call per shot/entity, then
  compositing) is an architectural idea from this research, not a demonstrated result.

**Layered/vector animation generation (closest to the project's own AniSVG target):**
- **AniClipart** (Wu et al., IJCV 2024, arXiv 2404.12347). Turns a *static clipart image* into
  motion using Bézier curves over keypoints plus a text-to-video diffusion prior (Score
  Distillation Sampling) as the motion signal, with an explicit **rigidity constraint** absent from
  prior sketch-animation work — the stated reason prior methods distort delicate vector edges.
  Single-subject, not multi-character/background composition. Directly relevant as a technique for
  "how to move a vector character believably without a temporal video model," which is exactly the
  gap between the project's AniSVG generator and a believable render.
  Source: https://ar5iv.labs.arxiv.org/html/2404.12347 , https://link.springer.com/article/10.1007/s11263-024-02306-1
- **Differentiable Motion Trajectory (DMT) for vector sketch animation** (Zhu et al., CGF 2025/2026,
  onlinelibrary.wiley.com/doi/10.1111/cgf.70335): differentiable polynomial trajectories over stroke
  control points, end-to-end automatic — another SDS-based, single-character sketch animator.
  Neither AniClipart nor DMT is shown handling multiple independent characters plus a background in
  the sources found — **multi-character vector-animation composition looks like an open gap in the
  published literature**, not just a gap in this project.

---

## 4. Task C — Datasets for stick-figure / cel / vector 2D animation

| dataset | size | license | fit for this project |
|---|---|---|---|
| **Sakuga-42M** (arXiv 2405.07425, github.com/KytraScript/SakugaDataset) | 42M keyframes from >150k cartoon videos, ~1.4M clips, captions ~40 words/clip (BLIP-v2 + LLM-linked) | **CC BY-NC-SA 4.0, academic only** | Largest hand-drawn-animation corpus found; captions exist but are auto-generated (BLIP-v2/LLM), not pose-labeled — would need a pose extractor on top (e.g. the project's own YOLO11-pose route) to get motion labels; non-commercial license matches "research paper" framing but blocks any commercial use later. |
| **AnimeRun** (arXiv 2211.05709) | train 1760 frames / test 1059 frames, synthetic from open-source 3D movies, dense optical-flow + region-wise correspondence labels | license not confirmed in sources checked (**unverified**, CC-BY-NC-SA appears on associated figures only) | Small; correspondence/flow labels are richer than ATD-12K's per the paper (avg. 237 segments/frame vs <50), useful for shape-correspondence work directly analogous to what killed raster-tracing in Attempt 11 — but too small to train a generative model on alone. |
| **ATD-12K** | 12,000 real-cartoon frame triplets, optical-flow annotated | unverified | Built for frame interpolation, not generation; small. |
| **CreativeFlow+** | 124K+ train frames, 10K test frames, synthetic optical flow | unverified | Synthetic (likely 3D-rendered 2D-style), largest of the flow-labeled sets; same caveat as AnimeRun — flow/correspondence data, not appearance+caption data for generation. |
| **LottieAnimation-660K** (already in use, `LottieGPT/LottieAnimation-660K`) | 660k captioned Lotties, project already extracted 91,387 clips / 198.3M tokens | **CC-BY-NC-SA-4.0** | Already the project's live corpus (svg/README.md); confirmed here as still the right choice — it is vector, captioned, and has exact shape correspondence for free, unlike anything above. Its limitation is already known and logged (Attempt 16: motion-graphics iconography, no characters). |
| **HumanML3D** (github.com/EricGuo5513/HumanML3D) | 14,616 motions, 44,970 text descriptions, 28.59 h total, avg. 7.1 s/motion, avg. 12-word captions | derived from AMASS; **cannot be redistributed directly** (must be regenerated from AMASS via provided scripts) | Text→3D-motion pairs at a scale a small model can train on; the retargeting question (3D motion → 2D stick rig) is Task D's open question. |
| **AMASS** | large (union of many mocap datasets); size not restated here | per-dataset licenses, distribution-restricted — must rebuild HumanML3D locally from it | Underlying raw mocap; same access friction as any AMASS-derived work. |
| **Motion-X / Motion-X++** (NeurIPS 2023 / arXiv 2501.05098) | "much larger than HumanML3D," whole-body incl. hands/face; IDEA400 and Kungfu subsets already reformatted to HumanML3D's spec | unverified license | Bigger than HumanML3D and already has HumanML3D-compatible subsets (IDEA400, Kungfu) — lower-friction than re-deriving from AMASS. |

**Reading for Task D's premise:** none of HumanML3D/AMASS/Motion-X are 2D or stick-figure native —
they are 3D joint-rotation/position data. Using them means solving retargeting (Task D, below), not
a free lunch.

---

## 5. Task D — Small-model motion representation (text-to-motion as a trainable route)

**Parameter counts found (all far smaller than any video diffusion model, all plausibly trainable
on 8 GB):**
- **MoMask**: 44.85M parameters (hierarchical residual VQ-VAE: 6 quantizers x 4 scales, codebook
  512 x dim 512, encoder/decoder width 512 depth 3; Stage-2 transformer: 8 layers, 6 heads, dim
  384; trained 500 epochs, batch 256, lr 8e-4 on HumanML3D). Source: arXiv 2412.11193 (Light-T2M,
  which cites MoMask's param count directly).
- **Light-T2M**: 4.48M parameters — stated as **10% of MoMask's trainable parameters**, explicitly
  positioned as the lightweight option in this family. arXiv 2412.11193.
- **TriC-Motion**: 13.86M parameters. arXiv 2602.08462.
- **MDM** (Tevet et al., "Human Motion Diffusion Model"): transformer denoiser, 8 layers, 4 heads,
  latent dim 512, FF dim 1024; batch 256, 750,000 iterations, EMA, no weight decay. Param count not
  restated numerically in sources checked but this architecture class is tens of millions of
  params, consistent with the family above (unverified exact count).
- **MotionGPT** (OpenMotionLab, NeurIPS 2023, github.com/OpenMotionLab/MotionGPT): unifies motion
  and language via a shared LLM backbone — this one is **not** small by construction (built on an
  LLM base), so it does not belong in the "trainable on 8GB from scratch" tier the way MoMask/
  Light-T2M/TriC-Motion do; treat as the outlier of this group.

**Training cost:** MDM-class training is reported at ~20 hours on a single GPU for text-to-motion
(unspecified card, unverified VRAM); a different report cites diffusion motion models trained in
~13 hours on an RTX 5090 (32 GB) — neither confirms an 8 GB figure directly, but the *parameter
counts* (4-45M) are two to three orders of magnitude below even Qwen3-0.6B (600M) already trained
successfully on this project's card per ATTEMPTS.md Attempt 14, so VRAM is very unlikely to be the
constraint for these motion models — batch size and sequence length would dominate instead
(unverified without a direct run).
Sources: https://arxiv.org/pdf/2412.11193 , https://arxiv.org/pdf/2602.08462 , https://github.com/GuyTevet/motion-diffusion-model , https://github.com/OpenMotionLab/MotionGPT

**Is retargeting 3D human motion onto a 2D stick rig credible?** Partially, with a caveat this
research did not find addressed directly: HumanML3D/AMASS motion is full 3D joint rotation/position
data; the project's rig is a 14-d 2D vector (hip x/y, 9 joint angles, ball position+flag) or,
in the AniSVG/COCO-18 direction, 2D keypoints. Projecting a 3D pose to a fixed 2D viewpoint (e.g.
side-on, as the project's own Muybridge data already is) is a solved, cheap geometric operation —
orthographic or perspective projection of joint positions, not a learned step. The **credible route**
is: train (or reuse a pretrained) small text-to-motion model like MoMask/Light-T2M on
HumanML3D/Motion-X's 3D output space, then apply a fixed camera projection to get 2D joint tracks,
then map those onto the project's existing 14-d/COCO-18 pose format the same way `pose_map.py`
already maps COCO-17→14-d today (ATTEMPTS.md #8). This reuses existing project code for the last
mile and only adds the 3D-motion-generation step as new work. Not found in the literature: a
published system doing exactly this projection-and-retarget step for stylized 2D animation — this
is a genuine, answerable-only-by-experiment gap (carried into section 6).

---

## 6. Task E — Evaluation metrics used in published animation-generation papers

- **FVD (Fréchet Video Distance)**: the standard distributional-distance metric between real and
  generated video features from a pretrained spatiotemporal network (typically I3D), analogous to
  FID for images. Known **content bias** issue documented in "On the Content Bias in Fréchet Video
  Distance" (CVPR 2024) — motivates the newer **content-debiased FVD**
  (github.com/songweige/content-debiased-fvd) and **JEDi** (github.com/oooolga/JEDi, "Beyond FVD").
  Reference implementations: Google's original
  (github.com/google-research/google-research/tree/master/frechet_video_distance),
  a PyTorch port (github.com/ragor114/PyTorch-Frechet-Video-Distance), and a combined
  FVD/PSNR/SSIM/LPIPS toolkit (github.com/JunyaoHu/common_metrics_on_video_quality).
- **CLIPSIM (CLIP Score for video)**: measures text-frame alignment via CLIP embeddings; rooted in
  the original CLIPScore paper (EMNLP 2021, github.com/jmhessel/clipscore) and first applied to
  video in GODIVA (arXiv 2021). A newer alternative, **VQAScore/t2v_metrics**
  (github.com/linzhiqiu/t2v_metrics), is reported in recent literature as more robust than CLIPSIM
  for prompt alignment — worth using alongside or instead of CLIPSIM for a 2026 paper.
- **Warping error (temporal consistency)**: standard blind-temporal-consistency metric — compute
  optical flow between consecutive real/generated frame pairs, warp frame t by that flow, take
  pixel-wise difference from frame t+1, average over all pairs. This is exactly the quantity
  Attempt 18 already approximated with its "inter-frame change" ratio (1.29 guides vs 30.44 output,
  23.6x) — formalizing it with a proper optical-flow warp (e.g. RAFT) rather than raw pixel-diff
  would make the number comparable to published warping-error figures.
- **Fréchet Video Motion Distance (FVMD)** (github.com/DSL-Lab/FVMD-frechet-video-motion-distance):
  a metric specifically for *motion* consistency (as distinct from FVD's overall appearance+motion
  blend) — closer in spirit to what this project needs to isolate, since the project's own failure
  mode is specifically motion/identity instability, not overall visual quality.
  Reference implementation available.
- **EvalCrafter** (arXiv 2310.11440): a full benchmarking suite for T2V models, aggregating multiple
  metrics (visual quality, motion quality, text alignment, temporal consistency) with human-study
  correlation — useful as a reference for which metric combination correlates with human judgment,
  relevant when picking a minimal defensible metric set for a small paper.
- **User studies**: standard in this literature as a supplement to automated metrics (referenced
  across VideoDirectorGPT, AniClipart, EvalCrafter) — typically pairwise preference (A/B) on
  motion faithfulness, identity consistency, and text alignment; no special tooling needed, but
  should be planned for since automated metrics alone (per EvalCrafter's own framing) do not fully
  correlate with perceived quality.

Practical recommendation for this project's paper: **FVMD or the raw warping-error ratio** (already
computed once, in Attempt 18) for temporal/motion stability — this is the axis the project's own
failures are on; **CLIPSIM or VQAScore** for text-motion alignment; **content-debiased FVD** only if
compute allows generating enough clips for a stable Fréchet estimate (FVD needs a nontrivial sample
size to be meaningful — exact minimum not found, unverified, but published FVD studies typically use
hundreds to thousands of clips per condition, which may itself be a budget problem worth flagging to
the builder).

---

## 7. Ruled out and why

- **ControlVideo's full cross-frame attention** — every frame attends to every other frame,
  O(n²) in frame count; the specific VRAM number that rules it out was not found in this pass
  (unverified), but the mechanism itself is the most expensive of the zero-shot group by
  construction and should be assumed to blow the 8 GB budget at more than a handful of frames until
  proven otherwise locally — do not adopt without a local measurement first.
- **Text2Video-Zero at its stated operating point** — repo states **12 GB minimum VRAM**, which is
  above this project's 8 GB card as published; only worth revisiting if a quantized/tiled variant is
  found to close that specific 4 GB gap.
- **TokenFlow for anything beyond very short clips** — 5091 s to process 240 frames on an RTX 3090
  (24 GB) is the documented number; even before considering whether it fits in 8 GB, the wall-clock
  cost alone rules it out for iterative development on a laptop with a damaged fan (50% GPU duty).
- **Training a video diffusion backbone from scratch, or full-parameter fine-tuning one** — already
  ruled out in ATTEMPTS.md ("Rejected outright... Not feasible to train on one 8GB laptop GPU") and
  confirmed again here: even the smallest LoRA-training report found for Wan2.1-1.3B used a 24 GB
  card and did not use half of it, meaning the actual 8 GB floor for LoRA training on this backbone
  is unmeasured and should not be assumed to work — the number that would rule it out (peak VRAM at
  batch size 1, 8 GB card) was not found and needs a local test before committing budget to it.
- **LTX-2 LoRA training** — already logged in ATTEMPTS.md Attempt 17 as needing 32-80 GB; this
  research did not find a contradicting figure, so it stays ruled out.
- **MotionGPT (OpenMotionLab)** for the "small, locally-trainable" tier specifically — it is built on
  an LLM backbone, which puts it in a different size class than MoMask/Light-T2M/TriC-Motion (4-45M
  params); it may still be useful as a heavier reference implementation, but does not belong in the
  same feasibility bucket as the other three.
- **Sakuga-42M and LottieAnimation-660K's non-commercial license (CC BY-NC-SA)** — this does not rule
  out research/paper use (matches the project's stated "research paper deliverable" goal per user
  memory) but rules out any future commercial deployment without renegotiating data rights; flagged
  so the choice is made knowingly, not rediscovered later.
- **MMSVG-Character** — the one dataset that would most directly answer "real anime vector character
  data," is not yet released (Icon and Illustration subsets are public; Character is "planned for
  release," unverified timeline) — cannot be used now regardless of size/license.

---

## 8. Open questions for the builder

1. **What is the actual peak VRAM of AnimateDiff (or AnimateDiff-Lightning) + ControlNet-OpenPose +
   a ReferenceNet-class identity module, all three stacked, on this exact 4060 8GB card at the
   project's target resolution/frame count?** No source combined all three; published numbers are
   per-component and inconsistent (8-21 GB range for AnimateDiff alone depending on settings). This
   determines whether route 1+2 from the executive summary is a single pipeline or needs staged
   inference.
2. **Does fixing the skeleton to true COCO-18 topology (per the rig's own exact joints) actually make
   ControlNet-OpenPose follow the pose, and does it change the 23.6x flicker number at all?**
   Attempt 18 predicted this fix but did not measure it — cheapest possible next experiment, isolates
   Failure 1 from Failure 2.
3. **With identity anchored by a ReferenceNet or IP-Adapter-FaceID, does the flicker ratio drop
   toward the 1.29x seen in the pose guides themselves, or does a residual gap persist that only a
   motion module (not identity anchoring) can close?** This tells the builder whether identity and
   temporal-smoothness are actually separable failure modes or coupled — Attempt 18's write-up
   treats them as two failures, but no experiment yet isolates which fix addresses which number.
4. **What is the minimum clip count needed for a stable FVD/FVMD estimate at this project's scale,**
   and is generating that many clips (each requiring the full diffusion+ControlNet pipeline)
   affordable under 50% GPU duty? Published FVD work generally uses corpora far larger than a small
   project can likely produce; if not affordable, the paper should lead with warping-error/FVMD-style
   per-clip metrics and user study, not FVD.
5. **Does 3D-motion→2D-projection→pose-retarget (Task D's proposed route) actually produce motion
   that looks correct when driving the project's cel-style rig**, or does the fixed-camera projection
   lose information the 2D rig needed (e.g. depth-driven limb overlap, foreshortening)? No published
   system does exactly this for stylized 2D output — it is untested even in the literature, not just
   in this project.
6. **Is Wan2.1-1.3B LoRA training (not just inference) actually feasible under 8 GB**, given the one
   training report found used a 24 GB card without needing most of it? If the true floor is, say,
   14-16 GB, this forecloses route where the "renderer" stage of a VideoDirectorGPT-style pipeline is
   fine-tuned locally, and inference-only use (already confirmed to fit) becomes the only option.
