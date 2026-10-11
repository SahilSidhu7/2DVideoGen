# Release Notes: 2DVideoGen, first release

**Date:** 2026-10-11 (draft)
**Scope:** the full history, `ebc68c8` to `628a95e` (13 commits, 2026-09-01 to 2026-09-29). The repository has no tags and no changelog, so everything is treated as one first release.

> Draft generated from the git log and cross-checked against `README.md`, `ATTEMPTS.md` and `release/MANIFEST.json`. Please edit before publishing.

## Summary

2DVideoGen is a scene-script compositor plus a small model that writes scripts for it, built and run on a single 8 GB consumer GPU. The project logs 24 attempts at small-scale 2D animation generation, including the failures, and is now **closed**: Attempt 24 tested capacity, the last untried lever, and found it was not the bottleneck.

## Highlights

- **24 logged attempts** in [`ATTEMPTS.md`](ATTEMPTS.md), with failures and retracted diagnoses kept (`ebc68c8`, `d63f263`).
- **Scene-script compositor** `scenescript.py`: multi-character (cast, timeline, props, occlusion), CPU only (`ebc68c8`).
- **Script-writing model**: a 60M-parameter t5-small with grammar, synthesis and probes under `model/`, three preserved checkpoints with provenance in [`model/MODELS.md`](model/MODELS.md). The shipped checkpoint is `v3-staged` (`ebc68c8`, `3fd7611`).
- **Critic harness** (`critic/`): 8 metrics, built to return unwelcome verdicts. It independently reproduced a logged 23.6x flicker figure as 23.59x (`ebc68c8`).
- **Released assets** in [`release/MANIFEST.json`](release/MANIFEST.json): two generated icon models (`v1-8b-icons`, `v2-1.7b-icons`) and two procedural training corpora (`corpus-stickman`, `corpus-anime`).
- **Attempt 24 / t5-base (222.9M)** trained on byte-identical data, with `train.py` gaining `--base`, `--gpu`, `--grad-accum`, `--grad-checkpoint` and `--resume` (`d63f263`).

## Results and findings

- **Capacity is not the bottleneck (`d63f263`).** 3.7x the parameters (60M to 220M) improved every aggregate:
  - in-distribution prompt-match 46.0% to 56.7%
  - actions per character 72.7% to 89.3%
  - structurally valid without repair 94.0% to 99.3%
  - staging 0.879 to 0.911
  - out of distribution (n=24) 75.0% to 79.2%

  The rendered clip did not improve: `a24_six` vs `a23_six` is a tie, with the character-stacking failure relocated rather than removed.
- Beam-search staging moves 0.182 to 0.689 at 220M, so the Attempt 23 collapse attributed to decoding is partly a property of the 60M model (`d63f263`).
- The sampling-costs-action-fidelity tradeoff recorded for v3 dissolves at 220M (`d63f263`).
- Earlier findings from the first release commit: objectives interfere rather than compound; a model can look incapable when its training data never asked it to be capable; an apparent model defect can be a decoding artefact (`ebc68c8`).
- **Project closed** (`d63f263`; see the README).
- Reel `out/2dvideogen_reel.mp4` (44 s) stitches Attempts 22, 23 and 24 (`96223d2`; the commit says its sources live in `out/stitch/`, which is not tracked in this checkout).

## Documentation and paper

- `paper/`: write-up (`PAPER.md`), two literature surveys (`RESEARCH.md`, `RESEARCH2.md`) and an arXiv LaTeX package (`ebc68c8`).
- Bibliography completed: 48 incomplete BibTeX entries filled, 0 TODOs, all 60 cite keys in `main.tex` resolve (`3bf4e5e`).
- [`DATASETS.md`](DATASETS.md): provenance and licensing for every dataset (`d63f263`).
- [`PRACTICES.md`](PRACTICES.md): the ten rules that survived (`d63f263`).
- [`paper/SCALING.md`](paper/SCALING.md): parameters vs data across five controlled comparisons, all numbers taken from `ATTEMPTS.md`, `model/MODELS.md` and `DATASETS.md` (`628a95e`).
- README: new "Does this already exist?" section stating that Sora, Veo, Runway and others do text-to-video better. The editable, deterministic intermediate representation is argued as a design position and flagged as unmeasured, as no head-to-head baseline was run (`96223d2`).

## Corrections

- **README figures** mixed three checkpoints (530 s v1, 86.0% v2, 75.0% v3 sampled). All now describe the shipped v3-staged checkpoint: 548 s training, 85.7% / 70.8% under beam search, 75.0% out-of-distribution under sampling. The table is now beam vs sampling (`3fd7611`).
- **Misattributed reference:** arXiv:2009.10939 was credited to "Zhao et al." in `RESEARCH2.md` and `PAPER.md`; the authors are Ivgi, Benny, Ben-David, Berant and Wolf. Fixed in `PAPER.md` and `refs.bib`, and a paraphrased title replaced with the published one. Four further `refs.bib` entries are flagged for the author rather than changed (`3bf4e5e`).
- **LinkedIn post 2** wrongly implied the measurement work was hand-run; it was directed by the author and run by Claude Code (`966261d`). Both posts now credit Claude Code for most of the code (`7eebd80`).

## Housekeeping

- `.gitignore` now ignores the HF Space working copy `hf_space/`. `3bf4e5e` had accidentally committed it, and `0510928` removed it again.
- LinkedIn write-up in `docs/LINKEDIN_POST.md`: split into two shorter posts (`9c022dc`), reframed as a small-model fine-tuning experiment (`0b5afed`), datasets named (`54d4206`), made paste-ready (`4d73324`).

## Known limitations

- Hardware: one 8 GB GPU with a damaged cooling fan, so every GPU run is capped at ~50% duty (via `tools/gpuguard.py`). Batch 16 OOMs the card; the 220M model uses batch 4 x grad-accum 4.
- The project is closed.
- Roughly half of the 24 attempts failed. The causes are logged in `ATTEMPTS.md`.
- At 220M the rendered clip is a tie with 60M; the character-stacking failure remains.
- The best-looking clip (`a21_park`) is hand-written, not model-generated.
- No head-to-head baseline against hosted text-to-video systems was run.
- The generated icon models (`v1-8b-icons`, `v2-1.7b-icons`) have abstract geometry.

## Compatibility and assets

- The text-to-video path (`model/scene_infer.py` to `scenescript.py` to `critic/evaluate.py`) uses no GPU. The 60M script writer trains in 548 s on CPU. The 220M model needs the GPU.
- `train.py` now defaults to CPU; use `--gpu` to opt in (`d63f263`). `save_total_limit` is 2.
- Asset inventory: [`release/MANIFEST.json`](release/MANIFEST.json).

| Version | Origin | Clips | Frames |
|---|---|--:|--:|
| `v1-8b-icons` | Qwen3-8B LoRA, 1024 ctx; 94% valid | 6 | 172 |
| `v2-1.7b-icons` | Qwen3-1.7B LoRA, 4096 ctx; 100% valid | 6 | 410 |
| `corpus-stickman` | procedural training data | 6 | 304 |
| `corpus-anime` | procedural training data | 6 | 304 |
