# prompt2video model

A small trained model that turns a plain-English prompt into a math-equation
video. It does **not** generate pixels directly (that needs huge models and GPUs).
Instead it learns the cheap, learnable part — understanding what the user wants —
and hands a compact spec to the `geovid` renderer.

```
 prompt ── t5-small ──▶ DSL spec ── grammar.py ──▶ geovid scene ──▶ mp4
 "slow glowing blue      "arch=rose color=blue     {view, layers,       rose.mp4
  rose with 5 petals"     speed=slow sym=5 ...}      equations...}
```

## Why this design

- **Runs on medium hardware.** The model is `t5-small` (60M params). Training is
  a few epochs over synthetic data on CPU (minutes on 16 cores). Inference is one
  short beam search — sub-second on CPU.
- **Always valid output.** The model emits a tiny DSL, not raw JSON. Everything
  is validated by `grammar.py`; unusable output falls back to a rule parser, so a
  prompt *always* produces a renderable video.
- **No labeled data existed**, so we synthesize it from the grammar itself.

## Files

| file               | role                                                        |
|--------------------|-------------------------------------------------------------|
| `grammar.py`       | 9 archetypes, slot vocab, DSL↔spec, `build_scene`, rule parser |
| `synth_dataset.py` | generate `data/train.jsonl`, `data/val.jsonl`               |
| `train.py`         | fine-tune `t5-small`: prompt → DSL                          |
| `infer.py`         | `PromptModel`: prompt → spec/scene (model + rule fallback)  |
| `eval.py`          | exact-match + per-slot accuracy on the val set             |

## Reproduce

```bash
cd model
python synth_dataset.py --n 9000 --val 800      # build dataset
USE_TF=0 python train.py --epochs 3 --batch 32  # fine-tune -> checkpoint/
python eval.py --ckpt checkpoint                # measure
```

Then from the project root:

```bash
python prompt2video.py "a slow glowing blue rose with 5 petals" -o rose.mp4
```

> Note: set `USE_TF=0` when training — transformers otherwise tries to import a
> TensorFlow/Keras backend that isn't compatible here.

## The DSL

A flat, order-free string. Slots:

| slot    | values                                                        |
|---------|---------------------------------------------------------------|
| `arch`  | lissajous, rose, spirograph, superformula, harmonograph, spiral, interference, mandala, waves |
| `color` | blue, cyan, teal, red, orange, amber, yellow, green, lime, purple, violet, pink, magenta, white, gold (line layers) |
| `cmap`  | fire, ice, inferno, magma, viridis (field layers)            |
| `speed` | slow, med, fast                                              |
| `sym`   | integer 2–16 (petals / fold symmetry / arms)                 |
| `mood`  | glow, dark, minimal                                          |

Add an archetype by extending `ARCH_NAMES` + `build_scene` in `grammar.py`, then
regenerate data and retrain.

## Limits / honest notes

- The model generalizes over phrasing but the space of *outputs* is the fixed
  grammar — it composes archetypes and slots, it doesn't invent new equations.
- For truly novel equations, edit a scene JSON directly (see the top-level README).
- This is a controllable, resource-cheap system, not a diffusion video model.
