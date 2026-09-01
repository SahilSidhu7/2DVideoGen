# svg/ — AniSVG pipeline

Era 4 of this project: generate 2D animation as **vector frames** rather than
equations or pixels. See `../ATTEMPTS.md` for what came before and why it
stopped.

## The representation

`anisvg.py` defines **AniSVG**: a shot declares a cast of shapes once, then
every frame lists only the edits.

```
H 256 256 15 48                       w, h, fps, frame count
P #F48D47 #FF9F57 #FFEEE6             palette
S 0 2 3 120 84 1 2 0 3 -2 1 ...       shape 0, palette 2, stroke 0.3, rel pts
@ 1 t 0 2 -1 r 3 -4                   frame 1: move shape 0, rotate shape 3
```

All integers, all deltas. A shape not mentioned in a frame holds its pose.

Ops: `t` translate, `r` rotate (about the shape's centroid), `s` scale,
`o` opacity, `v` per-vertex morph, `h`/`w` hide/show.

Measured cost: **~6 tokens per moving shape per frame** (Qwen3 tokenizer). A
36-shape animation is ~213 tok/frame; a 10-shape character is ~60. Full-frame
raster tracing, for comparison, cost ~4000 tok/frame.

## Modules

| file | role |
|---|---|
| `anisvg.py` | the format — parse, replay, render to per-frame SVG |
| `lottie.py` | Lottie/Bodymovin evaluator: keyframes, easing, paths, parenting |
| `lottie_to_anisvg.py` | Lottie -> AniSVG converter |
| `fit.py` | similarity fit + residual, so motion is 4 numbers not a vertex list |
| `raster.py` | dependency-free SVG rasteriser (cairo has no usable Windows build) |
| `stream_corpus.py` | stream LottieAnimation-660K metadata shards -> filtered AniSVG |
| `pack_dataset.py` | shards -> train/val jsonl: dedup, trim to budget, split |
| `train_lora.py` | LoRA fine-tune on this box (chunked loss, 4-bit option) |
| `to_soup.py` | packed clips -> Soup's Alpaca format |
| `generate.py` | caption -> AniSVG -> PNG/GIF, with a structural score |
| `soup_pilot.yaml` | Qwen3-8B layer-streaming pilot config |

Kept from the raster-tracing experiment (still useful, see ATTEMPTS.md #11):
`fetch_clips.py`, `prep.py`, `palette.py`, `trace.py`, `layers.py`,
`build_corpus.py`, `svgtok.py`, and the three `bench_*.py` scripts.

## Run

```bash
# convert Lottie files, with token statistics
USE_TF=0 python svg/lottie_to_anisvg.py datasets/lottie/*.json \
    --tokens -o svg/data/anisvg.jsonl

# render a converted animation back to frames
USE_TF=0 python - <<'EOF'
import sys, json; sys.path.insert(0, 'svg')
from anisvg import Anim
import raster
rec = json.loads(open('svg/data/anisvg.jsonl', encoding='utf-8').readline())
anim = Anim.from_text(rec['text'])
raster.render(anim.to_svgs()[24], width=256).save('svg/out/f24.png')
EOF
```

`USE_TF=0` is required project-wide — see the note in the root README.

## Corpus status

The pipeline is built for **`LottieGPT/LottieAnimation-660K`** (660k
text-captioned Lottie animations, CC-BY-NC-SA-4.0 - non-commercial).

Only part of it is worth fetching:

| files | size | needed? |
|---|---|---|
| `data/videos-*-of-00068.tar.zst` | 129.10 GB | no - mp4 previews, we never train on pixels |
| `data/metadata-*-of-00068.jsonl.zst` | 2.96 GB | **yes** - full Lottie JSON, captions, tags |

Two caveats follow from that split:

* `load_dataset("LottieGPT/LottieAnimation-660K")` fails with
  `FileFormatMismatchBetweenSplitsError` - the two splits use different
  formats (`json` vs `webdataset`). Fetch the metadata shards by filename
  instead of relying on split auto-detection.
* Converting all 660k yields ~10 GB of AniSVG text (~2 GB gzipped). A
  filtered subset - stable shape count, cast <= 20 shapes, character-like
  captions - is ~20-50k clips and ~750 MB, which is plenty for a 0.6B LoRA.

The dataset is **gated**. To unblock:

1. Accept the licence at
   <https://huggingface.co/datasets/LottieGPT/LottieAnimation-660K>
2. `hf auth login` (the old `huggingface-cli login` crashes on Windows cp1252
   while printing its own deprecation warning)

## Building the corpus

```bash
# check the record schema without downloading a shard
USE_TF=0 python svg/stream_corpus.py --probe 3

# stream, convert, filter, gzip - resumable per shard
USE_TF=0 python svg/stream_corpus.py -o svg/data/corpus     --max-clips 40000 --progress 2000
```

Nothing but the output touches disk: shards are decompressed off the socket.
Each shard writes `anisvg-000NN.jsonl.gz.part` and is renamed on completion, so
an interrupt redoes at most one shard.

Defaults reject a clip if it has a cast outside 2-20 shapes, under 8 frames,
motion in under half its frames, over 60k characters, or **any unsupported
Lottie feature** - a dropped precomp or trim path renders wrong rather than
failing, which is worse than losing the clip. `--allow-partial` keeps them.

Measured on 1500 real records: 20% kept, 0.052 s/record across a process pool,
~0.8 KB gzipped per clip.

Final corpus: **68/68 shards, 91,387 clips, 96 MB gzipped**.

## Training

```bash
# corpus -> train/val jsonl (dedup, trim, deterministic split)
USE_TF=0 python svg/pack_dataset.py --corpus svg/data/corpus -o svg/data/train
# -> 78,971 clips, 198.3M tokens, median 2531 tok/clip

# local LoRA (fits this 8 GB card)
USE_TF=0 python svg/train_lora.py --base Qwen/Qwen3-1.7B-Base --data svg/data/train

# or 8B via Soup layer streaming (needs the 3.12 venv)
python svg/to_soup.py --data svg/data/train -o svg/data/soup_pilot_data --limit 400
.venv-soup\Scripts\soup train --config svg/soup_pilot.yaml --yes
```

Over-long clips are **trimmed**, not dropped: AniSVG frames are individually
addressable lines and `to_text` derives the header count from the frame list,
so a six-second clip re-emits as three. At a 4096 cap that moved the loss from
37% of clips to 11%.

### What fits on 8 GB

Peak VRAM here is set by the **loss**, not the weights: `Qwen3ForCausalLM`
upcasts logits to fp32, so one ~1850-token sample against a 151,936 vocab is
~1.05 GB plus the same again for its gradient. `train_lora.py` scores the
sequence in slices (`--loss-chunk`, default 512) so the full logit tensor never
exists - 4.92 -> 3.16 GB at 0.6B, same objective to four decimals.

| base | quant | peak VRAM | s/clip |
|---|---|---|---|
| Qwen3-0.6B | bf16 | 3.16 GB | 0.76 |
| Qwen3-1.7B | bf16 | 5.30 GB | 1.32 |
| Qwen3-4B | NF4 | 6.28 GB | 4.00 |
| Qwen3-8B | NF4 + layer streaming | ~3.3 GB | see pilot |

soup-cli requires Python <3.13; this box defaults to 3.13, hence `.venv-soup`
built with 3.12.

## Scoring

Loss is a poor guide for a structured format - a model can look converged and
still emit text that will not parse. `generate.py` scores what the format
demands and renders the result:

```bash
python svg/generate.py --adapter svg/data/soup_pilot -c "a cat waving its paw"
python svg/generate.py --adapter svg/data/soup_pilot --eval 32   # held-out
```

Each generation is graded parses / declares shapes / anything moves, written as
`.anisvg`, and rendered to PNG frames plus a GIF.
