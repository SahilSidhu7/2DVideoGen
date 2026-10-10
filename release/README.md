# release/

The publishable AniSVG output set: four labelled reels of six clips each. Every
clip is stamped with its origin, so a figure cannot drift away from its
provenance. Everything in this directory is tracked in git.

## Versions

| directory | origin | what it shows |
|---|---|---|
| `v1-8b-icons/` | GENERATED, Qwen3-8B LoRA, 1024 ctx | Trained on LottieAnimation-660K. 94% valid; colour weak, geometry abstract. |
| `v2-1.7b-icons/` | GENERATED, Qwen3-1.7B LoRA, 4096 ctx | 7× the animation signal. 100% valid; colour and object count correct, geometry still abstract. |
| `corpus-stickman/` | PROCEDURAL, training data, not model output | Articulated rig to AniSVG. Shows what the format carries. |
| `corpus-anime/` | PROCEDURAL, training data, not model output | Cel-style character, fixed 16-part schema. |

The two `corpus-*` directories are what the format can carry. Only `v1` and
`v2` show what a model produced.

## Contents of each directory

- `clip00.anisvg` … `clip05.anisvg`, the six clips, as text.
- `<version>.mp4`, a showcase reel of those clips.

## `MANIFEST.json`

A list with one entry per version:

| key | meaning |
|---|---|
| `version` | directory name |
| `origin` | `GENERATED …` or `PROCEDURAL …` label |
| `note` | one-line summary of the result |
| `clips` | number of clips kept |
| `frames` | frames in the reel video |
| `video` | path to the reel, as recorded when it was built (Windows-style `release\…` separators) |
| `shapes` | shape count per clip |
| `clip_frames` | frame count per clip |

## The `.anisvg` format

A line-oriented text format, all integers. A cast of shapes is stated once, and
each frame spends tokens only on what changes. The full specification is the
docstring of [`../svg/anisvg.py`](../svg/anisvg.py). In brief:

```
H <w> <h> <fps> <nframes>               header
P <hex> ...                             palette, one line
S <id> <pal> <sw> <x0> <y0> <dx dy>...  cast member: path as relative points
@ <frame> <op> ...                      per-frame edits; unlisted shapes hold
```

Per-frame ops are `t` translate, `r` rotate, `s` scale, `o` opacity, `v`
per-vertex morph, `h` hide and `w` show, all as deltas.

## Regenerating

[`../svg/build_release.py`](../svg/build_release.py) rebuilds the whole set:

```bash
python svg/build_release.py -o release
```

Options: `--width` (384), `--per-version` (6), `--loops` (2), `--hold` (4),
`--seed` (4). It needs `ffmpeg` on PATH.

The `v1` and `v2` clips are drawn from `svg/out/gen_con2` and `svg/out/gen17`,
the corpus reels from `svg/data/chars/chars.jsonl.gz` and
`svg/data/anime/anime.jsonl.gz`. `svg/out/` and `svg/data/` are git-ignored, so
a fresh clone cannot rebuild the reels. For any missing source the script
prints `skip <version> (missing …)` and leaves that version out. Note that the
manifest it writes is rebuilt from only the versions that were rendered.

The default output directory is `release`, relative to where you run it, so run
it from the repository root.
