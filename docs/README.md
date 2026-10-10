# docs/

Material for sharing the project, as opposed to the technical write-ups in
[`../paper/`](../paper/).

## `LINKEDIN_POST.md`

Two paste-ready LinkedIn posts, written in plain text because LinkedIn renders
markdown literally.

- **Post 1**, a video post, uses `out/2dvideogen_reel.mp4`: the three clips with
  title cards.
- **Post 2**, an image post two to three days later, uses
  `docs/media/bench_loss_vs_quality.png`, `docs/media/bench_staging_decode.png`
  and `docs/media/bench_60m_vs_220m.png`.

## `media/`

Figures used by the root [`README.md`](../README.md) and
[`../paper/SCALING.md`](../paper/SCALING.md).

| file | what it shows | source |
|---|---|---|
| `a20_anime_diffusion.gif` | Attempt 20, the styled-anime AnimateDiff route that was closed by measurement | Not referenced by any document in the repository; see Attempt 20 in `ATTEMPTS.md` |
| `a21_park.gif` | The hand-written park scene | GIF of `out/a21_park.mp4`, rendered from `scenes/park_meet.scene` |
| `a22_generated.gif` | The first model-written scene | GIF of `out/a22_generated.mp4` (`scenes/a22_generated.scene`) |
| `a23_six.gif` | Six characters, script written by the 60M model | GIF of `out/a23_six.mp4` (`scenes/a23_six.scene`) |
| `a24_six.gif` | The same prompt at 220M parameters | GIF of `out/a24_six.mp4` (`scenes/a24_six.scene`) |
| `bench_loss_vs_quality.png` | `eval_loss` against out-of-distribution accuracy across checkpoints | Plots the loss and accuracy figures recorded in `ATTEMPTS.md` |
| `bench_60m_vs_220m.png` | The 60M and 220M models compared on the shipped sampling+rerank configuration | Plots the 60M and 220M figures recorded in `ATTEMPTS.md` (Attempt 24) |
| `bench_staging_decode.png` | Staging score under beam search and sampling | Plots the staging scores recorded in `ATTEMPTS.md` (Attempts 22–23) |

The repository does not contain a script that makes the GIFs or the three
`bench_*.png` charts, so how they were produced is not recorded here. The clips
they come from are re-renderable with `scenescript.py`; see
[`../scenes/README.md`](../scenes/README.md). The numbers behind the charts are
produced by `model/scene_eval.py` and the critic in [`../critic/`](../critic/).
