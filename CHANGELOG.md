# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). The project
has no tags yet and does not follow semantic versioning, so everything sits
under `[Unreleased]`. For the narrative version, with results and known
limitations, see [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## [Unreleased]

Draft v0.1.0. Covers the whole history, `ebc68c8` (2026-09-01) through
`5fe7579` (2026-11-06). The version number is a suggestion for the maintainer
to confirm.

### Added

- Initial drop: 23 logged attempts at 2D animation generation on an 8 GB
  consumer GPU, with the critic, script-writing model, renderers, AniSVG
  pipeline, web UI, GPU guard and paper. ([`ebc68c8`][ebc68c8])
- Attempt 24: a t5-base `v4-base` checkpoint trained on the same data as
  `v3-staged`, its evaluation in `model/eval_out4/`, plus
  [`DATASETS.md`](DATASETS.md), [`PRACTICES.md`](PRACTICES.md) and the
  benchmark charts. Capacity is not the bottleneck; the project is closed.
  See [`ATTEMPTS.md`](ATTEMPTS.md). ([`d63f263`][d63f263])
- `out/2dvideogen_reel.mp4`, stitching the Attempt 22, 23 and 24 clips, and a
  README section "Does this already exist?" that states the prior art
  plainly. ([`96223d2`][96223d2])
- [`paper/SCALING.md`](paper/SCALING.md): parameters vs data across the
  project. ([`628a95e`][628a95e])
- [`scenes/README.md`](scenes/README.md) (the `.scene` format reference),
  [`tools/README.md`](tools/README.md) (`gpuguard.py`), and a complete
  repository index in the README. ([`2ab5a3b`][2ab5a3b], merged in
  [`5fe7579`][5fe7579], [PR #7][pr7])

### Changed

- Split the LinkedIn write-up into two shorter posts. ([`9c022dc`][9c022dc])
- Reframed the posts as a small-model fine-tuning experiment.
  ([`0b5afed`][0b5afed])
- Named the datasets in the posts. ([`54d4206`][54d4206])
- Attributed the code to Claude Code plainly in both posts.
  ([`7eebd80`][7eebd80])
- Made the posts paste-ready and loosened the voice. ([`4d73324`][4d73324])

### Fixed

- Completed the bibliography and corrected a misattributed reference
  (arXiv:2009.10939) in `PAPER.md` and `refs.bib`. ([`3bf4e5e`][3bf4e5e])
- README results figures that mixed three checkpoints now all describe
  `v3-staged`. ([`3fd7611`][3fd7611])
- Post 2 now says the analysis was directed, not hand-run.
  ([`966261d`][966261d])

### Removed

- The Hugging Face Space working copy (`hf_space/`, `hf_space_progress.md`),
  which duplicates project files and lives on the Hub; it is now gitignored.
  ([`0510928`][0510928])

[Unreleased]: https://github.com/SahilSidhu7/2DVideoGen/commits/master
[pr7]: https://github.com/SahilSidhu7/2DVideoGen/pull/7
[ebc68c8]: https://github.com/SahilSidhu7/2DVideoGen/commit/ebc68c8
[3bf4e5e]: https://github.com/SahilSidhu7/2DVideoGen/commit/3bf4e5e
[0510928]: https://github.com/SahilSidhu7/2DVideoGen/commit/0510928
[3fd7611]: https://github.com/SahilSidhu7/2DVideoGen/commit/3fd7611
[d63f263]: https://github.com/SahilSidhu7/2DVideoGen/commit/d63f263
[9c022dc]: https://github.com/SahilSidhu7/2DVideoGen/commit/9c022dc
[96223d2]: https://github.com/SahilSidhu7/2DVideoGen/commit/96223d2
[0b5afed]: https://github.com/SahilSidhu7/2DVideoGen/commit/0b5afed
[54d4206]: https://github.com/SahilSidhu7/2DVideoGen/commit/54d4206
[7eebd80]: https://github.com/SahilSidhu7/2DVideoGen/commit/7eebd80
[4d73324]: https://github.com/SahilSidhu7/2DVideoGen/commit/4d73324
[966261d]: https://github.com/SahilSidhu7/2DVideoGen/commit/966261d
[628a95e]: https://github.com/SahilSidhu7/2DVideoGen/commit/628a95e
[2ab5a3b]: https://github.com/SahilSidhu7/2DVideoGen/commit/2ab5a3b
[5fe7579]: https://github.com/SahilSidhu7/2DVideoGen/commit/5fe7579
