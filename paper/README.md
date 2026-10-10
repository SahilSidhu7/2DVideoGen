# paper/

The written-up work. [`../ATTEMPTS.md`](../ATTEMPTS.md) is the primary source;
everything here is derived from it.

| file | what it is |
|---|---|
| [`PAPER.md`](PAPER.md) | "Two Routes to 2D Animation on an 8 GB Consumer GPU". The full working draft, with a source note on every table. Its status line says it is current through Attempt 23, so it predates Attempt 24. |
| [`SCALING.md`](SCALING.md) | "Parameters or Data? What Actually Moved a Small 2D Animation Model". A short note comparing the two levers across the project, with five controlled comparisons. |
| [`RESEARCH.md`](RESEARCH.md) | Literature survey written to unblock Attempt 18: temporal stability, small-model motion and comparable evaluation, with three routes ranked. |
| [`RESEARCH2.md`](RESEARCH2.md) | Second survey, written to unblock Attempt 22: the staging, counting and global-constraint problems of the scene-script model. Extends `RESEARCH.md`. |
| [`anisvg.html`](anisvg.html) | A standalone HTML page about AniSVG (title "AniSVG"). It loads its fonts from Google Fonts. |
| [`arxiv/`](arxiv/) | LaTeX source for an arXiv version: `main.tex`, `refs.bib`, and `_abstract_plain.txt` (the abstract as plain text). The author line in `main.tex` is still a TODO placeholder. |

## Suggested reading order

1. The root [`README.md`](../README.md) for the result.
2. [`SCALING.md`](SCALING.md), the shortest piece, for the closing finding.
3. [`PAPER.md`](PAPER.md) for the full account.
4. [`RESEARCH.md`](RESEARCH.md) and [`RESEARCH2.md`](RESEARCH2.md) for the prior art and why routes were ruled out.
5. [`anisvg.html`](anisvg.html) for the vector-animation format.
6. [`arxiv/main.tex`](arxiv/main.tex) only if you are preparing the submission.

## Building the arXiv version

`arxiv/` holds source only. If you have a TeX Live installation, the standard
route is `pdflatex`/`bibtex` from inside that directory; no build script is
included.
