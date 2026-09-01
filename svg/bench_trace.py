"""Measure SVG size in *tokens* across trace settings.

The whole representation choice hinges on this number: a small model has a
finite context, so tokens-per-frame decides whether per-frame SVG is viable
at 15 fps or whether frames must be delta-encoded.
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import trace as T          # noqa: E402
from prep import prep      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    src = Image.open(args.image)
    print("source %dx%d" % src.size)
    rows = []
    for width in (192, 256, 320, 480):
        for levels in (4, 6, 8):
            for preset in ("tiny", "compact"):
                img = prep(src, width=width, levels=levels)
                tmp = os.path.join(args.outdir, "_prep.png")
                img.save(tmp)
                svg = T.shrink(T.trace_file(tmp, preset), decimals=0 if preset == "tiny" else 1)
                st = T.stats(svg, tok)
                rows.append((width, levels, preset, st))
                print("w=%-4d lv=%d %-8s paths=%-5d pts=%-6d tok=%-6d  %.1f KB"
                      % (width, levels, preset, st["paths"], st["points"],
                         st["tokens"], st["bytes"] / 1024))
                name = "trace_w%d_l%d_%s.svg" % (width, levels, preset)
                with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as fh:
                    fh.write(svg)

    print("\n-- frames that fit a context window (tokens/frame) --")
    for ctx in (4096, 8192, 32768):
        best = min(rows, key=lambda r: r[3]["tokens"])
        print("ctx %-6d -> %d frames at best setting (%d tok/frame) = %.1f s @15fps"
              % (ctx, ctx // best[3]["tokens"], best[3]["tokens"],
                 (ctx // best[3]["tokens"]) / 15))


if __name__ == "__main__":
    main()
