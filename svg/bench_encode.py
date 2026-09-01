"""Compare token cost of encodings on a real traced corpus.

Decides the training representation: raw SVG vs compact absolute vs compact
relative vs inter-frame delta.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P     # noqa: E402
import svgtok as S      # noqa: E402


def abs_encode(paths, pal_index, quant=1):
    lines = []
    for fill, pts in paths:
        if len(pts) < 3:
            continue
        body = [str(pal_index(fill))]
        for x, y in pts:
            body += [str(round(x / quant)), str(round(y / quant))]
        lines.append(" ".join(body))
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--quant", type=int, default=1)
    ap.add_argument("--shots", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    n = lambda s: len(tok(s)["input_ids"])

    tot = dict(raw=0, absolute=0, relative=0, delta=0)
    frames = 0
    with open(args.jsonl, encoding="utf-8") as fh:
        for li, line in enumerate(fh):
            if li >= args.shots:
                break
            rec = json.loads(line)
            hexes = rec["palette"]
            cache = {}

            def pal_index(h, hexes=hexes, cache=cache):
                if h not in cache:
                    cache[h] = P.index_of(hexes, h)
                return cache[h]

            prev = None
            for svg in rec["frames"]:
                paths = S.svg_paths(svg)
                tot["raw"] += n(svg)
                tot["absolute"] += n(abs_encode(paths, pal_index, args.quant))
                tot["relative"] += n(S.encode_frame(paths, pal_index, args.quant))
                tot["delta"] += n(S.encode_delta(prev, paths, pal_index, args.quant))
                prev = paths
                frames += 1

    print("quant=%d  %d frames\n" % (args.quant, frames))
    base = tot["raw"] / frames
    for k in ("raw", "absolute", "relative", "delta"):
        per = tot[k] / frames
        print("%-9s %7.0f tok/frame  %5.2fx smaller  %6.1f tok/s of video @15fps"
              % (k, per, base / per, per * 15))
    print("\nseconds of video per context window (delta encoding):")
    per = tot["delta"] / frames
    for ctx in (4096, 8192, 16384, 32768):
        print("  ctx %-6d -> %5.1f s" % (ctx, ctx / per / 15))


if __name__ == "__main__":
    main()
