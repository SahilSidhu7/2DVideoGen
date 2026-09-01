"""How much cheaper is a shot when the static background is factored out?"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layers as L         # noqa: E402
import palette as P        # noqa: E402
import svgtok as S         # noqa: E402
import trace as T          # noqa: E402
from build_corpus import read_frames, shot_split   # noqa: E402
from prep import shot_box                            # noqa: E402

HOLE_RGB = (255, 0, 255)


def prep_idx(frames, width, colors):
    box = shot_box(frames[:: max(1, len(frames) // 6)])
    ch = max(1, round((box[3] - box[1]) * width / (box[2] - box[0])))
    small = [f.crop(box).convert("RGB").resize((width, ch), Image.LANCZOS)
             for f in frames]
    pal = P.fit(small[:: max(1, len(small) // 8)], colors=colors)
    idxs = [P.apply(s, pal)[1] for s in small]
    return pal, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--preset", default="compact")
    ap.add_argument("--quant", type=int, default=2)
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    n = lambda s: len(tok(s)["input_ids"])
    tmp = "svg/data/_layer.png"
    os.makedirs("svg/data", exist_ok=True)

    frames = list(read_frames(args.video, 15, args.max_seconds))
    shots = shot_split(frames, min_len=12)[: args.shots]
    print("%d shots\n" % len(shots))

    tot_full = tot_fg = tot_bg = tot_frames = 0
    cov = []
    for si, (a, b) in enumerate(shots):
        seg = frames[a: min(b, a + 45)]
        pal, idxs = prep_idx(seg, args.width, args.colors)
        hexes = P.to_hex(pal)
        cache = {}
        pal_index = lambda h: cache.setdefault(h, P.index_of(hexes, h))

        bg = L.background(idxs)
        L.to_image(bg, pal).save(tmp)
        bg_tok = n(S.encode_frame(S.svg_paths(T.shrink(T.trace_file(tmp, args.preset))),
                                  pal_index, args.quant))

        full = fg = 0
        for idx in idxs:
            mask = L.moving_mask(idx, bg)
            cov.append(L.coverage(mask))
            L.to_image(idx, pal).save(tmp)
            full += n(S.encode_frame(S.svg_paths(T.shrink(T.trace_file(tmp, args.preset))),
                                     pal_index, args.quant))
            L.to_image(L.compose(idx, bg, mask), pal, hole_rgb=HOLE_RGB).save(tmp)
            paths = [p for p in S.svg_paths(T.shrink(T.trace_file(tmp, args.preset)))
                     if p[0] != "#FF00FF"]
            fg += n(S.encode_frame(paths, pal_index, args.quant))
        tot_full += full
        tot_fg += fg
        tot_bg += bg_tok
        tot_frames += len(seg)
        print("shot %-2d %2d frames  bg=%-5d  full=%-5.0f/f  fg=%-5.0f/f  moving=%.0f%%"
              % (si, len(seg), bg_tok, full / len(seg), fg / len(seg),
                 100 * np.mean(cov[-len(seg):])))

    print("\n%d frames, quant=%d" % (tot_frames, args.quant))
    print("full frame each step : %6.0f tok/frame" % (tot_full / tot_frames))
    print("bg once + fg layer   : %6.0f tok/frame (+%d once per shot)"
          % (tot_fg / tot_frames, tot_bg / len(shots)))
    print("saving               : %6.2fx" % (tot_full / max(tot_fg, 1)))
    per = tot_fg / tot_frames
    print("\nseconds of video per context (fg layer only):")
    for ctx in (4096, 8192, 16384, 32768):
        print("  ctx %-6d -> %5.1f s" % (ctx, ctx / per / 15))


if __name__ == "__main__":
    main()
