"""Build a demo reel from generated AniSVG clips.

Renders each clip, lays them out in a grid with its caption, and writes one
looping animation. Clips differ in length, so shorter ones hold their last
frame rather than restarting - a grid where every tile loops on its own period
reads as noise.
"""
import argparse
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anisvg import Anim                          # noqa: E402
from generate import extract, score              # noqa: E402
import raster                                    # noqa: E402


def frames_of(text, width):
    anim = Anim.from_text(extract(text))
    return [raster.render(s, width=width) for s in anim.to_svgs()], anim.fps


def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="svg/out/gen_con2")
    ap.add_argument("--val", default="svg/data/train_1024/val.jsonl")
    ap.add_argument("-o", "--out", default="svg/out/demo/anisvg_8b_demo.gif")
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=6)
    args = ap.parse_args()

    captions = [json.loads(l)["caption"]
                for l in open(args.val, encoding="utf-8")]
    picked = []
    for path in sorted(glob.glob(os.path.join(args.gen, "g*.anisvg"))):
        idx = int(os.path.basename(path)[1:4])
        text = open(path, encoding="utf-8").read()
        v = score(text)
        if not v["ok"]:
            continue
        picked.append((idx, text, captions[idx] if idx < len(captions) else "", v))
    # A clip can be structurally valid and still draw almost nothing, so rank on
    # what actually reaches the canvas: how much ink there is, and how much of it
    # moves between frames. Both are measured on the render, not on the ops.
    import numpy as np
    scored = []
    for idx, text, cap, v in picked:
        imgs, _ = frames_of(text, 96)
        arr = np.stack([np.asarray(im).mean(2) for im in imgs])
        ink = float((arr < 200).mean())
        motion = float((np.abs(np.diff(arr, axis=0)) > 24).mean()) if len(arr) > 1 else 0.0
        if ink < 0.004:                    # essentially a blank canvas
            print("skip g%03d - blank (ink %.4f)" % (idx, ink))
            continue
        scored.append((ink, motion, idx, text, cap, v))
    scored.sort(key=lambda s: -(s[1] * 100 + min(s[0], 0.25)))
    picked = [(idx, text, cap, v) for _, _, idx, text, cap, v in scored][:args.max_clips]
    if not picked:
        raise SystemExit("no valid clips in %s" % args.gen)

    fnt = font(12)
    strips, labels = [], []
    for idx, text, cap, v in picked:
        imgs, fps = frames_of(text, args.tile)
        strips.append(imgs)
        labels.append("%s  [%d shapes, %d frames]" % (cap[:70], v["shapes"], v["frames"]))
        print("g%03d  %2d shapes %2d frames %2d moving  %s"
              % (idx, v["shapes"], v["frames"], v["moving"], cap[:56]))

    n = len(strips)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    probe = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(probe)
    cap_h = 4 + 14 * max(len(wrap(d0, l, fnt, args.tile - 8)) for l in labels)
    cell_w, cell_h = args.tile, args.tile + cap_h
    total = max(len(s) for s in strips)

    out = []
    for f in range(total):
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for i, strip in enumerate(strips):
            x, y = (i % cols) * cell_w, (i // cols) * cell_h
            canvas.paste(strip[min(f, len(strip) - 1)], (x, y))
            ty = y + args.tile + 2
            for line in wrap(draw, labels[i], fnt, cell_w - 8):
                draw.text((x + 4, ty), line, fill=(40, 40, 40), font=fnt)
                ty += 14
        out.append(canvas)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out[0].save(args.out, save_all=True, append_images=out[1:],
                duration=int(1000 / 15), loop=0, optimize=False)
    print("\n%d clips, %d frames -> %s (%.1f KB)"
          % (n, total, args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
