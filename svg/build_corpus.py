"""video -> shots -> 15 fps traced-SVG sequences.

Output is one JSONL record per shot:
    {clip, shot, fps, width, height, palette: [hex...], frames: [svg...]}

Shots matter: a palette and any delta encoding are only valid inside a single
continuous shot, so cuts are detected and sequences never cross them.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P        # noqa: E402
import trace as T          # noqa: E402
from prep import crop_bars  # noqa: E402


def read_frames(path, fps, max_seconds=None):
    """Decode a video at a fixed fps, yielding PIL RGB frames."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(src_fps / fps))
    limit = int(max_seconds * src_fps) if max_seconds else None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit and i > limit):
            break
        if i % step == 0:
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()


def shot_split(frames, thresh=18.0, min_len=8):
    """Split on hard cuts using mean abs difference of tiny grayscale frames."""
    keys = [np.asarray(f.convert("L").resize((32, 32)), dtype=np.float32)
            for f in frames]
    cuts = [0]
    for i in range(1, len(keys)):
        if np.abs(keys[i] - keys[i - 1]).mean() > thresh:
            cuts.append(i)
    cuts.append(len(frames))
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_len]


def trace_shot(frames, width, colors, preset, decimals, tmp):
    pal = P.fit(frames[:: max(1, len(frames) // 8)], colors=colors)
    hexes = P.to_hex(pal)
    svgs = []
    for f in frames:
        img = crop_bars(f)
        h = max(1, round(img.height * width / img.width))
        img = img.convert("RGB").resize((width, h), Image.LANCZOS)
        img, _ = P.apply(img, pal)
        img.save(tmp)
        svgs.append(T.shrink(T.trace_file(tmp, preset), decimals=decimals))
    return hexes, svgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("-o", "--out", default="svg/data/shots.jsonl")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--preset", default="compact")
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--max-shot-frames", type=int, default=90)
    ap.add_argument("--min-shot-frames", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = os.path.join(os.path.dirname(args.out), "_frame.png")
    n_shots = n_frames = 0
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for vid in args.videos:
            name = os.path.splitext(os.path.basename(vid))[0]
            frames = list(read_frames(vid, args.fps, args.max_seconds))
            shots = shot_split(frames, min_len=args.min_shot_frames)
            print("%s: %d frames @%dfps -> %d shots"
                  % (name, len(frames), args.fps, len(shots)))
            for si, (a, b) in enumerate(shots):
                seg = frames[a: min(b, a + args.max_shot_frames)]
                hexes, svgs = trace_shot(seg, args.width, args.colors,
                                         args.preset, args.decimals, tmp)
                w, h = T.canvas_size(svgs[0])
                fh.write(json.dumps(dict(
                    clip=name, shot=si, start=a, fps=args.fps,
                    width=w, height=h, palette=hexes, frames=svgs)) + "\n")
                n_shots += 1
                n_frames += len(svgs)
                print("  shot %-3d %3d frames  %5.1f s elapsed"
                      % (si, len(svgs), time.time() - t0))
    print("\n%d shots / %d frames -> %s (%.1f MB, %.0f s)"
          % (n_shots, n_frames, args.out,
             os.path.getsize(args.out) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
