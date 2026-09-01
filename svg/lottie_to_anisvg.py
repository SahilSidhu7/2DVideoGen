"""Lottie JSON -> AniSVG.

Sampling the same Lottie shape tree at every timestep gives shape
correspondence *for free* - shape k at frame t is the same shape at frame t+1.
That is precisely what raster tracing could not provide (see ATTEMPTS.md), and
it is why this corpus is worth the conversion work.

Per frame each shape is fitted with a similarity transform against its cast
pose; only the leftover becomes a vertex morph.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit                                     # noqa: E402
import lottie as L                             # noqa: E402
from anisvg import Anim, Shape                 # noqa: E402


def resample(pts, n):
    """Resample a closed polyline to exactly n points, evenly by arc length.

    Fixed point counts keep vertex indices comparable between frames, which the
    morph residual depends on.
    """
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return np.repeat(p if len(p) else np.zeros((1, 2)), n, axis=0)
    loop = np.vstack([p, p[:1]])
    seg = np.sqrt(((loop[1:] - loop[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    want = np.linspace(0, total, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, want, side="right") - 1, 0, len(seg) - 1)
    frac = ((want - cum[idx]) / np.maximum(seg[idx], 1e-9))[:, None]
    return loop[idx] + (loop[idx + 1] - loop[idx]) * frac


def quantise_palette(fills, max_colors):
    """Unique fills, capped; returns (palette, mapping fill -> index)."""
    order = sorted(set(fills))
    if len(order) <= max_colors:
        return order, {c: i for i, c in enumerate(order)}
    rgb = np.array([[int(c[1:][i:i + 2], 16) for i in (0, 2, 4)] for c in order],
                   dtype=np.float64)
    keep = [0]
    for _ in range(max_colors - 1):
        d = ((rgb[:, None] - rgb[keep][None]) ** 2).sum(-1).min(1)
        keep.append(int(d.argmax()))
    pal = [order[i] for i in keep]
    prgb = rgb[keep]
    mapping = {c: int(((prgb - rgb[i]) ** 2).sum(1).argmin())
               for i, c in enumerate(order)}
    return pal, mapping


def convert(doc, fps=15, width=256, points=16, max_colors=12,
            max_frames=90, morph_tol=1.2, move_tol=0.5):
    times, _ = L.times(doc, fps)
    times = times[:max_frames]
    if not times:
        raise ValueError("empty animation")

    frames_raw, skipped = [], set()
    for t in times:
        got, sk = L.sample(doc, t)
        skipped |= sk
        frames_raw.append(got)
    n_shapes = len(frames_raw[0])
    if n_shapes == 0:
        raise ValueError("no filled shapes")
    # Shape count must be stable for index-based correspondence to hold.
    frames_raw = [f for f in frames_raw if len(f) == n_shapes]
    if len(frames_raw) < 4:
        raise ValueError("shape count unstable across time")

    cw = float(doc.get("w", width) or width)
    ch = float(doc.get("h", width) or width)
    anim = encode(frames_raw, cw, ch, fps, width, points, max_colors,
                  morph_tol, move_tol)
    return anim, skipped


def encode(frames_raw, cw, ch, fps=15, width=256, points=16, max_colors=12,
           morph_tol=1.2, move_tol=0.5):
    """Per-frame shape samples -> an Anim, as integer deltas.

    `frames_raw[t][k]` is `(points, colour_hex, opacity, stroke_width)` for
    shape k at frame t, in source units. Shape k must be the same shape at every
    t - that correspondence is the whole reason this encoding is cheap, and it
    is what raster tracing could not provide.

    Kept separate from `convert` so any source of corresponded shapes can use
    it: Lottie today, procedural character rigs next.
    """
    n_shapes = len(frames_raw[0])
    scale = width / cw
    height = max(1, int(round(ch * scale)))

    # A shape already at its final vertex count is left alone; resampling a
    # two-point limb to sixteen would spend tokens describing a straight line.
    polys = [[(np.asarray(sh[0], dtype=np.float64) if len(sh[0]) == points
               else resample(sh[0], points)) * scale for sh in f]
             for f in frames_raw]
    fills = [sh[1] for sh in frames_raw[0]]
    strokes = [sh[3] for sh in frames_raw[0]]
    pal, pmap = quantise_palette(fills, max_colors)

    cast = [Shape(i, pmap[fills[i]], polys[0][i],
                  round(strokes[i] * scale * 10)) for i in range(n_shapes)]

    frames = []
    # `emitted` tracks the pose the *decoder* will have after replaying the ops
    # so far, which is not the true pose: every op is rounded to an integer.
    # Diffing against the emitted pose rather than the true one keeps rounding
    # error from accumulating, and lets a sub-unit change emit nothing at all
    # instead of a stream of no-op `t id 0 0`.
    emitted = [[0.0, 0.0, 0.0, 100.0] for _ in range(n_shapes)]
    prev_alpha = [sh[2] for sh in frames_raw[0]]
    prev_morph = [np.zeros((points, 2)) for _ in range(n_shapes)]
    for fi, poly in enumerate(polys):
        ops = []
        for si in range(n_shapes):
            tx, ty, rot, sc = fit.fit_similarity(polys[0][si], poly[si])
            state = emitted[si]

            dx, dy = int(round(tx - state[0])), int(round(ty - state[1]))
            if (dx or dy) and max(abs(tx - state[0]), abs(ty - state[1])) >= move_tol:
                ops.append(("t", si, dx, dy))
                state[0] += dx
                state[1] += dy
            dr = int(round(rot - state[2]))
            if dr:
                ops.append(("r", si, dr))
                state[2] += dr
            ds = int(round(sc - state[3]))
            if ds:
                ops.append(("s", si, ds))
                state[3] += ds

            res = fit.residual(polys[0][si], poly[si], state[0], state[1],
                               state[2], state[3])
            step = np.round(res - prev_morph[si]).astype(int)
            if fit.rms(res - prev_morph[si]) >= morph_tol and step.any():
                ops.append(("v", si) + tuple(int(v) for v in step.reshape(-1)))
                prev_morph[si] = prev_morph[si] + step

            alpha = frames_raw[fi][si][2]
            da = int(round(alpha - prev_alpha[si]))
            if abs(da) >= 5:
                ops.append(("o", si, da))
                prev_alpha[si] += da
        frames.append(ops)

    return Anim(width, height, fps, pal, cast, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lottie", nargs="+")
    ap.add_argument("-o", "--out", default="svg/data/anisvg.jsonl")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--tokens", action="store_true", help="report token stats")
    args = ap.parse_args()

    tok = None
    if args.tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ok = bad = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for path in args.lottie:
            try:
                with open(path, encoding="utf-8") as src:
                    doc = json.load(src)
                anim, skipped = convert(doc, args.fps, args.width, args.points)
            except Exception as exc:
                bad += 1
                print("skip %-24s %s" % (os.path.basename(path), str(exc)[:70]))
                continue
            text = anim.to_text()
            ok += 1
            fh.write(json.dumps(dict(name=os.path.basename(path),
                                     text=text, skipped=sorted(skipped))) + "\n")
            note = ""
            if tok:
                n = len(tok(text)["input_ids"])
                note = "  %5d tok total, %4.0f tok/frame" % (n, n / len(anim.frames))
            print("ok   %-24s %2d shapes %3d frames%s  skipped=%s"
                  % (os.path.basename(path), len(anim.shapes), len(anim.frames),
                     note, ",".join(sorted(skipped)) or "-"))
    print("\n%d converted, %d skipped -> %s" % (ok, bad, args.out))


if __name__ == "__main__":
    main()
