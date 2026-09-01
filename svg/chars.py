"""Procedural character animation -> AniSVG, with captions.

The Lottie corpus taught the model a grammar and a motion vocabulary of
rotation and translation, because that is what motion-graphics iconography
contains. It has no characters in it, so no amount of training on it produces
one.

This generates the missing data instead. `stickman.StickFigure` is an
articulated rig from an earlier era of this project: it maps an action and a
progress value to real joint positions. Sampling it over time gives exactly the
shape correspondence AniSVG needs - limb k is the same limb at every frame - so
clips come out at the same token cost as the Lottie ones and can be mixed with
them.

Unlike a scraped corpus this is unbounded, balanced by construction, and
captioned from the parameters that generated it, so the caption is true rather
than inferred.
"""
import argparse
import gzip
import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "svg"))
sys.path.insert(0, ROOT)

from lottie_to_anisvg import encode              # noqa: E402
from stickman import StickFigure                 # noqa: E402

ACTIONS = ["walk", "run", "jump", "kick", "wave", "dance"]

# Limb order is fixed so shape k means the same body part in every clip - the
# model can then learn "shape 3 is a shin" rather than re-deriving it per clip.
LIMBS = [
    ("spine", "hip", "shoulder"),
    ("thigh_R", "hip", "knee_R"),
    ("shin_R", "knee_R", "foot_R"),
    ("thigh_L", "hip", "knee_L"),
    ("shin_L", "knee_L", "foot_L"),
    ("uarm_R", "shoulder", "elbow_R"),
    ("farm_R", "elbow_R", "hand_R"),
    ("uarm_L", "shoulder", "elbow_L"),
    ("farm_L", "elbow_L", "hand_L"),
]

INK = [
    ("#000000", "black"), ("#1A1A1A", "black"), ("#123A6B", "dark blue"),
    ("#2E7D32", "green"), ("#8E24AA", "purple"), ("#C62828", "red"),
    ("#00695C", "teal"), ("#4E342E", "brown"),
]

SUBJECT = ["stick figure", "line-drawn figure", "simple character",
           "minimal human figure", "line-art person"]

MOVES = {
    "walk": ["walks steadily", "strides forward", "walks at an even pace"],
    "run":  ["runs quickly", "sprints forward", "runs with long strides"],
    "jump": ["jumps upward and lands", "leaps into the air", "hops upward"],
    "kick": ["swings one leg in a kick", "kicks forward", "performs a kick"],
    "wave": ["raises one arm and waves", "waves a hand overhead",
             "lifts an arm and waves it"],
    "dance": ["dances, swaying side to side", "moves rhythmically",
              "sways and steps in a dance"],
}


def circle(cx, cy, r, n=12):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def clip_frames(fig, action, n_frames, cycles):
    """Sample the rig; returns per-frame shape lists in world units."""
    out = []
    for i in range(n_frames):
        progress = (i / float(n_frames)) * cycles % 1.0
        joints, props = fig.pose(action, progress)
        shapes = [circle(joints["head_c"][0], joints["head_c"][1], fig.head_r)]
        for _, a, b in LIMBS:
            shapes.append(np.array([joints[a], joints[b]], dtype=np.float64))
        out.append((shapes, props))
    return out


def to_canvas(frames, width, margin=0.10, fill=0.72):
    """Place the rig on the canvas: sized by the body, moved by the motion.

    Scaling to the box that spans every frame would make a running figure tiny,
    because its travel dwarfs its body. So scale comes from the *median frame*
    (the figure) and translation comes from the trajectory, which is compressed
    toward its own mean only as far as needed to stay on canvas. The character
    stays large and the movement stays visible; only its amplitude is bounded.
    """
    per_frame = [np.concatenate(sh) for sh, _ in frames]
    spans = [max(float(p[:, 1].max() - p[:, 1].min()), 1e-6) for p in per_frame]
    body = float(np.median(spans))
    scale = (width * fill) / body

    centres = np.array([(p.min(0) + p.max(0)) / 2.0 for p in per_frame])
    mid = centres.mean(0)

    # How far the trajectory may stray before the figure clips the frame.
    half_body = np.array([
        max(float(p[:, 0].max() - p[:, 0].min()) for p in per_frame),
        body]) * scale / 2.0
    room = np.maximum(width * (1 - margin) / 2.0 - half_body, 0.0)
    travel = np.abs((centres - mid) * scale).max(0)
    squeeze = np.where(travel > room, room / np.maximum(travel, 1e-6), 1.0)

    half = width / 2.0

    def place(p, c):
        off = (c - mid) * scale * squeeze          # bounded trajectory
        local = (p - c) * scale                    # body, at full size
        x = half + local[:, 0] + off[0]
        y = half - local[:, 1] - off[1]            # world is y-up, canvas y-down
        return np.stack([x, y], axis=1)

    return [[place(s, centres[i]) for s in sh]
            for i, (sh, _) in enumerate(frames)]


def caption(action, colour_name, subject, cycles, stroke):
    move = random.choice(MOVES[action])
    bg = random.choice(["a plain white background", "a blank background",
                        "an empty white background"])
    weight = "thick" if stroke >= 3.4 else ("thin" if stroke <= 2.2 else "")
    desc = " ".join(x for x in [weight, colour_name, subject] if x)
    tail = ""
    if cycles >= 2 and action in ("walk", "run", "dance"):
        tail = ", repeating the motion"
    return "A %s %s against %s%s." % (desc, move, bg, tail)


def make_clip(rng, width=256, fps=15):
    action = rng.choice(ACTIONS)
    fig = StickFigure(width=854, height=480)
    # Vary proportions so the model sees a body plan, not one memorised figure.
    fig.L_thigh *= rng.uniform(0.85, 1.15)
    fig.L_shin *= rng.uniform(0.85, 1.15)
    fig.L_uarm *= rng.uniform(0.85, 1.15)
    fig.L_farm *= rng.uniform(0.85, 1.15)
    fig.torso *= rng.uniform(0.9, 1.1)
    fig.head_r *= rng.uniform(0.85, 1.2)

    cycles = rng.choice([1, 1, 2, 2, 3])
    n_frames = rng.randint(16, 34)
    raw = clip_frames(fig, action, n_frames, cycles)
    polys = to_canvas(raw, width)

    ink, ink_name = INK[rng.randrange(len(INK))]
    stroke = rng.uniform(1.8, 4.0)
    subject = rng.choice(SUBJECT)

    # Head is a filled disc; limbs are stroked segments.
    frames_raw = []
    for shapes in polys:
        row = [(shapes[0], ink, 100.0, 0.0)]
        row += [(s, ink, 100.0, stroke) for s in shapes[1:]]
        frames_raw.append(row)

    anim = encode(frames_raw, width, width, fps=fps, width=width,
                  points=12, morph_tol=1.0, move_tol=0.5)
    return anim, caption(action, ink_name, subject, cycles, stroke), action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=20000)
    ap.add_argument("-o", "--out", default="svg/data/chars/chars.jsonl.gz")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    per_action = {a: 0 for a in ACTIONS}
    kept = bad = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        while kept < args.count:
            try:
                anim, cap, action = make_clip(rng, args.width, args.fps)
            except Exception as exc:
                bad += 1
                if bad < 4:
                    print("skip:", str(exc)[:70])
                if bad > args.count:
                    raise SystemExit("generator failing: %s" % exc)
                continue
            text = anim.to_text()
            fh.write(json.dumps(dict(
                name="chr%06d" % kept, caption=cap, tags=[action],
                shapes=len(anim.shapes), frames=len(anim.frames),
                text=text, skipped=[])) + "\n")
            per_action[action] += 1
            kept += 1
            if kept % 2000 == 0:
                print("  %6d clips" % kept)

    size = os.path.getsize(args.out) / 1e6
    print("\n%d clips, %d failed -> %s (%.1f MB gz)" % (kept, bad, args.out, size))
    print("per action: " + ", ".join("%s %d" % kv for kv in sorted(per_action.items())))


if __name__ == "__main__":
    main()
