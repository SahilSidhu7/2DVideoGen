"""Cel-style character animation -> AniSVG, with captions.

`chars.py` proves the rig drives AniSVG correctly, but a stick figure is not
what "2D anime" means. This dresses the same joints in the idiom the style
actually uses: an oversized head, a hair silhouette drawn over it, large eyes,
a tapered body and limbs with real width, and optional clothing.

The construction is deliberate about cost. Every part is a polygon whose vertex
count is fixed for the whole clip, so shape k is the same body part at every
frame and the encoder charges a few transform ops rather than a coordinate
dump. A full figure is 16-18 shapes, which at ~6 tokens per moving shape per
frame keeps a two-second clip inside a 4096-token context.

Anatomy is anime convention rather than realism: head about a quarter of body
height, eyes low and wide on the face.
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

# Fixed, in paint order. Shape index k is PART_ORDER[k] in every clip, which is
# what lets `guides.py` rebuild a skeleton from generated AniSVG.
PART_ORDER = ["torso", "skirt",
              "thigh_R", "shin_R", "thigh_L", "shin_L",
              "uarm_R", "farm_R", "uarm_L", "farm_L",
              "hair_back", "face", "fringe", "eye_R", "eye_L", "mouth"]

SKIN = [("#F6D7C4", "pale"), ("#E8BE9C", "fair"), ("#C98F६B".replace("६","6"), "tan"),
        ("#8D5A3B", "brown")]
HAIR = [("#1A1A1A", "black"), ("#3B2313", "dark brown"), ("#B8860B", "blonde"),
        ("#7B2D26", "red"), ("#2E4A7D", "blue"), ("#5B2E6E", "purple"),
        ("#D96BA0", "pink"), ("#3F7D5A", "green")]
CLOTH = [("#2C3E7B", "blue"), ("#7B2C3E", "red"), ("#2F5D3A", "green"),
         ("#3A3A44", "grey"), ("#6B4B8A", "purple"), ("#1F1F1F", "black")]

MOVES = {
    "walk": ["walks steadily", "walks forward at an even pace"],
    "run":  ["runs quickly", "sprints forward"],
    "jump": ["jumps upward and lands", "leaps into the air"],
    "kick": ["kicks forward", "swings a leg in a high kick"],
    "wave": ["raises an arm and waves", "waves a hand overhead"],
    "dance": ["dances, swaying side to side", "sways and steps in a dance"],
}


def poly_circle(cx, cy, r, n=12, squash=1.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * squash * np.sin(a)], axis=1)


def capsule(p0, p1, w0, w1, n=8):
    """A tapered quad along p0->p1, so a limb has thickness, not just a line."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ln = math.hypot(*d) or 1e-6
    nx, ny = -d[1] / ln, d[0] / ln
    steps = np.linspace(0, 1, n // 2)
    wide = w0 + (w1 - w0) * steps
    left = np.stack([p0[0] + d[0] * steps + nx * wide,
                     p0[1] + d[1] * steps + ny * wide], axis=1)
    right = np.stack([p0[0] + d[0] * steps[::-1] - nx * wide[::-1],
                      p0[1] + d[1] * steps[::-1] - ny * wide[::-1]], axis=1)
    return np.vstack([left, right])


def hair_shape(cx, cy, r, style, facing):
    """A silhouette drawn over the skull; the style changes its outline."""
    n = 14
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rad = np.full(n, r * 1.16)
    upper = np.cos(a) > -0.15                       # the top of the head
    rad[upper] *= 1.06
    if style == "long":
        low = np.sin(a) < 0
        rad[low] *= 1.85
    elif style == "twin":
        side = np.abs(np.cos(a)) > 0.75
        rad[side] *= 1.7
    elif style == "spiky":
        rad *= (1.0 + 0.3 * (np.arange(n) % 2) * upper)
    pts = np.stack([cx + rad * np.cos(a), cy + rad * np.sin(a)], axis=1)
    pts[:, 0] += facing * r * 0.06                  # fringe leans with facing
    return pts


def build_figure(joints, fig, style, facing, has_skirt):
    """Joints -> ordered part polygons. Order is fixed for every clip."""
    hip = np.array(joints["hip"], float)
    sho = np.array(joints["shoulder"], float)
    head = np.array(joints["head_c"], float)
    hr = fig.head_r * 1.95                          # anime head: large

    body_w = fig.torso * 0.30
    limb_w = fig.torso * 0.11

    parts = []
    # torso, then limbs, then head group: painted back-to-front by list order.
    # Every part is emitted every time, in this order, for every clip - the
    # garment varies in width, never in existence. A constant schema means
    # shape index k is the same body part in generated output too, so a pose
    # skeleton can be recovered from what the model writes, not only from what
    # this generator produced.
    parts.append(("torso", capsule(hip, sho, body_w * 0.78, body_w, 10)))
    w = body_w * (1.55 if has_skirt else 0.85)
    drop = fig.torso * (0.42 if has_skirt else 0.20)
    down = (hip - sho)
    down = down / (np.linalg.norm(down) or 1e-6) * drop
    parts.append(("skirt", np.array([
        hip + [-body_w * 0.8, 0], hip + [body_w * 0.8, 0],
        hip + down + [w, 0], hip + down + [-w, 0]], float)))
    for name, a, b, w0, w1 in (
            ("thigh_R", "hip", "knee_R", limb_w * 1.25, limb_w),
            ("shin_R", "knee_R", "foot_R", limb_w, limb_w * 0.7),
            ("thigh_L", "hip", "knee_L", limb_w * 1.25, limb_w),
            ("shin_L", "knee_L", "foot_L", limb_w, limb_w * 0.7),
            ("uarm_R", "shoulder", "elbow_R", limb_w, limb_w * 0.8),
            ("farm_R", "elbow_R", "hand_R", limb_w * 0.8, limb_w * 0.6),
            ("uarm_L", "shoulder", "elbow_L", limb_w, limb_w * 0.8),
            ("farm_L", "elbow_L", "hand_L", limb_w * 0.8, limb_w * 0.6)):
        parts.append((name, capsule(joints[a], joints[b], w0, w1, 8)))

    # Paint order is the whole trick on a face: silhouette, then skin, then the
    # fringe across the brow only, and the eyes last so nothing can cover them.
    parts.append(("hair_back", hair_shape(head[0], head[1], hr, style, facing)))
    parts.append(("face", poly_circle(head[0], head[1], hr, 12, squash=1.08)))
    parts.append(("fringe", hair_shape(head[0], head[1] + hr * 0.62,
                                       hr * 0.52, style, facing)))
    # Eyes sit low and wide - the single strongest cue of the style.
    ex, ey = hr * 0.40, -hr * 0.20
    for side in (-1, 1):
        parts.append(("eye", poly_circle(head[0] + side * ex + facing * hr * 0.06,
                                         head[1] + ey, hr * 0.24, 8, squash=1.45)))
    parts.append(("mouth", poly_circle(head[0] + facing * hr * 0.08,
                                       head[1] - hr * 0.60, hr * 0.13, 6,
                                       squash=0.55)))
    return parts


def clip_frames(fig, action, n_frames, cycles, style, facing, skirt):
    out = []
    for i in range(n_frames):
        progress = (i / float(n_frames)) * cycles % 1.0
        joints, _ = fig.pose(action, progress)
        out.append(build_figure(joints, fig, style, facing, skirt))
    return out


def to_canvas(frames, width, margin=0.10, fill=0.80):
    """Size by the body, translate by the motion (see chars.to_canvas)."""
    per = [np.concatenate([p for _, p in f]) for f in frames]
    body = float(np.median([max(p[:, 1].max() - p[:, 1].min(), 1e-6) for p in per]))
    scale = (width * fill) / body
    centres = np.array([(p.min(0) + p.max(0)) / 2.0 for p in per])
    mid = centres.mean(0)
    half_body = np.array([max(float(p[:, 0].max() - p[:, 0].min()) for p in per),
                          body]) * scale / 2.0
    room = np.maximum(width * (1 - margin) / 2.0 - half_body, 0.0)
    travel = np.abs((centres - mid) * scale).max(0)
    squeeze = np.where(travel > room, room / np.maximum(travel, 1e-6), 1.0)
    half = width / 2.0

    out = []
    for i, f in enumerate(frames):
        off = (centres[i] - mid) * scale * squeeze
        row = []
        for name, p in f:
            local = (p - centres[i]) * scale
            row.append((name, np.stack([half + local[:, 0] + off[0],
                                        half - local[:, 1] - off[1]], axis=1)))
        out.append(row)
    return out


def caption(action, hair_name, cloth_name, style, skirt, facing):
    move = random.choice(MOVES[action])
    style_word = {"long": "long", "twin": "twin-tailed", "spiky": "spiky",
                  "short": "short"}[style]
    outfit = "a skirt" if skirt else "shorts"
    side = "to the right" if facing > 0 else "to the left"
    return ("An anime-style character with %s %s hair, wearing %s %s outfit, "
            "%s %s on a plain background."
            % (style_word, hair_name, cloth_name, outfit, move, side))


def make_clip(rng, width=256, fps=15):
    action = rng.choice(ACTIONS)
    fig = StickFigure(width=854, height=480)
    fig.L_thigh *= rng.uniform(0.9, 1.1)
    fig.L_shin *= rng.uniform(0.9, 1.1)
    fig.L_uarm *= rng.uniform(0.9, 1.1)
    fig.L_farm *= rng.uniform(0.9, 1.1)
    fig.torso *= rng.uniform(0.95, 1.05)
    fig.head_r *= rng.uniform(0.95, 1.15)

    style = rng.choice(["short", "long", "twin", "spiky"])
    facing = rng.choice([-1, 1])
    skirt = rng.random() < 0.45
    cycles = rng.choice([1, 1, 2, 2])
    n_frames = rng.randint(16, 30)

    frames = clip_frames(fig, action, n_frames, cycles, style, facing, skirt)
    placed = to_canvas(frames, width)

    hair_hex, hair_name = HAIR[rng.randrange(len(HAIR))]
    skin_hex, _ = SKIN[rng.randrange(len(SKIN))]
    cloth_hex, cloth_name = CLOTH[rng.randrange(len(CLOTH))]
    line = "#1A1A1A"

    colour = {"torso": cloth_hex, "skirt": cloth_hex,
              "thigh_R": cloth_hex, "thigh_L": cloth_hex,
              "shin_R": skin_hex, "shin_L": skin_hex,
              "uarm_R": cloth_hex, "uarm_L": cloth_hex,
              "farm_R": skin_hex, "farm_L": skin_hex,
              "hair_back": hair_hex, "face": skin_hex,
              "eye": line, "fringe": hair_hex, "mouth": "#8A3A3A"}

    frames_raw = []
    for row in placed:
        frames_raw.append([(pts, colour[name], 100.0, 0.0) for name, pts in row])

    anim = encode(frames_raw, width, width, fps=fps, width=width,
                  points=12, morph_tol=1.0, move_tol=0.5)
    return anim, caption(action, hair_name, cloth_name, style, skirt, facing), action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=20000)
    ap.add_argument("-o", "--out", default="svg/data/anime/anime.jsonl.gz")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=11)
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
                    print("skip:", str(exc)[:80])
                if bad > max(50, args.count):
                    raise SystemExit("generator failing: %s" % exc)
                continue
            fh.write(json.dumps(dict(
                name="ani%06d" % kept, caption=cap, tags=[action, "anime"],
                shapes=len(anim.shapes), frames=len(anim.frames),
                text=anim.to_text(), skipped=[])) + "\n")
            per_action[action] += 1
            kept += 1
            if kept % 2000 == 0:
                print("  %6d clips" % kept)

    print("\n%d clips, %d failed -> %s (%.1f MB gz)"
          % (kept, bad, args.out, os.path.getsize(args.out) / 1e6))
    print("per action: " + ", ".join("%s %d" % kv for kv in sorted(per_action.items())))


if __name__ == "__main__":
    main()
