"""AniSVG -> ControlNet conditioning frames.

The hybrid split: the language model writes *motion* as vectors, a diffusion
model paints *style* per frame. This module produces the conditioning images
that join them.

Normally ControlNet conditioning is *estimated* from a rendered image by a
preprocessor - an openpose detector guessing joints, a line detector guessing
edges - and that estimate is a source of error and of flicker, because it is
recomputed independently per frame. Here both signals are exact and already
temporally coherent: the same declared cast moves, so limb k is limb k in every
frame by construction. Nothing is detected, so nothing can be detected
differently from one frame to the next.

Three guides are produced:

  lineart   shape outlines, for a lineart/scribble ControlNet
  pose      the Attempt-18 skeleton: limb axes in the OpenPose palette. Kept
            only as the baseline the COCO-18 guide is measured against - the
            ControlNet ignored it (see `coco18`).
  coco18    a true COCO-18 OpenPose skeleton: 18 named keypoints, 17 limbs,
            canonical colours and stick constants (`svg/coco18.py`). This is
            the conditioning the openpose ControlNet was actually trained on.
  flat      the ordinary colour render, for img2img or a reference adapter
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anisvg import Anim, _transform               # noqa: E402
from generate import extract                      # noqa: E402
import raster                                     # noqa: E402
from anime_chars import PART_ORDER                # noqa: E402
import coco18 as coco18_mod                       # noqa: E402

# OpenPose limb colours; a ControlNet trained on openpose expects these, so the
# skeleton is drawn in the palette the model was conditioned on.
POSE_COLOURS = {
    "torso": (0, 0, 255), "thigh_R": (0, 255, 0), "shin_R": (0, 255, 85),
    "thigh_L": (255, 85, 0), "shin_L": (255, 170, 0),
    "uarm_R": (0, 170, 255), "farm_R": (0, 85, 255),
    "uarm_L": (255, 0, 0), "farm_L": (255, 0, 85),
}
JOINT_COLOUR = (255, 255, 85)


def frame_shapes(anim, state):
    """Resolved polygons for one frame, in canvas coordinates."""
    out = []
    for sh in anim.shapes:
        st = state.get(sh.id)
        if st is None or not st.visible:
            continue
        out.append((sh, np.asarray(_transform(sh, st), dtype=np.float64)))
    return out


def axis_ends(pts):
    """The two ends of an elongated polygon, along its principal axis.

    A limb is built as a tapered quad, so its endpoints are not vertices - they
    are the extremes of the long axis, which is what a skeleton needs.
    """
    p = np.asarray(pts, dtype=np.float64)
    c = p.mean(0)
    q = p - c
    # principal direction via the 2x2 covariance, in closed form
    cov = q.T @ q
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    t = q @ axis
    return c + axis * t.min(), c + axis * t.max()


def lineart(anim, state, width=512, thickness=2, bg=(255, 255, 255),
            ink=(0, 0, 0)):
    """Outlines only - what a lineart ControlNet expects."""
    scale = width / float(anim.width)
    img = Image.new("RGB", (width, int(round(anim.height * scale))), bg)
    draw = ImageDraw.Draw(img)
    for _, pts in frame_shapes(anim, state):
        xy = [(float(x) * scale, float(y) * scale) for x, y in pts]
        if len(xy) >= 2:
            draw.line(xy + [xy[0]], fill=ink, width=thickness, joint="curve")
    return img


def pose(anim, state, width=512, thickness=8, radius=5):
    """OpenPose-style skeleton, recovered from the fixed part schema.

    Returns None when the clip does not match the schema - a Lottie icon has no
    skeleton, and guessing one would be worse than declining.
    """
    if len(anim.shapes) != len(PART_ORDER):
        return None
    scale = width / float(anim.width)
    img = Image.new("RGB", (width, int(round(anim.height * scale))), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    by_part = {}
    for sh, pts in frame_shapes(anim, state):
        if sh.id < len(PART_ORDER):
            by_part[PART_ORDER[sh.id]] = pts

    joints = []
    for part, colour in POSE_COLOURS.items():
        pts = by_part.get(part)
        if pts is None or len(pts) < 3:
            continue
        a, b = axis_ends(pts)
        draw.line([(a[0] * scale, a[1] * scale), (b[0] * scale, b[1] * scale)],
                  fill=colour, width=thickness)
        joints += [a, b]
    face = by_part.get("face")
    if face is not None:
        c = face.mean(0)
        r = max(float(np.abs(face - c).max()), 1.0) * 0.5
        draw.ellipse([(c[0] - r) * scale, (c[1] - r) * scale,
                      (c[0] + r) * scale, (c[1] + r) * scale],
                     outline=(255, 255, 255), width=thickness // 2 or 1)
        joints.append(c)
    for j in joints:
        draw.ellipse([j[0] * scale - radius, j[1] * scale - radius,
                      j[0] * scale + radius, j[1] * scale + radius],
                     fill=JOINT_COLOUR)
    return img


def pose_coco18(anim, state, width=512, side=None):
    """True COCO-18 OpenPose skeleton for one frame.

    `pose()` above draws the limb axes of the part schema in the OpenPose
    palette, which is a different thing from an OpenPose skeleton: no neck,
    nose, eyes or ears, joints at limb endpoints rather than at named body
    points, and 9 limbs instead of 17. Attempt 18 measured what that costs -
    the ControlNet ignored the pose entirely. This maps the same exact joints
    onto the canonical 18-keypoint topology instead. Returns None off-schema.
    """
    kp = coco18_mod.from_anim(anim, state, side=side)
    if kp is None:
        return None
    h = int(round(anim.height * (width / float(anim.width))))
    return coco18_mod.draw(kp, anim.width, anim.height, width, h)


def flat(anim, svg, width=512):
    return raster.render(svg, width=width)


def build(text, out_dir, width=512, kinds=("lineart", "pose", "flat"),
          stem="f"):
    anim = Anim.from_text(extract(text))
    states = list(anim.states())
    svgs = anim.to_svgs()
    made = {k: 0 for k in kinds}
    for k in kinds:
        os.makedirs(os.path.join(out_dir, k), exist_ok=True)
    # Computed once per clip, never per frame: see coco18.clip_side.
    side = coco18_mod.clip_side(anim) if "coco18" in kinds else None
    for i, st in enumerate(states):
        if "lineart" in kinds:
            lineart(anim, st, width).save(
                os.path.join(out_dir, "lineart", "%s%04d.png" % (stem, i)))
            made["lineart"] += 1
        if "pose" in kinds:
            im = pose(anim, st, width)
            if im is not None:
                im.save(os.path.join(out_dir, "pose", "%s%04d.png" % (stem, i)))
                made["pose"] += 1
        if "coco18" in kinds:
            im = pose_coco18(anim, st, width, side=side)
            if im is not None:
                im.save(os.path.join(out_dir, "coco18", "%s%04d.png" % (stem, i)))
                made["coco18"] += 1
        if "flat" in kinds:
            flat(anim, svgs[i], width).save(
                os.path.join(out_dir, "flat", "%s%04d.png" % (stem, i)))
            made["flat"] += 1
    return anim, made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help=".anisvg file, or a jsonl(.gz) row index as path#N")
    ap.add_argument("-o", "--out", default="svg/out/guides")
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    path, _, idx = args.clip.partition("#")
    if path.endswith(".gz") or path.endswith(".jsonl"):
        import gzip
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == int(idx or 0):
                    row = json.loads(line)
                    break
        text, cap = row["text"], row.get("caption", "")
    else:
        text, cap = open(path, encoding="utf-8").read(), ""

    anim, made = build(text, args.out, args.width,
                       kinds=("lineart", "pose", "coco18", "flat"))
    print("%d shapes, %d frames -> %s" % (len(anim.shapes), len(anim.frames), args.out))
    print("guides: " + ", ".join("%s %d" % kv for kv in sorted(made.items())))
    if made.get("coco18", 0) == 0:
        print("  (no pose: clip does not match the %d-part character schema)"
              % len(PART_ORDER))
    if cap:
        print("caption: %s" % cap)


if __name__ == "__main__":
    main()
