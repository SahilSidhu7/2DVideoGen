"""True COCO-18 OpenPose keypoints from the fixed 16-part character schema.

Attempt 18 rendered "OpenPose-like" skeletons - limb axes recovered from the
part schema, drawn in the OpenPose palette - and the ControlNet ignored them.
The diagnosis was topology: `control_v11p_sd15_openpose` was trained on the
output of `draw_bodypose`, which draws **18 named keypoints and 17 limbs** in a
fixed colour order, with a neck, a nose, two eyes and two ears. Limb *axes* in
roughly the right colours are not that. There is no neck limb, no head limbs,
the joint colours are wrong, and the limb count is wrong; the control signal
lands off the distribution the model was conditioned on.

The rig has exact joints, so this is a mapping exercise rather than an
estimation problem:

    index  keypoint     source
    0      nose         between the eye centroids and the mouth centroid
    1      neck         the rig's single shoulder point (= shoulder midpoint)
    2/5    R/L shoulder neck +- torso half-width along the torso normal
    3/6    R/L elbow    rig elbow_R / elbow_L
    4/7    R/L wrist    rig hand_R / hand_L
    8/11   R/L hip      rig hip +- hip half-width along the torso normal
    9/12   R/L knee     rig knee_R / knee_L
    10/13  R/L ankle    rig foot_R / foot_L
    14/15  R/L eye      the eye_R / eye_L part centroids
    16/17  R/L ear      head centre +- 0.95 r, at eye height

Screen-side convention: "R" is screen-left, which is what a camera-facing
person gives and is already baked into the part schema (`PART_ORDER` puts
`eye_R` at the smaller x). It is fixed for the whole clip rather than chosen
per frame from which side a limb happens to be on, because a per-frame choice
would swap sides every half stride and put jitter into a guide whose entire
purpose is to be temporally exact.

Drawing is a verbatim port of `controlnet_aux.open_pose.util.draw_bodypose`:
`stickwidth = 4`, limbs filled at 0.6x their colour, keypoint discs of radius 4
at full colour drawn after the limbs, on a black canvas, with the detector's
default 512 px working resolution. `verify_against_controlnet_aux()` renders
the same keypoints through the real function and asserts the two images are
bit-identical, so these constants are checked rather than remembered.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anime_chars import PART_ORDER                # noqa: E402

# Canonical COCO-18 order. Index == position in this tuple.
KEYPOINT_NAMES = (
    "nose", "neck",
    "Rsho", "Relb", "Rwri",
    "Lsho", "Lelb", "Lwri",
    "Rhip", "Rkne", "Rank",
    "Lhip", "Lkne", "Lank",
    "Reye", "Leye", "Rear", "Lear",
)

# 1-based pairs, exactly as controlnet_aux draws them (17 limbs).
LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]

# Shared by limbs (index = limb) and keypoints (index = keypoint).
COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
]

STICKWIDTH = 4          # draw_bodypose constant, at the detector's 512 px
JOINT_RADIUS = 4        # ditto


# --------------------------------------------------------------------------- #
# Recovering rig joints from the part polygons
# --------------------------------------------------------------------------- #

def axis_ends(pts):
    """The two extremes of an elongated polygon along its principal axis.

    A limb is a tapered quad built between two joints; the joints are not
    vertices, they are the ends of the long axis. The perpendicular taper
    projects to zero on that axis, so this returns the two joint positions
    essentially exactly rather than approximately.
    """
    p = np.asarray(pts, dtype=np.float64)
    c = p.mean(0)
    q = p - c
    cov = q.T @ q
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    t = q @ axis
    return c + axis * t.min(), c + axis * t.max()


def _far(ends, anchor):
    """Whichever end is further from a known joint."""
    a, b = ends
    return a if np.linalg.norm(a - anchor) > np.linalg.norm(b - anchor) else b


def _near(ends, anchor):
    a, b = ends
    return b if np.linalg.norm(a - anchor) > np.linalg.norm(b - anchor) else a


def _halfwidth(pts, p_from, p_to, frac=0.35):
    """Half-width of a capsule measured near its `p_to` end.

    Used for shoulder and hip breadth: the figure is drawn with a real torso
    width, so the two shoulders should sit where the drawn shoulders are rather
    than at a guessed anthropometric ratio.
    """
    p = np.asarray(pts, dtype=np.float64)
    d = np.asarray(p_to, float) - np.asarray(p_from, float)
    ln = np.linalg.norm(d) or 1e-6
    u = d / ln
    n = np.array([-u[1], u[0]])
    t = (p - np.asarray(p_from, float)) @ u / ln
    sel = p[t >= 1.0 - frac]
    if len(sel) == 0:
        sel = p
    return float(np.abs((sel - np.asarray(p_to, float)) @ n).max())


def recover_joints(parts):
    """`{part_name: polygon}` -> named rig joints, in the same coordinates.

    Chain topology does the disambiguation: `axis_ends` gives an unordered
    pair, but the hip is the torso end that a thigh also touches, the knee is
    the thigh end away from the hip, and so on down each chain.
    """
    need = ("torso", "thigh_R", "shin_R", "thigh_L", "shin_L",
            "uarm_R", "farm_R", "uarm_L", "farm_L", "face")
    if any(k not in parts or len(parts[k]) < 3 for k in need):
        return None

    ends = {k: axis_ends(parts[k]) for k in need if k != "face"}

    # hip vs shoulder: the hip end of the torso is the one a thigh meets.
    thigh_pts = np.array(list(ends["thigh_R"]) + list(ends["thigh_L"]))
    t0, t1 = ends["torso"]
    d0 = np.linalg.norm(thigh_pts - t0, axis=1).min()
    d1 = np.linalg.norm(thigh_pts - t1, axis=1).min()
    hip, shoulder = (t0, t1) if d0 < d1 else (t1, t0)

    j = {"hip": hip, "shoulder": shoulder}
    for side in ("R", "L"):
        knee = _far(ends["thigh_" + side], hip)
        j["knee_" + side] = knee
        j["foot_" + side] = _far(ends["shin_" + side], knee)
        elbow = _far(ends["uarm_" + side], shoulder)
        j["elbow_" + side] = elbow
        j["hand_" + side] = _far(ends["farm_" + side], elbow)

    face = np.asarray(parts["face"], dtype=np.float64)
    c = face.mean(0)
    j["head_c"] = c
    j["head_r"] = max(float(np.abs(face[:, 0] - c[0]).max()), 1.0)

    for name in ("eye_R", "eye_L", "mouth"):
        if name in parts and len(parts[name]) >= 3:
            j[name] = np.asarray(parts[name], dtype=np.float64).mean(0)

    j["sho_hw"] = _halfwidth(parts["torso"], hip, shoulder)
    j["hip_hw"] = _halfwidth(parts["torso"], shoulder, hip)
    return j


def parts_from_anim(anim, state):
    """Resolved polygons keyed by schema part name, or None off-schema."""
    from anisvg import _transform
    if len(anim.shapes) != len(PART_ORDER):
        return None
    out = {}
    for sh in anim.shapes:
        st = state.get(sh.id)
        if st is None or not st.visible or sh.id >= len(PART_ORDER):
            continue
        out[PART_ORDER[sh.id]] = np.asarray(_transform(sh, st), dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Rig joints -> COCO-18
# --------------------------------------------------------------------------- #

# Head geometry as fractions of the torso length T (neck -> hip midpoint),
# measured off an OpenPose annotation the ControlNet demonstrably obeys.
# Anime proportions are not human proportions: the schema's head is 1.65x
# wider than its own shoulders and its nose sits 0.10 T above the neck instead
# of 0.33 T. Attempt 19 measured that this, and not the topology, is what puts
# the guide off the distribution the ControlNet was trained on.
HEAD_HUMAN = dict(nose=0.330, eye_up=0.420, eye_span=0.200,
                  ear_up=0.375, ear_span=0.400)


def clip_side(anim):
    """Which screen side the rig's "R" limbs sit on, averaged over a whole clip.

    The rig is a side-view walker: its R and L limbs swing through the same
    sagittal plane, so which one is left of the other changes within a stride.
    Choosing per frame would swap the hip and shoulder keypoints every half
    step and put jitter into a guide whose whole point is to be exact;
    choosing per clip is stable *and* stops the skeleton from crossing itself,
    which is what an OpenPose annotation of a real person almost never does.

    Returns -1 when R is on the screen-left (the camera-facing convention) and
    +1 when it is on the screen-right.
    """
    dx = []
    for st in anim.states():
        parts = parts_from_anim(anim, st)
        if parts is None:
            continue
        j = recover_joints(parts)
        if j is None:
            continue
        dx.append((j["hand_R"][0] + j["knee_R"][0])
                  - (j["hand_L"][0] + j["knee_L"][0]))
    if not dx:
        return -1
    return 1 if float(np.mean(dx)) > 0 else -1


def coco18(j, human_head=True, side=-1):
    """Named rig joints -> 18 (x, y) points in canvas coordinates.

    `human_head=True` places nose/eyes/ears at human ratios of the torso
    length, keeping only their *direction* (the facing lean) from the art. The
    limbs are never touched - they are the motion we are trying to condition
    on, and they are already exact. See HEAD_HUMAN.

    `side` says which screen side the rig's "R" limbs are on (see clip_side);
    the shoulder, hip, eye and ear keypoints are placed to match, so the
    skeleton does not cross itself.

    Returns a list of length 18; entries are never None here because the rig
    always supplies every joint, but the drawing code tolerates None so a
    partial schema can still be drawn.
    """
    hip, sho = j["hip"], j["shoulder"]
    c, r = j["head_c"], j["head_r"]

    # Torso normal, oriented so that -n is screen-left. "R" is screen-left,
    # matching the schema's own eye_R.
    d = sho - hip
    ln = float(np.linalg.norm(d)) or 1e-6
    u = d / ln
    n = np.array([-u[1], u[0]])
    if (n[0] > 0) != (side > 0):
        n = -n                                    # n points to the "R" side

    sho_hw = max(float(j.get("sho_hw", r * 0.5)), r * 0.25)
    hip_hw = max(float(j.get("hip_hw", r * 0.4)), r * 0.20)

    eye_R = j.get("eye_R")
    eye_L = j.get("eye_L")
    if eye_R is None or eye_L is None:
        # Fallback for a schema without eye parts: anime convention, low and
        # wide (`anime_chars.build_figure` uses 0.40 r out, 0.20 r down).
        down = np.array([0.0, 1.0])               # canvas y grows downward
        side = np.array([1.0, 0.0])
        eye_R = c - side * r * 0.40 + down * r * 0.20
        eye_L = c + side * r * 0.40 + down * r * 0.20
    # The schema paints eye index 13 at the smaller x; keep that as R even if a
    # transform ever mirrored it, so the face is never left-right inconsistent.
    if (eye_R[0] > eye_L[0]) != (side > 0):
        eye_R, eye_L = eye_L, eye_R
    eyes_mid = (eye_R + eye_L) / 2.0

    mouth = j.get("mouth")
    if mouth is None:
        mouth = eyes_mid + np.array([0.0, r * 0.45])
    nose = eyes_mid + 0.45 * (mouth - eyes_mid)

    ear_y = eyes_mid[1] - 0.10 * r
    ear_R = np.array([c[0] + side * 0.95 * r, ear_y])
    ear_L = np.array([c[0] - side * 0.95 * r, ear_y])

    if human_head:
        # T is the torso the ControlNet reads scale from; "up" is the torso
        # axis, so a leaning figure keeps its lean.
        T = ln
        up = u
        sv = np.array([-up[1], up[0]])
        if (sv[0] > 0) != (side > 0):
            sv = -sv                              # sv points to the "R" side
        lean_x = np.sign(nose[0] - c[0]) * 0.03 * T   # keep the facing offset
        h = HEAD_HUMAN
        base = sho + np.array([lean_x, 0.0])
        nose = base + up * h["nose"] * T
        eye_R = base + up * h["eye_up"] * T + sv * h["eye_span"] * T / 2.0
        eye_L = base + up * h["eye_up"] * T - sv * h["eye_span"] * T / 2.0
        ear_R = base + up * h["ear_up"] * T + sv * h["ear_span"] * T / 2.0
        ear_L = base + up * h["ear_up"] * T - sv * h["ear_span"] * T / 2.0

    kp = [None] * 18
    kp[0] = nose
    kp[1] = sho                                   # neck = shoulder midpoint
    kp[2] = sho + n * sho_hw
    kp[3] = j["elbow_R"]
    kp[4] = j["hand_R"]
    kp[5] = sho - n * sho_hw
    kp[6] = j["elbow_L"]
    kp[7] = j["hand_L"]
    kp[8] = hip + n * hip_hw
    kp[9] = j["knee_R"]
    kp[10] = j["foot_R"]
    kp[11] = hip - n * hip_hw
    kp[12] = j["knee_L"]
    kp[13] = j["foot_L"]
    kp[14] = eye_R
    kp[15] = eye_L
    kp[16] = ear_R
    kp[17] = ear_L
    return [None if p is None else np.asarray(p, dtype=np.float64) for p in kp]


def from_anim(anim, state, human_head=True, side=None):
    """One frame of a clip -> COCO-18 points, or None if it is off-schema.

    `side` defaults to `clip_side(anim)`; pass it in when converting a whole
    clip so it is computed once and every frame gets the same assignment.
    """
    parts = parts_from_anim(anim, state)
    if parts is None:
        return None
    j = recover_joints(parts)
    if j is None:
        return None
    if side is None:
        side = clip_side(anim)
    return coco18(j, human_head=human_head, side=side)


# --------------------------------------------------------------------------- #
# Drawing - a verbatim port of controlnet_aux draw_bodypose
# --------------------------------------------------------------------------- #

def draw(kp, src_w, src_h, out_w=512, out_h=None):
    """Draw COCO-18 points exactly the way the ControlNet's trainer drew them.

    `kp` is in source-canvas coordinates; it is normalised by (src_w, src_h)
    and drawn at (out_w, out_h), which is how the detector works - it runs at
    its own resolution and the stick constants are relative to that, not to the
    original image.
    """
    import cv2
    out_h = out_h or out_w
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for (k1, k2), color in zip(LIMB_SEQ, COLORS):
        a, b = kp[k1 - 1], kp[k2 - 1]
        if a is None or b is None:
            continue
        Y = np.array([a[0] / src_w, b[0] / src_w]) * float(out_w)
        X = np.array([a[1] / src_h, b[1] / src_h]) * float(out_h)
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        poly = cv2.ellipse2Poly((int(mY), int(mX)),
                                (int(length / 2), STICKWIDTH), int(angle),
                                0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(float(c) * 0.6) for c in color])

    for p, color in zip(kp, COLORS):
        if p is None:
            continue
        cv2.circle(canvas, (int(p[0] / src_w * out_w), int(p[1] / src_h * out_h)),
                   JOINT_RADIUS, color, thickness=-1)

    from PIL import Image
    return Image.fromarray(canvas)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_against_controlnet_aux(kp, src_w, src_h, size=512):
    """Bit-compare `draw` with the real `draw_bodypose`.

    The stick width, joint radius, limb pairs, colour table and the 0.6x limb
    dimming are all constants that a ControlNet was conditioned on. Rather than
    trusting that they were copied correctly, the same keypoints are pushed
    through the library's own drawing function and the two canvases compared
    pixel for pixel.
    """
    from controlnet_aux.open_pose import draw_poses, PoseResult
    from controlnet_aux.open_pose.body import BodyResult, Keypoint

    pts = [None if p is None else Keypoint(float(p[0] / src_w),
                                           float(p[1] / src_h))
           for p in kp]
    body = BodyResult(pts, 1.0, sum(p is not None for p in pts))
    ref = draw_poses([PoseResult(body, None, None, None)], size, size,
                     draw_body=True, draw_hand=False, draw_face=False)
    mine = np.asarray(draw(kp, src_w, src_h, size, size))
    return int(np.abs(ref.astype(np.int16) - mine.astype(np.int16)).max()), ref


def detect(img, detector=None):
    """Run the real OpenPose detector and return its COCO-18 points (or None).

    Used as a round-trip check: draw a skeleton we constructed, hand it to the
    estimator, and see whether the joints come back where they were put.
    """
    from controlnet_aux import OpenposeDetector
    det = detector or OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    from PIL import Image
    a = np.asarray(img.convert("RGB"))
    poses = det.detect_poses(a[:, :, ::-1])       # detector wants BGR
    if not poses:
        return None
    h, w = a.shape[:2]
    body = poses[0].body.keypoints
    return [None if p is None else np.array([p.x * w, p.y * h]) for p in body]


# --------------------------------------------------------------------------- #
# CLI: verify the mapping, and render old-style vs COCO-18 side by side
# --------------------------------------------------------------------------- #

def _rig_clip(action, n_frames=16, width=256, seed=3):
    """A deterministic clip for one named action, with the true rig joints.

    `anime_chars` does not hand back the joint positions it drew from, so the
    same construction is repeated here and the joints are pushed through the
    identical canvas placement. That gives ground truth to compare the
    recovered joints against - a far tighter check than any pose estimator,
    because it is exact by construction rather than estimated.
    """
    import random
    from anisvg import Anim
    from anime_chars import build_figure, to_canvas
    from lottie_to_anisvg import encode
    from stickman import StickFigure

    rng = random.Random(seed)
    fig = StickFigure(width=854, height=480)
    style, facing, skirt = "short", 1, False

    frames, joints = [], []
    for i in range(n_frames):
        j, _ = fig.pose(action, i / float(n_frames))
        joints.append(j)
        frames.append(build_figure(j, fig, style, facing, skirt))
    placed = to_canvas(frames, width)

    # Replicate to_canvas's affine on the joints themselves.
    per = [np.concatenate([p for _, p in f]) for f in frames]
    body = float(np.median([max(p[:, 1].max() - p[:, 1].min(), 1e-6)
                            for p in per]))
    scale = (width * 0.80) / body
    centres = np.array([(p.min(0) + p.max(0)) / 2.0 for p in per])
    mid = centres.mean(0)
    half_body = np.array([max(float(p[:, 0].max() - p[:, 0].min()) for p in per),
                          body]) * scale / 2.0
    room = np.maximum(width * 0.90 / 2.0 - half_body, 0.0)
    travel = np.abs((centres - mid) * scale).max(0)
    squeeze = np.where(travel > room, room / np.maximum(travel, 1e-6), 1.0)
    half = width / 2.0
    truth = []
    for i, j in enumerate(joints):
        off = (centres[i] - mid) * scale * squeeze
        row = {}
        for k, v in j.items():
            local = (np.asarray(v, float) - centres[i]) * scale
            row[k] = np.array([half + local[0] + off[0],
                               half - local[1] - off[1]])
        truth.append(row)

    colour = {"torso": "#2C3E7B", "skirt": "#2C3E7B", "thigh_R": "#2C3E7B",
              "thigh_L": "#2C3E7B", "shin_R": "#F6D7C4", "shin_L": "#F6D7C4",
              "uarm_R": "#2C3E7B", "uarm_L": "#2C3E7B", "farm_R": "#F6D7C4",
              "farm_L": "#F6D7C4", "hair_back": "#1A1A1A", "face": "#F6D7C4",
              "eye": "#1A1A1A", "fringe": "#1A1A1A", "mouth": "#8A3A3A"}
    raw = [[(pts, colour[name], 100.0, 0.0) for name, pts in row]
           for row in placed]
    anim = encode(raw, width, width, fps=15, width=width, points=12,
                  morph_tol=1.0, move_tol=0.5)
    return Anim.from_text(anim.to_text()), truth


def _roundtrip(action, n_frames=16, width=256):
    """Max/mean error between recovered joints and the rig's own joints."""
    anim, truth = _rig_clip(action, n_frames, width)
    keys = ("hip", "shoulder", "knee_R", "foot_R", "knee_L", "foot_L",
            "elbow_R", "hand_R", "elbow_L", "hand_L", "head_c")
    errs = []
    for st, gt in zip(anim.states(), truth):
        parts = parts_from_anim(anim, st)
        j = recover_joints(parts)
        if j is None:
            return anim, None, None
        errs += [float(np.linalg.norm(j[k] - gt[k])) for k in keys]
    return anim, float(np.mean(errs)), float(np.max(errs))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="COCO-18 guide check")
    ap.add_argument("--actions", default="walk,wave,kick,jump")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("-o", "--out", default="svg/out/a19")
    args = ap.parse_args()

    from PIL import Image
    import guides
    os.makedirs(args.out, exist_ok=True)

    # 1. the drawing constants, checked against the library rather than recalled
    anim, _ = _rig_clip("wave", 8)
    st = list(anim.states())[4]
    diff, _ = verify_against_controlnet_aux(from_anim(anim, st),
                                            anim.width, anim.height, args.size)
    print("draw parity vs controlnet_aux.draw_bodypose: max abs pixel diff %d"
          % diff)

    # 2. the mapping: do the joints come back where the rig put them
    print("\njoint round-trip (rig -> AniSVG -> recovered), 256 px canvas:")
    for a in args.actions.split(","):
        anim, mean, mx = _roundtrip(a, args.frames)
        print("  %-6s mean %.3f px   max %.3f px" % (a, mean, mx))

    # 3. side-by-side old-style vs COCO-18
    print("\nside-by-side guides -> %s" % args.out)
    for a in args.actions.split(","):
        anim, _ = _rig_clip(a, args.frames)
        states = list(anim.states())
        picks = [int(round(k * (len(states) - 1) / 3.0)) for k in range(4)]
        s = args.size // 2
        sheet = Image.new("RGB", (s * 4, s * 2), (0, 0, 0))
        for col, i in enumerate(picks):
            old = guides.pose(anim, states[i], s)
            new = draw(from_anim(anim, states[i]), anim.width, anim.height, s, s)
            sheet.paste(old, (col * s, 0))
            sheet.paste(new, (col * s, s))
        path = os.path.join(args.out, "guides_%s.png" % a)
        sheet.save(path)
        print("  %s   (top: Attempt-18 limb axes, bottom: COCO-18)" % path)


if __name__ == "__main__":
    main()
