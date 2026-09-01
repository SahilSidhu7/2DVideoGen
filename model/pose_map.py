"""
pose_map.py - bridge real pose data into our equation skeleton.

A pose estimator (YOLO-pose, trained on real COCO images) returns 17 keypoints
per person. This module converts those keypoints into our 14-d pose vector, which
StickFigure.equations_from_vector turns into equations. That is the concrete
"real image / video -> equations" step.

COCO-17 index order:
 0 nose 1 l_eye 2 r_eye 3 l_ear 4 r_ear 5 l_sho 6 r_sho 7 l_elb 8 r_elb
 9 l_wri 10 r_wri 11 l_hip 12 r_hip 13 l_kne 14 r_kne 15 l_ank 16 r_ank

Angles use our convention: 0 = straight down, +ccw, measured in image space with
y pointing down (so the math matches StickFigure._pt after we flip y).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stickman as S   # noqa: E402

NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI = 0, 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16


def _limb_angle(parent, child):
    """Our angle (rad) from parent->child in image coords (y-down). 0=down."""
    return math.atan2(child[0] - parent[0], child[1] - parent[1])


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _ok(conf, *idx, thr=0.3):
    return all(conf[i] >= thr for i in idx)


def coco_to_vector(kpts, conf, ground=1.15, view_w=10.0, view_h=5.62,
                   leg=1.22, torso=0.85):
    """
    kpts: (17,2) image pixel coords. conf: (17,) confidences (0..1).
    Returns a 14-d pose vector (same layout as StickFigure.pose_vector) plus the
    world hip/scale so callers can place the figure. Missing joints fall back to
    a neutral standing angle.
    """
    a = {"lean": 0.0, "thigh_R": 0.0, "knee_R": 0.0, "thigh_L": 0.0,
         "knee_L": 0.0, "uarm_R": math.radians(8), "elbow_R": math.radians(10),
         "uarm_L": math.radians(-8), "elbow_L": math.radians(10)}

    hip_mid = _mid(kpts[LHIP], kpts[RHIP]) if _ok(conf, LHIP, RHIP) else None
    sho_mid = _mid(kpts[LSHO], kpts[RSHO]) if _ok(conf, LSHO, RSHO) else None

    # torso lean (shoulder is above hip; measure tilt from vertical)
    if hip_mid and sho_mid:
        dx = sho_mid[0] - hip_mid[0]
        dy_up = hip_mid[1] - sho_mid[1]        # +ve because shoulder higher
        a["lean"] = math.atan2(dx, max(dy_up, 1e-6))

    def leg_angles(hip_i, kne_i, ank_i):
        thigh = knee = None
        if hip_mid and _ok(conf, kne_i):
            thigh = _limb_angle(hip_mid, kpts[kne_i])
        if _ok(conf, kne_i, ank_i):
            foot = _limb_angle(kpts[kne_i], kpts[ank_i])
            if thigh is not None:
                knee = foot - thigh
        return thigh, knee

    def arm_angles(sho_i, elb_i, wri_i):
        ua = el = None
        if _ok(conf, sho_i, elb_i):
            ua = _limb_angle(kpts[sho_i], kpts[elb_i])
        if _ok(conf, elb_i, wri_i) and ua is not None:
            fa = _limb_angle(kpts[elb_i], kpts[wri_i])
            el = fa - ua
        return ua, el

    tR, kR = leg_angles(RHIP, RKNE, RANK)
    tL, kL = leg_angles(LHIP, LKNE, LANK)
    uR, eR = arm_angles(RSHO, RELB, RWRI)
    uL, eL = arm_angles(LSHO, LELB, LWRI)
    for key, val in [("thigh_R", tR), ("knee_R", kR), ("thigh_L", tL),
                     ("knee_L", kL), ("uarm_R", uR), ("elbow_R", eR),
                     ("uarm_L", uL), ("elbow_L", eL)]:
        if val is not None:
            a[key] = val

    # horizontal placement from the hip's image x (normalized to our view width)
    if hip_mid:
        hip_x = view_w * min(0.92, max(0.08, hip_mid[0] / kpts[:, 0].max()
                                       if kpts[:, 0].max() > 0 else 0.5))
    else:
        hip_x = view_w / 2
    hip_y = ground + leg

    keys = S.StickFigure.ANGLE_KEYS
    vec = [hip_x / view_w, hip_y / view_h] + [a[k] / math.pi for k in keys]
    vec += [0.0, 0.0, 0.0]                     # no ball for a real person
    return vec


def vector_to_equations(vec, width=854, height=480):
    fig = S.StickFigure(width, height)
    return fig, fig.equations_from_vector(vec)
