"""Decompose a moving polyline into rigid motion plus a residual morph.

Whatever the source of the animation (Lottie today, traced footage or a
character rig later), a shape usually *moves* far more than it *changes*. This
fits a similarity transform - translate, rotate, uniform scale - between the
cast pose and the frame's pose, so those four numbers absorb most of the
motion, and only the leftover residual is charged as a per-vertex morph.

That split is exactly what makes AniSVG cheap: a walking character costs a few
transform ops per frame instead of a full vertex list.
"""
import math

import numpy as np


def fit_similarity(src, dst):
    """Least-squares similarity src -> dst (Umeyama, uniform scale).

    Returns (tx, ty, rot_degrees, scale_percent) applied about src's centroid,
    matching `anisvg._transform`.
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ca, cb = a.mean(0), b.mean(0)
    a0, b0 = a - ca, b - cb

    var = (a0 ** 2).sum()
    if var < 1e-9:
        return float(cb[0] - ca[0]), float(cb[1] - ca[1]), 0.0, 100.0

    # 2x2 cross-covariance -> rotation angle in closed form
    sxx = float((a0[:, 0] * b0[:, 0] + a0[:, 1] * b0[:, 1]).sum())
    sxy = float((a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum())
    rot = math.atan2(sxy, sxx)
    scale = math.hypot(sxx, sxy) / var

    ca_t = _apply(a, ca, rot, scale)
    off = b.mean(0) - ca_t.mean(0)
    return float(off[0]), float(off[1]), math.degrees(rot), scale * 100.0


def _apply(pts, centre, rot, scale):
    ca, sa = math.cos(rot), math.sin(rot)
    p = np.asarray(pts, dtype=np.float64) - centre
    r = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                  p[:, 0] * sa + p[:, 1] * ca], axis=1) * scale
    return r + centre


def residual(src, dst, tx, ty, rot_deg, scale_pct):
    """Per-vertex leftover after the similarity fit, in destination units."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    moved = _apply(a, a.mean(0), math.radians(rot_deg), scale_pct / 100.0)
    moved = moved + np.array([tx, ty])
    return b - moved


def rms(vec):
    v = np.asarray(vec, dtype=np.float64)
    return float(np.sqrt((v ** 2).sum(1).mean())) if v.size else 0.0
