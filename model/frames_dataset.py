"""
frames_dataset.py - self-distillation corpus generated from the equation engine.

Real anime->equation labels don't exist, so we let the engine (stickman.py) be
the teacher: sample every action across the whole motion cycle and record, for
each frame,

    descriptor features   ->  input for the generative Stage-B model (PoseNet)
    pose vector (14-d)     ->  target (uniquely maps to that frame's equations)
    rendered frame image   ->  input for the image->equation model (ImgPoseNet)
    tags (action, phase)   ->  the "with tags" part

This is the "anime images/videos translated to equations with tags" dataset,
synthesized with perfect ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import geovid          # noqa: E402
import stickman as S   # noqa: E402

ACTIONS = ["walk", "run", "jump", "kick", "wave", "dance"]
FREQS = [1, 2, 3, 4, 6, 8, 12]          # Fourier features of progress
FEAT_LEN = len(ACTIONS) + 1 + 2 * len(FREQS)   # onehot + progress + sin/cos
VEC_LEN = S.StickFigure.VEC_LEN                 # 14


def descriptor(action_idx, progress):
    """Per-frame input features for the generative pose model."""
    f = np.zeros(FEAT_LEN, dtype=np.float32)
    f[action_idx] = 1.0
    f[len(ACTIONS)] = progress
    base = len(ACTIONS) + 1
    for i, k in enumerate(FREQS):
        f[base + 2 * i] = np.sin(2 * np.pi * k * progress)
        f[base + 2 * i + 1] = np.cos(2 * np.pi * k * progress)
    return f


def gen_pose_samples(n, seed=0):
    """(features, pose_vector) pairs — cheap, no rendering."""
    rng = np.random.default_rng(seed)
    fig = S.StickFigure(854, 480)
    X = np.zeros((n, FEAT_LEN), np.float32)
    Y = np.zeros((n, VEC_LEN), np.float32)
    A = np.zeros(n, np.int64)
    for i in range(n):
        ai = int(rng.integers(len(ACTIONS)))
        p = float(rng.random())
        X[i] = descriptor(ai, p)
        Y[i] = fig.pose_vector(ACTIONS[ai], p)
        A[i] = ai
    return X, Y, A


def render_gray(fig, action, progress, size):
    """Render one frame small + grayscale (input for the image model)."""
    j, props = fig.pose(action, progress)
    spec = S.build_scene_spec(fig, S.frame_equations(j, props), size, size,
                              color=(255, 255, 255), bg=(0, 0, 0))
    spec["supersample"] = 1
    img = geovid.Scene(spec).render_frame(0).convert("L")
    return np.asarray(img, np.float32) / 255.0


def gen_image_samples(n, size=80, seed=1):
    """(image, pose_vector, action_idx) — renders frames (slower)."""
    rng = np.random.default_rng(seed)
    fig = S.StickFigure(size, size)          # square view so figure fills frame
    imgs = np.zeros((n, size, size), np.float32)
    Y = np.zeros((n, VEC_LEN), np.float32)
    A = np.zeros(n, np.int64)
    for i in range(n):
        ai = int(rng.integers(len(ACTIONS)))
        p = float(rng.random())
        imgs[i] = render_gray(fig, ACTIONS[ai], p, size)
        Y[i] = fig.pose_vector(ACTIONS[ai], p)
        A[i] = ai
        if (i + 1) % 500 == 0:
            print(f"\r  rendered {i+1}/{n}", end="", flush=True)
    print()
    return imgs, Y, A


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=40000)
    ap.add_argument("--images", type=int, default=3000)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    X, Y, A = gen_pose_samples(args.poses)
    np.savez_compressed(out / "poses.npz", X=X, Y=Y, A=A)
    print(f"poses  -> {out}/poses.npz  X{X.shape} Y{Y.shape}")

    imgs, Yi, Ai = gen_image_samples(args.images, size=args.size)
    np.savez_compressed(out / "images.npz", imgs=imgs, Y=Yi, A=Ai)
    print(f"images -> {out}/images.npz  imgs{imgs.shape}")
