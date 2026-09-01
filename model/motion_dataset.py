"""
motion_dataset.py - extract real motion SEQUENCES from video clips.

Each clip -> per-frame best-person pose vector (YOLO-pose + pose_map), temporally
smoothed -> one variable-length sequence of 14-d pose vectors, tagged with an
action guessed from the filename. This is the real-motion training corpus.

    python motion_dataset.py datasets/videos/mp4 -o data/real_motion.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import stickman as S       # noqa: E402
import pose_map as PMAP    # noqa: E402
import vectorize as V      # noqa: E402

ACTION_HINTS = {"walkcycle": "walk", "walking": "walk", "walk": "walk",
                "shackle": "walk", "stairs": "walk", "ascending": "walk",
                "march": "walk",
                "run": "run", "running": "run", "jog": "run", "gallop": "run",
                "somersault": "jump", "leapfrog": "jump", "jump": "jump",
                "danc": "dance", "dance": "dance"}


def guess_action(name):
    n = name.lower()
    for k, a in ACTION_HINTS.items():
        if k in n:
            return a
    return "walk"


def clip_sequence(path, smooth=0.5, min_joints=9):
    fig = S.StickFigure(854, 480)
    seq, prev = [], None
    for res in V.yolo()(str(path), stream=True, verbose=False):
        got = V._best_person(res)
        if not got or (got[1] >= 0.3).sum() < min_joints:
            if prev is not None:
                seq.append(prev)                 # hold last good pose
            continue
        vec = np.array(PMAP.coco_to_vector(got[0], got[1], ground=fig.ground,
                       view_w=fig.W, view_h=fig.H, leg=fig.leg, torso=fig.torso),
                       np.float32)
        if prev is not None:
            vec = prev + (1 - smooth) * (vec - prev)   # temporal smoothing
        prev = vec
        seq.append(vec.copy())
    return np.array(seq, np.float32)


def build(src_dir, out_npz):
    files = sorted(p for p in Path(src_dir).glob("*.mp4"))
    seqs, actions = [], []
    for p in files:
        s = clip_sequence(p)
        if len(s) < 20:
            print(f"  skip {p.name} (only {len(s)} frames)")
            continue
        act = guess_action(p.name)
        seqs.append(s); actions.append(act)
        print(f"  {p.name[:48]:48s}  {len(s):4d} frames  action={act}")
    Path(out_npz).parent.mkdir(exist_ok=True)
    np.savez_compressed(out_npz,
                        seqs=np.array(seqs, dtype=object),
                        actions=np.array(actions))
    print(f"real motion -> {out_npz}  {len(seqs)} clips, "
          f"{sum(len(s) for s in seqs)} frames")
    return seqs, actions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="data/real_motion.npz")
    args = ap.parse_args()
    build(args.src, args.out)
