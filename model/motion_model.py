"""
motion_model.py - a motion prior LEARNED FROM REAL VIDEO.

RealMotionNet maps (action, phase in [0,1]) -> pose vector, fit to the real pose
sequences extracted from video (motion_dataset.py). Because it is a smooth
function of phase, generation can never drift or collapse: sweeping phase 0..1
replays the learned real motion smoothly. This replaces the hand-scripted gaits
with motion distilled from real footage.

    python motion_model.py train
    python motion_model.py generate walk -o ../out/realmotion_walk.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import geovid          # noqa: E402
import stickman as S   # noqa: E402

CKPT = HERE / "checkpoint"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FREQS = [1, 2, 3, 4, 5, 6, 8]
VEC_LEN = S.StickFigure.VEC_LEN


def _feat(action_idx, n_actions, phase):
    f = np.zeros(n_actions + 1 + 2 * len(FREQS), np.float32)
    f[action_idx] = 1.0
    f[n_actions] = phase
    b = n_actions + 1
    for i, k in enumerate(FREQS):
        f[b + 2 * i] = np.sin(2 * np.pi * k * phase)
        f[b + 2 * i + 1] = np.cos(2 * np.pi * k * phase)
    return f


class RealMotionNet(nn.Module):
    def __init__(self, din, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, VEC_LEN))

    def forward(self, x):
        return self.net(x)


def _load_motion():
    d = np.load(HERE.joinpath("data", "real_motion.npz"), allow_pickle=True)
    return list(d["seqs"]), list(d["actions"])


def train(epochs=400, bs=256, lr=1.5e-3):
    seqs, actions = _load_motion()
    ACT = sorted(set(actions))
    print(f"actions: {ACT}  clips: {len(seqs)}  device: {DEV}")
    X, Y = [], []
    for seq, act in zip(seqs, actions):
        ai = ACT.index(act)
        n = len(seq)
        for i in range(n):
            X.append(_feat(ai, len(ACT), i / (n - 1)))
            Y.append(seq[i])
    X = np.array(X, np.float32); Y = np.array(Y, np.float32)
    din = X.shape[1]
    model = RealMotionNet(din).to(DEV)
    Xt = torch.from_numpy(X).to(DEV); Yt = torch.from_numpy(Y).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    ntr = int(len(Xt) * 0.9)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=DEV)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(model(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = lossf(model(Xt[ntr:]), Yt[ntr:]).item()
            print(f"  epoch {ep:3d}  val {va:.5f}")
    torch.save({"state": model.cpu().state_dict(), "actions": ACT, "din": din},
               CKPT / "realmotion.pt")
    print(f"saved -> {CKPT/'realmotion.pt'}  actions={ACT}")


def load():
    ck = torch.load(CKPT / "realmotion.pt", weights_only=False)
    m = RealMotionNet(ck["din"]); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["actions"]


@torch.no_grad()
def generate(action, out, fps=25, seconds=6, width=854, height=480,
             color=(120, 220, 255), loops=3, verbose=True):
    model, ACT = load()
    if action not in ACT:
        raise SystemExit(f"action '{action}' not learned; have {ACT}")
    ai = ACT.index(action)
    fig = S.StickFigure(width, height)
    n = int(fps * seconds)
    out = Path(out); jsonl = out.with_suffix(".equations.jsonl")
    ffmpeg = geovid.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with jsonl.open("w", encoding="utf-8") as jf:
        for i in range(n):
            phase = (i / n * loops) % 1.0          # loop the learned motion
            f = torch.from_numpy(_feat(ai, len(ACT), phase)[None])
            vec = model(f)[0].numpy().astype(float)
            if action in ("walk", "run"):          # add locomotion across screen
                vec[0] = 0.1 + 0.8 * (i / max(1, n - 1))
            eqs = fig.equations_from_vector(vec)
            jf.write(json.dumps({"frame": i, "action": action, "phase": round(phase, 3),
                                 "source": "realmotion", "equations": eqs}) + "\n")
            proc.stdin.write(geovid.Scene(
                S.build_scene_spec(fig, eqs, width, height, color)).render_frame(0).tobytes())
            if verbose and i % fps == 0:
                print(f"\r  frame {i}/{n}", end="", flush=True)
    proc.stdin.close(); proc.wait()
    if verbose:
        print(f"\nvideo -> {out}\nequations -> {jsonl}")
    return {"equations_jsonl": str(jsonl), "frames": n, "action": action}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    g = sub.add_parser("generate"); g.add_argument("action")
    g.add_argument("-o", "--out", default="../out/realmotion.mp4")
    g.add_argument("--seconds", type=float, default=6)
    args = ap.parse_args()
    if args.cmd == "train":
        train()
    else:
        generate(args.action, args.out, seconds=args.seconds)


if __name__ == "__main__":
    main()
