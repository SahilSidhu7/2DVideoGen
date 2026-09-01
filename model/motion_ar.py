"""
motion_ar.py - AUTOREGRESSIVE motion model learned from real video.

Unlike the phase->pose model (which averages a clip into one loop), this GRU
learns local dynamics: given a short window of recent poses (+ the action), it
predicts the next pose. Generation seeds with a few real frames and rolls out,
producing non-looping, evolving motion. This handles multiple differing clips of
the same action without blurring them together.

    python motion_ar.py train
    python motion_ar.py generate walk -o ../out/ar_walk.mp4
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
VEC = S.StickFigure.VEC_LEN
WIN = 16                     # training window length
SEED = 8                     # frames used to seed generation


class MotionAR(nn.Module):
    def __init__(self, n_act, hidden=160):
        super().__init__()
        self.n_act = n_act
        self.gru = nn.GRU(VEC + n_act, hidden, batch_first=True)
        self.head = nn.Linear(hidden, VEC)      # predicts pose DELTA

    def forward(self, poses, act1h, h0=None):
        # poses (B,T,VEC), act1h (B,n_act) -> next-pose deltas (B,T,VEC)
        a = act1h[:, None, :].expand(-1, poses.shape[1], -1)
        y, h = self.gru(torch.cat([poses, a], dim=-1), h0)
        return self.head(y), h


def _load():
    d = np.load(HERE.joinpath("data", "real_motion.npz"), allow_pickle=True)
    return list(d["seqs"]), list(map(str, d["actions"]))


def train(epochs=600, bs=128, lr=1.5e-3, noise=0.02):
    seqs, actions = _load()
    ACT = sorted(set(actions))
    ai = {a: i for i, a in enumerate(ACT)}
    print(f"actions {ACT}  clips {len(seqs)}  device {DEV}")
    # build windows: X poses[t:t+WIN], target delta poses[t+1:t+WIN+1]-poses[t:t+WIN]
    Xs, Ds, As = [], [], []
    for seq, act in zip(seqs, actions):
        s = np.asarray(seq, np.float32)
        for t in range(0, len(s) - WIN - 1):
            Xs.append(s[t:t + WIN])
            Ds.append(s[t + 1:t + WIN + 1] - s[t:t + WIN])
            As.append(ai[act])
    X = torch.tensor(np.array(Xs), device=DEV)
    D = torch.tensor(np.array(Ds), device=DEV)
    A = torch.eye(len(ACT), device=DEV)[torch.tensor(As, device=DEV)]
    print(f"windows: {len(X)}")
    model = MotionAR(len(ACT)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(X); ntr = int(n * 0.9)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(ntr, device=DEV)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            xb = X[idx] + noise * torch.randn_like(X[idx])   # scheduled-sampling noise
            opt.zero_grad()
            pred, _ = model(xb, A[idx])
            loss = lossf(pred, D[idx])
            loss.backward(); opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = lossf(model(X[ntr:], A[ntr:])[0], D[ntr:]).item()
            print(f"  epoch {ep:3d}  val {va:.6f}")
    # store a real seed window per action (first SEED frames of the longest clip)
    seeds = {}
    for a in ACT:
        clips = [np.asarray(s, np.float32) for s, ac in zip(seqs, actions) if ac == a]
        best = max(clips, key=len)
        seeds[a] = best[:SEED]
    torch.save({"state": model.cpu().state_dict(), "actions": ACT,
                "seeds": {a: seeds[a] for a in ACT}}, CKPT / "motion_ar.pt")
    print(f"saved -> {CKPT/'motion_ar.pt'}")


def load():
    ck = torch.load(CKPT / "motion_ar.pt", weights_only=False)
    m = MotionAR(len(ck["actions"])); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["actions"], ck["seeds"]


@torch.no_grad()
def rollout(model, ACT, seeds, action, n):
    ai = ACT.index(action)
    a1h = torch.eye(len(ACT))[ai][None]
    seed = torch.tensor(seeds[action], dtype=torch.float32)[None]   # (1,SEED,VEC)
    # warm up the hidden state on the seed
    _, h = model(seed, a1h)
    cur = seed[:, -1:, :]                        # last seed pose
    out = [seeds[action][i] for i in range(len(seeds[action]))]
    for _ in range(n - len(out)):
        delta, h = model(cur, a1h, h)
        cur = (cur + delta).clamp(-2.5, 2.5)
        out.append(cur[0, 0].numpy())
    return np.array(out, np.float32)


@torch.no_grad()
def generate(action, out, fps=25, seconds=6, width=854, height=480,
             color=(120, 220, 255), verbose=True):
    model, ACT, seeds = load()
    if action not in ACT:
        raise SystemExit(f"action '{action}' not learned; have {ACT}")
    n = int(fps * seconds)
    poses = rollout(model, ACT, seeds, action, n)
    fig = S.StickFigure(width, height)
    out = Path(out); jsonl = out.with_suffix(".equations.jsonl")
    ffmpeg = geovid.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with jsonl.open("w", encoding="utf-8") as jf:
        for i, vec in enumerate(poses):
            vec = vec.astype(float).copy()
            if action in ("walk", "run"):
                vec[0] = 0.1 + 0.8 * (i / max(1, len(poses) - 1))
            eqs = fig.equations_from_vector(vec)
            jf.write(json.dumps({"frame": i, "action": action,
                                 "source": "motion-ar", "equations": eqs}) + "\n")
            proc.stdin.write(geovid.Scene(
                S.build_scene_spec(fig, eqs, width, height, color)).render_frame(0).tobytes())
            if verbose and i % fps == 0:
                print(f"\r  frame {i}/{len(poses)}", end="", flush=True)
    proc.stdin.close(); proc.wait()
    if verbose:
        print(f"\nvideo -> {out}\nequations -> {jsonl}")
    return {"equations_jsonl": str(jsonl), "frames": len(poses), "action": action}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    g = sub.add_parser("generate"); g.add_argument("action")
    g.add_argument("-o", "--out", default="../out/ar.mp4")
    g.add_argument("--seconds", type=float, default=6)
    args = ap.parse_args()
    if args.cmd == "train":
        train()
    else:
        generate(args.action, args.out, seconds=args.seconds)


if __name__ == "__main__":
    main()
