"""
pose_model.py - the trained Stage-B models that output the equations.

Both learn a low-dimensional POSE VECTOR (14 numbers); the exact equations for a
frame are then reconstructed deterministically by
StickFigure.equations_from_vector. Learning the pose manifold (not raw equation
text) keeps the numbers stable and the nets tiny / CPU-trainable.

  PoseNet    : frame descriptor (action + Fourier(progress))  -> pose vector
               (the generative path used by the prompt->video pipeline)
  ImgPoseNet : rendered frame image (1x80x80)                 -> pose vector
               (literally "image translated to equations")

    python pose_model.py train-pose      -> checkpoint/posenet.pt
    python pose_model.py train-image     -> checkpoint/imgposenet.pt   (needs data/images.npz)
    python pose_model.py eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import frames_dataset as FD   # noqa: E402
import stickman as S          # noqa: E402

CKPT = HERE / "checkpoint"
CKPT.mkdir(exist_ok=True)
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class PoseNet(nn.Module):
    def __init__(self, din=FD.FEAT_LEN, dout=FD.VEC_LEN, h=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class ImgPoseNet(nn.Module):
    def __init__(self, dout=FD.VEC_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),      # 80->40
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),     # 40->20
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),     # 20->10
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),     # 10->5
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                nn.Linear(128, dout))

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# horizontal-flip remap for the 14-d pose vector (swap L/R + negate x-angles)
_FLIP_PERM = [0, 1, 2, 5, 6, 3, 4, 9, 10, 7, 8, 11, 12, 13]
_FLIP_SIGN = torch.tensor([1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1.],
                          dtype=torch.float32)


def flip_batch(imgs, Y):
    """Mirror images horizontally and remap the pose vector accordingly."""
    imgs_f = torch.flip(imgs, dims=[-1])
    Yf = Y[:, _FLIP_PERM] * _FLIP_SIGN.to(Y.device)
    Yf[:, 0] = 1.0 - Yf[:, 0]          # hip_x
    Yf[:, 11] = 1.0 - Yf[:, 11]        # ball_x
    return imgs_f, Yf


def _fit(model, X, Y, epochs, bs, lr, name, device=DEV,
         augment_flip=False, noise=0.0):
    model = model.to(device)
    X = torch.from_numpy(X); Y = torch.from_numpy(Y)
    n = len(X); ntr = int(n * 0.9)
    Xtr, Ytr = X[:ntr].to(device), Y[:ntr].to(device)
    Xva, Yva = X[ntr:].to(device), Y[ntr:].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        tot = 0.0
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], Ytr[idx]
            if augment_flip and torch.rand(1).item() < 0.5:
                xb, yb = flip_batch(xb, yb)
            if noise:
                xb = (xb + noise * torch.randn_like(xb)).clamp(0, 1)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = lossf(model(Xva), Yva).item()
            print(f"  [{name}] epoch {ep:3d}  train {tot/ntr:.5f}  val {va:.5f}")
    return model.cpu()


def train_pose(n=40000, epochs=120, bs=256, lr=2e-3):
    X, Y, _ = FD.gen_pose_samples(n)
    print(f"PoseNet: {len(X)} samples, feat {X.shape[1]} -> vec {Y.shape[1]}")
    model = PoseNet()
    _fit(model, X, Y, epochs, bs, lr, "pose")
    torch.save(model.state_dict(), CKPT / "posenet.pt")
    print(f"saved -> {CKPT/'posenet.pt'}  ({_nparams(model)} params)")


def train_image(epochs=60, bs=128, lr=1e-3):
    d = np.load(HERE.joinpath("data", "images.npz"))
    imgs, Y = d["imgs"][:, None, :, :].astype(np.float32), d["Y"].astype(np.float32)
    print(f"ImgPoseNet: {len(imgs)} images {imgs.shape[2]}x{imgs.shape[3]} -> vec {Y.shape[1]}")
    model = ImgPoseNet()
    _fit(model, imgs, Y, epochs, bs, lr, "img")
    torch.save(model.state_dict(), CKPT / "imgposenet.pt")
    print(f"saved -> {CKPT/'imgposenet.pt'}  ({_nparams(model)} params)")


def train_real(epochs=200, bs=64, lr=1e-3, mix_synth=True):
    """Train ImgPoseNet on REAL person crops (YOLO-pose-labelled COCO)."""
    d = np.load(HERE.joinpath("data", "real_imgs.npz"))
    imgs, Y = d["imgs"].astype(np.float32), d["Y"].astype(np.float32)
    if mix_synth and HERE.joinpath("data", "images.npz").exists():
        s = np.load(HERE.joinpath("data", "images.npz"))
        # center synthetic figures too so hip_x distribution matches real crops
        sy = s["Y"].astype(np.float32).copy(); sy[:, 0] = 0.5
        imgs = np.concatenate([imgs, s["imgs"].astype(np.float32)])
        Y = np.concatenate([Y, sy])
    idx = np.random.default_rng(0).permutation(len(imgs))
    imgs, Y = imgs[idx][:, None], Y[idx]
    print(f"train_real on {DEV}: {len(imgs)} images "
          f"({len(d['imgs'])} real + synthetic)  -> vec {Y.shape[1]}")
    model = ImgPoseNet()
    _fit(model, imgs, Y, epochs, bs, lr, "real", augment_flip=True, noise=0.05)
    torch.save(model.state_dict(), CKPT / "imgposenet_real.pt")
    print(f"saved -> {CKPT/'imgposenet_real.pt'}  ({_nparams(model)} params)")


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

def load_posenet():
    m = PoseNet(); m.load_state_dict(torch.load(CKPT / "posenet.pt")); m.eval()
    return m


def load_imgposenet():
    m = ImgPoseNet(); m.load_state_dict(torch.load(CKPT / "imgposenet.pt")); m.eval()
    return m


@torch.no_grad()
def predict_vec(model, action_idx, progress):
    f = FD.descriptor(action_idx, progress)[None, :]
    return model(torch.from_numpy(f))[0].numpy()


def evaluate():
    fig = S.StickFigure(854, 480)
    m = load_posenet()
    # per-action reconstruction error vs the engine (teacher)
    print("PoseNet vs engine (mean abs angle error in degrees):")
    for ai, act in enumerate(FD.ACTIONS):
        errs = []
        for p in np.linspace(0, 1, 60):
            pred = predict_vec(m, ai, float(p))
            true = np.array(fig.pose_vector(act, float(p)))
            errs.append(np.abs(pred[2:11] - true[2:11]) * 180)  # rad/pi->deg
        print(f"  {act:6s}  {np.mean(errs):5.2f} deg")
    if (CKPT / "imgposenet.pt").exists():
        d = np.load(HERE.joinpath("data", "images.npz"))
        im = load_imgposenet()
        X = torch.from_numpy(d["imgs"][:, None].astype(np.float32))
        with torch.no_grad():
            pred = im(X).numpy()
        mae = np.abs(pred[:, 2:11] - d["Y"][:, 2:11]).mean() * 180
        print(f"ImgPoseNet image->pose angle MAE: {mae:.2f} deg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train-pose":
        train_pose()
    elif cmd == "train-image":
        train_image()
    elif cmd == "train-real":
        train_real()
    elif cmd == "eval":
        evaluate()
    else:
        print("usage: pose_model.py [train-pose|train-image|train-real|eval]")
