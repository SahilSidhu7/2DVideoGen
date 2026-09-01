"""One fixed colour palette per shot.

Quantising each frame independently makes colours drift frame to frame, which
both looks wrong and destroys the temporal coherence that delta encoding
depends on. So the palette is fitted once over sampled frames of a shot and
then applied to every frame of it.
"""
import numpy as np
from PIL import Image


def fit(frames, colors=12, sample=20000, seed=0):
    """K-means over pixels of sampled frames -> (colors, 3) uint8 palette."""
    rng = np.random.default_rng(seed)
    px = []
    for f in frames:
        a = np.asarray(f.convert("RGB"), dtype=np.float32).reshape(-1, 3)
        take = min(len(a), max(1, sample // len(frames)))
        px.append(a[rng.choice(len(a), take, replace=False)])
    x = np.concatenate(px)

    # k-means++ seeding keeps rare but important colours (skin, props)
    cen = x[rng.integers(len(x))][None, :]
    for _ in range(colors - 1):
        d = ((x[:, None, :] - cen[None]) ** 2).sum(-1).min(1)
        s = d.sum()
        cen = np.vstack([cen, x[rng.choice(len(x), p=d / s) if s > 0
                                 else rng.integers(len(x))]])

    for _ in range(25):
        lab = ((x[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([x[lab == k].mean(0) if (lab == k).any() else cen[k]
                        for k in range(colors)])
        done = np.allclose(new, cen, atol=0.5)
        cen = new
        if done:
            break
    order = np.argsort(cen.sum(1))          # dark -> light, stable index meaning
    return np.clip(cen[order], 0, 255).astype(np.uint8)


def apply(img, pal):
    """Map an image onto a fixed palette (nearest colour, no dither)."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    h, w, _ = a.shape
    flat = a.reshape(-1, 1, 3)
    idx = ((flat - pal[None].astype(np.int16)) ** 2).sum(-1).argmin(1)
    return Image.fromarray(pal[idx].reshape(h, w, 3)), idx.reshape(h, w).astype(np.uint8)


def to_hex(pal):
    return ["#%02X%02X%02X" % tuple(int(v) for v in c) for c in pal]


def index_of(hexes, color):
    """Nearest palette slot for an arbitrary hex colour (tracer output)."""
    c = color.lstrip("#")
    rgb = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)
    pal = np.array([[int(h[1:][i:i + 2], 16) for i in (0, 2, 4)] for h in hexes],
                   dtype=np.int16)
    return int(((pal - rgb) ** 2).sum(1).argmin())
