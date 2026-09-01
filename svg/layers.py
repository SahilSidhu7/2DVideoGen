"""Split a shot into a static background plate and a per-frame moving layer.

This mirrors how cel animation was actually made: one painted background held
for the whole shot, with only the character cels redrawn each frame. It matters
here because the background is what makes naive per-frame tracing expensive —
it is ~80% of the traced points and it does not change, so paying for it 15
times a second is pure waste.
"""
import numpy as np
from PIL import Image


def background(idx_frames, mode="median"):
    """Per-pixel modal/median palette index over a shot -> static plate."""
    stack = np.stack(idx_frames)                       # (n, h, w) uint8
    if mode == "median":
        return np.median(stack, axis=0).astype(np.uint8)
    n_col = int(stack.max()) + 1
    counts = np.zeros((n_col,) + stack.shape[1:], dtype=np.int32)
    for c in range(n_col):
        counts[c] = (stack == c).sum(0)
    return counts.argmax(0).astype(np.uint8)


def moving_mask(idx, bg, dilate=2):
    """Pixels of a frame that differ from the background plate."""
    m = idx != bg
    if dilate:
        m = _dilate(m, dilate)
    return m


def _dilate(m, r):
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | out)
    return out


def compose(idx, bg, mask, hole=255):
    """Frame indices with static pixels knocked out, ready to trace alone."""
    out = np.full_like(idx, hole)
    out[mask] = idx[mask]
    return out


def to_image(idx, pal, hole=255, hole_rgb=(255, 0, 255)):
    """Palette-index map -> RGB image; the hole colour marks 'transparent'."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[: len(pal)] = pal
    lut[hole] = hole_rgb
    return Image.fromarray(lut[idx])


def coverage(mask):
    return float(mask.mean())
