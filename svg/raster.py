"""Rasterise the polygon SVGs this project produces. No native deps.

vtracer's polygon presets emit only `M`/`L`/`Z` subpaths with a flat fill and an
optional translate, so a scanline even-odd fill covers the whole dialect. Using
our own rasteriser avoids cairo (no Windows wheel) and keeps the training loop's
reconstruction metric dependency-free.
"""
import re

import numpy as np
from PIL import Image, ImageDraw

_TAG = re.compile(r"<path[^>]*>")
_D = re.compile(r'\bd="([^"]*)"')
_FILL = re.compile(r'\bfill="([^"]*)"')
_TRANS = re.compile(r"translate\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_SIZE = re.compile(r'<svg[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"')
_NUM = re.compile(r"-?\d+\.?\d*")
_STROKE = re.compile(r'\bstroke="([^"]*)"')
_SW = re.compile(r'\bstroke-width="([\d.]+)"')


def _hex(c):
    c = (c or "#000000").strip()
    if not c.startswith("#"):
        return (0, 0, 0)
    c = c[1:]
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def subpaths(d, dx=0.0, dy=0.0, closed=True):
    """Split a path's `d` into lists of (x, y). Only M/L/Z appear here.

    `closed=False` keeps two-point runs, which open stroked paths need.
    """
    out, cur = [], []
    for token in re.finditer(r"([MLZmlz])([^MLZmlz]*)", d):
        cmd, rest = token.group(1), token.group(2)
        nums = [float(n) for n in _NUM.findall(rest)]
        least = 3 if closed else 2
        if cmd in "Zz":
            if len(cur) >= least:
                out.append(cur)
            cur = []
            continue
        pts = [(nums[i] + dx, nums[i + 1] + dy) for i in range(0, len(nums) - 1, 2)]
        if cmd in "Mm" and cur:
            if len(cur) >= least:
                out.append(cur)
            cur = []
        cur.extend(pts)
    if len(cur) >= (3 if closed else 2):
        out.append(cur)
    return out


def _fill(buf, polys, rgb, scale):
    """Even-odd scanline fill of all subpaths of one path, together."""
    edges = []
    for poly in polys:
        p = [(x * scale, y * scale) for x, y in poly]
        for i in range(len(p)):
            x0, y0 = p[i]
            x1, y1 = p[(i + 1) % len(p)]
            if y0 != y1:
                edges.append((y0, y1, x0, x1))
    if not edges:
        return
    e = np.array(edges, dtype=np.float64)
    y0, y1, x0, x1 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    ylo, yhi = np.minimum(y0, y1), np.maximum(y0, y1)
    h, w = buf.shape[:2]
    top = max(0, int(np.floor(ylo.min())))
    bot = min(h - 1, int(np.ceil(yhi.max())))
    for y in range(top, bot + 1):
        yc = y + 0.5
        hit = (ylo <= yc) & (yhi > yc)
        if not hit.any():
            continue
        xs = x0[hit] + (yc - y0[hit]) * (x1[hit] - x0[hit]) / (y1[hit] - y0[hit])
        xs = np.sort(xs)
        for i in range(0, len(xs) - 1, 2):
            a = max(0, int(np.ceil(xs[i] - 0.5)))
            b = min(w - 1, int(np.floor(xs[i + 1] - 0.5)))
            if b >= a:
                buf[y, a:b + 1] = rgb


def render(svg, width=None, bg=(255, 255, 255)):
    m = _SIZE.search(svg)
    sw, sh = (float(m.group(1)), float(m.group(2))) if m else (320.0, 240.0)
    scale = (width / sw) if width else 1.0
    w, h = max(1, round(sw * scale)), max(1, round(sh * scale))
    buf = np.zeros((h, w, 3), dtype=np.uint8)
    buf[:, :] = bg
    img = None
    for tag in _TAG.findall(svg):
        d = _D.search(tag)
        if not d:
            continue
        t = _TRANS.search(tag)
        dx, dy = (float(t.group(1)), float(t.group(2))) if t else (0.0, 0.0)
        polys = subpaths(d.group(1), dx, dy, closed=False)
        st = _STROKE.search(tag)
        if st and st.group(1) != "none":
            # Strokes need real line joins, so hand these to PIL rather than
            # the scanline filler - line art is mostly strokes.
            if img is None:
                img = Image.fromarray(buf)
            drw = ImageDraw.Draw(img)
            wid = max(1, round(float(_SW.search(tag).group(1)) * scale)) if _SW.search(tag) else 1
            for poly in polys:
                pts = [(x * scale, y * scale) for x, y in poly]
                drw.line(pts, fill=_hex(st.group(1)), width=wid, joint="curve")
            buf = np.asarray(img).copy()
            img = None
            continue
        fl = _FILL.search(tag)
        if fl and fl.group(1) == "none":
            continue
        _fill(buf, [p for p in polys if len(p) >= 3],
              _hex(fl.group(1) if fl else None), scale)
    return Image.fromarray(buf)


def render_file(path, width=None):
    with open(path, "r", encoding="utf-8") as fh:
        return render(fh.read(), width)
