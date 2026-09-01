"""Raster frame -> compact SVG, tuned for cel animation.

Cel animation (flat fills, hard edges, small palette) vectorises far more
cleanly than photography, which is why this project traces anime rather than
live action. Everything here optimises for *token count* as much as fidelity,
because the traced SVG is training data for a small language model.
"""
import argparse
import os
import re
import tempfile

import vtracer

# Tuned for flat cel art: few colours, polygon mode, aggressive speckle filter.
PRESETS = {
    "fine": dict(colormode="color", hierarchical="stacked", mode="spline",
                 filter_speckle=4, color_precision=6, layer_difference=16,
                 corner_threshold=60, length_threshold=4.0, max_iterations=10,
                 splice_threshold=45, path_precision=2),
    "compact": dict(colormode="color", hierarchical="stacked", mode="polygon",
                    filter_speckle=16, color_precision=4, layer_difference=32,
                    corner_threshold=70, length_threshold=8.0, max_iterations=10,
                    splice_threshold=60, path_precision=1),
    "tiny": dict(colormode="color", hierarchical="stacked", mode="polygon",
                 filter_speckle=48, color_precision=3, layer_difference=48,
                 corner_threshold=80, length_threshold=16.0, max_iterations=10,
                 splice_threshold=80, path_precision=0),
}

_NUM = re.compile(r"-?\d+\.?\d*")
_PATH = re.compile(r'<path[^>]*\bd="([^"]*)"[^>]*/?>')
_FILL = re.compile(r'fill="([^"]*)"')


def trace_file(src, preset="compact"):
    """Trace an image file, return SVG text."""
    out = tempfile.mktemp(suffix=".svg")
    try:
        vtracer.convert_image_to_svg_py(src, out, **PRESETS[preset])
        with open(out, "r", encoding="utf-8") as fh:
            return fh.read()
    finally:
        if os.path.exists(out):
            os.remove(out)


def shrink(svg, decimals=1, min_area=0.0):
    """Strip vtracer boilerplate and round coordinates.

    vtracer emits verbose headers and full float precision; neither survives
    into training data. Rounding coordinates is the single biggest token win.
    """
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"\s+", " ", svg).strip()

    def round_num(m):
        txt = "%.*f" % (decimals, float(m.group(0)))
        if "." in txt:                      # never strip zeros from an integer
            txt = txt.rstrip("0").rstrip(".")
        return txt or "0"

    def fix_path(m):
        d = _NUM.sub(round_num, m.group(1))
        d = re.sub(r"\s+", " ", d).strip()
        return m.group(0).replace(m.group(1), d)

    svg = _PATH.sub(fix_path, svg)
    if min_area > 0:
        svg = _drop_small(svg, min_area)
    return svg


def _drop_small(svg, min_area):
    """Remove paths whose bounding box is below min_area (fraction of canvas)."""
    w, h = canvas_size(svg)
    if not w or not h:
        return svg
    thresh = min_area * w * h

    def keep(m):
        nums = [float(x) for x in _NUM.findall(m.group(1))]
        xs, ys = nums[0::2], nums[1::2]
        if len(xs) < 2 or len(ys) < 2:
            return ""
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        return m.group(0) if area >= thresh else ""

    return _PATH.sub(keep, svg)


def canvas_size(svg):
    m = re.search(r'width="(\d+\.?\d*)"[^>]*height="(\d+\.?\d*)"', svg)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'viewBox="[\d.\- ]*?([\d.]+) ([\d.]+)"', svg)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def paths(svg):
    """[(fill, d)] in document order."""
    out = []
    for m in re.finditer(r"<path[^>]*>", svg):
        tag = m.group(0)
        d = _PATH.search(tag)
        f = _FILL.search(tag)
        if d:
            out.append((f.group(1) if f else "#000000", d.group(1)))
    return out


def stats(svg, tokenizer=None):
    p = paths(svg)
    pts = sum(len(_NUM.findall(d)) // 2 for _, d in p)
    n_tok = len(tokenizer(svg)["input_ids"]) if tokenizer else None
    return dict(bytes=len(svg), paths=len(p), points=pts, tokens=n_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out")
    ap.add_argument("--preset", default="compact", choices=list(PRESETS))
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument("--min-area", type=float, default=0.0)
    args = ap.parse_args()

    svg = shrink(trace_file(args.image, args.preset), args.decimals, args.min_area)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(svg)
    print(args.preset, stats(svg))


if __name__ == "__main__":
    main()
