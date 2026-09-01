"""Frame preprocessing before tracing.

Two jobs, both essential for cel material from film scans:
  * strip pillar/letterbox bars (archive prints are matted 4:3 inside 16:9)
  * downscale + posterise, which flattens painted-background texture so the
    tracer emits large flat regions instead of thousands of noise paths
"""
from PIL import Image, ImageFilter


def bar_box(img, thresh=24):
    """Bounding box of the image once near-black borders are removed."""
    g = img.convert("L")
    w, h = g.size
    px = g.load()
    step = max(1, min(w, h) // 120)

    def row_dark(y):
        return max(px[x, y] for x in range(0, w, step)) < thresh

    def col_dark(x):
        return max(px[x, y] for y in range(0, h, step)) < thresh

    top = 0
    while top < h - 1 and row_dark(top):
        top += 1
    bot = h - 1
    while bot > top and row_dark(bot):
        bot -= 1
    left = 0
    while left < w - 1 and col_dark(left):
        left += 1
    right = w - 1
    while right > left and col_dark(right):
        right -= 1
    return (left, top, right + 1, bot + 1)


def crop_bars(img, thresh=24):
    return img.crop(bar_box(img, thresh))


def shot_box(frames, thresh=24):
    """One crop box for a whole shot.

    Per-frame cropping makes frames different sizes, which breaks any pixel-wise
    comparison across a shot (background plates, motion masks). The union of the
    per-frame boxes over a few samples keeps every frame aligned.
    """
    boxes = [bar_box(f, thresh) for f in frames]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def posterize(img, levels=6):
    """Quantise to a small palette. Cel art is already flat; this removes the
    film-grain and paint texture that would otherwise dominate the trace."""
    return img.quantize(colors=levels, method=Image.MEDIANCUT, dither=Image.NONE).convert("RGB")


def prep(img, width=320, levels=8, smooth=1.2, bars=True):
    if bars:
        img = crop_bars(img)
    if width and img.width != width:
        h = max(1, round(img.height * width / img.width))
        img = img.convert("RGB").resize((width, h), Image.LANCZOS)
    if smooth:
        img = img.filter(ImageFilter.GaussianBlur(smooth))
    if levels:
        img = posterize(img, levels)
    return img
