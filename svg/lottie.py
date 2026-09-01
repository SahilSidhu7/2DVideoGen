"""A small Lottie (Bodymovin JSON) evaluator - enough of the format to sample
shape geometry at arbitrary times.

Lottie is the only large corpus of *human-authored vector animation*, and it is
already structured the way AniSVG wants: persistent shapes plus keyframed
motion. Rendering it is out of scope; all we need is, for a given time, the
world-space polyline and fill colour of every filled shape. Unsupported
features (text layers, images, mattes, gradients, trim paths, repeaters) are
skipped and reported, so corpus coverage is measurable rather than assumed.
"""
import math

import numpy as np

SUPPORTED_SHAPES = {"sh", "rc", "el"}


class Unsupported(Exception):
    pass


# --------------------------------------------------------------- properties


def _bez_ease(t, o, i):
    """Solve the Lottie keyframe easing curve for progress t in [0, 1]."""
    x1 = _first(o.get("x", 0.333))
    y1 = _first(o.get("y", 0.0))
    x2 = _first(i.get("x", 0.667))
    y2 = _first(i.get("y", 1.0))
    lo, hi = 0.0, 1.0
    for _ in range(18):                       # bisection is plenty at 15 fps
        mid = (lo + hi) / 2
        x = _cubic(mid, 0.0, x1, x2, 1.0)
        if x < t:
            lo = mid
        else:
            hi = mid
    return _cubic((lo + hi) / 2, 0.0, y1, y2, 1.0)


def _cubic(t, p0, p1, p2, p3):
    u = 1 - t
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3


def _first(v):
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def value_at(prop, t, default=None):
    """Evaluate a Lottie animatable property at frame time `t`."""
    if prop is None:
        return default
    if not isinstance(prop, dict):
        return prop
    k = prop.get("k")
    # Bodymovin frequently omits the "a" (animated) flag on properties that are
    # in fact keyframed, so trust the structure instead: a keyframe list is a
    # list of dicts carrying a time field.
    keyed = (isinstance(k, list) and k and isinstance(k[0], dict)
             and ("t" in k[0] or "s" in k[0]))
    if not keyed:
        return k
    if t <= k[0].get("t", 0):
        return _kf_start(k[0])
    for a, b in zip(k, k[1:]):
        t0, t1 = a.get("t", 0), b.get("t", 0)
        if t0 <= t < t1:
            s, e = _kf_start(a), _kf_end(a, b)
            if s is None or e is None or t1 == t0:
                return s
            f = _bez_ease((t - t0) / (t1 - t0), a.get("o", {}), a.get("i", {}))
            return _lerp(s, e, f)
    return _kf_end(k[-2], k[-1]) if len(k) > 1 else _kf_start(k[-1])


def _kf_start(kf):
    return kf.get("s", kf.get("e"))


def _kf_end(a, b):
    return a.get("e", _kf_start(b))


def _lerp(s, e, f):
    if isinstance(s, list) and isinstance(e, list):
        if s and isinstance(s[0], dict):       # bezier path keyframe
            return _lerp_path(s[0], e[0], f)
        return [_lerp(x, y, f) for x, y in zip(s, e)]
    if isinstance(s, dict) and isinstance(e, dict):
        return _lerp_path(s, e, f)
    try:
        return float(s) + (float(e) - float(s)) * f
    except (TypeError, ValueError):
        return s


def _lerp_path(s, e, f):
    out = {"c": s.get("c", False)}
    for key in ("v", "i", "o"):
        a, b = s.get(key, []), e.get(key, [])
        out[key] = [[p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f]
                    for p, q in zip(a, b)] or a
    return out


# ------------------------------------------------------------------ shapes


def flatten_path(path, steps=6):
    """Cubic bezier path dict -> polyline."""
    # Animated path keyframes wrap the shape dict one list deep; static ones
    # do not. Unwrap before deciding whether this is already a polyline.
    if isinstance(path, list) and path and isinstance(path[0], dict):
        path = path[0]
    if isinstance(path, list):
        return [(float(p[0]), float(p[1])) for p in path]
    if not isinstance(path, dict):
        return []
    v = path.get("v") or []
    i_t = path.get("i") or [[0, 0]] * len(v)
    o_t = path.get("o") or [[0, 0]] * len(v)
    closed = path.get("c", True)
    if not v:
        return []
    pts = []
    n = len(v)
    last = n if closed else n - 1
    for a in range(last):
        b = (a + 1) % n
        p0 = v[a]
        p1 = [v[a][0] + o_t[a][0], v[a][1] + o_t[a][1]]
        p2 = [v[b][0] + i_t[b][0], v[b][1] + i_t[b][1]]
        p3 = v[b]
        straight = abs(o_t[a][0]) + abs(o_t[a][1]) + abs(i_t[b][0]) + abs(i_t[b][1]) < 1e-6
        k = 1 if straight else steps
        for s in range(k):
            t = s / k
            pts.append((_cubic(t, p0[0], p1[0], p2[0], p3[0]),
                        _cubic(t, p0[1], p1[1], p2[1], p3[1])))
    if not closed:
        pts.append((float(v[-1][0]), float(v[-1][1])))
    return pts


def _rect(sh, t):
    cx, cy = _vec2(value_at(sh.get("p"), t, [0, 0]))
    sw, sh_ = _vec2(value_at(sh.get("s"), t, [0, 0]))
    hw, hh = sw / 2, sh_ / 2
    return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]


def _ellipse(sh, t, steps=20):
    cx, cy = _vec2(value_at(sh.get("p"), t, [0, 0]))
    sw, sh_ = _vec2(value_at(sh.get("s"), t, [0, 0]))
    rx, ry = sw / 2, sh_ / 2
    return [(cx + rx * math.cos(2 * math.pi * i / steps),
             cy + ry * math.sin(2 * math.pi * i / steps)) for i in range(steps)]


def geometry(sh, t):
    ty = sh.get("ty")
    if ty == "sh":
        return flatten_path(value_at(sh.get("ks"), t, {}) or {})
    if ty == "rc":
        return _rect(sh, t)
    if ty == "el":
        return _ellipse(sh, t)
    return []


# -------------------------------------------------------------- transforms


def _vec2(v, tr=None, t=0.0, default=(0.0, 0.0)):
    """Coerce whatever a Lottie property yields into an (x, y) pair.

    Position shows up as a plain pair, as a split {x, y} pair of separate
    properties, and occasionally wrapped one list deep; normalising here keeps
    `matrix` readable.
    """
    if isinstance(v, dict):
        if tr is not None and ("px" in tr or "py" in tr):
            return (float(value_at(tr.get("px"), t, 0) or 0),
                    float(value_at(tr.get("py"), t, 0) or 0))
        return (float(_first(v.get("x", default[0]))),
                float(_first(v.get("y", default[1]))))
    if isinstance(v, (list, tuple)):
        if not v:
            return default
        if isinstance(v[0], dict):
            return _vec2(v[0], tr, t, default)
        if len(v) == 1:
            return (float(v[0]), float(v[0]))
        return (float(v[0]), float(v[1]))
    if v is None:
        return default
    return (float(v), float(v))


def _scalar(v, default=0.0):
    while isinstance(v, (list, tuple)):
        if not v:
            return default
        v = v[0]
    if isinstance(v, dict) or v is None:
        return default
    return float(v)


def matrix(tr, t):
    """Lottie transform block -> 3x3 affine matrix at time t."""
    ax, ay = _vec2(value_at(tr.get("a"), t, [0, 0]), tr, t)
    px, py = _vec2(value_at(tr.get("p"), t, [0, 0]), tr, t)
    sx, sy = _vec2(value_at(tr.get("s"), t, [100, 100]), None, t, (100.0, 100.0))
    rad = math.radians(_scalar(value_at(tr.get("r"), t, 0)))
    sx, sy = sx / 100.0, sy / 100.0
    ca, sa = math.cos(rad), math.sin(rad)
    m = np.array([[ca * sx, -sa * sy, px],
                  [sa * sx, ca * sy, py],
                  [0, 0, 1.0]])
    anchor = np.array([[1, 0, -ax], [0, 1, -ay], [0, 0, 1.0]])
    return m @ anchor


def opacity(tr, t):
    return _scalar(value_at(tr.get("o"), t, 100), 100.0)


def transform_pts(pts, m):
    if not pts:
        return []
    a = np.array([[p[0], p[1], 1.0] for p in pts]).T
    r = (m @ a)[:2].T
    return [(float(x), float(y)) for x, y in r]


# ----------------------------------------------------------------- walking


def _color_hex(fl, t):
    c = value_at(fl.get("c"), t, [0, 0, 0]) or [0, 0, 0]
    if isinstance(c, dict):
        return "#000000"
    vals = [float(v) for v in c[:3] if isinstance(v, (int, float))]
    while len(vals) < 3:
        vals.append(0.0)
    # Bodymovin writes colours as 0..1 floats in newer exports and as 0..255
    # bytes in older ones, with no version marker; infer from the values.
    scale = 1.0 if max(vals) > 1.0 else 255.0
    rgb = [max(0, min(255, int(round(v * scale)))) for v in vals]
    return "#%02X%02X%02X" % tuple(rgb)


def _walk_group(items, t, parent, out, skipped):
    """Collect (polyline, colour, opacity, stroke_width) from one shape group.

    Lottie puts a group's own `tr` block *last* in its item list, but it applies
    to everything in the group - including nested groups declared before it. So
    the transform is resolved in a first pass before any geometry is walked.
    """
    m = parent
    alpha = 100.0
    for it in items:
        if it.get("ty") == "tr":
            m = parent @ matrix(it, t)
            alpha = opacity(it, t)
            break

    geo, fill, stroke, stroke_w = [], None, None, 0.0
    for it in items:
        ty = it.get("ty")
        if ty == "gr":
            _walk_group(it.get("it", []), t, m, out, skipped)
        elif ty in SUPPORTED_SHAPES:
            geo.append(geometry(it, t))
        elif ty == "fl":
            fill = _color_hex(it, t)
            alpha = min(alpha, _scalar(value_at(it.get("o"), t, 100), 100.0))
        elif ty == "st":
            stroke = _color_hex(it, t)
            stroke_w = _scalar(value_at(it.get("w"), t, 1), 1.0)
            alpha = min(alpha, _scalar(value_at(it.get("o"), t, 100), 100.0))
        elif ty in ("gs", "gf", "tm", "rp", "mm", "sr"):
            skipped.add(ty)

    for g in geo:
        if len(g) < 2:
            continue
        world = transform_pts(g, m)
        if fill is not None:
            out.append((world, fill, alpha, 0.0))
        if stroke is not None:
            out.append((world, stroke, alpha, _scale_of(m) * stroke_w))


def _scale_of(m):
    """Uniform scale factor a transform applies, for stroke width."""
    return float(np.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])) or 1.0)


def sample(doc, t):
    """All filled polylines of a Lottie doc at frame time t, in comp space.

    Returns (shapes, skipped_feature_names) where each shape is
    (points, colour_hex, opacity_percent, stroke_width); a stroke width of 0
    means the path is filled.
    """
    out, skipped = [], set()
    layers = doc.get("layers", [])
    by_index = {ly.get("ind"): ly for ly in layers if "ind" in ly}
    # Lottie paints the layer list front-to-back; SVG paints in document order,
    # so the list is reversed to put the backdrop down first.
    for ly in reversed(layers):
        ty = ly.get("ty")
        if ty != 4:                       # 4 = shape layer
            # Null layers (3) hold no art at all - they exist to be parented
            # to, and the chain below already applies their transform. Calling
            # them "skipped" would flag two thirds of the corpus as lossy.
            if ty != 3:
                skipped.add("layer%s" % ty)
            continue
        if t < ly.get("ip", 0) or t >= ly.get("op", 1e9):
            continue
        m = np.eye(3)
        chain, cur, guard = [], ly, 0
        while cur is not None and guard < 12:
            chain.append(cur)
            cur = by_index.get(cur.get("parent"))
            guard += 1
        for node in reversed(chain):
            m = m @ matrix(node.get("ks", {}), t)
        alpha = opacity(ly.get("ks", {}), t)
        got = []
        _walk_group(ly.get("shapes", []), t, m, got, skipped)
        out.extend((p, c, min(a, alpha), w) for p, c, a, w in got)
    return out, skipped


def times(doc, fps):
    """Lottie frame times that correspond to sampling at `fps`."""
    src = float(doc.get("fr", 30) or 30)
    ip, op = float(doc.get("ip", 0)), float(doc.get("op", src))
    n = max(1, int(round((op - ip) / src * fps)))
    return [ip + i * src / fps for i in range(n)], src
