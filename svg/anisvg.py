"""AniSVG - the animation representation this project trains on.

Raster tracing failed because it pays for the *whole picture* every frame
(~4000 tokens at 15 fps = ~60k tokens per second of video; see ATTEMPTS.md).
AniSVG instead states a cast of shapes **once**, then spends tokens only on
what actually changes per frame - which is how cel animation, Flash and After
Effects all work, and it puts a frame back in the ~40-60 token range a small
model can realistically generate.

Text format (line oriented, whitespace separated, all integers):

    H <w> <h> <fps> <nframes>          header
    P <hex> ...                        palette, one line, index = position
    S <id> <pal> <sw> <x0> <y0> <dx dy>...  cast member: path, relative pts
                                       sw = stroke width x10, 0 = filled
    @ <frame> <op> ...                 per-frame edits; unlisted shapes hold

Per-frame ops:

    t <id> <dx> <dy>     translate by delta (grid units)
    r <id> <ddeg>        rotate by delta, about the shape's own centroid
    s <id> <dpct>        uniform scale delta, percent
    o <id> <dpct>        opacity delta, percent
    v <id> <dx dy>...    per-vertex delta (a morph - the expensive op)
    h <id>               hide
    w <id>               show

Deltas rather than absolutes throughout: an animated shape moves a few units
per frame, and small integers cost one BPE token where three-digit absolute
coordinates cost three.
"""
import math

QUANT = 1

_OPS = {"t": 2, "r": 1, "s": 1, "o": 1, "h": 0, "w": 0}


class Shape:
    __slots__ = ("id", "pal", "pts", "stroke")

    def __init__(self, sid, pal, pts, stroke=0):
        self.id = sid
        self.pal = pal
        self.pts = [(float(x), float(y)) for x, y in pts]
        # Tenths of a unit; 0 means the path is filled rather than stroked.
        # Line art - which is what anime actually is - is mostly strokes, so
        # this is not an optional extra.
        self.stroke = int(stroke)

    def centroid(self):
        n = len(self.pts) or 1
        return (sum(p[0] for p in self.pts) / n, sum(p[1] for p in self.pts) / n)


class State:
    """Live transform of one cast member."""
    __slots__ = ("tx", "ty", "rot", "scale", "opacity", "visible", "morph")

    def __init__(self, n_pts=0):
        self.tx = self.ty = self.rot = 0.0
        self.scale = 100.0
        self.opacity = 100.0
        self.visible = True
        self.morph = [(0.0, 0.0)] * n_pts

    def copy(self):
        s = State()
        s.tx, s.ty, s.rot = self.tx, self.ty, self.rot
        s.scale, s.opacity, s.visible = self.scale, self.opacity, self.visible
        s.morph = list(self.morph)
        return s


class Anim:
    def __init__(self, width, height, fps, palette, shapes, frames):
        self.width = width
        self.height = height
        self.fps = fps
        self.palette = list(palette)
        self.shapes = list(shapes)
        self.frames = list(frames)          # frames[i] = list of op tuples

    # ------------------------------------------------------------ encoding

    def to_text(self, quant=QUANT):
        out = ["H %d %d %d %d" % (self.width, self.height, self.fps,
                                  len(self.frames))]
        out.append("P " + " ".join(self.palette))
        for sh in self.shapes:
            q = [(round(x / quant), round(y / quant)) for x, y in sh.pts]
            body = ["S", str(sh.id), str(sh.pal), str(sh.stroke),
                    str(q[0][0]), str(q[0][1])]
            px, py = q[0]
            for x, y in q[1:]:
                body += [str(x - px), str(y - py)]
                px, py = x, y
            out.append(" ".join(body))
        for i, ops in enumerate(self.frames):
            if not ops:
                continue
            parts = ["@", str(i)]
            for op in ops:
                parts += [str(v) for v in op]
            out.append(" ".join(parts))
        return "\n".join(out)

    @staticmethod
    def from_text(text, quant=QUANT):
        w = h = fps = n = 0
        pal, shapes, frames = [], [], {}
        for line in text.splitlines():
            f = line.split()
            if not f:
                continue
            if f[0] == "H":
                w, h, fps, n = (int(v) for v in f[1:5])
            elif f[0] == "P":
                pal = f[1:]
            elif f[0] == "S":
                sid, p, sw = int(f[1]), int(f[2]), int(f[3])
                x, y = int(f[4]), int(f[5])
                pts = [(x * quant, y * quant)]
                for i in range(6, len(f) - 1, 2):
                    x += int(f[i])
                    y += int(f[i + 1])
                    pts.append((x * quant, y * quant))
                shapes.append(Shape(sid, p, pts, sw))
            elif f[0] == "@":
                frames[int(f[1])] = _parse_ops(f[2:])
        return Anim(w, h, fps, pal, shapes, [frames.get(i, []) for i in range(n)])

    # -------------------------------------------------------------- replay

    def states(self):
        """Yield the resolved per-shape State dict for every frame."""
        cur = {sh.id: State(len(sh.pts)) for sh in self.shapes}
        for ops in self.frames:
            for op in ops:
                _apply(cur, op)
            yield {k: v.copy() for k, v in cur.items()}

    def frame_svg(self, state):
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
                 % (self.width, self.height)]
        for sh in self.shapes:
            st = state.get(sh.id)
            if st is None or not st.visible or st.opacity <= 0:
                continue
            pts = _transform(sh, st)
            d = "M" + " L".join("%.1f,%.1f" % p for p in pts)
            colour = self.palette[sh.pal] if sh.pal < len(self.palette) else "#000000"
            fade = "" if st.opacity >= 100 else ' opacity="%.2f"' % (st.opacity / 100.0)
            if sh.stroke > 0:
                paint = ('fill="none" stroke="%s" stroke-width="%.1f" '
                         'stroke-linecap="round" stroke-linejoin="round"'
                         % (colour, sh.stroke / 10.0 * st.scale / 100.0))
            else:
                d += " Z"
                paint = 'fill="%s"' % colour
            parts.append('<path d="%s" %s%s/>' % (d, paint, fade))
        parts.append("</svg>")
        return "".join(parts)

    def to_svgs(self):
        return [self.frame_svg(s) for s in self.states()]


# ---------------------------------------------------------------- internals


def _is_op(tok):
    return tok in _OPS or tok == "v"


def _parse_ops(fields):
    ops, i = [], 0
    while i < len(fields):
        kind = fields[i]
        if i + 1 >= len(fields):
            # A generated clip can stop mid-op when it hits the token limit.
            # The frames before it are perfectly good, so drop the stub rather
            # than fail the whole document.
            break
        if kind == "v":                       # variable length, runs to next op
            sid = int(fields[i + 1])
            j = i + 2
            vals = []
            while j < len(fields) and not _is_op(fields[j]):
                vals.append(int(fields[j]))
                j += 1
            ops.append(("v", sid) + tuple(vals))
            i = j
        elif kind in _OPS:
            k = _OPS[kind]
            ops.append((kind, int(fields[i + 1]))
                       + tuple(int(v) for v in fields[i + 2: i + 2 + k]))
            i += 2 + k
        else:
            i += 1
    return ops


# Minimum tuple length per op, counting the op name and shape id. A generated
# clip can end mid-op, and a truncated `t` used to crash replay rather than
# report an invalid clip - so arity is checked once, here, for every caller.
_ARITY = {"t": 4, "r": 3, "s": 3, "o": 3, "v": 4, "h": 2, "w": 2}


def valid_op(op):
    return len(op) >= _ARITY.get(op[0], 2) if op else False


def _apply(cur, op):
    if not valid_op(op):
        return
    kind, sid = op[0], op[1]
    st = cur.get(sid)
    if st is None:
        return
    if kind == "t":
        st.tx += op[2]
        st.ty += op[3]
    elif kind == "r":
        st.rot += op[2]
    elif kind == "s":
        st.scale += op[2]
    elif kind == "o":
        st.opacity = max(0.0, min(100.0, st.opacity + op[2]))
    elif kind == "h":
        st.visible = False
    elif kind == "w":
        st.visible = True
    elif kind == "v":
        vals = op[2:]
        st.morph = [(st.morph[i][0] + vals[2 * i], st.morph[i][1] + vals[2 * i + 1])
                    if 2 * i + 1 < len(vals) else st.morph[i]
                    for i in range(len(st.morph))]


def _transform(sh, st):
    cx, cy = sh.centroid()
    a = math.radians(st.rot)
    ca, sa = math.cos(a), math.sin(a)
    k = st.scale / 100.0
    out = []
    for i, (x, y) in enumerate(sh.pts):
        mx, my = st.morph[i] if i < len(st.morph) else (0.0, 0.0)
        x, y = x + mx - cx, y + my - cy
        out.append((cx + (x * ca - y * sa) * k + st.tx,
                    cy + (x * sa + y * ca) * k + st.ty))
    return out
