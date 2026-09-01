"""Compact text encoding of a traced frame, and frame-to-frame deltas.

Raw SVG is a terrible training target: `<path d="M120,84 L121,86 ..." fill=
"#728FAF"/>` spends ~7 tokens per point on markup and 3-digit coordinates. Two
changes cut that hard:

  * **relative points** — traced outlines move a few pixels per step, so
    deltas are small integers, which a BPE tokenizer encodes in one token
  * **palette indices** instead of hex fills, since the shot has a fixed palette

On top of that, consecutive frames of a shot share most of their geometry, so a
frame can be expressed as edits to the previous one.

Frame grammar (one line per path, `;` separated):

    <pal> <x0> <y0> <dx dy>*        e.g.  "3 120 84 1 2 0 3 -2 1"

Delta grammar (ops against the previous frame's path list, in order):

    =            keep path unchanged
    ~<pal> <x0> <y0> <dx dy>*   replace path
    +<pal> ...   insert path
    -            delete path
"""
import re

_NUM = re.compile(r"-?\d+\.?\d*")
_TAG = re.compile(r"<path[^>]*>")
_D = re.compile(r'\bd="([^"]*)"')
_FILL = re.compile(r'\bfill="([^"]*)"')


def _pts(d):
    n = [float(v) for v in _NUM.findall(d)]
    return [(n[i], n[i + 1]) for i in range(0, len(n) - 1, 2)]


def svg_paths(svg):
    """[(hex_fill, [(x, y), ...])] in draw order."""
    out = []
    for tag in _TAG.findall(svg):
        d = _D.search(tag)
        if not d:
            continue
        f = _FILL.search(tag)
        out.append(((f.group(1) if f else "#000000").upper(), _pts(d.group(1))))
    return out


def encode_frame(paths, pal_index, quant=1):
    """paths -> compact text. `pal_index(hex) -> int`."""
    lines = []
    for fill, pts in paths:
        if len(pts) < 3:
            continue
        q = [(round(x / quant), round(y / quant)) for x, y in pts]
        body = [str(pal_index(fill)), str(q[0][0]), str(q[0][1])]
        px, py = q[0]
        for x, y in q[1:]:
            body += [str(x - px), str(y - py)]
            px, py = x, y
        lines.append(" ".join(body))
    return " ; ".join(lines)


def decode_frame(text, hexes, width, height, quant=1):
    """compact text -> SVG document."""
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
           % (width, height)]
    for line in text.split(";"):
        n = line.split()
        if len(n) < 5:
            continue
        pal = int(n[0])
        x, y = int(n[1]), int(n[2])
        pts = [(x, y)]
        for i in range(3, len(n) - 1, 2):
            x += int(n[i])
            y += int(n[i + 1])
            pts.append((x, y))
        d = "M" + " L".join("%d,%d" % (px * quant, py * quant) for px, py in pts) + " Z"
        out.append('<path d="%s" fill="%s"/>' % (d, hexes[min(pal, len(hexes) - 1)]))
    out.append("</svg>")
    return "".join(out)


def _key(fill, pts):
    """Cheap identity for a path: colour + rounded centroid + point count."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (fill, round(sum(xs) / len(xs) / 8), round(sum(ys) / len(ys) / 8),
            len(pts) // 4)


def encode_delta(prev, cur, pal_index, quant=1):
    """Express `cur` paths as edits to `prev` paths."""
    if prev is None:
        return encode_frame(cur, pal_index, quant)

    def one(fill, pts):
        return encode_frame([(fill, pts)], pal_index, quant)

    ops, i, j = [], 0, 0
    pk = [_key(f, p) for f, p in prev]
    ck = [_key(f, p) for f, p in cur]
    while i < len(prev) and j < len(cur):
        if prev[i] == cur[j]:
            ops.append("=")
            i, j = i + 1, j + 1
        elif pk[i] == ck[j]:
            ops.append("~" + one(*cur[j]))
            i, j = i + 1, j + 1
        elif ck[j] in pk[i + 1:]:          # prev path vanished
            ops.append("-")
            i += 1
        else:                              # new path
            ops.append("+" + one(*cur[j]))
            j += 1
    ops += ["-"] * (len(prev) - i)
    ops += ["+" + one(*cur[j2]) for j2 in range(j, len(cur))]
    return " ; ".join(ops)


def apply_delta(prev, text, hexes, quant=1):
    """Reconstruct a path list from `prev` plus a delta string."""
    out, i = [], 0
    for op in text.split(";"):
        op = op.strip()
        if not op:
            continue
        if op == "=":
            out.append(prev[i])
            i += 1
        elif op == "-":
            i += 1
        elif op[0] in "~+":
            svg = decode_frame(op[1:], hexes, 1, 1, quant)
            got = svg_paths(svg)
            if got:
                out.append(got[0])
            if op[0] == "~":
                i += 1
    out.extend(prev[i:])
    return out
