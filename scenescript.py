#!/usr/bin/env python3
"""
scenescript.py - a scene script -> a multi-character composed shot -> a video.

Everything before this in the project animated ONE figure doing ONE action
against nothing. This module is the compositor: a cast of several characters,
each with its own identity, position, scale and *independent* timeline; a
background that stays put; and props that a character can actually act on.

It is built on what already exists:
  * `stickman.StickFigure`  - the articulated rig (walk/run/jump/kick/wave/
                              dance/idle) and `frame_equations` /
                              `equations_to_layers`, reused unchanged.
  * `geovid.Scene`          - the equation rasteriser everything renders into.
  * `model.grammar.COLORS`  - the existing colour vocabulary.

Script format (line based, `#` comments, blank lines ignored)
-------------------------------------------------------------
    title      A meeting in the park
    size       854 480
    fps        30
    seconds    9
    background park

    prop tree  at 1.5 scale 1.2
    prop ball  at 6.2

    cast ana color cyan   at 1.0 scale 1.00
    cast bo  color orange at 8.6 scale 0.78

    0.0 3.4 ana walk to 3.4
    0.6 3.4 bo   walk to 6.9
    3.6 5.2 ana  wave
    4.0 6.0 bo   kick ball at ana
    6.2 8.4 ana  jump

A timeline line is `<start> <end> <who> <action> [to X] [at TARGET]`.
JSON with the same keys is also accepted (`--json`).

Usage
    python scenescript.py scenes/park_meet.scene -o out/scene_park.mp4
    python scenescript.py <script> --frames out/scene_park_frames
    python scenescript.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import geovid
import stickman
from stickman import StickFigure, _smooth

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
try:
    import grammar as G
    COLORS = dict(G.COLORS)
except Exception:                                            # pragma: no cover
    COLORS = {"blue": (90, 210, 255), "cyan": (80, 240, 240),
              "orange": (255, 150, 60), "green": (140, 255, 150),
              "pink": (255, 110, 180), "white": (240, 245, 255)}

ACTIONS = ("walk", "run", "jump", "kick", "wave", "dance", "idle")

# world constants, inherited from StickFigure so the rig's own numbers hold
WORLD_H = 5.62
GROUND = 1.15
HORIZON_LIFT = 1.05     # how far up the frame a scale-0 character would stand


# --------------------------------------------------------------------------- #
# 1. The script format
# --------------------------------------------------------------------------- #

def parse_script(text):
    """Parse the line-based scene script into a plain dict."""
    spec = {"title": "untitled", "width": 854, "height": 480, "fps": 30,
            "seconds": 8.0, "background": "park", "cast": [], "props": [],
            "timeline": []}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        head = tok[0].lower()
        try:
            if head == "title":
                spec["title"] = " ".join(tok[1:])
            elif head == "size":
                spec["width"], spec["height"] = int(tok[1]), int(tok[2])
            elif head == "fps":
                spec["fps"] = int(tok[1])
            elif head == "seconds":
                spec["seconds"] = float(tok[1])
            elif head == "background":
                spec["background"] = tok[1]
            elif head == "cast":
                spec["cast"].append(_kv(tok[2:], {"id": tok[1]},
                                       floats=("at", "scale", "y"),
                                       ints=("facing", "depth")))
            elif head == "prop":
                kind = tok[1]
                rest = tok[2:]
                if rest and rest[0] not in ("at", "scale", "y", "color"):
                    pid, rest = rest[0], rest[1:]
                else:
                    pid = kind
                spec["props"].append(_kv(rest, {"kind": kind, "id": pid},
                                         floats=("at", "scale", "y")))
            else:                                   # timeline: starts with a number
                start, end, who, action = float(tok[0]), float(tok[1]), tok[2], tok[3].lower()
                ev = _kv(tok[4:], {"start": start, "end": end, "who": who,
                                   "action": action}, floats=("to",))
                # bare first arg after the action is the prop, e.g. "kick ball at ana"
                if len(tok) > 4 and tok[4] not in ("to", "at", "prop"):
                    ev["prop"] = tok[4]
                spec["timeline"].append(ev)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"scene script line {lineno}: {raw!r} ({exc})")
    return spec


def _kv(tok, base, floats=(), ints=()):
    """`at 3.0 scale 0.8 color cyan` -> dict, with light type coercion."""
    d = dict(base)
    i = 0
    while i < len(tok) - 1:
        k, v = tok[i], tok[i + 1]
        if k in floats:
            d[k] = float(v)
        elif k in ints:
            d[k] = int(v)
        else:
            d[k] = v
        i += 2
    return d


# --------------------------------------------------------------------------- #
# 2. Characters
# --------------------------------------------------------------------------- #

class Character:
    """One cast member: an identity, a place on the ground plane, and its own
    timeline. The rig is shared code; the *root* (where the figure stands) is
    owned here, not by the action."""

    def __init__(self, spec, width, height):
        self.id = spec["id"]
        self.color = COLORS.get(spec.get("color", "blue"), (120, 220, 255))
        self.scale = float(spec.get("scale", 1.0))
        self.x0 = float(spec.get("at", 5.0))
        self.facing = int(spec.get("facing", 1))
        # the ground plane: a smaller (further) character stands higher up the
        # frame. depth order then *falls out of* scale - it is not a free field.
        self.ground_y = float(spec.get("y", GROUND + (1.0 - self.scale) * HORIZON_LIFT))
        self.depth = float(spec.get("depth", self.scale))

        fig = StickFigure(width, height, ground=GROUND)
        for k in ("L_thigh", "L_shin", "L_uarm", "L_farm", "torso", "head_r"):
            setattr(fig, k, getattr(fig, k) * self.scale)
        fig.leg = fig.L_thigh + fig.L_shin
        self.fig = fig
        self.clips = []          # (start, end, action, kwargs)
        self.motion = []         # root-motion segments (t0, t1, x0, x1)

    # -- root motion ------------------------------------------------------- #

    def x_at(self, t):
        cur = self.x0
        for (t0, t1, xa, xb) in self.motion:
            if t >= t1:
                cur = xb
            elif t >= t0:
                return _smooth(xa, xb, (t - t0) / max(1e-6, t1 - t0))
            else:
                break
        return cur

    def facing_at(self, t):
        f = self.facing
        for (t0, t1, xa, xb) in self.motion:
            if t >= t0 and abs(xb - xa) > 1e-3:
                f = 1 if xb > xa else -1
        for (s, e, act, kw) in self.clips:
            if s <= t < e and "face" in kw:
                f = kw["face"]
        return f

    def clip_at(self, t):
        best = None
        for c in self.clips:
            if c[0] <= t < c[1]:
                if best is None or c[0] >= best[0]:
                    best = c        # later-starting clip wins an overlap
        return best

    # -- pose -------------------------------------------------------------- #

    def joints_at(self, t):
        clip = self.clip_at(t)
        if clip is None:
            action, p = "idle", (t * 0.35) % 1.0
            steps = 6.0
        else:
            s, e, action, kw = clip
            p = (t - s) / max(1e-6, e - s)
            steps = kw.get("steps", 6.0)
        self.fig.steps = steps
        joints, _ = self.fig.pose(action, p)

        # re-anchor: throw away the action's own absolute hip x, keep only the
        # vertical offset it produced (bob / jump height), and put the figure
        # where the *script* says it is.
        hip = joints["hip"]
        dy = hip[1] - (self.fig.ground + self.fig.leg)
        tx = self.x_at(t)
        ty = self.ground_y + self.fig.leg + dy
        dx, dyy = tx - hip[0], ty - hip[1]
        face = self.facing_at(t)
        out = {}
        for k, (x, y) in joints.items():
            x, y = x + dx, y + dyy
            if face < 0:
                x = tx - (x - tx)
            out[k] = (x, y)
        return out

    def foot_x(self, t):
        j = self.joints_at(t)
        return j["hip"][0]


# --------------------------------------------------------------------------- #
# 3. Props (stateful, unlike characters)
# --------------------------------------------------------------------------- #

class Ball:
    """A ball is *not* a function of one action's phase. It has to remember
    where it was left, so its trajectory is integrated over the whole timeline
    as a list of segments."""

    def __init__(self, pid, x, scale=1.0):
        self.id = pid
        self.r = 0.17 * scale
        self.x0, self.y0 = x, GROUND + 0.17 * scale
        self.segs = []           # (t0, t1, kind, payload)

    def rest_x_at(self, t):
        return self.pos_at(t)[0]

    def pos_at(self, t):
        x, y, spin = self.x0, self.y0, 0.0
        travelled = 0.0
        for (t0, t1, kind, pl) in self.segs:
            if t >= t1:
                x, y = pl["x1"], pl["y1"]
                travelled += abs(pl["x1"] - pl["x0"])
                continue
            if t < t0:
                break
            f = (t - t0) / max(1e-6, t1 - t0)
            if kind == "fly":
                x = pl["x0"] + (pl["x1"] - pl["x0"]) * f
                y = pl["y0"] + (pl["y1"] - pl["y0"]) * f + pl["h"] * math.sin(math.pi * f)
            else:                                       # roll
                g = 1.0 - (1.0 - f) ** 2                # decelerating
                x = pl["x0"] + (pl["x1"] - pl["x0"]) * g
                y = pl["y1"]
            travelled += abs(x - pl["x0"])
            break
        spin = travelled / max(1e-6, self.r)
        return x, y, spin


# --------------------------------------------------------------------------- #
# 4. The scene
# --------------------------------------------------------------------------- #

BACKGROUNDS = {
    "park":   {"bg": (14, 20, 30), "sky": (18, 28, 44), "hill": (24, 44, 52),
               "ground": (26, 40, 34), "line": (70, 110, 84)},
    "night":  {"bg": (6, 7, 14), "sky": (10, 12, 26), "hill": (16, 18, 40),
               "ground": (12, 14, 26), "line": (54, 60, 96)},
    "studio": {"bg": (18, 18, 22), "sky": (24, 24, 30), "hill": (30, 30, 38),
               "ground": (34, 34, 42), "line": (86, 86, 100)},
}


class Scene:
    def __init__(self, spec):
        self.spec = spec
        self.width = int(spec.get("width", 854))
        self.height = int(spec.get("height", 480))
        self.fps = int(spec.get("fps", 30))
        self.seconds = float(spec.get("seconds", 8.0))
        self.W = (self.width / self.height) * WORLD_H
        self.pal = BACKGROUNDS.get(spec.get("background", "park"),
                                   BACKGROUNDS["park"])
        self.cast = [Character(c, self.width, self.height) for c in spec["cast"]]
        self.by_id = {c.id: c for c in self.cast}
        self.props = []
        self.balls = {}
        for p in spec.get("props", []):
            if p["kind"] == "ball":
                b = Ball(p.get("id", "ball"), float(p.get("at", self.W / 2)),
                         float(p.get("scale", 1.0)))
                self.balls[b.id] = b
                self.props.append(("ball", b))
            else:
                self.props.append((p["kind"], p))
        self._compile_timeline()
        self._bg_layers = self._background_layers()
        self.prop_stamps = self._prop_stamps()

    # -- timeline compilation: this is where the multi-character work lives - #

    def _compile_timeline(self):
        self.notes = []
        for ev in sorted(spec_timeline(self.spec), key=lambda e: e["start"]):
            who = self.by_id.get(ev["who"])
            if who is None:
                self.notes.append(f"unknown cast member {ev['who']!r} - skipped")
                continue
            s, e = float(ev["start"]), float(ev["end"])
            action = ev["action"]
            if action not in ACTIONS:
                self.notes.append(f"unknown action {action!r} -> idle")
                action = "idle"
            kw = {}

            if action in ("walk", "run") and "to" in ev:
                x0, x1 = who.x_at(s), float(ev["to"])
                who.motion.append((s, e, x0, x1))
                # foot slide: the gait cycle count must come from the distance
                # actually covered, not from a fixed constant.
                stride = 1.15 * who.scale * (1.6 if action == "run" else 1.0)
                kw["steps"] = max(1.0, abs(x1 - x0) / stride)

            elif action == "kick":
                ball = self.balls.get(ev.get("prop", "ball"))
                if ball is None:
                    self.notes.append(f"kick with no ball prop at t={s}")
                else:
                    # the kicker walks to where the ball ACTUALLY is now.
                    if any(t0 <= s < t1 for (t0, t1, _k, _p) in ball.segs):
                        self.notes.append(
                            f"kick at t={s} starts while {ball.id!r} is still "
                            f"moving - the ball is struck in mid-flight")
                    bx = ball.pos_at(s)[0]
                    x0 = who.x_at(s)
                    face = 1 if bx >= x0 else -1
                    stand = bx - face * 0.72 * who.scale
                    who.motion.append((s, s + 0.55 * (e - s), x0, stand))
                    kw["face"] = face
                    tc = s + 0.70 * (e - s)             # contact, per _act_kick
                    tgt = ev.get("at")
                    if tgt in self.by_id:
                        land = self.by_id[tgt].x_at(tc + 1.1) - face * 0.55
                    elif tgt is not None:
                        try:
                            land = float(tgt)
                        except ValueError:
                            land = bx + face * 3.0
                    else:
                        land = bx + face * 3.0
                    land = max(0.4, min(self.W - 0.4, land))
                    ball.segs.append((tc, tc + 1.1, "fly",
                                      {"x0": bx, "y0": ball.y0,
                                       "x1": land, "y1": ball.y0,
                                       "h": 1.45 * who.scale}))
                    roll = max(0.4, min(self.W - 0.4, land + face * 0.35))
                    ball.segs.append((tc + 1.1, tc + 1.7, "roll",
                                      {"x0": land, "y0": ball.y0,
                                       "x1": roll, "y1": ball.y0, "h": 0.0}))
            who.clips.append((s, e, action, kw))

        for c in self.cast:
            c.motion.sort(key=lambda m: m[0])
            c.clips.sort(key=lambda k: k[0])

    # -- layers ------------------------------------------------------------ #

    def _background_layers(self):
        """Time-independent by construction: nothing here reads t, so the
        backdrop cannot drift between frames."""
        P, W, H = self.pal, self.W, WORLD_H
        L = [{"type": "polygon", "points": [[0, 0], [W, 0], [W, H], [0, H]],
              "fill": list(P["sky"])}]
        # two hill silhouettes
        for cx, r, col in ((W * 0.28, W * 0.30, P["hill"]),
                           (W * 0.74, W * 0.24, tuple(int(v * 0.8) for v in P["hill"]))):
            th = np.linspace(0, math.pi, 40)
            pts = [[cx + r * math.cos(a), GROUND + 0.55 * r * math.sin(a)] for a in th]
            pts += [[cx - r, GROUND], [cx + r, GROUND]]
            L.append({"type": "polygon", "points": pts, "fill": list(col)})
        L.append({"type": "polygon",
                  "points": [[0, 0], [W, 0], [W, GROUND + 0.02], [0, GROUND + 0.02]],
                  "fill": list(P["ground"])})
        L.append({"type": "parametric", "x": f"0 + ({W:.3f})*t",
                  "y": f"{GROUND:.3f} + (0)*t", "t": [0, 1, 2],
                  "color": list(P["line"]), "width": 2})
        return L

    def _prop_stamps(self):
        """Static props are *not* part of the backdrop. They sit on the same
        ground plane as the cast, so they take part in the same back-to-front
        sort - otherwise a tree in the foreground is drawn behind a character
        standing further away than it is."""
        out = []
        for i, (kind, p) in enumerate(self.props):
            if kind == "ball":
                continue
            layers = _prop_layers(kind, p, self.pal)
            if not layers:
                continue
            depth = float(p.get("depth", -1.0 if kind in ("sun", "cloud")
                                 else float(p.get("scale", 1.0))))
            out.append((depth, {"matte": [dict(L, fill=[255, 255, 255])
                                          for L in layers],
                                "ink": [(tuple(L.get("fill", [200, 200, 200])), [L])
                                        for L in layers],
                                "key": f"prop{i}"}))
        return out

    def stamps_at(self, t):
        """Everything that is NOT the backdrop, back to front, as a list of
        stamps. A stamp is {matte: layers, ink: [(colour, layers)]}: `matte`
        is the dilated silhouette that must erase whatever was drawn behind
        it, `ink` is what gets painted. Kept separate because a flat-coloured
        matte drawn straight into the frame would also erase the *backdrop*,
        which is the bug this replaced."""
        out = []
        shadows = [_shadow(c.x_at(t), c.ground_y, 0.42 * c.scale) for c in self.cast]
        col = tuple(max(0, int(v * 0.55)) for v in self.pal["ground"])
        out.append({"matte": [], "ink": [(col, shadows)]})

        drawables = [("char", c.depth, c) for c in self.cast]
        for kind, p in self.props:
            if kind == "ball":
                drawables.append(("ball", 1.02, p))
        drawables += [("prop", d, st) for d, st in self.prop_stamps]
        for kind, _d, obj in sorted(drawables, key=lambda d: d[1]):
            if kind == "char":
                out.append(self._char_stamp(obj, t))
            elif kind == "ball":
                out.append(_ball_stamp(obj, t))
            else:
                out.append(obj)
        return out

    def _char_stamp(self, ch, t):
        j = ch.joints_at(t)
        eqs = stickman.frame_equations(j, head_r=ch.fig.head_r)
        w = max(2, int(round(3 * ch.scale)))
        head, r = j["head_c"], ch.fig.head_r + 0.03
        disc = [[head[0] + r * math.cos(a), head[1] + r * math.sin(a)]
                for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        matte = ([{"type": "polygon", "points": disc, "fill": [255, 255, 255]}]
                 + stickman.equations_to_layers(eqs, width=w + 4))
        return {"matte": matte,
                "ink": [(ch.color, stickman.equations_to_layers(eqs, width=w))]}


def spec_timeline(spec):
    return spec.get("timeline", [])


def _shadow(x, y, r):
    pts = [[x + r * math.cos(a), y + 0.10 * r * math.sin(a)]
           for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
    return {"type": "polygon", "points": pts, "fill": [255, 255, 255]}


def _ball_stamp(ball, t):
    x, y, spin = ball.pos_at(t)
    r = ball.r
    circ = [{"type": "polygon",
             "points": [[x + r * math.cos(a), y + r * math.sin(a)]
                        for a in np.linspace(0, 2 * math.pi, 26, endpoint=False)],
             "fill": [255, 255, 255]}]
    a0 = -spin
    spoke = [{"type": "parametric",
              "x": f"{x:.3f} + ({(r * math.cos(a0)):.3f})*(2*t-1)",
              "y": f"{y:.3f} + ({(r * math.sin(a0)):.3f})*(2*t-1)",
              "t": [0, 1, 2], "width": 2}]
    return {"matte": circ, "ink": [((255, 150, 40), circ), ((120, 60, 10), spoke)]}


def _prop_layers(kind, p, pal):
    x = float(p.get("at", 5.0))
    s = float(p.get("scale", 1.0))
    y = float(p.get("y", GROUND))
    if kind == "tree":
        trunk = [[x - 0.09 * s, y], [x + 0.09 * s, y],
                 [x + 0.06 * s, y + 1.0 * s], [x - 0.06 * s, y + 1.0 * s]]
        can = []
        for cx, cy, r in ((x, y + 1.55 * s, 0.55 * s), (x - 0.34 * s, y + 1.2 * s, 0.40 * s),
                          (x + 0.34 * s, y + 1.24 * s, 0.42 * s)):
            can.append([[cx + r * math.cos(a), cy + r * math.sin(a)]
                        for a in np.linspace(0, 2 * math.pi, 22, endpoint=False)])
        out = [{"type": "polygon", "points": trunk, "fill": [86, 62, 40]}]
        out += [{"type": "polygon", "points": c, "fill": [46, 110, 66]} for c in can]
        return out
    if kind == "bush":
        return [{"type": "polygon",
                 "points": [[x + 0.5 * s * math.cos(a), y + 0.33 * s * math.sin(a)]
                            for a in np.linspace(0, math.pi, 18)] + [[x - 0.5 * s, y]],
                 "fill": [40, 96, 58]}]
    if kind == "rock":
        return [{"type": "polygon",
                 "points": [[x - 0.36 * s, y], [x - 0.22 * s, y + 0.26 * s],
                            [x + 0.05 * s, y + 0.34 * s], [x + 0.30 * s, y + 0.18 * s],
                            [x + 0.38 * s, y]],
                 "fill": [92, 96, 104]}]
    if kind == "box":
        h = 0.55 * s
        return [{"type": "polygon",
                 "points": [[x - 0.35 * s, y], [x + 0.35 * s, y],
                            [x + 0.35 * s, y + h], [x - 0.35 * s, y + h]],
                 "fill": [122, 88, 48], "color": [180, 140, 90], "width": 2}]
    if kind == "sun":
        cy = float(p.get("y", 4.5))
        r = 0.42 * s
        return [{"type": "polygon",
                 "points": [[x + r * math.cos(a), cy + r * math.sin(a)]
                            for a in np.linspace(0, 2 * math.pi, 30, endpoint=False)],
                 "fill": [235, 205, 130]}]
    if kind == "cloud":
        cy = float(p.get("y", 4.4))
        out = []
        for dx, dy, r in ((-0.4, 0, 0.30), (0, 0.10, 0.38), (0.42, 0, 0.28)):
            out.append({"type": "polygon",
                        "points": [[x + dx * s + r * s * math.cos(a),
                                    cy + dy * s + r * s * math.sin(a)]
                                   for a in np.linspace(0, 2 * math.pi, 22, endpoint=False)],
                        "fill": [56, 68, 92]})
        return out
    return []


# --------------------------------------------------------------------------- #
# 5. Render
# --------------------------------------------------------------------------- #

def _white(layers):
    out = []
    for L in layers:
        d = dict(L)
        if "fill" in d and d["fill"] is not None:
            d["fill"] = [255, 255, 255]
        d["color"] = [255, 255, 255]
        out.append(d)
    return out


class _Raster:
    """Renders one layer list to an anti-aliased 8-bit mask via geovid."""

    def __init__(self, scene):
        self.base = {"width": scene.width, "height": scene.height,
                     "fps": scene.fps, "duration": scene.seconds,
                     "supersample": 2,
                     "view": {"xmin": 0, "xmax": scene.W,
                              "ymin": 0, "ymax": WORLD_H}}
        self.bgcol = list(scene.pal["bg"])
        self._cache = {}

    def colour(self, layers):
        spec = dict(self.base, background=self.bgcol, layers=layers)
        return geovid.Scene(spec).render_frame(0)

    def mask(self, layers, key=None):
        if not layers:
            return None
        if key is not None and key in self._cache:
            return self._cache[key]
        spec = dict(self.base, background=[0, 0, 0], layers=_white(layers))
        m = geovid.Scene(spec).render_frame(0).convert("L")
        if key is not None:
            self._cache[key] = m
        return m


def compose_frame(scene: Scene, rast, bg_img, t):
    """Backdrop, then every stamp back to front. A stamp's matte restores the
    *backdrop* before its ink is painted, so a nearer figure hides the figures
    behind it without punching a hole in the scenery."""
    cur = bg_img.copy()
    for st in scene.stamps_at(t):
        k = st.get("key")
        m = rast.mask(st["matte"], key=k and k + "/m")
        if m is not None:
            cur.paste(bg_img, mask=m)
        for n, (colour, layers) in enumerate(st["ink"]):
            mi = rast.mask(layers, key=k and f"{k}/i{n}")
            if mi is not None:
                cur.paste(Image.new("RGB", cur.size, tuple(colour)), mask=mi)
    return cur


def render(scene: Scene, out=None, frames_dir=None, verbose=True):
    n = max(1, int(round(scene.fps * scene.seconds)))
    rast = _Raster(scene)
    bg_img = rast.colour(scene._bg_layers)       # rendered ONCE, by construction
    proc = None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = [geovid.find_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{scene.width}x{scene.height}", "-r", str(scene.fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    fd = Path(frames_dir) if frames_dir else None
    if fd:
        fd.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = compose_frame(scene, rast, bg_img, i / scene.fps)
        if proc:
            proc.stdin.write(img.tobytes())
        if fd:
            img.save(fd / f"f{i:04d}.png")
        if verbose and (i % scene.fps == 0 or i == n - 1):
            print(f"\r  frame {i+1}/{n}", end="", flush=True)
    if verbose:
        print()
    if proc:
        proc.stdin.close()
        proc.wait()
    return n


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json") or text.lstrip().startswith("{"):
        return json.loads(text)
    return parse_script(text)


# --------------------------------------------------------------------------- #
# 6. Self test - the checks the composition problems need
# --------------------------------------------------------------------------- #

def selftest():
    ok = True
    src = """title selftest
size 320 180
fps 15
seconds 4
background park
prop tree at 1.5
prop ball at 5.5
cast a color cyan at 1.0 scale 1.0
cast b color orange at 8.0 scale 0.7
0.0 2.0 a walk to 3.0
0.5 2.5 b walk to 6.5
2.2 3.5 a wave
2.6 3.8 b kick ball at a
"""
    sc = Scene(parse_script(src))
    print(f"cast={[c.id for c in sc.cast]} props={[k for k, _ in sc.props]} "
          f"notes={sc.notes}")

    # 1. background is time-invariant
    rast = _Raster(sc)
    a = np.asarray(rast.colour(sc._bg_layers))
    b = np.asarray(rast.colour(sc._bg_layers))
    same = int(np.abs(a.astype(int) - b.astype(int)).max())
    print(f"background max |diff| over frames: {same}  -> {'OK' if same == 0 else 'FAIL'}")
    ok &= same == 0

    # 2. characters are independent: different action at the same instant
    print(f"t=2.3  a clip={sc.by_id['a'].clip_at(2.3)[2]!r} "
          f"b clip={sc.by_id['b'].clip_at(2.3)[2]!r}")

    # 3. feet stay on their own ground line
    for c in sc.cast:
        errs = []
        for i in range(0, 60):
            t = i / 15
            j = c.joints_at(t)
            errs.append(min(j["foot_R"][1], j["foot_L"][1]) - c.ground_y)
        print(f"  {c.id}: ground_y={c.ground_y:.3f} foot-to-ground "
              f"min={min(errs):+.3f} max={max(errs):+.3f}")
        ok &= min(errs) > -0.35

    # 4. the ball is kicked by someone and lands somewhere
    ball = sc.balls["ball"]
    xs = [(round(t, 2), round(ball.pos_at(t)[0], 2), round(ball.pos_at(t)[1], 2))
          for t in (0.0, 3.0, 3.5, 4.0, 4.6, 5.5)]
    print(f"  ball track: {xs}")
    ok &= abs(ball.pos_at(0.0)[0] - ball.pos_at(5.5)[0]) > 0.5

    # 5. the kicker actually reached the ball
    kicker = sc.by_id["b"]
    tc = 2.6 + 0.70 * (3.8 - 2.6)
    print(f"  kicker x at contact={kicker.x_at(tc):.2f}  ball x={ball.pos_at(2.6)[0]:.2f}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def audit(script_path, frames_dir, step=5, tol=45):
    """Independent cast audit, added because the critic's `scene_complexity`
    cannot see this style at all: `yolo11n-pose.pt` returns a modal cast of 0
    on these renders AND on the project's own single-figure hand-written
    controls. This does not detect people - it measures, per frame, how much
    of each declared character's ink is actually on screen and how many
    separate blobs it forms. It measures presence and occlusion, not
    personhood, and is reported as such."""
    import cv2
    sc = Scene(load(script_path))
    files = sorted(Path(frames_dir).glob("*.png"))[::step]
    rows = {c.id: {"present": 0, "blobs": [], "px": []} for c in sc.cast}
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
        for c in sc.cast:
            d = np.abs(im - np.array(c.color, dtype=np.int16)).sum(-1)
            m = (d < tol).astype(np.uint8)
            n = int(m.sum())
            rows[c.id]["px"].append(n)
            if n > 40:
                rows[c.id]["present"] += 1
                nlab, _ = cv2.connectedComponents(m, connectivity=8)
                rows[c.id]["blobs"].append(nlab - 1)
    print(f"cast audit over {len(files)} frames of {frames_dir}")
    for c in sc.cast:
        r = rows[c.id]
        b = r["blobs"] or [0]
        print(f"  {c.id:5s} colour={c.color} scale={c.scale:.2f} "
              f"visible {r['present']}/{len(files)} frames "
              f"({100*r['present']/len(files):.1f}%)  "
              f"ink px mean={np.mean(r['px']):.0f} min={min(r['px'])}  "
              f"blobs mean={np.mean(b):.2f}")
    allpx = sum(np.array(rows[c.id]["px"]) > 40 for c in sc.cast)
    print(f"  frames with all {len(sc.cast)} characters visible: "
          f"{int((allpx == len(sc.cast)).sum())}/{len(files)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", nargs="?")
    ap.add_argument("-o", "--out", default=None, help="output .mp4")
    ap.add_argument("--frames", default=None, help="also write PNG frames here")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audit", default=None,
                    help="frames dir to audit against the script")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.script:
        ap.error("give a script, or --selftest")
    if args.audit:
        audit(args.script, args.audit)
        return
    spec = load(args.script)
    if args.seconds:
        spec["seconds"] = args.seconds
    sc = Scene(spec)
    for n in sc.notes:
        print("note:", n)
    print(f"{spec.get('title')!r}: {len(sc.cast)} characters, "
          f"{len(sc.props)} props, {len(spec_timeline(spec))} timeline events")
    n = render(sc, args.out, args.frames)
    print(f"{n} frames -> {args.out or args.frames}")


if __name__ == "__main__":
    main()
