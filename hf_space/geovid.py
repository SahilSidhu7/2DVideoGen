#!/usr/bin/env python3
"""
geovid - generate videos from geometric / mathematical equations, frame by frame.

Design goals:
  * Light on resources. Pure numpy compute + Pillow rasterization + a raw pipe
    to the system ffmpeg. No matplotlib, no Manim.
  * Scene-driven. A scene is a small JSON (or Python dict) describing a viewport,
    a duration, and a list of "layers" (curves / fields) whose equations may
    depend on a time variable, so the plot animates.

Layer types
-----------
parametric : x(t, time), y(t, time)          -- t swept over a range, drawn as a polyline
explicit   : y(x, time)                       -- y as a function of x across the viewport
polar      : r(theta, time)                   -- radius as a function of angle
field      : f(x, y, time)                    -- scalar field over the whole frame -> colormap

Every equation is a string evaluated with numpy. Available names:
  variables : t, x, y, theta, r, time, progress (time/duration), frame, duration
  constants : pi, tau, e
  functions : sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt
              abs sign floor ceil round mod fmod hypot clip where minimum maximum
              gauss (gaussian bump), pulse, smoothstep
  plus any keys you put in the layer's "params" object.

Usage
-----
  python geovid.py render scene.json -o out.mp4
  python geovid.py render scene.json -o out.mp4 --width 1280 --height 720 --fps 60
  python geovid.py preview scene.json -o frame.png --at 0.5      # single frame at t=0.5s
  python geovid.py demo lissajous -o out.mp4                     # built-in demo scene
  python geovid.py demos                                         # list built-in demos
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Expression evaluation
# --------------------------------------------------------------------------- #

def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _gauss(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def _pulse(x, period=1.0, duty=0.5):
    return (np.mod(x, period) < period * duty).astype(float)


# numpy-backed function namespace shared by every expression
_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "round": np.round, "mod": np.mod, "fmod": np.fmod, "hypot": np.hypot,
    "clip": np.clip, "where": np.where, "minimum": np.minimum, "maximum": np.maximum,
    "gauss": _gauss, "pulse": _pulse, "smoothstep": _smoothstep,
    "pi": np.pi, "tau": 2 * np.pi, "e": math.e,
}


class Expr:
    """A compiled equation string, evaluated against a variable dict."""

    def __init__(self, source, params=None):
        self.source = str(source)
        self.params = dict(params or {})
        self.code = compile(self.source, "<equation>", "eval")

    def __call__(self, **variables):
        env = dict(_FUNCS)
        env.update(self.params)
        env.update(variables)
        return eval(self.code, {"__builtins__": {}}, env)  # noqa: S307 - local authored scenes


# --------------------------------------------------------------------------- #
# Viewport: world coordinates <-> pixel coordinates
# --------------------------------------------------------------------------- #

class Viewport:
    def __init__(self, width, height, xmin, xmax, ymin, ymax):
        self.w, self.h = width, height
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def to_px(self, x, y):
        px = (np.asarray(x) - self.xmin) / (self.xmax - self.xmin) * (self.w - 1)
        py = (self.ymax - np.asarray(y)) / (self.ymax - self.ymin) * (self.h - 1)
        return px, py

    def grid(self):
        xs = np.linspace(self.xmin, self.xmax, self.w)
        ys = np.linspace(self.ymax, self.ymin, self.h)
        return np.meshgrid(xs, ys)


# --------------------------------------------------------------------------- #
# Color helpers
# --------------------------------------------------------------------------- #

# compact perceptual-ish colormaps as control-point lists (t in 0..1 -> rgb)
_COLORMAPS = {
    "inferno": [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
                (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
                (180, 222, 44), (253, 231, 37)],
    "magma": [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
              (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135),
              (252, 253, 191)],
    "ice": [(4, 6, 20), (10, 40, 90), (20, 90, 160), (40, 160, 210),
            (150, 220, 240), (240, 250, 255)],
    "fire": [(0, 0, 0), (60, 0, 0), (140, 20, 0), (220, 80, 0),
             (255, 170, 30), (255, 240, 180), (255, 255, 255)],
}


def apply_colormap(values, name):
    """values in 0..1 (2D) -> uint8 RGB image via named colormap."""
    stops = np.array(_COLORMAPS.get(name, _COLORMAPS["viridis"]), dtype=np.float64)
    n = len(stops) - 1
    v = np.clip(values, 0.0, 1.0) * n
    lo = np.floor(v).astype(int)
    lo = np.clip(lo, 0, n - 1)
    frac = (v - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return rgb.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

class Scene:
    def __init__(self, spec):
        self.width = int(spec.get("width", 854))
        self.height = int(spec.get("height", 480))
        self.fps = int(spec.get("fps", 30))
        self.duration = float(spec.get("duration", 8.0))
        self.background = tuple(spec.get("background", [8, 8, 16]))
        v = spec.get("view", {})
        # default view keeps square aspect matched to frame
        aspect = self.width / self.height
        self.view_spec = {
            "xmin": v.get("xmin", -aspect * 2),
            "xmax": v.get("xmax", aspect * 2),
            "ymin": v.get("ymin", -2.0),
            "ymax": v.get("ymax", 2.0),
        }
        self.axes = spec.get("axes", False)
        self.supersample = int(spec.get("supersample", 1))
        self.layers = spec.get("layers", [])

    def viewport(self, scale=1):
        return Viewport(self.width * scale, self.height * scale,
                        **self.view_spec)

    # -- per-layer rasterization ------------------------------------------- #

    def _draw_polyline(self, draw, vp, xs, ys, color, width, close=False):
        px, py = vp.to_px(xs, ys)
        pts = np.stack([px, py], axis=-1)
        # split on non-finite points so asymptotes don't draw across the frame
        finite = np.isfinite(pts).all(axis=1)
        seg = []
        for i, ok in enumerate(finite):
            if ok:
                seg.append((float(pts[i, 0]), float(pts[i, 1])))
            else:
                if len(seg) > 1:
                    draw.line(seg, fill=color, width=width, joint="curve")
                seg = []
        if len(seg) > 1:
            draw.line(seg, fill=color, width=width, joint="curve")
        if close and len(seg) > 1:
            draw.line([seg[-1], seg[0]], fill=color, width=width)

    def _render_field(self, layer, vp, time, progress, frame):
        X, Y = vp.grid()
        expr = Expr(layer["f"], layer.get("params"))
        vals = expr(x=X, y=Y, time=time, progress=progress, frame=frame,
                    duration=self.duration,
                    r=np.hypot(X, Y), theta=np.arctan2(Y, X))
        vals = np.asarray(vals, dtype=np.float64)
        lo = layer.get("min", None)
        hi = layer.get("max", None)
        if lo is None:
            lo = np.nanmin(vals)
        if hi is None:
            hi = np.nanmax(vals)
        norm = (vals - lo) / (hi - lo + 1e-12)
        rgb = apply_colormap(norm, layer.get("cmap", "viridis"))
        return rgb

    def render_frame(self, frame_idx, scale=None):
        scale = scale if scale is not None else self.supersample
        vp = self.viewport(scale)
        time = frame_idx / self.fps
        progress = time / self.duration if self.duration else 0.0

        # base image: start from any field layer, else solid background
        base = None
        overlay_layers = []
        for layer in self.layers:
            if layer.get("type") == "field":
                rgb = self._render_field(layer, vp, time, progress, frame_idx)
                base = rgb if base is None else rgb  # last field wins as base
            else:
                overlay_layers.append(layer)

        if base is None:
            img = Image.new("RGB", (vp.w, vp.h), self.background)
        else:
            img = Image.fromarray(base, "RGB")

        draw = ImageDraw.Draw(img)

        if self.axes:
            self._draw_axes(draw, vp)

        for layer in overlay_layers:
            self._render_curve(draw, layer, vp, time, progress, frame_idx)

        if scale != 1:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        return img

    def _draw_axes(self, draw, vp):
        col = (60, 60, 80)
        x0, y0 = vp.to_px(0, 0)
        draw.line([(0, float(y0)), (vp.w, float(y0))], fill=col, width=1)
        draw.line([(float(x0), 0), (float(x0), vp.h)], fill=col, width=1)

    def _render_curve(self, draw, layer, vp, time, progress, frame):
        typ = layer.get("type", "parametric")
        color = tuple(layer.get("color", [120, 200, 255]))
        width = int(layer.get("width", 2)) * (vp.w // self.width or 1)
        params = layer.get("params")

        if typ == "parametric":
            spec = layer.get("t", [0, 2 * math.pi, 2000])
            t = np.linspace(spec[0], spec[1], int(spec[2]))
            xs = Expr(layer["x"], params)(t=t, time=time, progress=progress,
                                          frame=frame, duration=self.duration)
            ys = Expr(layer["y"], params)(t=t, time=time, progress=progress,
                                          frame=frame, duration=self.duration)
            xs = np.broadcast_to(np.asarray(xs, dtype=float), t.shape)
            ys = np.broadcast_to(np.asarray(ys, dtype=float), t.shape)
            self._draw_polyline(draw, vp, xs, ys, color, width,
                                close=layer.get("close", False))

        elif typ == "explicit":
            n = int(layer.get("samples", vp.w))
            xs = np.linspace(vp.xmin, vp.xmax, n)
            ys = Expr(layer["y"], params)(x=xs, time=time, progress=progress,
                                          frame=frame, duration=self.duration)
            ys = np.broadcast_to(np.asarray(ys, dtype=float), xs.shape)
            self._draw_polyline(draw, vp, xs, ys, color, width)

        elif typ == "polar":
            spec = layer.get("theta", [0, 2 * math.pi, 2000])
            th = np.linspace(spec[0], spec[1], int(spec[2]))
            r = Expr(layer["r"], params)(theta=th, time=time, progress=progress,
                                         frame=frame, duration=self.duration)
            r = np.broadcast_to(np.asarray(r, dtype=float), th.shape)
            self._draw_polyline(draw, vp, r * np.cos(th), r * np.sin(th),
                                color, width, close=layer.get("close", False))
        elif typ == "polygon":
            # filled polygon in world coordinates. Added for scenescript.py:
            # backgrounds, props and occlusion masks need *fills*, and every
            # other layer type here is stroke-only.
            pts = np.asarray(layer["points"], dtype=float)
            px, py = vp.to_px(pts[:, 0], pts[:, 1])
            poly = [(float(a), float(b)) for a, b in zip(px, py)]
            if len(poly) < 3:
                return
            fill = tuple(layer["fill"]) if layer.get("fill") is not None else None
            outline = tuple(layer["color"]) if layer.get("color") else None
            draw.polygon(poly, fill=fill, outline=outline)
            if outline and width > 1:
                draw.line(poly + [poly[0]], fill=outline, width=width,
                          joint="curve")

        else:
            raise ValueError(f"unknown layer type: {typ!r}")


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found on PATH. Install ffmpeg and retry.")
    return exe


def render_video(scene: Scene, out_path, crf=20, verbose=True):
    ffmpeg = find_ffmpeg()
    n_frames = max(1, int(round(scene.duration * scene.fps)))
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{scene.width}x{scene.height}",
        "-r", str(scene.fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for i in range(n_frames):
            img = scene.render_frame(i)
            proc.stdin.write(img.tobytes())
            if verbose and (i % scene.fps == 0 or i == n_frames - 1):
                pct = (i + 1) / n_frames * 100
                print(f"\r  rendering {i+1}/{n_frames} frames ({pct:4.0f}%)",
                      end="", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    if verbose:
        print()
    return out_path


# --------------------------------------------------------------------------- #
# Built-in demo scenes
# --------------------------------------------------------------------------- #

DEMOS = {
    "lissajous": {
        "width": 854, "height": 480, "fps": 30, "duration": 10,
        "background": [6, 6, 14], "axes": False, "supersample": 2,
        "view": {"xmin": -1.4, "xmax": 1.4, "ymin": -0.8, "ymax": 0.8},
        "layers": [
            {"type": "parametric", "t": [0, 6.283185, 3000],
             "x": "sin(3*t + 0.5*time)", "y": "sin(2*t)",
             "color": [90, 210, 255], "width": 2},
            {"type": "parametric", "t": [0, 6.283185, 3000],
             "x": "sin(4*t)", "y": "sin(3*t + 0.3*time)",
             "color": [255, 110, 180], "width": 2},
        ],
    },
    "rose": {
        "width": 720, "height": 720, "fps": 30, "duration": 12,
        "background": [8, 4, 12], "supersample": 2,
        "view": {"xmin": -1.2, "xmax": 1.2, "ymin": -1.2, "ymax": 1.2},
        "layers": [
            {"type": "polar", "theta": [0, 25.13274, 4000],
             "r": "cos((2 + 2*progress) * theta)",
             "color": [255, 160, 60], "width": 2},
        ],
    },
    "spirograph": {
        "width": 720, "height": 720, "fps": 30, "duration": 12,
        "background": [4, 6, 10], "supersample": 2,
        "view": {"xmin": -1.3, "xmax": 1.3, "ymin": -1.3, "ymax": 1.3},
        "layers": [
            {"type": "parametric", "t": [0, 125.66, 8000],
             "params": {"R": 0.7, "r": 0.23, "d": 0.55},
             "x": "(R-r)*cos(t) + d*cos(((R-r)/r)*t + 0.4*time)",
             "y": "(R-r)*sin(t) - d*sin(((R-r)/r)*t + 0.4*time)",
             "color": [140, 255, 200], "width": 1},
        ],
    },
    "superformula": {
        "width": 720, "height": 720, "fps": 30, "duration": 12,
        "background": [10, 10, 18], "supersample": 2,
        "view": {"xmin": -1.4, "xmax": 1.4, "ymin": -1.4, "ymax": 1.4},
        "layers": [
            {"type": "parametric", "t": [0, 6.283185, 4000],
             "params": {"m": 7, "n1": 0.3, "n2": 1.7, "n3": 1.7, "a": 1, "b": 1},
             "x": "cos(t) * ((abs(cos(m*(t+0.2*time)/4)/a)**n2 + abs(sin(m*(t+0.2*time)/4)/b)**n3)**(-1/n1))",
             "y": "sin(t) * ((abs(cos(m*(t+0.2*time)/4)/a)**n2 + abs(sin(m*(t+0.2*time)/4)/b)**n3)**(-1/n1))",
             "color": [255, 210, 120], "width": 2},
        ],
    },
    "interference": {
        "width": 854, "height": 480, "fps": 30, "duration": 10,
        "supersample": 1,
        "view": {"xmin": -6, "xmax": 6, "ymin": -3.37, "ymax": 3.37},
        "layers": [
            {"type": "field",
             "f": "sin(4*hypot(x-2*cos(time), y-2*sin(time))) "
                  "+ sin(4*hypot(x+2*cos(time), y+2*sin(time)))",
             "min": -2, "max": 2, "cmap": "inferno"},
        ],
    },
    "waves": {
        "width": 854, "height": 480, "fps": 30, "duration": 10,
        "background": [6, 8, 16], "axes": True,
        "view": {"xmin": -6.283, "xmax": 6.283, "ymin": -2.5, "ymax": 2.5},
        "layers": [
            {"type": "explicit", "y": "sin(x - time) * exp(-0.05*x*x)",
             "color": [120, 220, 255], "width": 2},
            {"type": "explicit", "y": "0.6*sin(2*x + 1.5*time)",
             "color": [255, 140, 90], "width": 2},
            {"type": "explicit",
             "y": "sin(x - time)*exp(-0.05*x*x) + 0.6*sin(2*x + 1.5*time)",
             "color": [170, 255, 150], "width": 1},
        ],
    },
    "mandala": {
        "width": 720, "height": 720, "fps": 30, "duration": 14,
        "supersample": 1,
        "view": {"xmin": -3.14, "xmax": 3.14, "ymin": -3.14, "ymax": 3.14},
        "layers": [
            {"type": "field",
             "f": "sin(6*theta + 3*time) * sin(5*r - time) + cos(4*r + theta)",
             "min": -2, "max": 2, "cmap": "magma"},
        ],
    },
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_scene(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return Scene(spec)


def apply_overrides(scene, args):
    if args.width:
        scene.width = args.width
    if args.height:
        scene.height = args.height
    if args.fps:
        scene.fps = args.fps
    if args.duration:
        scene.duration = args.duration
    return scene


def cmd_render(args):
    scene = load_scene(args.scene)
    apply_overrides(scene, args)
    print(f"scene: {scene.width}x{scene.height} @ {scene.fps}fps, "
          f"{scene.duration}s, {len(scene.layers)} layer(s)")
    render_video(scene, args.out, crf=args.crf)
    print(f"done -> {args.out}")


def cmd_preview(args):
    scene = load_scene(args.scene)
    apply_overrides(scene, args)
    frame_idx = int(round(args.at * scene.fps))
    img = scene.render_frame(frame_idx)
    img.save(args.out)
    print(f"preview frame at t={args.at}s -> {args.out}")


def cmd_demo(args):
    if args.name not in DEMOS:
        sys.exit(f"unknown demo {args.name!r}. options: {', '.join(DEMOS)}")
    scene = Scene(DEMOS[args.name])
    apply_overrides(scene, args)
    print(f"demo '{args.name}': {scene.width}x{scene.height} @ {scene.fps}fps, "
          f"{scene.duration}s")
    render_video(scene, args.out, crf=args.crf)
    print(f"done -> {args.out}")


def cmd_demos(args):
    print("built-in demos:")
    for k, v in DEMOS.items():
        types = ", ".join(sorted({l.get("type", "parametric") for l in v["layers"]}))
        print(f"  {k:14s} [{types}]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="geovid",
                                description="Generate videos from geometric equations.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--width", type=int)
    common.add_argument("--height", type=int)
    common.add_argument("--fps", type=int)
    common.add_argument("--duration", type=float)

    r = sub.add_parser("render", parents=[common], help="render a scene JSON to mp4")
    r.add_argument("scene")
    r.add_argument("-o", "--out", default="out.mp4")
    r.add_argument("--crf", type=int, default=20, help="x264 quality (lower=better)")
    r.set_defaults(func=cmd_render)

    pv = sub.add_parser("preview", parents=[common], help="render a single frame to PNG")
    pv.add_argument("scene")
    pv.add_argument("-o", "--out", default="frame.png")
    pv.add_argument("--at", type=float, default=0.0, help="time in seconds")
    pv.set_defaults(func=cmd_preview)

    d = sub.add_parser("demo", parents=[common], help="render a built-in demo scene")
    d.add_argument("name")
    d.add_argument("-o", "--out", default="demo.mp4")
    d.add_argument("--crf", type=int, default=20)
    d.set_defaults(func=cmd_demo)

    sub.add_parser("demos", help="list built-in demos").set_defaults(func=cmd_demos)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
