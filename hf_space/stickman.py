#!/usr/bin/env python3
"""
stickman.py - animate a stick figure by emitting a *fresh set of equations for
every frame*, now with multiple actions.

Nothing is animated by a `time` variable. For each frame an action-specific
kinematics function computes concrete joint positions, and every body part
(plus props like a ball) is written as its own geometric equation with the
numbers baked in for that frame:

    limb / prop segment ->  x(t) = x0 + dx*t , y(t) = y0 + dy*t , t in [0,1]
    head / ball         ->  x(t) = cx + r*cos(t) , y(t) = cy + r*sin(t)

Actions: walk, run, jump, kick (football, with a ball), wave, dance.

Outputs
    <out>.mp4
    <out>.equations.jsonl  one JSON per frame: every equation used that frame
    <out>.equations.txt    human-readable sample

Usage
    python stickman.py -o walk.mp4 --action walk
    python stickman.py -o kick.mp4 --action kick --seconds 5
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import geovid

D = math.radians  # degrees -> radians shorthand


def _smooth(a, b, x):
    """smoothstep from a->b over x in [0,1]."""
    t = max(0.0, min(1.0, x))
    return a + (b - a) * (t * t * (3 - 2 * t))


# --------------------------------------------------------------------------- #
# Skeleton
# --------------------------------------------------------------------------- #

class StickFigure:
    def __init__(self, width=854, height=480, ground=1.15, steps=6.0):
        self.H = 5.62
        self.W = (width / height) * self.H if height else 10.0
        self.ground = ground
        self.steps = steps
        self.L_thigh, self.L_shin = 0.62, 0.60
        self.L_uarm, self.L_farm = 0.46, 0.42
        self.torso, self.head_r = 0.85, 0.22
        self.leg = self.L_thigh + self.L_shin

    @staticmethod
    def _pt(o, length, angle):
        """Point `length` from o at `angle` (0 = straight down, +ccw)."""
        return (o[0] + length * math.sin(angle), o[1] - length * math.cos(angle))

    def skeleton(self, hip, a):
        """Assemble joints from a hip point + a dict of joint angles (rad)."""
        lean = a.get("lean", D(8))
        shoulder = (hip[0] + self.torso * math.sin(lean),
                    hip[1] + self.torso * math.cos(lean))
        head_c = (shoulder[0] + (self.head_r + 0.03) * math.sin(lean),
                  shoulder[1] + (self.head_r + 0.03) * math.cos(lean))
        knee_R = self._pt(hip, self.L_thigh, a["thigh_R"])
        foot_R = self._pt(knee_R, self.L_shin, a["thigh_R"] + a["knee_R"])
        knee_L = self._pt(hip, self.L_thigh, a["thigh_L"])
        foot_L = self._pt(knee_L, self.L_shin, a["thigh_L"] + a["knee_L"])
        elbow_R = self._pt(shoulder, self.L_uarm, a["uarm_R"])
        hand_R = self._pt(elbow_R, self.L_farm, a["uarm_R"] + a["elbow_R"])
        elbow_L = self._pt(shoulder, self.L_uarm, a["uarm_L"])
        hand_L = self._pt(elbow_L, self.L_farm, a["uarm_L"] + a["elbow_L"])
        return {"hip": hip, "shoulder": shoulder, "head_c": head_c,
                "knee_R": knee_R, "foot_R": foot_R,
                "knee_L": knee_L, "foot_L": foot_L,
                "elbow_R": elbow_R, "hand_R": hand_R,
                "elbow_L": elbow_L, "hand_L": hand_L}

    # ---- actions: progress in [0,1] -> (joints, props) ------------------- #

    def _raw(self, action, progress):
        fn = getattr(self, f"_act_{action}", self._act_walk)
        return fn(progress)                       # -> (hip, angle_dict, props)

    def pose(self, action, progress):
        hip, a, props = self._raw(action, progress)
        return self.skeleton(hip, a), props

    # ---- low-dim pose vector (what the learned Stage-B model predicts) ---- #
    # layout: [hip_x/W, hip_y/H, lean, thigh_R, knee_R, thigh_L, knee_L,
    #          uarm_R, elbow_R, uarm_L, elbow_L, ball_x/W, ball_y/H, ball_flag]
    #  angles stored in radians / pi (so ~[-1,1]).
    ANGLE_KEYS = ["lean", "thigh_R", "knee_R", "thigh_L", "knee_L",
                  "uarm_R", "elbow_R", "uarm_L", "elbow_L"]
    VEC_LEN = 14

    def pose_vector(self, action, progress):
        hip, a, props = self._raw(action, progress)
        v = [hip[0] / self.W, hip[1] / self.H]
        v += [a[k] / math.pi for k in self.ANGLE_KEYS]
        if props:
            b = props[0]["center"]
            v += [b[0] / self.W, b[1] / self.H, 1.0]
        else:
            v += [0.0, 0.0, 0.0]
        return v

    def equations_from_vector(self, v):
        """Inverse of pose_vector: rebuild the frame's equations from a vector."""
        hip = (v[0] * self.W, v[1] * self.H)
        a = {k: v[2 + i] * math.pi for i, k in enumerate(self.ANGLE_KEYS)}
        joints = self.skeleton(hip, a)
        props = []
        if v[13] > 0.5:
            props = [{"kind": "circle", "part": "ball",
                      "center": (v[11] * self.W, v[12] * self.H), "r": 0.17,
                      "color": [255, 150, 40]}]
        return frame_equations(joints, props)

    def _act_walk(self, p):
        phi = 2 * math.pi * self.steps * p
        hip = (0.9 + (self.W - 1.8) * p,
               self.ground + self.leg + 0.05 * math.cos(2 * phi))
        a = {"lean": D(8),
             "thigh_R": D(30) * math.sin(phi),
             "thigh_L": D(30) * math.sin(phi + math.pi),
             "knee_R": max(0, D(55) * math.sin(phi + math.pi / 2)),
             "knee_L": max(0, D(55) * math.sin(phi + 3 * math.pi / 2)),
             "uarm_R": D(26) * math.sin(phi + math.pi),
             "uarm_L": D(26) * math.sin(phi),
             "elbow_R": D(22), "elbow_L": D(22)}
        return hip, a, []

    def _act_run(self, p):
        phi = 2 * math.pi * (self.steps * 1.4) * p
        hip = (0.9 + (self.W - 1.8) * p,
               self.ground + self.leg - 0.12 + 0.09 * math.cos(2 * phi))
        a = {"lean": D(24),
             "thigh_R": D(48) * math.sin(phi),
             "thigh_L": D(48) * math.sin(phi + math.pi),
             "knee_R": max(0, D(95) * (0.5 + 0.5 * math.sin(phi + 1.2))),
             "knee_L": max(0, D(95) * (0.5 + 0.5 * math.sin(phi + 1.2 + math.pi))),
             "uarm_R": D(45) * math.sin(phi + math.pi),
             "uarm_L": D(45) * math.sin(phi),
             "elbow_R": D(75), "elbow_L": D(75)}
        return hip, a, []

    def _act_jump(self, p):
        hops = 4.0
        frac = (p * hops) % 1.0
        h = 0.7 * max(0.0, math.sin(math.pi * frac))       # height off ground
        hip = (0.9 + (self.W - 1.8) * p, self.ground + self.leg + h)
        tuck = D(70) * (h / 0.7)                            # tuck legs at apex
        a = {"lean": D(6),
             "thigh_R": D(18) * (h / 0.7), "thigh_L": D(18) * (h / 0.7),
             "knee_R": tuck, "knee_L": tuck,
             "uarm_R": D(150) - D(40) * (1 - h / 0.7),      # arms swing up
             "uarm_L": D(150) - D(40) * (1 - h / 0.7),
             "elbow_R": D(20), "elbow_L": D(20)}
        return hip, a, []

    def _act_kick(self, p):
        """Approach a ball, plant, and kick it flying (football)."""
        approach = _smooth(0.0, 1.0, p / 0.55)
        stand_x = 1.4 + 1.8 * approach
        # subtle run-in bob while approaching
        phi = 2 * math.pi * 3 * min(p, 0.55)
        hip = (stand_x, self.ground + self.leg + 0.04 * math.cos(2 * phi))

        # right leg = kicking leg: wind back then swing through around p=0.6
        if p < 0.5:
            thigh_R = D(20) * math.sin(phi)                # jogging
            knee_R = max(0, D(50) * math.sin(phi + 1.4))
        else:
            sw = _smooth(0.0, 1.0, (p - 0.5) / 0.2)        # windup->follow
            thigh_R = D(-35) + D(105) * sw                 # behind -> forward(+x)
            knee_R = D(60) * (1 - sw)                      # extend at contact
        thigh_L, knee_L = D(-8), D(12)                     # planted support leg

        a = {"lean": D(14),
             "thigh_R": thigh_R, "knee_R": knee_R,
             "thigh_L": thigh_L, "knee_L": knee_L,
             "uarm_R": D(30), "uarm_L": D(-35),            # arms balance
             "elbow_R": D(30), "elbow_L": D(30)}

        # ball: rests ahead of the plant foot, launches after contact (p~0.7)
        r = 0.17
        rest = (stand_x + 0.72, self.ground + r)
        contact = 0.7
        if p < contact:
            ball = rest
        else:
            f = (p - contact) / (1 - contact)
            ball = (rest[0] + (self.W - 0.6 - rest[0]) * f,
                    self.ground + r + 1.7 * math.sin(math.pi * min(1, f)))
        props = [{"kind": "circle", "part": "ball", "center": ball, "r": r,
                  "color": [255, 150, 40]}]
        return hip, a, props

    def _act_wave(self, p):
        cx = self.W / 2
        sway = D(4) * math.sin(2 * math.pi * 1.5 * p)
        hip = (cx, self.ground + self.leg)
        wob = 2 * math.pi * 4 * p                          # fast hand wave
        a = {"lean": sway,
             "thigh_R": D(10), "thigh_L": D(-10),
             "knee_R": D(6), "knee_L": D(6),
             "uarm_R": D(150) + D(22) * math.sin(wob),     # raised, waving
             "uarm_L": D(-12),
             "elbow_R": D(18) + D(18) * math.sin(wob),
             "elbow_L": D(10)}
        return hip, a, []

    def _act_idle(self, p):
        """Standing rest with a slow breath. Added for scenescript.py: a cast
        member with no scripted action still has to be *somewhere*, and a
        frozen figure reads as a rendering bug rather than as a character."""
        breath = 2 * math.pi * 0.6 * p
        hip = (self.W / 2, self.ground + self.leg + 0.015 * math.sin(breath))
        a = {"lean": D(5) + D(1.5) * math.sin(breath),
             "thigh_R": D(7), "thigh_L": D(-7),
             "knee_R": D(4), "knee_L": D(4),
             "uarm_R": D(9) + D(2) * math.sin(breath),
             "uarm_L": D(-9) - D(2) * math.sin(breath),
             "elbow_R": D(14), "elbow_L": D(14)}
        return hip, a, []

    def _act_dance(self, p):
        cx = self.W / 2
        beat = 2 * math.pi * 2.0 * p
        hip = (cx + 0.18 * math.sin(beat),
               self.ground + self.leg - 0.06 + 0.06 * abs(math.sin(beat)))
        a = {"lean": D(10) * math.sin(beat),
             "thigh_R": D(18) * math.sin(beat), "thigh_L": D(-18) * math.sin(beat),
             "knee_R": max(0, D(35) * math.sin(beat)),
             "knee_L": max(0, D(35) * math.sin(beat + math.pi)),
             "uarm_R": D(120) * (0.5 + 0.5 * math.sin(beat)),
             "uarm_L": D(120) * (0.5 + 0.5 * math.sin(beat + math.pi)),
             "elbow_R": D(40), "elbow_L": D(40)}
        return hip, a, []


# --------------------------------------------------------------------------- #
# Joints -> per-frame equations
# --------------------------------------------------------------------------- #

def _seg_eq(name, p0, p1, color=None):
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    e = {"part": name, "kind": "segment",
         "x": f"{x0:.3f} + ({dx:.3f})*t", "y": f"{y0:.3f} + ({dy:.3f})*t",
         "t": [0, 1, 2], "from": [round(x0, 3), round(y0, 3)],
         "to": [round(p1[0], 3), round(p1[1], 3)]}
    if color:
        e["color"] = list(color)
    return e


def _circle_eq(name, c, r, color=None):
    e = {"part": name, "kind": "circle",
         "x": f"{c[0]:.3f} + {r:.3f}*cos(t)", "y": f"{c[1]:.3f} + {r:.3f}*sin(t)",
         "t": [0, 6.28319, 48], "center": [round(c[0], 3), round(c[1], 3)],
         "r": round(r, 3)}
    if color:
        e["color"] = list(color)
    return e


def frame_equations(j, props=None, head_r=0.22):
    eqs = [
        _circle_eq("head", j["head_c"], head_r),
        _seg_eq("spine", j["hip"], j["shoulder"]),
        _seg_eq("thigh_R", j["hip"], j["knee_R"]),
        _seg_eq("shin_R", j["knee_R"], j["foot_R"]),
        _seg_eq("thigh_L", j["hip"], j["knee_L"]),
        _seg_eq("shin_L", j["knee_L"], j["foot_L"]),
        _seg_eq("uarm_R", j["shoulder"], j["elbow_R"]),
        _seg_eq("farm_R", j["elbow_R"], j["hand_R"]),
        _seg_eq("uarm_L", j["shoulder"], j["elbow_L"]),
        _seg_eq("farm_L", j["elbow_L"], j["hand_L"]),
    ]
    for pr in (props or []):
        eqs.append(_circle_eq(pr["part"], pr["center"], pr["r"],
                              color=pr.get("color")))
    return eqs


def equations_to_layers(eqs, color=(120, 220, 255), width=3):
    layers = []
    for e in eqs:
        layers.append({"type": "parametric", "x": e["x"], "y": e["y"],
                       "t": e["t"], "color": list(e.get("color", color)),
                       "width": width})
    return layers


# --------------------------------------------------------------------------- #
# Scene / render
# --------------------------------------------------------------------------- #

def build_scene_spec(fig, eqs, w, h, color=(120, 220, 255), bg=(10, 12, 20)):
    layers = equations_to_layers(eqs, color=color)
    layers.insert(0, {"type": "parametric",
                      "x": f"0 + ({fig.W:.3f})*t", "y": f"{fig.ground:.3f} + (0)*t",
                      "t": [0, 1, 2], "color": [70, 78, 96], "width": 2})
    return {"width": w, "height": h, "fps": 30, "duration": 1,
            "background": list(bg), "supersample": 2,
            "view": {"xmin": 0, "xmax": fig.W, "ymin": 0, "ymax": fig.H},
            "layers": layers}


def static_scene(progress, color, bg, width, height, steps=6.0, action="walk"):
    fig = StickFigure(width, height, steps=steps)
    j, props = fig.pose(action, progress)
    return build_scene_spec(fig, frame_equations(j, props), width, height,
                            color=color, bg=bg)


def render(out, fps, seconds, width, height, sample,
           color=(120, 220, 255), bg=(10, 12, 20), steps=6.0, action="walk",
           verbose=True):
    fig = StickFigure(width, height, steps=steps)
    n = max(1, int(round(fps * seconds)))
    out = Path(out)
    jsonl = out.with_suffix(".equations.jsonl")
    txt = out.with_suffix(".equations.txt")

    ffmpeg = geovid.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sample_idx = set(range(0, n, max(1, n // sample))) if sample else set()
    with jsonl.open("w", encoding="utf-8") as jf, \
         txt.open("w", encoding="utf-8") as tf:
        tf.write(f"# Stickman '{action}' - {n} frames, a distinct equation set "
                 f"per frame\n# view: x[0,{fig.W:.2f}] y[0,{fig.H:.2f}] "
                 f"ground y={fig.ground}\n\n")
        for i in range(n):
            progress = i / (n - 1) if n > 1 else 0.0
            j, props = fig.pose(action, progress)
            eqs = frame_equations(j, props)
            jf.write(json.dumps({"frame": i, "action": action,
                                 "progress": round(progress, 4),
                                 "equations": eqs}) + "\n")
            if i in sample_idx or sample == -1:
                tf.write(f"--- frame {i}  (progress {progress:.3f}) ---\n")
                for e in eqs:
                    tf.write(f"  {e['part']:8s}: x = {e['x']:<28s}  y = {e['y']}\n")
                tf.write("\n")
            scene = geovid.Scene(build_scene_spec(fig, eqs, width, height, color, bg))
            proc.stdin.write(scene.render_frame(0).tobytes())
            if verbose and (i % fps == 0 or i == n - 1):
                print(f"\r  frame {i+1}/{n}", end="", flush=True)
    if verbose:
        print(f"\nvideo     -> {out}\nequations -> {jsonl}  ({n} frames)"
              f"\nreadable  -> {txt}")
    proc.stdin.close()
    proc.wait()
    return {"equations_jsonl": str(jsonl), "equations_txt": str(txt),
            "frames": n, "action": action}


def render_from_spec(spec, out, fps=30, duration=6.0, width=854, height=480,
                     verbose=False):
    import grammar as G
    color = G.COLORS.get(spec.get("color", "blue"), (120, 220, 255))
    bg = G.MOOD_BG.get(spec.get("mood", "dark"), (10, 12, 20))
    steps = {"slow": 5.0, "med": 6.0, "fast": 8.0}.get(spec.get("speed"), 6.0)
    action = spec.get("action", "walk")
    return render(out, fps, duration, width, height, sample=4, color=color,
                  bg=bg, steps=steps, action=action, verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description="Animated stickman via per-frame equations.")
    ap.add_argument("-o", "--out", default="walk.mp4")
    ap.add_argument("--action", default="walk",
                    choices=["walk", "run", "jump", "kick", "wave", "dance", "idle"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--sample", type=int, default=4)
    args = ap.parse_args()
    render(args.out, args.fps, args.seconds, args.width, args.height,
           args.sample, action=args.action)


if __name__ == "__main__":
    main()
