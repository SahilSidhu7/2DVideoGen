"""
pipeline.py - the two-stage learned generator.

    prompt
      -> [Stage A: planner]  scene plan + per-frame descriptors
      -> [Stage B: PoseNet]  per-frame pose vector -> equations
      -> render

Stage A is the prompt understanding model (PromptModel: rules + t5-small) which
decides subject/action/style and lays out the frames. Stage B is the trained
PoseNet: for every frame it outputs the pose vector, which is turned into that
frame's equations. So the figure you see is drawn entirely from equations the
*model* produced, not from the hand-written kinematics.

Non-stickman prompts still route to the math-archetype engine (diversity).

    python pipeline.py "a stickman playing football" -o out/nn_kick.mp4
    python pipeline.py "a green stickman running" -o out/nn_run.mp4 --seconds 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import geovid          # noqa: E402
import grammar as G    # noqa: E402
import stickman as S   # noqa: E402
import render as R     # noqa: E402
import frames_dataset as FD      # noqa: E402
from infer import PromptModel    # noqa: E402
import pose_model as PM          # noqa: E402


class Pipeline:
    def __init__(self, ckpt=str(HERE / "checkpoint")):
        self.planner = PromptModel(ckpt)          # Stage A
        self.posenet = PM.load_posenet()          # Stage B

    # ---- Stage A: prompt -> plan --------------------------------------- #
    def plan(self, prompt, fps, duration):
        spec = self.planner.spec(prompt)
        n = max(1, int(round(fps * duration)))
        frames = None
        if spec.get("arch") == "stickman":
            act = spec.get("action", "walk")
            ai = FD.ACTIONS.index(act) if act in FD.ACTIONS else 0
            frames = [{"frame": i, "action": act, "action_idx": ai,
                       "progress": (i / (n - 1) if n > 1 else 0.0)}
                      for i in range(n)]
        return spec, frames, n

    # ---- Stage B: descriptor -> pose vector -> equations --------------- #
    def render(self, prompt, out, fps=30, duration=6.0, width=854, height=480,
               verbose=True):
        spec, frames, n = self.plan(prompt, fps, duration)
        return self._render(spec, frames, n, out, fps, duration,
                            width, height, verbose)

    def render_from_spec(self, spec, out, fps=30, duration=6.0,
                         width=854, height=480, verbose=False):
        """Render an already-planned spec (used by the web app)."""
        spec = G.validate_spec(spec)
        n = max(1, int(round(fps * duration)))
        frames = None
        if spec.get("arch") == "stickman":
            act = spec.get("action", "walk")
            ai = FD.ACTIONS.index(act) if act in FD.ACTIONS else 0
            frames = [{"frame": i, "action": act, "action_idx": ai,
                       "progress": (i / (n - 1) if n > 1 else 0.0)}
                      for i in range(n)]
        return self._render(spec, frames, n, out, fps, duration,
                            width, height, verbose)

    def _render(self, spec, frames, n, out, fps, duration, width, height, verbose):
        dsl = G.spec_to_dsl(spec)
        if frames is None:      # not a stickman -> math-archetype engine
            if verbose:
                print(f"plan: {dsl}  (math archetype, engine render)")
            R.render_spec(spec, out, fps=fps, duration=duration,
                          width=width, height=height, verbose=verbose)
            return {"spec": dsl, "path": "engine"}

        color = G.COLORS.get(spec.get("color", "blue"), (120, 220, 255))
        bg = G.MOOD_BG.get(spec.get("mood", "dark"), (10, 12, 20))
        fig = S.StickFigure(width, height)
        out = Path(out)
        jsonl = out.with_suffix(".equations.jsonl")
        if verbose:
            print(f"plan: {dsl}  ({n} frames via PoseNet)")

        ffmpeg = geovid.find_ffmpeg()
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with jsonl.open("w", encoding="utf-8") as jf:
            for fr in frames:
                vec = PM.predict_vec(self.posenet, fr["action_idx"], fr["progress"])
                eqs = fig.equations_from_vector([float(x) for x in vec])
                jf.write(json.dumps({"frame": fr["frame"], "action": fr["action"],
                                     "progress": round(fr["progress"], 4),
                                     "source": "posenet", "equations": eqs}) + "\n")
                scene = geovid.Scene(
                    S.build_scene_spec(fig, eqs, width, height, color, bg))
                proc.stdin.write(scene.render_frame(0).tobytes())
                if verbose and fr["frame"] % fps == 0:
                    print(f"\r  frame {fr['frame']+1}/{n}", end="", flush=True)
        if verbose:
            print(f"\nvideo -> {out}\nequations (model-produced) -> {jsonl}")
        proc.stdin.close(); proc.wait()
        return {"spec": dsl, "path": "posenet", "equations_jsonl": str(jsonl),
                "frames": n}


def image_to_equations(png_path, out_png, size=80):
    """Demo: read a frame image -> ImgPoseNet -> equations -> re-render it."""
    from PIL import Image
    im = PM.load_imgposenet()
    import torch
    arr = np.asarray(Image.open(png_path).convert("L").resize((size, size)),
                     np.float32) / 255.0
    with torch.no_grad():
        vec = im(torch.from_numpy(arr[None, None]))[0].numpy()
    fig = S.StickFigure(size, size)
    eqs = fig.equations_from_vector(vec)
    geovid.Scene(S.build_scene_spec(fig, eqs, 480, 480,
                 color=(120, 220, 255), bg=(10, 12, 20))).render_frame(0).save(out_png)
    print(f"image->equations: {png_path} -> {out_png}  ({len(eqs)} equations)")
    return eqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default="out/nn.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    Pipeline().render(args.prompt, args.out, fps=args.fps,
                      duration=args.seconds, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
