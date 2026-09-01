"""
render.py - one place that turns a validated grammar spec into a video file.

Both the web app and the CLI use this so the stickman special-case lives in a
single spot. Most archetypes render as a single geovid scene animated by `time`;
`stickman` is special: it is drawn per-frame from a fresh equation set, so it
goes through stickman.render_from_spec instead.

Returns a dict of extras (e.g. equation-list file paths for stickman); empty for
ordinary archetypes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import geovid          # noqa: E402
import grammar as G    # noqa: E402
import stickman        # noqa: E402

# When EQV_NEURAL=1, stickman frames are produced by the trained PoseNet
# (two-stage pipeline) instead of the hand-written kinematics.
_NEURAL = os.environ.get("EQV_NEURAL") == "1"
# When EQV_REALMOTION=1, actions the RealMotionNet learned from real video
# (e.g. walk, dance) are generated from that learned motion instead.
_REALMOTION = os.environ.get("EQV_REALMOTION") == "1"
_PIPE = None
_RM_ACTIONS = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        import pipeline
        _PIPE = pipeline.Pipeline()
    return _PIPE


# EQV_MOTION selects the real-motion backend: "ar" (autoregressive, default) or
# "phase" (RealMotionNet phase->pose).
_MOTION_BACKEND = os.environ.get("EQV_MOTION", "ar")


def _motion_mod():
    import motion_ar, motion_model
    return motion_ar if _MOTION_BACKEND == "ar" else motion_model


def _realmotion_actions():
    global _RM_ACTIONS
    if _RM_ACTIONS is None:
        try:
            _RM_ACTIONS = set(map(str, _motion_mod().load()[1]))
        except Exception:
            _RM_ACTIONS = set()
    return _RM_ACTIONS


def render_spec(spec: dict, out_path, fps=30, duration=6.0,
                width=854, height=480, verbose=False) -> dict:
    spec = G.validate_spec(spec)
    if spec.get("arch") in G.SPECIAL_ARCH:      # currently: stickman
        action = spec.get("action", "walk")
        if _REALMOTION and action in _realmotion_actions():
            color = G.COLORS.get(spec.get("color", "blue"), (120, 220, 255))
            return _motion_mod().generate(action, out_path, seconds=duration,
                                          width=width, height=height,
                                          color=color, verbose=verbose)
        if _NEURAL:
            return _pipeline().render_from_spec(
                spec, out_path, fps=fps, duration=duration,
                width=width, height=height, verbose=verbose)
        return stickman.render_from_spec(
            spec, out_path, fps=fps, duration=duration,
            width=width, height=height, verbose=verbose)

    scene_spec = G.build_scene(spec)
    scene_spec["duration"] = duration
    if width:
        scene_spec["width"] = width
    if height:
        scene_spec["height"] = height
    scene = geovid.Scene(scene_spec)
    geovid.render_video(scene, out_path, verbose=verbose)
    return {}
