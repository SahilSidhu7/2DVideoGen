#!/usr/bin/env python3
"""
prompt2video - turn a plain-English prompt into a math-equation video.

    python prompt2video.py "a slow glowing blue rose with 5 petals" -o rose.mp4
    python prompt2video.py "fast fire-colored water ripples" -o ripples.mp4
    python prompt2video.py "dark lissajous" --preview frame.png   # 1 frame, no encode

How it works
    prompt --[trained t5-small]--> DSL spec --[grammar]--> geovid scene --> mp4
If no trained checkpoint is present, a rule-based parser handles the prompt, so
it always works; the model just makes prompt understanding more flexible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "model"))

import geovid  # noqa: E402
import grammar as G  # noqa: E402
from infer import PromptModel  # noqa: E402

DEFAULT_CKPT = ROOT / "model" / "checkpoint"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="what to generate, in plain English")
    ap.add_argument("-o", "--out", default="out.mp4")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="trained model dir (falls back to rules if missing)")
    ap.add_argument("--preview", metavar="PNG",
                    help="render a single frame to PNG instead of a video")
    ap.add_argument("--at", type=float, default=1.0, help="preview time (s)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--show-spec", action="store_true",
                    help="print the parsed spec + scene JSON and exit")
    ap.add_argument("--rules-only", action="store_true",
                    help="skip the neural model, use the rule parser")
    args = ap.parse_args()

    ckpt = None if args.rules_only else args.ckpt
    model = PromptModel(ckpt)
    src = "t5-small model" if model.ok else "rule parser"
    spec, scene_spec = model.scene(args.prompt)

    if args.width:
        scene_spec["width"] = args.width
    if args.height:
        scene_spec["height"] = args.height
    if args.duration:
        scene_spec["duration"] = args.duration

    print(f"prompt : {args.prompt!r}")
    print(f"parsed : {G.spec_to_dsl(spec)}   [{src}]")

    if args.show_spec:
        print(json.dumps(scene_spec, indent=2))
        return

    scene = geovid.Scene(scene_spec)
    if args.preview:
        idx = int(round(args.at * scene.fps))
        scene.render_frame(idx).save(args.preview)
        print(f"preview -> {args.preview}")
    else:
        geovid.render_video(scene, args.out, crf=args.crf)
        print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
