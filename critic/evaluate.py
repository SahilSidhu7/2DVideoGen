"""critic/evaluate.py — the objective judge for whether project output is a
real, usable animation video.

    USE_TF=0 python critic/evaluate.py <video-or-frame-dir> [options]

Emits <out>/report.json (raw metrics) and prints a human-readable summary plus
the verdict from verdict.py. See critic/README.md for what each metric means,
why its threshold is what it is, and the baseline numbers measured on this
project's own outputs.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics                          # noqa: E402
import verdict as V                     # noqa: E402
from frames_io import load_frames       # noqa: E402


def _load_yolo(weights):
    if not weights:
        return None
    from ultralytics import YOLO
    return YOLO(weights)


def _guard(use_guard):
    if not use_guard:
        return None
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.gpuguard import Guard
    return Guard()


def _parse_colors(spec):
    """'#00CCCC,FFA500,FF6EB4' -> [(0,204,204), (255,165,0), (255,110,180)]."""
    if not spec:
        return None
    out = []
    for tok in spec.split(","):
        h = tok.strip().lstrip("#")
        if len(h) != 6:
            raise SystemExit("bad --character-colors entry: %r (want 6 hex digits)" % tok)
        out.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return out


def evaluate(path, guide_path=None, pose_guide_dir=None, yolo_weights=None,
             max_frames=200, no_gpu_guard=False, identity_max=60, caption=None,
             clip_max=30, anisvg_path=None, schema=None, character_colors=None):
    frames, fps, _files = load_frames(path, max_frames=max_frames)
    if not frames:
        raise SystemExit("no frames loaded from %s" % path)

    guide_frames = None
    if guide_path:
        guide_frames, _gfps, _ = load_frames(guide_path, max_frames=max_frames)

    pose_files = None
    if pose_guide_dir:
        import glob
        pose_files = sorted(glob.glob(os.path.join(pose_guide_dir, "*.png")) or
                            glob.glob(os.path.join(pose_guide_dir, "*.jpg")))

    anisvg_text = None
    if anisvg_path:
        with open(anisvg_path, encoding="utf-8") as fh:
            anisvg_text = fh.read()

    yolo = _load_yolo(yolo_weights)
    guard = _guard(yolo is not None and not no_gpu_guard)

    t0 = time.time()
    report = dict(
        input=os.path.abspath(path),
        guide=os.path.abspath(guide_path) if guide_path else None,
        pose_guide_dir=os.path.abspath(pose_guide_dir) if pose_guide_dir else None,
        anisvg=os.path.abspath(anisvg_path) if anisvg_path else None,
        caption=caption,
        n_frames=len(frames),
        fps=fps,
        resolution=[int(frames[0].shape[1]), int(frames[0].shape[0])],
        metrics=dict(
            temporal_stability=metrics.temporal_stability(frames, guide_frames),
            identity_preservation=metrics.identity_preservation(
                frames, yolo, guard, max_frames=identity_max),
            motion_presence=metrics.motion_presence(frames),
            pose_fidelity=metrics.pose_fidelity(frames, pose_files, yolo, guard),
            scene_complexity=metrics.scene_complexity(frames, yolo, guard),
            content_correctness=metrics.content_correctness(
                frames, caption, max_frames=clip_max),
            structural_correctness=metrics.structural_correctness(
                anisvg_text, caption, schema=schema),
            scene_presence_audit=metrics.scene_presence_audit(
                frames, character_colors),
        ),
    )
    report["eval_seconds"] = round(time.time() - t0, 2)
    if guard is not None:
        report["gpu_guard"] = guard.report()
        guard.release()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="video file, gif, or directory of frame images")
    ap.add_argument("--guide", default=None,
                    help="matching input-guide video/gif/dir, for the flicker "
                         "amplification ratio")
    ap.add_argument("--pose-guides", default=None,
                    help="directory of exact pose-guide PNGs (svg/guides.py "
                         "pose/ output), for pose fidelity")
    ap.add_argument("--yolo", default="yolo11n-pose.pt",
                    help="pose weights; pass '' to skip identity-box/pose/"
                         "scene detection entirely (falls back to center-crop "
                         "identity only)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--identity-max", type=int, default=60,
                    help="cap on frames compared pairwise for identity "
                         "(O(n^2))")
    ap.add_argument("--no-gpu-guard", action="store_true",
                    help="skip tools.gpuguard duty-cycling (only for quick "
                         "local tests, not real runs on this hardware)")
    ap.add_argument("--caption", default=None,
                    help="the clip's prompt/caption, for CLIPSIM content "
                         "correctness. Omit to skip that metric (N/A, not "
                         "gated) — meaningless for a multi-clip reel where "
                         "no single caption describes the whole file")
    ap.add_argument("--clip-max", type=int, default=30,
                    help="frames sampled for CLIPSIM (CPU-only, keep modest)")
    ap.add_argument("--anisvg", default=None,
                    help="path to the raw .anisvg source text for this clip, "
                         "for structural correctness (metric 7). Applies "
                         "ONLY to the symbolic AniSVG path — omit for raster/"
                         "diffusion output, which has no source to parse")
    ap.add_argument("--schema", choices=["stick", "anime"], default=None,
                    help="declare the expected fixed-part schema (chars.py="
                         "stick, anime_chars.py=anime) for the per-part/"
                         "paint-order/cast-stability checks. NOT auto-"
                         "inferred from shape count — see metrics.py "
                         "structural_correctness() for why")
    ap.add_argument("--character-colors", default=None,
                    help="comma-separated hex colours, one per declared cast "
                         "member (e.g. '00CCCC,FFA500,FF6EB4'), for the "
                         "detector-free scene_presence_audit (metric 8) — "
                         "the honest replacement for scene_complexity on "
                         "this project's non-photographic art")
    ap.add_argument("-o", "--out", default=None,
                    help="output dir for report.json (default: alongside "
                         "input, or critic/out/<stem>)")
    args = ap.parse_args()

    report = evaluate(args.input, args.guide, args.pose_guides,
                      args.yolo or None, args.max_frames, args.no_gpu_guard,
                      args.identity_max, args.caption, args.clip_max,
                      args.anisvg, args.schema, _parse_colors(args.character_colors))

    stem = os.path.splitext(os.path.basename(args.input.rstrip("\\/")))[0]
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out", stem)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("frames=%d fps=%.1f res=%s eval=%.1fs -> %s" % (
        report["n_frames"], report["fps"], report["resolution"],
        report["eval_seconds"], report_path))
    print(json.dumps(report["metrics"], indent=2))
    print()
    print(V.format_report(args.input, report))


if __name__ == "__main__":
    main()
