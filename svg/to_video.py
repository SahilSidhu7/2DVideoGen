"""AniSVG clips -> a real video file.

The representation is already frame-based: a clip declares its cast once, then
each frame carries only the deltas. Replaying it yields one SVG per frame, so
producing video is rasterise-then-encode.

Chaining matters as much as encoding. One clip is bounded by the model's
context - about 10 frames at the 1024 cap the 8B run used - so anything longer
than a second comes from playing clips in sequence. `--hold` repeats the last
frame of each clip so a cut does not land mid-gesture, and `--loops` replays
each clip, which is how a 0.7 s cycle becomes a few seconds of motion.

    python svg/to_video.py svg/out/gen_con2/*.anisvg -o reel.mp4
    python svg/to_video.py clip.anisvg -o clip.mp4 --loops 4 --width 512
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anisvg import Anim                          # noqa: E402
from generate import extract                     # noqa: E402
import raster                                    # noqa: E402


def clip_frames(text, width, loops=1, hold=0):
    anim = Anim.from_text(extract(text))
    base = [raster.render(s, width=width) for s in anim.to_svgs()]
    if not base:
        return [], 15
    out = []
    for _ in range(max(1, loops)):
        out.extend(base)
        out.extend([base[-1]] * hold)
    return out, anim.fps or 15


def encode(frames, out, fps):
    """Write frames to a video. Falls back to GIF when ffmpeg is absent."""
    if not frames:
        raise SystemExit("no frames to encode")
    if shutil.which("ffmpeg") is None or out.lower().endswith(".gif"):
        gif = os.path.splitext(out)[0] + ".gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        return gif
    tmp = tempfile.mkdtemp(prefix="anisvg_")
    try:
        for i, im in enumerate(frames):
            # yuv420p needs even dimensions, and most players need yuv420p.
            w, h = im.size
            if w % 2 or h % 2:
                im = im.resize((w - w % 2, h - h % 2))
            im.save(os.path.join(tmp, "f%05d.png" % i))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(tmp, "f%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out]
        subprocess.run(cmd, check=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help=".anisvg files (globs allowed)")
    ap.add_argument("-o", "--out", default="svg/out/demo/anisvg.mp4")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--loops", type=int, default=1, help="replay each clip N times")
    ap.add_argument("--hold", type=int, default=0,
                    help="repeat each clip's last frame N times before the cut")
    ap.add_argument("--fps", type=int, default=0, help="override the clip fps")
    args = ap.parse_args()

    paths = []
    for pat in args.clips:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    frames, fps = [], args.fps or 15
    for path in paths:
        try:
            got, cfps = clip_frames(open(path, encoding="utf-8").read(),
                                    args.width, args.loops, args.hold)
        except Exception as exc:
            print("skip %-28s %s" % (os.path.basename(path), str(exc)[:60]))
            continue
        if not got:
            continue
        frames.extend(got)
        fps = args.fps or cfps
        print("%-28s %3d frames" % (os.path.basename(path), len(got)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = encode(frames, args.out, fps)
    print("\n%d frames at %d fps = %.1fs -> %s (%.1f KB)"
          % (len(frames), fps, len(frames) / float(fps), written,
             os.path.getsize(written) / 1024))


if __name__ == "__main__":
    main()
