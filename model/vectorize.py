"""
vectorize.py - turn a REAL image or video into our equation skeleton.

    real frame --[YOLO-pose, trained on real COCO images]--> 17 keypoints
               --[pose_map.coco_to_vector]--> 14-d pose vector
               --[StickFigure.equations_from_vector]--> equations --> render

This is the "anime images / videos translated to equations with tags" step,
done on real footage. It also dumps a dataset of real pose vectors (poses.npz)
that our own models can be trained/fine-tuned on.

    python vectorize.py image path/to/photo.jpg -o out/vec.png
    python vectorize.py video path/to/clip.mp4  -o out/vec.mp4
    python vectorize.py dataset videos_dir/ -o data/real_poses.npz
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
import stickman as S   # noqa: E402
import pose_map as PMAP  # noqa: E402

_YOLO = None
WEIGHTS = "yolo11n-pose.pt"     # nano pose model, auto-downloads on first use


def yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(WEIGHTS)
    return _YOLO


def _best_person(result):
    """Return (kpts(17,2), conf(17,)) for the highest-confidence person, or None."""
    kp = result.keypoints
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return None
    xy = kp.xy.cpu().numpy()            # (n,17,2)
    cf = kp.conf.cpu().numpy() if kp.conf is not None else np.ones(xy.shape[:2])
    best = int(cf.mean(axis=1).argmax())
    return xy[best], cf[best]


def _fig_and_vec(kpts, conf, width, height):
    fig = S.StickFigure(width, height)
    vec = PMAP.coco_to_vector(kpts, conf, ground=fig.ground, view_w=fig.W,
                              view_h=fig.H, leg=fig.leg, torso=fig.torso)
    return fig, vec


def vectorize_image(path, out_png, width=854, height=480, color=(120, 220, 255)):
    res = yolo()(path, verbose=False)[0]
    got = _best_person(res)
    if not got:
        raise SystemExit("no person detected")
    fig, vec = _fig_and_vec(got[0], got[1], width, height)
    eqs = fig.equations_from_vector(vec)
    geovid.Scene(S.build_scene_spec(fig, eqs, width, height, color,
                 (10, 12, 20))).render_frame(0).save(out_png)
    print(f"{path} -> {out_png}  ({len(eqs)} equations)")
    return vec, eqs


def vectorize_image_to_video(path, out_mp4, seconds=3, fps=30, width=854,
                             height=480, color=(120, 220, 255)):
    """Vectorize a still image and write it as a short static clip (for the app)."""
    res = yolo()(path, verbose=False)[0]
    got = _best_person(res)
    if not got:
        raise SystemExit("no person detected")
    fig, vec = _fig_and_vec(got[0], got[1], width, height)
    eqs = fig.equations_from_vector([float(x) for x in vec])
    out = Path(out_mp4)
    jsonl = out.with_suffix(".equations.jsonl")
    frame = geovid.Scene(S.build_scene_spec(fig, eqs, width, height, color,
                         (10, 12, 20))).render_frame(0).tobytes()
    n = max(1, int(seconds * fps))
    ffmpeg = geovid.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(n):
        proc.stdin.write(frame)
    proc.stdin.close(); proc.wait()
    import json as _json
    jsonl.write_text(_json.dumps({"frame": 0, "source": "yolo-pose",
                                  "equations": eqs}), encoding="utf-8")
    return {"equations_jsonl": str(jsonl), "frames": n, "n_equations": len(eqs)}


def vectorize_video(path, out_mp4, fps=30, width=854, height=480,
                    color=(120, 220, 255), smooth=0.4):
    out = Path(out_mp4)
    jsonl = out.with_suffix(".equations.jsonl")
    ffmpeg = geovid.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fig = S.StickFigure(width, height)
    prev = None
    vecs = []
    i = 0
    with jsonl.open("w", encoding="utf-8") as jf:
        for res in yolo()(path, stream=True, verbose=False):
            got = _best_person(res)
            if not got:
                if prev is None:
                    continue
                vec = prev
            else:
                _, vec = _fig_and_vec(got[0], got[1], width, height)
                if prev is not None:      # temporal smoothing to reduce jitter
                    vec = [prev[k] + (1 - smooth) * (vec[k] - prev[k])
                           for k in range(len(vec))]
                prev = vec
            vecs.append(vec)
            eqs = fig.equations_from_vector([float(x) for x in vec])
            jf.write(json.dumps({"frame": i, "source": "yolo-pose",
                                 "equations": eqs}) + "\n")
            proc.stdin.write(geovid.Scene(
                S.build_scene_spec(fig, eqs, width, height, color,
                                   (10, 12, 20))).render_frame(0).tobytes())
            i += 1
            if i % 30 == 0:
                print(f"\r  frame {i}", end="", flush=True)
    proc.stdin.close(); proc.wait()
    print(f"\nvideo -> {out}\nequations -> {jsonl}  ({i} frames)")
    return np.array(vecs, np.float32)


def build_dataset(src_dir, out_npz, width=854, height=480):
    """Extract real pose vectors from every image/video in a folder."""
    src = Path(src_dir)
    files = [p for p in src.rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".mov",
                                     ".avi", ".webm", ".mkv")]
    all_vecs = []
    for p in files:
        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
            for res in yolo()(str(p), stream=True, verbose=False):
                got = _best_person(res)
                if got:
                    _, v = _fig_and_vec(got[0], got[1], width, height)
                    all_vecs.append(v)
        else:
            res = yolo()(str(p), verbose=False)[0]
            got = _best_person(res)
            if got:
                _, v = _fig_and_vec(got[0], got[1], width, height)
                all_vecs.append(v)
    arr = np.array(all_vecs, np.float32)
    Path(out_npz).parent.mkdir(exist_ok=True)
    np.savez_compressed(out_npz, vecs=arr)
    print(f"real poses -> {out_npz}  {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    im = sub.add_parser("image"); im.add_argument("path"); im.add_argument("-o", "--out", default="out/vec.png")
    vi = sub.add_parser("video"); vi.add_argument("path"); vi.add_argument("-o", "--out", default="out/vec.mp4"); vi.add_argument("--fps", type=int, default=30)
    ds = sub.add_parser("dataset"); ds.add_argument("src"); ds.add_argument("-o", "--out", default="data/real_poses.npz")
    args = ap.parse_args()
    if args.cmd == "image":
        vectorize_image(args.path, args.out)
    elif args.cmd == "video":
        vectorize_video(args.path, args.out, fps=args.fps)
    else:
        build_dataset(args.src, args.out)


if __name__ == "__main__":
    main()
