"""Load frames from a video file, a GIF, or a directory of images.

Kept separate from metrics.py because every metric needs the same frame list
and nothing else in this module needs torch/ultralytics.
"""
import glob
import os

import numpy as np
from PIL import Image

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _load_video(path, max_frames=None):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open video: %s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def _load_gif(path, max_frames=None):
    im = Image.open(path)
    frames = []
    fps = 15.0
    try:
        while True:
            frames.append(np.asarray(im.convert("RGB")))
            if max_frames and len(frames) >= max_frames:
                break
            dur = im.info.get("duration", 66)
            fps = 1000.0 / max(dur, 1)
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return frames, fps


def _load_dir(path, max_frames=None, fps=15.0):
    files = sorted(f for f in glob.glob(os.path.join(path, "*"))
                    if f.lower().endswith(IMAGE_EXT))
    if max_frames:
        files = files[:max_frames]
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    return frames, fps, files


def load_frames(path, max_frames=None, fps_hint=15.0):
    """Returns (frames: list[HxWx3 uint8 RGB], fps: float, files: list[str]|None)."""
    if os.path.isdir(path):
        frames, fps, files = _load_dir(path, max_frames, fps_hint)
        return frames, fps, files
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        frames, fps = _load_gif(path, max_frames)
    elif ext in VIDEO_EXT:
        frames, fps = _load_video(path, max_frames)
    else:
        raise SystemExit("unrecognised input: %s (need a video, gif, or frame dir)"
                          % path)
    return frames, fps, None
