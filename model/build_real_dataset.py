"""
build_real_dataset.py - label real photos with YOLO-pose, store (crop, pose vec).

Runs the real-data-trained YOLO-pose over a folder of real images, and for every
detected person saves a grayscale 80x80 crop plus our 14-d pose vector (from
pose_map). This is a genuine (real image -> equations-parameters) dataset our
compact ImgPoseNet is then trained on.

    python build_real_dataset.py <images_dir> -o data/real_imgs.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import stickman as S       # noqa: E402
import pose_map as PMAP    # noqa: E402
import vectorize as V      # noqa: E402

MIN_CONF_JOINTS = 10       # need at least this many confident keypoints


def gray_crop(img_bgr, box, size=80, pad=0.15):
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - pad * bw)); x1 = min(w, int(x1 + pad * bw))
    y0 = max(0, int(y0 - pad * bh)); y1 = min(h, int(y1 + pad * bh))
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    from PIL import Image
    g = Image.fromarray(crop[:, :, ::-1]).convert("L").resize((size, size))
    return np.asarray(g, np.float32) / 255.0


def build(src_dir, out_npz, size=80):
    files = [p for p in Path(src_dir).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    fig = S.StickFigure(854, 480)
    imgs, Y = [], []
    model = V.yolo()
    for k, p in enumerate(files):
        res = model(str(p), verbose=False)[0]
        if res.keypoints is None or res.boxes is None:
            continue
        xy = res.keypoints.xy.cpu().numpy()
        cf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
              else np.ones(xy.shape[:2]))
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        orig = res.orig_img
        for i in range(len(xy)):
            if int(cls[i]) != 0:               # person class
                continue
            if (cf[i] >= 0.3).sum() < MIN_CONF_JOINTS:
                continue
            g = gray_crop(orig, boxes[i], size)
            if g is None:
                continue
            vec = PMAP.coco_to_vector(xy[i], cf[i], ground=fig.ground,
                                      view_w=fig.W, view_h=fig.H,
                                      leg=fig.leg, torso=fig.torso)
            vec[0] = 0.5                        # crop is centered on the person
            imgs.append(g); Y.append(vec)
        if (k + 1) % 20 == 0:
            print(f"\r  {k+1}/{len(files)} images, {len(imgs)} persons", end="", flush=True)
    imgs = np.array(imgs, np.float32); Y = np.array(Y, np.float32)
    Path(out_npz).parent.mkdir(exist_ok=True)
    np.savez_compressed(out_npz, imgs=imgs, Y=Y)
    print(f"\nreal dataset -> {out_npz}  imgs{imgs.shape}  Y{Y.shape}")
    return imgs, Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="data/real_imgs.npz")
    args = ap.parse_args()
    build(args.src, args.out)
