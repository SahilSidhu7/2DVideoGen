"""Assemble the publishable output set, labelled by origin.

Two kinds of artefact live in this project and they must never be confused in
public material:

  GENERATED   written by a trained model from a caption it had not seen
  PROCEDURAL  written by the rig in `chars.py` / `anime_chars.py` - this is
              training data, and it shows what the *format* can carry, not
              what the model can produce

Every clip is stamped with which it is, and the manifest records it, so a
figure cannot drift away from its provenance.
"""
import argparse
import glob
import gzip
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anisvg import Anim                           # noqa: E402
from generate import extract, score               # noqa: E402
import raster                                     # noqa: E402
from PIL import Image, ImageDraw, ImageFont       # noqa: E402


def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def clip_texts_from_jsonl(path, n, rng, min_shapes=0):
    op = gzip.open if path.endswith(".gz") else open
    rows = []
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("shapes", 0) >= min_shapes:
                rows.append(r)
            if len(rows) > 4000:
                break
    rng.shuffle(rows)
    return [(r["text"], r.get("caption", ""), r.get("tags", [])) for r in rows[:n]]


def clip_texts_from_dir(path, n, rng):
    out = []
    for f in sorted(glob.glob(os.path.join(path, "g*.anisvg"))):
        t = open(f, encoding="utf-8").read()
        if score(t)["ok"]:
            out.append((t, "", []))
    return out[:n]


def ink_of(imgs):
    import numpy as np
    arr = np.stack([np.asarray(im.convert("L")) for im in imgs])
    return float((arr < 200).mean())


def render_clip(text, width):
    anim = Anim.from_text(extract(text))
    return [raster.render(s, width=width) for s in anim.to_svgs()], anim


def build_reel(clips, out_mp4, width, origin, label_lines, loops, hold, fps=15):
    """One video per version, each clip stamped with its origin."""
    fnt, small = font(15), font(12)
    frames = []
    for (imgs, cap) in clips:
        stamped = []
        for im in imgs:
            c = Image.new("RGB", (width, width + 46), (250, 250, 250))
            c.paste(im, (0, 0))
            d = ImageDraw.Draw(c)
            d.rectangle([0, width, width, width + 46], fill=(24, 26, 30))
            d.text((8, width + 5), origin, fill=(110, 220, 190), font=small)
            d.text((8, width + 22), cap[:74], fill=(228, 232, 237), font=small)
            stamped.append(c)
        for _ in range(loops):
            frames.extend(stamped)
            frames.extend([stamped[-1]] * hold)
    if not frames:
        return None
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="rel_")
    try:
        for i, im in enumerate(frames):
            w, h = im.size
            if w % 2 or h % 2:
                im = im.resize((w - w % 2, h - h % 2))
            im.save(os.path.join(tmp, "f%05d.png" % i))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", out_mp4], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(frames)


VERSIONS = [
    dict(key="v1-8b-icons", origin="GENERATED - Qwen3-8B LoRA, 1024 ctx",
         src=("dir", "svg/out/gen_con2"),
         note="Trained on LottieAnimation-660K. 94% valid; colour weak, geometry abstract."),
    dict(key="v2-1.7b-icons", origin="GENERATED - Qwen3-1.7B LoRA, 4096 ctx",
         src=("dir", "svg/out/gen17"),
         note="7x the animation signal. 100% valid; colour and object count correct, geometry still abstract."),
    dict(key="corpus-stickman", origin="PROCEDURAL - training data, not model output",
         src=("jsonl", "svg/data/chars/chars.jsonl.gz"),
         note="Articulated rig -> AniSVG. Shows what the format carries."),
    dict(key="corpus-anime", origin="PROCEDURAL - training data, not model output",
         src=("jsonl", "svg/data/anime/anime.jsonl.gz"),
         note="Cel-style character, fixed 16-part schema. Target for the next model."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="release")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--per-version", type=int, default=6)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for v in VERSIONS:
        kind, path = v["src"]
        if not os.path.exists(path):
            print("skip %-18s (missing %s)" % (v["key"], path))
            continue
        raw = (clip_texts_from_dir(path, args.per_version * 3, rng) if kind == "dir"
               else clip_texts_from_jsonl(path, args.per_version * 3, rng, min_shapes=8))
        chosen = []
        for text, cap, tags in raw:
            try:
                imgs, anim = render_clip(text, args.width)
            except Exception:
                continue
            if len(imgs) < 4 or ink_of(imgs) < 0.004:
                continue                    # valid but draws nothing
            chosen.append(((imgs, cap or ", ".join(tags)), anim, text))
            if len(chosen) >= args.per_version:
                break
        if not chosen:
            print("skip %-18s (no renderable clips)" % v["key"])
            continue

        vdir = os.path.join(args.out, v["key"])
        os.makedirs(vdir, exist_ok=True)
        mp4 = os.path.join(vdir, "%s.mp4" % v["key"])
        n = build_reel([c[0] for c in chosen], mp4, args.width, v["origin"],
                       None, args.loops, args.hold)
        for i, (_, anim, text) in enumerate(chosen):
            with open(os.path.join(vdir, "clip%02d.anisvg" % i), "w",
                      encoding="utf-8") as fh:
                fh.write(extract(text))
        entry = dict(version=v["key"], origin=v["origin"], note=v["note"],
                     clips=len(chosen), frames=n, video=os.path.relpath(mp4),
                     shapes=[len(a.shapes) for _, a, _ in chosen],
                     clip_frames=[len(a.frames) for _, a, _ in chosen])
        manifest.append(entry)
        print("%-18s %d clips, %3d frames -> %s (%.0f KB)"
              % (v["key"], len(chosen), n, mp4, os.path.getsize(mp4) / 1024))

    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nmanifest -> %s/MANIFEST.json" % args.out)


if __name__ == "__main__":
    main()
