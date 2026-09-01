"""Corpus shards -> train/val jsonl for caption -> AniSVG training.

Three jobs the raw shards do not do: drop duplicates (the source corpus repeats
animations under different ids), drop clips too long for the context we can
afford on 8 GB, and split deterministically so a rerun cannot leak validation
clips into training.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anisvg import Anim                          # noqa: E402

PROMPT = "### animation\n%s\n\n### anisvg\n"

_WS = re.compile(r"\s+")


def clean(caption):
    return _WS.sub(" ", caption or "").strip()


def body_key(text):
    """Hash the shape/motion body, ignoring the header.

    Two clips with the same art but a different canvas size are still the same
    animation for training purposes.
    """
    body = "\n".join(ln for ln in text.split("\n") if not ln.startswith("H "))
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()


def split_of(name, val_frac):
    """Deterministic per-clip split - stable across reruns and shard order."""
    h = int(hashlib.blake2b(str(name).encode(), digest_size=8).hexdigest(), 16)
    return "val" if (h % 10000) < val_frac * 10000 else "train"


def rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="svg/data/corpus")
    ap.add_argument("-o", "--out", default="svg/data/train")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="drop clips whose prompt+body exceeds this")
    ap.add_argument("--min-caption", type=int, default=15)
    ap.add_argument("--min-frames", type=int, default=12,
                    help="never trim a clip below this; drop it instead")
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    paths = sorted(glob.glob(os.path.join(args.corpus, "anisvg-*.jsonl.gz")))
    if not paths:
        raise SystemExit("no shards in %s" % args.corpus)
    print("%d shards" % len(paths))

    os.makedirs(args.out, exist_ok=True)
    seen_name, seen_body = set(), set()
    drop = {"dup-name": 0, "dup-body": 0, "caption": 0, "too-long": 0,
            "trimmed": 0}
    kept = {"train": 0, "val": 0}
    lens = []

    fh = {s: open(os.path.join(args.out, "%s.jsonl" % s), "w",
                  encoding="utf-8") for s in kept}
    try:
        batch = []
        for rec in rows(paths):
            name = str(rec["name"])
            if name in seen_name:
                drop["dup-name"] += 1
                continue
            seen_name.add(name)
            caption = clean(rec.get("caption"))
            if len(caption) < args.min_caption:
                drop["caption"] += 1
                continue
            key = body_key(rec["text"])
            if key in seen_body:
                drop["dup-body"] += 1
                continue
            seen_body.add(key)
            batch.append((name, caption, rec))
            if len(batch) >= 256:
                flush(batch, tok, args, fh, kept, drop, lens)
                batch = []
        flush(batch, tok, args, fh, kept, drop, lens)
    finally:
        for f in fh.values():
            f.close()

    lens.sort()
    total = kept["train"] + kept["val"]
    print("\nkept %d  (train %d, val %d)" % (total, kept["train"], kept["val"]))
    print("dropped: " + ", ".join(
        "%s %d" % kv for kv in sorted(drop.items()) if kv[0] != "trimmed"))
    print("trimmed to fit: %d" % drop["trimmed"])
    if lens:
        print("tokens: median %d  p90 %d  max %d  total %.1fM"
              % (lens[len(lens) // 2], lens[int(0.9 * len(lens))], lens[-1],
                 sum(lens) / 1e6))
    print("-> %s/{train,val}.jsonl" % args.out)


def trim(text, keep):
    """Re-emit an AniSVG clip with only its first `keep` frames.

    The header's frame count is derived from the frame list, so slicing and
    re-emitting produces a valid, shorter clip. Dropping the tail of a six
    second animation is a far better trade than dropping the animation.
    """
    anim = Anim.from_text(text)
    anim.frames = anim.frames[:keep]
    return anim.to_text(), len(anim.frames)


def flush(batch, tok, args, fh, kept, drop, lens):
    """Tokenise a batch at once - one call beats 256 - trimming what overruns."""
    if not batch:
        return
    items = [[name, caption, rec["text"], rec["frames"], rec["shapes"]]
             for name, caption, rec in batch]
    todo = list(range(len(items)))
    counts = [0] * len(items)
    cut = set()

    for _ in range(4):
        if not todo:
            break
        ids = tok([PROMPT % items[i][1] + items[i][2] for i in todo],
                  add_special_tokens=False)["input_ids"]
        nxt = []
        for i, seq in zip(todo, ids):
            n = len(seq) + 1                   # + eos
            counts[i] = n
            if n <= args.max_tokens:
                continue
            # Frames are roughly uniform in cost, so scale the frame count by
            # how far over budget we are, with a margin, then re-measure.
            frames = items[i][3]
            keep = int(frames * (args.max_tokens / float(n)) * 0.92)
            if keep < args.min_frames:
                counts[i] = -1                 # cast alone busts the budget
                continue
            items[i][2], items[i][3] = trim(items[i][2], keep)
            if i not in cut:                   # count clips, not iterations
                cut.add(i)
                drop["trimmed"] += 1
            nxt.append(i)
        todo = nxt

    for i in todo:                             # never converged
        counts[i] = -1

    for (name, caption, text, frames, shapes), n in zip(items, counts):
        if n < 0 or n > args.max_tokens:
            drop["too-long"] += 1
            continue
        split = split_of(name, args.val_frac)
        fh[split].write(json.dumps(dict(
            name=name, caption=caption, text=text,
            shapes=shapes, frames=frames, tokens=n)) + "\n")
        kept[split] += 1
        lens.append(n)


if __name__ == "__main__":
    main()
