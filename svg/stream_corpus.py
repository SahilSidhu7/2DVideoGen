"""Stream LottieAnimation-660K -> AniSVG, one shard at a time.

The dataset is 132 GB, but 129 GB of that is `videos-*.tar.zst` - rendered mp4
previews we never train on. Only `metadata-*.jsonl.zst` (2.96 GB, 68 shards)
carries the Lottie JSON, captions and tags. This script pulls those shards over
HTTP, decompresses and converts them in flight, and never writes the source to
disk: peak disk cost is the gzipped AniSVG output alone.

`datasets.load_dataset` cannot be used here - the repo's two splits have
different formats (json vs webdataset) and auto-detection fails with
FileFormatMismatchBetweenSplitsError - so shards are addressed by filename.

Resume is per shard: each shard writes to `<out>/anisvg-000NN.jsonl.gz.part`
and is renamed only once the shard is fully consumed, so an interrupted run
redoes at most one shard.
"""
import argparse
import glob
import gzip
import http.client
import io
import json
import os
import re
import sys
import time

import requests
import urllib3
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lottie as L                              # noqa: E402
from lottie_to_anisvg import convert            # noqa: E402

REPO = "LottieGPT/LottieAnimation-660K"
SHARDS = 68
URL = ("https://huggingface.co/datasets/%s/resolve/main/"
       "data/metadata-%05d-of-%05d.jsonl.zst")

CAPTION_KEYS = ("caption", "captions", "text", "description", "prompt", "title")
NAME_KEYS = ("id", "name", "key", "file", "filename", "path", "uuid")


def token():
    """HF token from the CLI login, the env, or nothing."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


# A mid-stream drop surfaces as urllib3's ProtocolError, which is *not* a
# requests exception because the body is read off `resp.raw` directly, and as a
# bare ConnectionResetError on Windows. Catch the lot.
NET_ERRS = (requests.RequestException, urllib3.exceptions.HTTPError,
            zstandard.ZstdError, http.client.HTTPException, OSError)


def shard_lines(index, tok, timeout=60, retries=6):
    """Yield decoded JSON lines of one metadata shard, streaming.

    The zstd frame is decompressed as it arrives, so memory stays at the
    decompressor's window size regardless of shard size.

    A dropped connection reconnects and replays the shard, discarding the lines
    already delivered without parsing them. The bytes are re-fetched - a zstd
    frame cannot be resumed from the middle - but conversion is ~50x the cost of
    the download, so what matters is that the caller never sees a line twice and
    never redoes a conversion.
    """
    url = URL % (REPO, index, SHARDS)
    headers = {"Authorization": "Bearer %s" % tok} if tok else {}
    done = 0
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 401:
                raise SystemExit(
                    "HTTP 401 on %s\n"
                    "The dataset is gated. Accept the licence at\n"
                    "  https://huggingface.co/datasets/%s\n"
                    "then run `hf auth login` (or set HF_TOKEN)."
                    % (os.path.basename(url), REPO))
            resp.raise_for_status()
            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            with dctx.stream_reader(resp.raw) as raw:
                for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
                    if i < done:
                        continue
                    done = i + 1
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return
        except NET_ERRS as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 3 * 2 ** attempt)
            print("  shard %02d dropped after %d lines (%s); reconnecting in %ds"
                  % (index, done, str(exc)[:70], wait))
            time.sleep(wait)


def _is_lottie(obj):
    return (isinstance(obj, dict) and isinstance(obj.get("layers"), list)
            and ("fr" in obj or "v" in obj))


def find_lottie(rec, depth=3):
    """Locate the Lottie document inside a metadata record.

    The record schema is not published, and the field may hold the document
    itself or a JSON string of it, so probe rather than hard-code a key.
    """
    if _is_lottie(rec):
        return rec
    if depth <= 0:
        return None
    for val in (rec.values() if isinstance(rec, dict) else
                rec if isinstance(rec, list) else []):
        if isinstance(val, str) and val.lstrip()[:1] == "{":
            try:
                val = json.loads(val)
            except ValueError:
                continue
        if _is_lottie(val):
            return val
        if isinstance(val, (dict, list)):
            got = find_lottie(val, depth - 1)
            if got is not None:
                return got
    return None


def _first(rec, keys):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0].strip()
    return ""


def describe(rec):
    """Best-effort (name, caption, tags) from an unknown record schema."""
    tags = rec.get("tag") or rec.get("tags") or rec.get("keywords") or []
    if isinstance(tags, str):
        tags = [t for t in re.split(r"[,;|]", tags) if t.strip()]
    return (_first(rec, NAME_KEYS), _first(rec, CAPTION_KEYS),
            [str(t).strip() for t in tags][:24])


class Filters(object):
    """Reject anything the token budget or the converter cannot carry."""

    def __init__(self, args):
        self.min_shapes = args.min_shapes
        self.max_shapes = args.max_shapes
        self.min_frames = args.min_frames
        self.max_chars = args.max_chars
        self.min_motion = args.min_motion
        self.allow_partial = args.allow_partial
        self.grep = re.compile(args.caption_grep, re.I) if args.caption_grep else None

    def pre(self, doc, caption, tags):
        """Cheap checks, before paying for a conversion."""
        if self.grep and not self.grep.search(caption + " " + " ".join(tags)):
            return "caption"
        n = len(doc.get("layers") or ())
        # A layer expands to >= 1 shape, so a layer count over the cap is a
        # guaranteed reject and skipping it here avoids the sampling cost.
        if n > self.max_shapes * 4:
            return "layers"
        return None

    def probe(self, doc, fps):
        """Verdict from a single sampled frame, or None to keep converting.

        Only rejects - a clip that passes here can still fail `post`, because a
        feature may appear later in the timeline than the frame probed.
        """
        times, _ = L.times(doc, fps)
        if not times:
            return "frames"
        got, skipped = L.sample(doc, times[0])
        if skipped and not self.allow_partial:
            return "lossy:" + ",".join(sorted(skipped))
        if not self.min_shapes <= len(got) <= self.max_shapes:
            return "shapes"
        return None

    def post(self, anim, skipped):
        if not self.min_shapes <= len(anim.shapes) <= self.max_shapes:
            return "shapes"
        if len(anim.frames) < self.min_frames:
            return "frames"
        moving = sum(1 for f in anim.frames if f)
        if moving < self.min_motion * len(anim.frames):
            return "static"
        if skipped and not self.allow_partial:
            # A dropped precomp, gradient fill or trim path does not fail
            # loudly - the clip just renders wrong, which is worse than losing
            # it. Roughly 62% of convertible clips are clean, so the corpus can
            # afford to be picky.
            return "lossy:" + ",".join(sorted(skipped))
        return None


_ARGS = None
_FILT = None


def _setup(args):
    """Pool initialiser: rebuild the per-process conversion config once."""
    global _ARGS, _FILT
    _ARGS, _FILT = args, Filters(args)


def convert_record(rec):
    """One metadata record -> (output dict, None) or (None, reject reason).

    Written as a free function taking one argument so it can be handed
    straight to a process pool; conversion is pure Python and dominates the
    runtime, so this is where the parallelism has to go.
    """
    name, caption, tags = describe(rec)
    doc = find_lottie(rec)
    if doc is None:
        return None, "no-lottie"
    why = _FILT.pre(doc, caption, tags)
    if why:
        return None, why
    # Most rejects are only visible after conversion, but conversion costs one
    # sample per frame - ~25x what a single sample costs. Cast size and
    # unsupported features are both decidable from one frame, and together they
    # account for the majority of rejects, so probe before paying for the rest.
    try:
        why = _FILT.probe(doc, _ARGS.fps)
    except Exception as exc:
        return None, "err:" + str(exc)[:32]
    if why:
        return None, why
    try:
        anim, skipped = convert(doc, fps=_ARGS.fps, width=_ARGS.width,
                                points=_ARGS.points,
                                max_frames=_ARGS.max_frames)
    except Exception as exc:
        return None, "err:" + str(exc)[:32]
    text = anim.to_text()
    if len(text) > _ARGS.max_chars:
        return None, "too-long"
    why = _FILT.post(anim, skipped)
    if why:
        return None, why
    return dict(name=name, caption=caption, tags=tags,
                shapes=len(anim.shapes), frames=len(anim.frames),
                text=text, skipped=sorted(skipped)), None


def _results(records, args):
    """Yield convert_record results, in a pool unless workers == 1."""
    if args.workers == 1:
        _setup(args)
        for rec in records:
            yield convert_record(rec)
        return
    import multiprocessing as mp
    n = args.workers or max(1, (os.cpu_count() or 2) - 1)
    pool = mp.Pool(n, initializer=_setup, initargs=(args,))
    try:
        # imap keeps the shard streaming lazily instead of materialising it.
        for got in pool.imap_unordered(convert_record, records, chunksize=8):
            yield got
    finally:
        pool.terminate()
        pool.join()


def run_shard(index, tok, args, budget):
    """Convert one shard. Returns (kept, reasons) or None if already done."""
    final = os.path.join(args.out, "anisvg-%05d.jsonl.gz" % index)
    if os.path.exists(final):
        return None
    part = final + ".part"
    reasons = {}
    kept = seen = 0
    t0 = time.time()

    def records():
        for rec in shard_lines(index, tok):
            yield rec

    with gzip.open(part, "wt", encoding="utf-8", compresslevel=6) as fh:
        for out, why in _results(records(), args):
            seen += 1
            if why:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            out["name"] = out["name"] or "%05d-%d" % (index, seen)
            fh.write(json.dumps(out) + "\n")
            kept += 1
            if args.progress and kept % args.progress == 0:
                print("    %5d kept / %6d seen  %4.0fs"
                      % (kept, seen, time.time() - t0))
            if kept >= budget:
                break
    os.rename(part, final)
    print("shard %02d: %5d kept / %6d seen  %5.1f MB  %4.0fs" %
          (index, kept, seen, os.path.getsize(final) / 1e6, time.time() - t0))
    return kept, reasons


def probe(tok, args):
    """Print the schema of the first few records without a full download."""
    for i, rec in enumerate(shard_lines(args.start, tok)):
        doc = find_lottie(rec)
        name, caption, tags = describe(rec)
        print("record %d" % i)
        print("  keys     : %s" % ", ".join(sorted(rec)[:20]))
        print("  name     : %s" % name)
        print("  caption  : %s" % caption[:100])
        print("  tags     : %s" % ", ".join(tags[:8]))
        if doc is None:
            print("  lottie   : NOT FOUND")
        else:
            print("  lottie   : %d layers, %sx%s, fr=%s, op=%s"
                  % (len(doc.get("layers") or ()), doc.get("w"), doc.get("h"),
                     doc.get("fr"), doc.get("op")))
        if i + 1 >= args.probe:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default="svg/data/corpus")
    ap.add_argument("--start", type=int, default=0, help="first shard index")
    ap.add_argument("--shards", type=int, default=SHARDS, help="shards to read")
    ap.add_argument("--max-clips", type=int, default=40000,
                    help="stop once this many clips are kept (0 = no cap). A "
                         "shard cut short by the cap still counts as done, so "
                         "raising the cap later resumes from the next shard.")
    ap.add_argument("--probe", type=int, metavar="N",
                    help="print the schema of N records and exit")
    # conversion
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=90)
    # filters
    ap.add_argument("--min-shapes", type=int, default=2)
    ap.add_argument("--max-shapes", type=int, default=20,
                    help="cast size cap; ~6 tokens per moving shape per frame")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-motion", type=float, default=0.5,
                    help="minimum fraction of frames carrying at least one op")
    ap.add_argument("--max-chars", type=int, default=60000,
                    help="roughly 18k tokens at this format's chars-per-token")
    ap.add_argument("--allow-partial", action="store_true",
                    help="keep clips that use an unsupported feature, and so "
                         "convert with content missing (default: reject them)")
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help="parallel converter processes (0 = auto, 1 = inline)")
    ap.add_argument("--caption-grep", help="keep only captions/tags matching this regex")
    ap.add_argument("--progress", type=int, default=0, metavar="N",
                    help="print a line every N kept clips")
    args = ap.parse_args()

    tok = token()
    if not tok:
        print("warning: no HF token found; gated shards will return 401\n"
              "  huggingface-cli login   (or set HF_TOKEN)\n")

    if args.probe:
        probe(tok, args)
        return

    os.makedirs(args.out, exist_ok=True)
    for stale in glob.glob(os.path.join(args.out, "*.part")):
        os.remove(stale)

    total = 0
    reasons = {}
    for index in range(args.start, min(args.start + args.shards, SHARDS)):
        budget = (args.max_clips - total) if args.max_clips else 10 ** 9
        if budget <= 0:
            break
        try:
            got = run_shard(index, tok, args, budget)
        except NET_ERRS as exc:
            # One unreachable shard should not cost the other 67.
            print("shard %02d: giving up (%s)" % (index, str(exc)[:70]))
            for stale in glob.glob(os.path.join(args.out, "*.part")):
                os.remove(stale)
            continue
        if got is None:
            print("shard %02d: done already" % index)
            continue
        kept, why = got
        total += kept
        for k, v in why.items():
            reasons[k] = reasons.get(k, 0) + v

    size = sum(os.path.getsize(p) for p in
               glob.glob(os.path.join(args.out, "anisvg-*.jsonl.gz")))
    print("\n%d clips kept, %.1f MB gzipped -> %s" % (total, size / 1e6, args.out))
    if reasons:
        print("rejected:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]:
            print("  %-40s %7d" % (k, v))


if __name__ == "__main__":
    main()
