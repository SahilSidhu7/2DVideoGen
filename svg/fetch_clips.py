"""Download public-domain cel-animation source clips from the Internet Archive.

Only PD material: Fleischer Superman (1941-42, copyright not renewed) and
early Japanese animation. Flat cel shading + hard edges = ideal for tracing.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DEST = os.path.join(os.path.dirname(__file__), "..", "datasets", "pd_anime")

# (item identifier, note). All verified public domain.
ITEMS = [
    ("the-arctic-giant-1942", "Fleischer Superman - The Arctic Giant"),
    ("destruction-inc.-1942_202406", "Fleischer Superman - Destruction Inc."),
    ("billion-dollar-limited-1942", "Fleischer Superman - Billion Dollar Limited"),
    ("superman-takes-a-vacation", "Superman compilation"),
    ("TheBettyBoopLimited", "Fleischer - The Betty Boop Limited (1932)"),
]

VIDEO_EXT = (".mp4", ".m4v", ".mkv", ".avi", ".ogv", ".mpeg", ".mpg")


def files_for(ident):
    url = "https://archive.org/metadata/" + urllib.parse.quote(ident)
    with urllib.request.urlopen(url, timeout=120) as fh:
        meta = json.load(fh)
    out = []
    for f in meta.get("files", []):
        name = f.get("name", "")
        if name.lower().endswith(VIDEO_EXT):
            out.append((name, int(f.get("size", 0) or 0)))
    return out


def pick(files, max_mb):
    """Smallest video that is still a full-length encode, under max_mb."""
    limit = max_mb * 1_000_000
    ok = [f for f in files if 5_000_000 < f[1] <= limit]
    if not ok:
        ok = sorted(files, key=lambda f: f[1])[:1]
    return max(ok, key=lambda f: f[1]) if ok else None


def download(ident, name, dest):
    url = "https://archive.org/download/%s/%s" % (
        urllib.parse.quote(ident), urllib.parse.quote(name))
    path = os.path.join(dest, ident + os.path.splitext(name)[1].lower())
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print("  have", os.path.basename(path))
        return path
    print("  get ", url)
    req = urllib.request.Request(url, headers={"User-Agent": "2dvideogen/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as w:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            w.write(chunk)
            done += len(chunk)
            sys.stdout.write("\r   %.0f MB" % (done / 1e6))
            sys.stdout.flush()
    print()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--max-mb", type=int, default=350)
    ap.add_argument("--limit", type=int, default=len(ITEMS))
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    got = []
    for ident, note in ITEMS[: args.limit]:
        print(ident, "-", note)
        try:
            f = pick(files_for(ident), args.max_mb)
        except Exception as exc:
            print("  metadata failed:", exc)
            continue
        if not f:
            print("  no suitable video file")
            continue
        try:
            got.append(download(ident, f[0], dest))
        except Exception as exc:
            print("  download failed:", exc)
    print("\n%d clips in %s" % (len(got), dest))


if __name__ == "__main__":
    main()
