"""Packed clips -> Soup's Alpaca format.

Soup masks the instruction and trains on `output`, which is the same objective
`train_lora.py` implements by hand - so the two paths stay comparable.

`--limit` exists because an 8B streamed run is priced per token: a pilot has to
be sized from a measured rate, not from a clip count.
"""
import argparse
import json
import os


def convert(src, dst, limit=0, max_tokens=0):
    n = tokens = 0
    with open(src, encoding="utf-8") as fh, \
            open(dst, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if max_tokens and r["tokens"] > max_tokens:
                continue
            out.write(json.dumps(dict(instruction=r["caption"], input="",
                                      output=r["text"])) + "\n")
            n += 1
            tokens += r["tokens"]
            if limit and n >= limit:
                break
    return n, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="svg/data/train")
    ap.add_argument("-o", "--out", default="svg/data/soup")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap training rows (pilot runs)")
    ap.add_argument("--val-limit", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="skip clips longer than this")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for split, limit in (("train", args.limit), ("val", args.val_limit)):
        n, tokens = convert(os.path.join(args.data, "%s.jsonl" % split),
                            os.path.join(args.out, "%s.jsonl" % split),
                            limit, args.max_tokens)
        total += tokens
        print("%-5s %6d rows  %8.2fM tokens" % (split, n, tokens / 1e6))
    print("\ntotal %.2fM tokens" % (total / 1e6))
    print("at 120 tok/s  -> %6.1f h" % (total / 120 / 3600))
    print("at 240 tok/s  -> %6.1f h" % (total / 240 / 3600))


if __name__ == "__main__":
    main()
