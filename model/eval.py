"""
eval.py - measure the trained model on the held-out val set.

Reports exact DSL match and per-slot accuracy (arch is the slot that matters
most; if arch is right the video is on-topic).

    python eval.py --ckpt checkpoint --val data/val.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import grammar as G
from infer import PromptModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.val).read_text(
        encoding="utf-8").splitlines() if l.strip()][:args.n]
    model = PromptModel(None if args.rules_only else args.ckpt)
    print(f"backend: {'model' if model.ok else 'rules'}  n={len(rows)}")

    slots = ["arch", "color", "cmap", "speed", "sym", "mood"]
    hit = {s: 0 for s in slots}
    tot = {s: 0 for s in slots}
    exact = 0
    for r in rows:
        gold = G.dsl_to_spec(r["dsl"])
        pred = model.spec(r["prompt"])
        if G.spec_to_dsl(pred) == G.spec_to_dsl(gold):
            exact += 1
        for s in slots:
            if s in gold:
                tot[s] += 1
                if pred.get(s) == gold[s]:
                    hit[s] += 1

    print(f"exact DSL match: {exact}/{len(rows)} = {exact/len(rows)*100:.1f}%")
    for s in slots:
        if tot[s]:
            print(f"  {s:6s} {hit[s]/tot[s]*100:5.1f}%  ({hit[s]}/{tot[s]})")


if __name__ == "__main__":
    main()
