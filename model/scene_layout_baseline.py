"""
scene_layout_baseline.py - is the `layout` rate evidence, or is it chance?

`scene_eval` reports how often a scene satisfies the spatial facts its prompt
stated. That number is meaningless on its own, because the facts are cheap to
satisfy by accident: with a cast of two, "character 0 is leftmost" is true half
the time whatever the model does.

This computes the standard control - **mismatched pairing**. Each prompt's
layout facts are scored against a DIFFERENT prompt's emitted scene, averaged
over every mismatched pair. If the matched rate is no better than the
mismatched rate, the model is not reading the spatial instruction, and any
apparent obedience is the geometry it would have produced anyway.

    USE_TF=0 python model/scene_layout_baseline.py \\
        model/eval_out3/out_of_distribution_sp_v3samp.json
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import scene_grammar as SG


def _facts(row, val_rows=None):
    """The layout facts the prompt stated, from the eval row or the val file."""
    return row.get("_layout") or []


def score(path, val_path=None):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    # intents live in the source file, not in the eval output, so re-read them
    src = None
    if val_path:
        p = Path(val_path)
        if p.suffix == ".json":
            src = json.loads(p.read_text(encoding="utf-8"))
        else:
            src = [json.loads(l) for l in
                   p.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_prompt = {}
    if src:
        for r in src:
            by_prompt[r["prompt"]] = (r.get("intent") or {}).get("layout") or []

    items = []
    for r in rows:
        if not r.get("parsed"):
            continue
        facts = by_prompt.get(r["prompt"], [])
        if not facts:
            continue
        try:
            spec, _ = SG.repair(SG.dsl_to_spec(r["raw"]))
        except SG.DSLError:
            continue
        items.append((facts, spec))

    if not items:
        return None
    matched = sum(all(SG.check_layout_fact(sp, f) for f in fs)
                  for fs, sp in items) / len(items)
    tot = hit = 0
    for i, (fs, _) in enumerate(items):
        for j, (_, sp) in enumerate(items):
            if i == j:
                continue
            tot += 1
            hit += all(SG.check_layout_fact(sp, f) for f in fs)
    return {"n": len(items), "matched": round(100 * matched, 1),
            "mismatched_chance": round(100 * hit / tot, 1),
            "lift": round(100 * matched - 100 * hit / tot, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+",
                    help="evaljson:valfile:name triples (colon-separated)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["%-22s %5s %9s %11s %7s"
             % ("cell", "n", "matched", "chance", "lift")]
    res = {}
    for spec in a.pairs:
        parts = spec.split("|")
        path, val = parts[0], parts[1]
        name = parts[2] if len(parts) > 2 else Path(path).stem
        s = score(path, val)
        if s is None:
            lines.append("%-22s  no layout facts in this set" % name)
            continue
        res[name] = s
        lines.append("%-22s %5d %8.1f%% %10.1f%% %6.1f" %
                     (name, s["n"], s["matched"], s["mismatched_chance"],
                      s["lift"]))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n\n" + json.dumps(res, indent=1),
                               encoding="utf-8")
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
