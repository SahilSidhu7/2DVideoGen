"""
scene_data_report.py - what is actually in a synthesised dataset.

Attempt 22's headline failure was a correlation nobody had measured until
after the model had learned it. This reports, for any `*_train.jsonl`:

  * the cast-size distribution
  * the clause-count / cast-size correlation (the Attempt 22 confound), both
    as an exact-agreement rate and as Pearson r
  * the staging score of the TARGETS themselves, via model/scene_staging.py
  * how often a spatial fact is stated, and by kind
  * tokenised target length, so --max-out can be set from a measurement

    USE_TF=0 python model/scene_data_report.py model/data/scene3_train.jsonl
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import scene_grammar as SG
import scene_staging as ST


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def report(path, stage_n=600, tok_n=400, tokenizer=None):
    rows = [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = ["%s  n=%d" % (Path(path).name, len(rows))]

    sizes = Counter()
    have_desc = [r for r in rows if "n_described" in r]
    for r in rows:
        try:
            sizes[len(SG.dsl_to_spec(r["dsl"])["cast"])] += 1
        except SG.DSLError:
            sizes["unparseable"] += 1
    out.append("  cast size: %s" % dict(sorted(sizes.items(), key=str)))

    if have_desc:
        nc = [r["n_cast"] for r in have_desc]
        nd = [r["n_described"] for r in have_desc]
        agree = sum(a == b for a, b in zip(nc, nd))
        out.append("  clause count == cast size: %d/%d = %.1f%%   pearson r=%.3f"
                   % (agree, len(nc), 100 * agree / len(nc), _pearson(nc, nd)))
    else:
        out.append("  clause count: not recorded in this dataset "
                   "(pre-Attempt-23 file); see ATTEMPTS.md for the historical "
                   "99.9% / 68.0% figures")

    lay = Counter()
    nlay = 0
    for r in rows:
        fs = (r.get("intent") or {}).get("layout") or []
        nlay += bool(fs)
        for f in fs:
            lay[f["kind"]] += 1
    out.append("  prompts stating a spatial fact: %d/%d = %.1f%%   kinds: %s"
               % (nlay, len(rows), 100 * nlay / len(rows), dict(lay.most_common())))

    vals, nan = [], 0
    for r in rows[:stage_n]:
        try:
            m = ST.stage_spec(SG.dsl_to_spec(r["dsl"]), n_samples=25)
        except Exception:
            continue
        if math.isnan(m["staging"]):
            nan += 1
        else:
            vals.append(m["staging"])
    if vals:
        vals.sort()
        out.append("  target staging (n=%d multi-cast of %d): mean %.3f  "
                   "median %.3f  p10 %.3f  frac<0.5 %.3f  frac<0.825 %.3f"
                   % (len(vals), min(stage_n, len(rows)), sum(vals) / len(vals),
                      vals[len(vals) // 2], vals[int(0.1 * len(vals))],
                      sum(v < 0.5 for v in vals) / len(vals),
                      sum(v < 0.825 for v in vals) / len(vals)))

    if tokenizer is not None:
        li = [len(tokenizer("prompt2scene: " + r["prompt"])["input_ids"])
              for r in rows[:tok_n]]
        lo = [len(tokenizer(text_target=r["dsl"])["input_ids"])
              for r in rows[:tok_n]]
        out.append("  tokens (n=%d): input mean %.1f max %d | target mean %.1f "
                   "p95 %d max %d" %
                   (len(li), sum(li) / len(li), max(li), sum(lo) / len(lo),
                    sorted(lo)[int(0.95 * len(lo))], max(lo)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = None
    if not a.no_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("t5-small")
    txt = "\n\n".join(report(p, tokenizer=tok) for p in a.paths)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
