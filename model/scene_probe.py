"""
scene_probe.py - the controlled cast-count probe, extended past Attempt 22.

Attempt 22 §5 isolated the counting failure by holding the *stated numeral* and
the *number of described characters* independent and asking for the cast size.
v1 emitted the clause count 15/15 times and ignored the numeral; v2, trained on
deconfounded data, read the numeral 14/15 times.

Attempt 23 extends the numeral range to 9, because the training range is now
1-8 (paper/RESEARCH2.md item 3) and the interesting question has split in two:

  * 6, 7, 8 are now INSIDE the trained range - does the deconfounding fix keep
    working there, or was it specific to small casts?
  * 9 is one step OUTSIDE it - RESEARCH2 §7 predicts it reproduces the same
    saturation seen at 6 in Attempt 22 rather than generalising. If 9 works,
    that contradicts the survey's reading of the compositional-generalisation
    literature and is its own result.

    USE_TF=0 python model/scene_probe.py --ckpt model/scene_ckpt3 \\
        --out model/eval_out3/clause_probe3.json
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
from scene_infer import SceneWriter

WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine"}
CLAUSE = ["one waves", "another jumps", "the third dances",
          "the fourth walks across", "the fifth just stands there"]
MAX_CLAUSES = 5


def probe(w, numerals=range(1, 10)):
    out = []
    for k in numerals:
        for c in range(1, min(k, MAX_CLAUSES) + 1):
            p = "%s people in the park: %s" % (WORD[k], ", then ".join(CLAUSE[:c]))
            raw = w.raw(p)
            try:
                n = len(SG.dsl_to_spec(raw)["cast"])
            except SG.DSLError as e:
                n = None
            out.append({"numeral": k, "clauses": c, "prompt": p,
                        "cast": n, "raw": raw})
            print("  numeral %d, %d clause(s) -> cast %s" % (k, c, n), flush=True)
    return out


def table(rows, numerals=range(1, 10)):
    cs = sorted(set(r["clauses"] for r in rows))
    ln = ["| stated numeral | " + " | ".join(str(c) for c in cs) + " |",
          "|---|" + "--:|" * len(cs)]
    for k in numerals:
        cells = []
        for c in cs:
            m = [r for r in rows if r["numeral"] == k and r["clauses"] == c]
            if not m:
                cells.append("")
            else:
                v = m[0]["cast"]
                cells.append("**%s**" % v if v == k else str(v))
        ln.append("| \"%s people...\" | " % WORD[k] + " | ".join(cells) + " |")
    hit = sum(r["cast"] == r["numeral"] for r in rows)
    ln.append("")
    ln.append("numeral read correctly: %d/%d" % (hit, len(rows)))
    for lo, hi, name in ((1, 5, "1-5, the Attempt 22 range"),
                         (6, 8, "6-8, new in the Attempt 23 range"),
                         (9, 9, "9, one past the trained ceiling")):
        m = [r for r in rows if lo <= r["numeral"] <= hi]
        if m:
            ln.append("  %-34s %d/%d" % (name,
                      sum(r["cast"] == r["numeral"] for r in m), len(m)))
    return "\n".join(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/scene_ckpt3")
    ap.add_argument("--out", default="model/eval_out3/clause_probe3.json")
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--max-in", type=int, default=160)
    ap.add_argument("--max-numeral", type=int, default=9)
    a = ap.parse_args()
    w = SceneWriter(a.ckpt, max_out=a.max_out, max_in=a.max_in)
    nums = range(1, a.max_numeral + 1)
    rows = probe(w, nums)
    txt = table(rows, nums)
    print("\n" + txt)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ckpt": a.ckpt, "rows": rows, "table": txt},
                            indent=1), encoding="utf-8")
    p.with_suffix(".txt").write_text(txt, encoding="utf-8")
    print("-> %s" % p)


if __name__ == "__main__":
    main()
