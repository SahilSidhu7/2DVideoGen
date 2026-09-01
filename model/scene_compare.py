"""
scene_compare.py - the Attempt 23 master table, built from saved scene_eval runs.

Two things it does that `scene_eval.summarise()` cannot:

* **`ALL_nolayout`.** v3's in-distribution val set carries a field v2's does not
  (`layout`, the stated spatial instruction), so a raw `ALL` comparison between
  them is not like-for-like - v3 is being scored on a question v2 was never
  asked. This recomputes "every stated field correct at once" with the layout
  field excluded, so v2 and v3 can be put in the same column honestly. Both
  numbers are reported.
* the "+ spacing pass" cell, replayed post-hoc (see scene_space_ablation.py).

    USE_TF=0 python model/scene_compare.py model/eval_out3 --out model/eval_out3/compare.txt
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
import scene_constrain as SC
import scene_staging as ST


def _rate(a, b):
    return float("nan") if not b else 100.0 * a / b


def cell(path, space=1.25):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    n = len(rows)
    o = {"file": Path(path).name, "n": n}
    o["parse"] = _rate(sum(r["parsed"] for r in rows), n)
    o["valid"] = _rate(sum(r.get("valid", False) for r in rows), n)
    o["render"] = _rate(sum(r.get("render", False) for r in rows), n)

    fields = {}
    allx = [0, 0]
    for r in rows:
        m = r.get("match") or {}
        if not m:
            continue
        for k, v in m.items():
            if k.startswith("_") or k == "ALL":
                continue
            a = fields.setdefault(k, [0, 0])
            a[1] += 1
            a[0] += bool(v)
        allx[1] += 1
        allx[0] += all(v for k, v in m.items()
                       if not k.startswith("_") and k not in ("ALL", "layout"))
    for k, (a, b) in fields.items():
        o[k] = _rate(a, b)
    o["ALL"] = _rate(sum(bool((r.get("match") or {}).get("ALL")) for r in rows),
                     sum(1 for r in rows if r.get("match")))
    o["ALL_nolayout"] = _rate(allx[0], allx[1])

    prob = Counter()
    for r in rows:
        for p in (r.get("problems") or []):
            prob[p.split("=")[0].split(" x")[0]] += 1
    o["dup_colour"] = prob.get("two characters share a colour", 0)
    o["problems"] = dict(prob.most_common(6))

    # staging, before and after the spacing pass
    for tag, use_space in (("", False), ("_spaced", True)):
        ms = []
        moved = 0
        for r in rows:
            if not r.get("parsed"):
                continue
            try:
                sp = SG.dsl_to_spec(r["raw"])
            except SG.DSLError:
                continue
            if use_space:
                sp, ch = SC.space_out_spec(sp, min_gap=space)
                moved += bool(ch)
            sp, _ = SG.repair(sp)
            m = ST.stage_spec(sp, n_samples=41)
            if not math.isnan(m["staging"]):
                xs = m["declared_x"]
                m["decl_min_sep"] = min(abs(a - b) for i, a in enumerate(xs)
                                        for b in xs[i + 1:])
                ms.append(m)
        if ms:
            for k in ("staging", "sep_score", "span_score", "occl_score",
                      "decl_min_sep", "min_sep", "mean_sep", "span_frac",
                      "occl_frac"):
                o[k + tag] = round(sum(x[k] for x in ms) / len(ms), 3)
            o["staging_ok" + tag] = _rate(
                sum(x["staging"] >= 0.825 for x in ms), len(ms))
            o["n_multicast"] = len(ms)
        if use_space:
            o["spacing_moved"] = moved
    return o


ROWS = ["n", "n_multicast", "parse", "valid", "render", "cast_count",
        "background", "actions_per_character", "actions_present",
        "ball_interaction", "props", "colors", "layout", "ALL_nolayout", "ALL",
        "dup_colour",
        "staging", "sep_score", "span_score", "occl_score",
        "decl_min_sep", "min_sep", "span_frac", "occl_frac", "staging_ok",
        "staging_spaced", "sep_score_spaced", "span_score_spaced",
        "occl_score_spaced", "decl_min_sep_spaced", "min_sep_spaced",
        "span_frac_spaced", "occl_frac_spaced", "staging_ok_spaced",
        "spacing_moved"]


def table(cells, names):
    w = max(len(n) for n in names) + 2
    out = ["%-24s" % "" + "".join("%*s" % (w, n) for n in names)]
    for k in ROWS:
        if not any(k in c for c in cells):
            continue
        vals = []
        for c in cells:
            v = c.get(k)
            vals.append("-" if v is None else
                        ("%.1f" % v if isinstance(v, float) and abs(v) > 3
                         else ("%.3f" % v if isinstance(v, float) else str(v))))
        out.append("%-24s" % k + "".join("%*s" % (w, v) for v in vals))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="path=name pairs or plain paths")
    ap.add_argument("--space", type=float, default=1.25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paths, names = [], []
    for f in a.files:
        if "=" in f:
            p, nm = f.split("=", 1)
        else:
            p, nm = f, Path(f).stem
        paths.append(p)
        names.append(nm)
    cells = [cell(p, a.space) for p in paths]
    txt = table(cells, names)
    print(txt)
    for c, nm in zip(cells, names):
        print("\n%s problems: %s" % (nm, c["problems"]))
    if a.out:
        Path(a.out).write_text(txt + "\n\n" + json.dumps(
            {n: c for n, c in zip(names, cells)}, indent=1), encoding="utf-8")
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
