"""
scene_eval.py - score the scene-writing model on the things that matter.

Loss is not a metric here. Attempt 2's lesson - and this project's standing
one - is that a structured target can look converged and still emit something
that will not parse. So this measures, on prompts the model never saw:

  parse rate      does the raw string parse under scene_grammar.dsl_to_spec
  valid rate      does the parsed spec pass validate() with ZERO problems
  render rate     does scenescript.py build and render frames from it
  prompt match    does the scene say what the prompt asked for, per field

and reports the in-distribution set and the out-of-distribution set
separately, because the gap between them is the actual result.

    USE_TF=0 python model/scene_eval.py --ckpt model/scene_ckpt --n 150
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import scene_grammar as SG
import scene_staging as ST
from scene_infer import SceneWriter


# ------------------------------------------------------------ prompt match #

def match(spec, intent):
    """Score a spec against ONLY what the prompt actually said."""
    f = {}
    if intent.get("n_cast") is not None:
        f["cast_count"] = len(spec["cast"]) == intent["n_cast"]
    if intent.get("bg"):
        f["background"] = spec["bg"] == intent["bg"]
    acts_by_i = defaultdict(set)
    ids = [c["id"] for c in spec["cast"]]
    idx = {cid: i for i, cid in enumerate(ids)}
    all_acts = set()
    for e in spec["timeline"]:
        all_acts.add(e["act"])
        if e["who"] in idx:
            acts_by_i[idx[e["who"]]].add(e["act"])
    # per-character actions (in-distribution intents carry these)
    per = intent.get("acts") or {}
    if per:
        hit = tot = 0
        for k, want in per.items():
            i = int(k)
            for a in set(want):
                tot += 1
                hit += a in acts_by_i.get(i, set())
        f["actions_per_character"] = (hit == tot)
        f["_actions_frac"] = hit / max(1, tot)
    # actions that must appear somewhere (out-of-distribution intents)
    any_ = intent.get("acts_any") or []
    if any_:
        hit = sum(a in all_acts for a in any_)
        f["actions_present"] = hit == len(any_)
        f["_actions_frac"] = hit / len(any_)
    if intent.get("kick") is not None:
        f["ball_interaction"] = bool(intent["kick"]) == ("kick" in all_acts)
    if intent.get("props"):
        kinds = set(p["kind"] for p in spec["props"])
        f["props"] = all(k in kinds for k in intent["props"])
    # Attempt 23: spatial instructions the prompt actually gave. Each fact is
    # machine-checkable against the emitted spec, so "did it stage what it was
    # told to stage" is a rate and not an impression.
    lay = intent.get("layout") or []
    if lay:
        ok = sum(SG.check_layout_fact(spec, fact) for fact in lay)
        f["layout"] = (ok == len(lay))
        f["_layout_frac"] = ok / len(lay)
    if intent.get("colors"):
        ok = True
        for k, want in intent["colors"].items():
            i = int(k)
            ok &= i < len(spec["cast"]) and spec["cast"][i]["color"] == want
        f["colors"] = ok
    f["ALL"] = all(v for k, v in f.items() if not k.startswith("_"))
    return f


# ---------------------------------------------------------------- rendering #

def render_check(scene_text, n_frames=3, size=(320, 180)):
    """Build the scene through scenescript.py and rasterise a few frames.
    Small size on purpose: this asks whether it renders, not how it looks."""
    import scenescript as SS
    spec = SS.parse_script(scene_text)
    spec["width"], spec["height"] = size
    sc = SS.Scene(spec)
    rast = SS._Raster(sc)
    bg = rast.colour(sc._bg_layers)
    for i in range(n_frames):
        t = sc.seconds * (i + 0.5) / n_frames
        SS.compose_frame(sc, rast, bg, t)
    return sc


def run(writer, rows, label, out_dir, render=True):
    res = {"label": label, "n": len(rows), "rows": []}
    t0 = time.time()
    for i, r in enumerate(rows):
        rec = writer.write(r["prompt"])
        row = {"prompt": r["prompt"], "group": r.get("group", "in-dist"),
               "raw": rec["raw"], "parsed": bool(rec["parsed"]),
               "parse_error": rec["parse_error"],
               "problems": rec["problems"], "repairs": rec["repairs"]}
        if rec["parsed"]:
            row["valid"] = (len(rec["problems"]) == 0)
            row["dsl_exact"] = (rec["raw"].strip() == r.get("dsl", "").strip())
            row["match"] = match(rec["spec"], r["intent"])
            row["spacing"] = rec.get("spacing")
            try:
                row["staging"] = ST.stage_spec(rec["spec"], n_samples=41)
            except Exception as e:
                row["staging"] = {"error": "%s: %s" % (type(e).__name__, e)}
            if render:
                try:
                    render_check(rec["scene"])
                    row["render"] = True
                except Exception as e:
                    row["render"] = False
                    row["render_error"] = "%s: %s" % (type(e).__name__, e)
            row["scene"] = rec["scene"]
        res["rows"].append(row)
        if (i + 1) % 10 == 0:
            print("  %s %d/%d  (%.1fs)" % (label, i + 1, len(rows),
                                           time.time() - t0), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s.json" % label)).write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    return res


def summarise(res, by_group=False):
    rows = res["rows"]
    n = len(rows)
    p = sum(r["parsed"] for r in rows)
    v = sum(r.get("valid", False) for r in rows)
    rd = sum(r.get("render", False) for r in rows)
    ex = sum(r.get("dsl_exact", False) for r in rows)
    fields = defaultdict(lambda: [0, 0])
    frac = []
    for r in rows:
        m = r.get("match") or {}
        for k, val in m.items():
            if k.startswith("_"):
                continue
            fields[k][1] += 1
            fields[k][0] += bool(val)
        if "_actions_frac" in m:
            frac.append(m["_actions_frac"])
    out = ["%-22s n=%d" % (res["label"], n),
           "  parse   %3d/%d = %5.1f%%" % (p, n, 100 * p / n),
           "  valid   %3d/%d = %5.1f%%  (zero structural problems, no repair)"
           % (v, n, 100 * v / n),
           "  render  %3d/%d = %5.1f%%  (after repair)" % (rd, n, 100 * rd / n),
           "  exact   %3d/%d = %5.1f%%  (byte-identical to the reference DSL)"
           % (ex, n, 100 * ex / n)]
    for k in sorted(fields):
        a, b = fields[k]
        out.append("  %-22s %3d/%-3d = %5.1f%%" % (k, a, b, 100 * a / max(1, b)))
    if frac:
        out.append("  action recall (partial credit) = %5.1f%%"
                   % (100 * sum(frac) / len(frac)))
    # --- Attempt 23: staging, over the multi-character outputs only
    st = [r["staging"] for r in rows
          if isinstance(r.get("staging"), dict) and "staging" in r["staging"]
          and not math.isnan(r["staging"]["staging"])]
    if st:
        def mn(k):
            return sum(x[k] for x in st) / len(st)
        out.append("  STAGING (n=%d multi-cast of %d): staging %.3f  "
                   "[sep %.3f span %.3f occl %.3f]  min_sep %.2f  "
                   "mean_sep %.2f  span_frac %.3f  occl_frac %.3f"
                   % (len(st), n, mn("staging"), mn("sep_score"),
                      mn("span_score"), mn("occl_score"), mn("min_sep"),
                      mn("mean_sep"), mn("span_frac"), mn("occl_frac")))
        good = sum(x["staging"] >= 0.825 for x in st)
        out.append("  staging >= 0.825 (worst hand-written scene): %d/%d = %.1f%%"
                   % (good, len(st), 100 * good / len(st)))
    nsp = sum(1 for r in rows if r.get("spacing"))
    if nsp:
        out.append("  spacing pass moved a character in %d/%d = %.1f%% of scenes"
                   % (nsp, n, 100 * nsp / n))
    if by_group:
        g = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            k = r["group"]
            g[k][2] += 1
            g[k][0] += r.get("render", False)
            g[k][1] += bool((r.get("match") or {}).get("ALL"))
        out.append("  by group (render / prompt-match / n):")
        for k in sorted(g):
            a, b, c = g[k]
            out.append("    %-12s %d/%d   %d/%d" % (k, a, c, b, c))
    errs = Counter(r["parse_error"] for r in rows if r["parse_error"])
    if errs:
        out.append("  parse errors: %s" % dict(errs))
    probs = Counter()
    for r in rows:
        for p_ in (r.get("problems") or []):
            probs[p_.split("=")[0].split(" x")[0]] += 1
    if probs:
        out.append("  structural problems: %s" % dict(probs.most_common(8)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/scene_ckpt")
    ap.add_argument("--val", default="model/data/scene_val.jsonl")
    ap.add_argument("--ood", default="model/ood_prompts.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="model/eval_out")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--max-out", type=int, default=176)
    ap.add_argument("--max-in", type=int, default=104)
    ap.add_argument("--unique-colours", action="store_true")
    ap.add_argument("--space", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--rerank", type=int, default=0)
    a = ap.parse_args()

    w = SceneWriter(a.ckpt, max_out=a.max_out, max_in=a.max_in,
                    unique_colours=a.unique_colours, space=a.space,
                    sample=a.sample, top_p=a.top_p,
                    temperature=a.temperature, seed=a.seed, rerank=a.rerank)
    out = Path(a.out)
    val = [json.loads(l) for l in
           Path(a.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    val = val[:a.n]
    ood = json.loads(Path(a.ood).read_text(encoding="utf-8"))

    tag = a.tag
    r1 = run(w, val, "in_distribution" + tag, out, render=not a.no_render)
    r2 = run(w, ood, "out_of_distribution" + tag, out, render=not a.no_render)
    hdr = ("ckpt=%s  unique_colours=%s  space=%s  max_in=%d max_out=%d\n"
           % (a.ckpt, a.unique_colours, a.space, a.max_in, a.max_out))
    if w.colour_proc is not None:
        hdr += "colour processor masked at %d decode steps\n" % w.colour_proc.hits
    txt = hdr + "\n" + summarise(r1) + "\n\n" + summarise(r2, by_group=True)
    print("\n" + txt)
    (out / ("summary%s.txt" % tag)).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
