"""
scene_decode_probe.py - is the position collapse in the MODEL or in the SEARCH?

Attempt 22 and Attempt 23 both decode with `num_beams=4`. Beam search returns
the (approximately) most likely *sequence*, which for a slot the prompt does
not constrain is by definition the marginal mode - so "the model emits x=9.4
every time" is consistent with two very different situations:

  (a) the model's conditional distribution over x has collapsed, or
  (b) the distribution is fine and the decoder is picking its peak.

They call for opposite fixes - (a) needs data or architecture, (b) needs one
line of `generate()` - so it is worth ten minutes to tell them apart. This
decodes the same prompts four ways and scores the staging of each:

    beam4     what every number in Attempts 22 and 23 uses
    greedy    num_beams=1
    sample    top-p 0.95, temperature 1.0
    sample_lo top-p 0.9, temperature 0.7

    USE_TF=0 python model/scene_decode_probe.py --ckpt model/scene_ckpt3 \\
        --val model/data/scene3_val.jsonl --n 30
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import scene_grammar as SG
import scene_staging as ST
from scene_infer import SceneWriter, PREFIX

MODES = {
    "beam4":     dict(num_beams=4, early_stopping=True),
    "greedy":    dict(num_beams=1, do_sample=False),
    "sample":    dict(num_beams=1, do_sample=True, top_p=0.95, temperature=1.0),
    "sample_lo": dict(num_beams=1, do_sample=True, top_p=0.90, temperature=0.7),
}


def decode(w, prompt, **kw):
    enc = w.tok(PREFIX + prompt, return_tensors="pt", truncation=True,
                max_length=w.max_in)
    with w.torch.no_grad():
        out = w.model.generate(**enc, max_length=w.max_out, **kw)
    return w.tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/scene_ckpt3")
    ap.add_argument("--val", default="model/data/scene3_val.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--max-in", type=int, default=160)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="model/eval_out3/decode_probe.json")
    a = ap.parse_args()

    w = SceneWriter(a.ckpt, max_out=a.max_out, max_in=a.max_in)
    w.torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in
            Path(a.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:a.n]

    res = {m: {"staging": [], "parse": 0, "xs": []} for m in MODES}
    for i, r in enumerate(rows):
        for m, kw in MODES.items():
            raw = decode(w, r["prompt"], **kw)
            try:
                sp = SG.dsl_to_spec(raw)
            except SG.DSLError:
                continue
            res[m]["parse"] += 1
            sp2, _ = SG.repair(sp)
            st = ST.stage_spec(sp2, n_samples=25)
            res[m]["xs"].append(st["declared_x"])
            if not math.isnan(st["staging"]):
                res[m]["staging"].append(st["staging"])
        print("  %d/%d" % (i + 1, len(rows)), flush=True)

    lines = ["decode probe: %s, n=%d prompts" % (a.ckpt, len(rows)),
             "%-10s %7s %9s %11s %11s" %
             ("mode", "parse", "staging", "distinct x", "modal x share")]
    for m in MODES:
        d = res[m]
        allx = [x for xs in d["xs"] for x in xs]
        modal = max(set(allx), key=allx.count) if allx else None
        lines.append("%-10s %6d/%d %9.3f %11d %10.1f%%  (mode %.1f)"
                     % (m, d["parse"], len(rows),
                        sum(d["staging"]) / max(1, len(d["staging"])),
                        len(set(allx)),
                        100 * allx.count(modal) / max(1, len(allx)),
                        modal if modal is not None else float("nan")))
    txt = "\n".join(lines)
    print("\n" + txt)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ckpt": a.ckpt, "table": txt,
                             "res": {m: {k: v for k, v in d.items()}
                                     for m, d in res.items()}}, indent=1),
                 encoding="utf-8")
    p.with_suffix(".txt").write_text(txt, encoding="utf-8")
    print("-> %s" % p)


if __name__ == "__main__":
    main()
