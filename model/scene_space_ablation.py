"""
scene_space_ablation.py - the "+ spacing pass" half of the Attempt 23 ablation,
computed post-hoc from a saved scene_eval run.

The spacing pass (scene_constrain.space_out) is applied AFTER decoding and
changes nothing about generation, so re-running the model with --space would
produce byte-identical raw strings. Rather than pay for four more beam-search
passes over 174 prompts, this replays the saved `raw` strings through exactly
the pipeline SceneWriter.write() uses when `space > 0`:

    dsl_to_spec(raw) -> space_out_spec() -> repair() -> stage

`--verify` checks that against a live SceneWriter on a handful of prompts, so
the equivalence is demonstrated rather than asserted.

    USE_TF=0 python model/scene_space_ablation.py \\
        model/eval_out3/in_distribution_v3.json --space 1.25
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
import scene_constrain as SC
import scene_staging as ST

KEYS = ("staging", "sep_score", "span_score", "occl_score",
        "decl_min_sep", "min_sep", "mean_sep", "span_frac", "occl_frac")


def _decl_min_sep(m):
    """Minimum pairwise distance between the DECLARED cast positions - what
    the spacing pass actually operates on. `min_sep` is the minimum over the
    whole clip, which walk destinations can still ruin; the two differing is
    itself a result."""
    xs = m["declared_x"]
    return min(abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:])


def replay(path, space=1.25):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    before, after, moved, layout_b, layout_a = [], [], 0, [0, 0], [0, 0]
    rows = []
    for r in d["rows"]:
        if not r.get("parsed"):
            continue
        try:
            spec = SG.dsl_to_spec(r["raw"])
        except SG.DSLError:
            continue
        b_spec, _ = SG.repair(json.loads(json.dumps(spec)))
        a_spec, changed = SC.space_out_spec(json.loads(json.dumps(spec)),
                                            min_gap=space)
        a_spec, _ = SG.repair(a_spec)
        mb = ST.stage_spec(b_spec, n_samples=41)
        ma = ST.stage_spec(a_spec, n_samples=41)
        for m in (mb, ma):
            m["decl_min_sep"] = _decl_min_sep(m) if m["n_cast"] > 1 else float("nan")
        moved += bool(changed)
        if not math.isnan(mb["staging"]):
            before.append(mb)
            after.append(ma)
        rows.append({"prompt": r["prompt"], "changed": changed,
                     "before": mb["staging"], "after": ma["staging"]})
    return d, before, after, moved, rows


def agg(ms):
    return {k: round(sum(m[k] for m in ms) / len(ms), 3) for k in KEYS} if ms \
        else {}


def verify(path, ckpt, space, k=4, **wkw):
    """Demonstrate the equivalence rather than asserting it: run a live
    SceneWriter with space=<space> on the first k prompts of a saved run and
    check the resulting cast x values match the offline replay exactly."""
    from scene_infer import SceneWriter
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    w = SceneWriter(ckpt, space=space, **wkw)
    ok = 0
    for r in d["rows"][:k]:
        live = w.write(r["prompt"])
        spec = SG.dsl_to_spec(r["raw"])
        off, _ = SC.space_out_spec(spec, min_gap=space)
        off, _ = SG.repair(off)
        a = [c["x"] for c in live["spec"]["cast"]]
        b = [c["x"] for c in off["cast"]]
        same = (a == b)
        ok += same
        print("  %-46s live=%s offline=%s %s"
              % (r["prompt"][:46], a, b, "OK" if same else "MISMATCH"))
    print("  equivalence: %d/%d" % (ok, k))
    return ok == k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--space", type=float, default=1.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-ckpt", default=None)
    ap.add_argument("--max-out", type=int, default=176)
    ap.add_argument("--max-in", type=int, default=104)
    a = ap.parse_args()
    if a.verify_ckpt:
        print("equivalence check against a live SceneWriter:")
        verify(a.paths[0], a.verify_ckpt, a.space,
               max_out=a.max_out, max_in=a.max_in)
        print()
    lines = []
    dump = {}
    for p in a.paths:
        d, before, after, moved, rows = replay(p, a.space)
        b, af = agg(before), agg(after)
        lines.append("%s  (min_gap=%s)" % (Path(p).name, a.space))
        lines.append("  spacing pass moved a character in %d/%d scenes"
                     % (moved, len(d["rows"])))
        lines.append("  %-8s %s" % ("", "  ".join("%9s" % k for k in KEYS)))
        lines.append("  %-8s %s" % ("before", "  ".join("%9s" % b[k] for k in KEYS)))
        lines.append("  %-8s %s" % ("after", "  ".join("%9s" % af[k] for k in KEYS)))
        gb = sum(m["staging"] >= 0.825 for m in before)
        ga = sum(m["staging"] >= 0.825 for m in after)
        lines.append("  staging >= 0.825 : before %d/%d = %.1f%%   after %d/%d = %.1f%%"
                     % (gb, len(before), 100 * gb / max(1, len(before)),
                        ga, len(after), 100 * ga / max(1, len(after))))
        dump[Path(p).name] = {"before": b, "after": af, "moved": moved,
                              "n_multicast": len(before), "rows": rows}
    txt = "\n\n".join(["\n".join(lines)])
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dump, indent=1), encoding="utf-8")
        Path(a.out).with_suffix(".txt").write_text(txt, encoding="utf-8")
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
