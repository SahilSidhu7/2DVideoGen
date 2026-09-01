"""verify_checkpoints.py - prove a preserved checkpoint still loads and still
emits the string Attempt 22 recorded for it.  Writes model/checkpoints/verify.json.

    USE_TF=0 python model/verify_checkpoints.py
"""
from __future__ import annotations
import os
os.environ.setdefault("USE_TF", "0")
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
import scene_grammar as SG
from scene_infer import SceneWriter

# the two prompts Attempt 22 published a raw string for, plus a control.
PROMPTS = [
 "two friends meet in the park, one waves, then they kick a ball around",
 "three friends in the park: all three dance at the same time, then two of them pass a ball while the third one just watches",
]
EXPECT = {  # section 8 of Attempt 22, v2 only
 "v2-deconfounded": {PROMPTS[0]:
   "bg park dur 91 | cast ana blue 96 100 ; bo amber 92 90 | "
   "prop ball 58 100 ; tree 69 110 | "
   "tl 1 26 bo walk 92 ; 8 28 ana wave ; 28 47 bo kick ball ana ; 57 79 bo walk 92"},
}

def main():
    ckdir = ROOT / "model" / "checkpoints"
    out = {}
    for name in sorted(p.name for p in ckdir.iterdir() if p.is_dir()):
        path = ckdir / name
        t0 = time.time()
        # v3 targets are longer; loading it at v2 sequence caps would
        # truncate and look like a checkpoint fault.
        kw = dict(max_out=256, max_in=160) if "v3" in name else {}
        w = SceneWriter(str(path), **kw)
        rec = {"load_seconds": round(time.time() - t0, 1), "prompts": {}}
        for p in PROMPTS:
            r = w.write(p)
            e = EXPECT.get(name, {}).get(p)
            rec["prompts"][p] = {
                "raw": r["raw"], "parsed": bool(r["parsed"]),
                "parse_error": r["parse_error"],
                "problems": r["problems"], "repairs": r["repairs"],
                "cast": None if not r["spec"] else
                        [(c["id"], c["color"], c["x"], c["scale"]) for c in r["spec"]["cast"]],
                "expected": e,
                "matches_attempt22": (None if e is None else r["raw"].strip() == e.strip()),
            }
            print("%-20s %-60s parsed=%s match=%s" %
                  (name, p[:60], bool(r["parsed"]),
                   rec["prompts"][p]["matches_attempt22"]), flush=True)
        out[name] = rec
    (ckdir / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("-> %s" % (ckdir / "verify.json"))

if __name__ == "__main__":
    main()
