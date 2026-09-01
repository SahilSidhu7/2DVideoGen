"""
scene_infer.py - natural-language prompt -> a .scene file scenescript.py renders.

    USE_TF=0 python model/scene_infer.py "two friends meet in the park, one
        waves, then they kick a ball around" -o scenes/gen_park.scene

Returns the raw model string as well as the parsed/repaired spec, because the
Attempt 22 evaluation reports the *raw* parse and validity rates separately
from the post-repair render rate. Nothing here silently launders a bad output.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")     # transformers 4.57 + Keras 3 dies

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_grammar as SG
import scene_constrain as SC

PREFIX = "prompt2scene: "


class SceneWriter:
    """Attempt 23 adds two OPTIONAL inference-time passes, both off by default
    so that every rate can be quoted with and without them:

      unique_colours  a logits processor that cannot emit a colour an earlier
                      cast member already took (paper/RESEARCH2.md item 1)
      space           a deterministic 1-D minimum-separation pass over the
                      emitted x-coordinates, 0 disables (RESEARCH2 item 2)
    """

    def __init__(self, ckpt="model/scene_ckpt", beams=4, max_out=176,
                 max_in=104, unique_colours=False, space=0.0,
                 sample=False, top_p=0.95, temperature=1.0, seed=23,
                 rerank=0):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model.eval()
        self.beams = beams
        self.max_out = max_out
        self.max_in = max_in
        self.space = float(space)
        # Attempt 23 §6: beam search reports the mode of an unconstrained slot,
        # which is what collapsed every position in Attempt 22. Sampling is the
        # alternative and is measured, not assumed, in scene_eval.
        self.sample = bool(sample)
        # Attempt 23 §9: sampling buys staging and costs `valid`. rerank=k
        # draws k samples and keeps the FIRST with the fewest structural
        # problems. The tie-break is draw order, NOT the staging score, so the
        # staging number stays an unbiased read of the model rather than
        # something selected for.
        self.rerank = int(rerank)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        if sample:
            torch.manual_seed(seed)
        self.colour_proc = SC.UniqueColourProcessor(self.tok)             if unique_colours else None

    def raw(self, prompt, beams=None):
        enc = self.tok(PREFIX + prompt, return_tensors="pt",
                       truncation=True, max_length=self.max_in)
        kw = {}
        if self.colour_proc is not None:
            from transformers import LogitsProcessorList
            kw["logits_processor"] = LogitsProcessorList([self.colour_proc])
        if self.sample:
            kw.update(do_sample=True, num_beams=1, top_p=self.top_p,
                      temperature=self.temperature)
        else:
            kw.update(num_beams=beams or self.beams, early_stopping=True)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_length=self.max_out, **kw)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def raw_reranked(self, prompt):
        best, best_n, tried = None, None, []
        for _ in range(max(1, self.rerank)):
            raw = self.raw(prompt)
            tried.append(raw)
            try:
                n = len(SG.validate(SG.dsl_to_spec(raw)))
            except SG.DSLError:
                n = 999
            if best_n is None or n < best_n:
                best, best_n = raw, n
            if best_n == 0:
                break
        return best, tried

    def write(self, prompt, title=None):
        """-> dict with every stage kept, so the failure mode is visible."""
        if self.rerank > 1:
            raw, tried = self.raw_reranked(prompt)
        else:
            raw, tried = self.raw(prompt), None
        r = {"prompt": prompt, "raw": raw, "n_drawn": len(tried or [1]), "parsed": None, "parse_error": None,
             "problems": None, "repairs": None, "spec": None, "scene": None}
        try:
            spec = SG.dsl_to_spec(raw)
        except SG.DSLError as e:
            r["parse_error"] = str(e)
            return r
        r["parsed"] = True
        r["problems"] = SG.validate(spec)
        r["spacing"] = []
        if self.space > 0:
            # after validate() - spacing is staging, not structural validity -
            # and before repair(), so repair sees the final positions.
            spec, r["spacing"] = SC.space_out_spec(spec, min_gap=self.space)
        spec2, fixed = SG.repair(spec)
        r["repairs"] = fixed
        r["spec"] = spec2
        r["scene"] = SG.spec_to_scene(spec2, title or prompt[:60])
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--ckpt", default="model/scene_ckpt")
    ap.add_argument("-o", "--out", default=None, help="write a .scene file")
    ap.add_argument("--max-out", type=int, default=176)
    ap.add_argument("--max-in", type=int, default=104)
    ap.add_argument("--unique-colours", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--rerank", type=int, default=0,
                    help="draw k samples, keep the first with zero structural "
                         "problems (tie-break = draw order, not staging)")
    ap.add_argument("--space", type=float, default=0.0,
                    help="minimum pairwise x separation to enforce (0 = off)")
    a = ap.parse_args()
    w = SceneWriter(a.ckpt, max_out=a.max_out, max_in=a.max_in,
                    unique_colours=a.unique_colours, space=a.space,
                    sample=a.sample, top_p=a.top_p,
                    temperature=a.temperature, seed=a.seed, rerank=a.rerank)
    r = w.write(a.prompt)
    print("raw:", r["raw"])
    if r["parse_error"]:
        print("PARSE FAILED:", r["parse_error"])
        raise SystemExit(1)
    print("problems:", r["problems"])
    print("spacing :", r.get("spacing"))
    print("repairs :", r["repairs"])
    print(r["scene"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(r["scene"], encoding="utf-8")
        print("->", a.out)


if __name__ == "__main__":
    main()
