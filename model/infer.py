"""
infer.py - prompt -> scene dict, using the trained model with a rule fallback.

    from infer import PromptModel
    m = PromptModel("model/checkpoint")      # or None -> rules only
    spec, scene = m.scene("a slow glowing blue rose with 5 petals")

Robustness: the model output is always parsed and validated by grammar.py. If
the model emits an unusable archetype, we merge in the rule-based reading of the
prompt so the result is always renderable.
"""
from __future__ import annotations

from pathlib import Path

import grammar as G

PREFIX = "prompt2scene: "


class PromptModel:
    def __init__(self, ckpt="checkpoint"):
        self.ok = False
        self.tok = self.model = None
        if ckpt and Path(ckpt).exists():
            try:
                import torch
                from transformers import (AutoTokenizer,
                                          AutoModelForSeq2SeqLM)
                self.torch = torch
                self.tok = AutoTokenizer.from_pretrained(ckpt)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
                self.model.eval()
                self.ok = True
            except Exception as e:  # pragma: no cover
                print(f"[infer] model load failed ({e}); using rules only")

    def _generate(self, prompt: str) -> str:
        enc = self.tok(PREFIX + prompt, return_tensors="pt",
                       truncation=True, max_length=64)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_length=48, num_beams=4,
                                      early_stopping=True)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def spec(self, prompt: str) -> dict:
        rules = G.parse_prompt_rules(prompt)
        if not self.ok:
            return rules
        raw = self._generate(prompt)
        model_spec = G.dsl_to_spec(raw)
        # merge: model wins on archetype; fall back to rules for empty slots
        merged = dict(rules)
        merged.update({k: v for k, v in model_spec.items() if v not in (None, "")})
        # stickman is a keyword-detected special archetype the model was not
        # trained on, so a confident rule match must override the model.
        if rules.get("arch") in G.SPECIAL_ARCH:
            merged["arch"] = rules["arch"]
        return G.validate_spec(merged)

    def scene(self, prompt: str):
        spec = self.spec(prompt)
        return spec, G.build_scene(spec)


if __name__ == "__main__":
    import sys
    m = PromptModel(sys.argv[1] if len(sys.argv) > 1 else "checkpoint")
    tests = [
        "a slow glowing blue rose with 5 petals",
        "fast red spirograph",
        "water ripples in fire colors",
        "make a calm purple mandala with 8 fold symmetry",
        "energetic green sine waves",
        "dark minimal lissajous",
    ]
    for t in tests:
        spec = m.spec(t)
        print(f"{t!r}\n   -> {G.spec_to_dsl(spec)}")
