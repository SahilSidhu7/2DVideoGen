"""
scene_constrain.py - the two inference-time passes from paper/RESEARCH2.md.

Both fix a defect Attempt 22 measured and left to `SG.repair()`. Both are
optional and off by default, so every number can be reported with and without
them - which is the point, because a post-decode pass that fixes the output is
NOT the model learning to do the thing, and the write-up must not blur those.

1. `UniqueColourProcessor` (RESEARCH2 §1 item 1, §4).
   A `transformers.LogitsProcessor` that masks colour tokens already used by an
   earlier cast member, so "no two characters share a colour" is enforced *during*
   decoding instead of patched afterwards. The constraint is a trivial DFA -
   state is the subset of ten colours emitted so far, at most 2^10 states - which
   is the class automata-guided constrained decoding (ABS, arXiv 2506.09701)
   handles exactly rather than probabilistically. Hand-written rather than pulled
   from Outlines/Guidance because it is one constraint over a ten-word vocabulary.

   The tokenizer detail that makes this fiddly: t5 encodes "cyan" as
   ['_', 'cyan'] - its first token is the *generic* space piece (id 3), shared
   with everything else, so banning first tokens naively would ban the space.
   The processor therefore bans, per colour, the first token in its sequence
   that is not that generic piece, keyed on how much of the word has already
   been emitted. All ten colours have distinct distinctive tokens (checked).

2. `space_out()` (RESEARCH2 §1 item 2, §2).
   A deterministic 1-D pass over the emitted cast x-coordinates that enforces a
   minimum pairwise separation and the frame margins, preserving the left-right
   ORDER the model chose (so any spatial instruction it obeyed survives) and
   moving positions as little as the constraint allows.

   It deliberately does NOT stretch the cast to a target span. RESEARCH2
   describes this pass as "minimum pairwise spacing + frame margins", and
   keeping it to that leaves `span_score` entirely attributable to the model -
   which is what makes the four-cell ablation in ATTEMPTS.md Attempt 23 mean
   anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_grammar as SG

GENERIC_SPACE_PIECE = "▁"          # sentencepiece's word-boundary marker


# ------------------------------------------------------- 1. unique colours #

def build_colour_bans(tok):
    """{colour: [(partial_text_already_emitted, token_id_to_ban), ...]}."""
    generic = tok.convert_tokens_to_ids(GENERIC_SPACE_PIECE)
    bans = {}
    for c in SG.CAST_COLORS:
        ids = tok(c, add_special_tokens=False)["input_ids"]
        out = []
        for k, tid in enumerate(ids):
            if tid == generic:
                continue                # cannot ban the bare space piece
            out.append((tok.decode(ids[:k]).strip(), tid))
        bans[c] = out
    return bans


def _colour_state(text):
    """(used_colours, partial) for a decoded prefix, or (…, None) if the next
    token is not part of a cast member's colour."""
    if "cast " not in text:
        return set(), None
    seg = text.split("cast ", 1)[1]
    if "|" in seg:
        return set(), None              # the cast section is already closed
    entries = seg.split(";")
    used = set()
    for e in entries[:-1]:
        w = e.split()
        if len(w) >= 2 and w[1] in SG.CAST_COLORS:
            used.add(w[1])
    cur = entries[-1].split()
    if len(cur) == 1:
        return used, ""                 # id emitted, colour is next
    if len(cur) == 2 and cur[1] not in SG.CAST_COLORS:
        return used, cur[1]             # colour part-way emitted
    return used, None


try:
    from transformers import LogitsProcessor as _LP
except Exception:                        # keep the module importable without torch
    _LP = object


class UniqueColourProcessor(_LP):
    """Mask colours already used by an earlier cast member in the same scene."""

    def __init__(self, tok):
        self.tok = tok
        self.bans = build_colour_bans(tok)
        self.hits = 0                    # how many times it actually masked

    def __call__(self, input_ids, scores):
        import torch
        for b in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[b], skip_special_tokens=True)
            used, partial = _colour_state(text)
            if partial is None or not used:
                continue
            ban = [tid for c in used for (p, tid) in self.bans[c] if p == partial]
            if ban:
                self.hits += 1
                scores[b, torch.tensor(ban, device=scores.device)] = -float("inf")
        return scores


# --------------------------------------------------------- 2. spacing pass #

def space_out(xs, min_gap=1.25, xmin=SG.XMIN, xmax=SG.XMAX, iters=200):
    """Push 1-D positions apart to `min_gap`, preserving their order, inside
    [xmin, xmax]. Returns a new list in the ORIGINAL input order."""
    n = len(xs)
    if n < 2:
        return [min(xmax, max(xmin, x)) for x in xs]
    gap = min(min_gap, (xmax - xmin) / (n - 1))   # always feasible
    order = sorted(range(n), key=lambda i: xs[i])
    s = [xs[i] for i in order]
    for _ in range(iters):
        moved = False
        for i in range(1, n):
            d = gap - (s[i] - s[i - 1])
            if d > 1e-9:
                s[i - 1] -= d / 2.0
                s[i] += d / 2.0
                moved = True
        # frame margins, then re-run: clamping can re-create a violation
        if s[0] < xmin:
            shift = xmin - s[0]
            s = [v + shift for v in s]
            moved = True
        if s[-1] > xmax:
            shift = s[-1] - xmax
            s = [v - shift for v in s]
            moved = True
        if not moved:
            break
    out = [0.0] * n
    for k, i in enumerate(order):
        out[i] = round(min(xmax, max(xmin, s[k])), 1)
    return out


def space_out_spec(spec, min_gap=1.25):
    """Apply the spacing pass to a spec in place. -> (spec, [what changed])."""
    xs = [c["x"] for c in spec["cast"]]
    new = space_out(xs, min_gap=min_gap)
    changed = []
    for c, a, b in zip(spec["cast"], xs, new):
        if abs(a - b) > 0.05:
            changed.append("%s x %s->%s" % (c["id"], a, b))
        c["x"] = b
    return spec, changed


if __name__ == "__main__":
    # the Attempt 22 deliverable's stack, and a five-way pile-up
    print(space_out([9.6, 9.2]))
    print(space_out([9.6, 9.2, 9.4, 9.5, 9.3]))
    print(space_out([1.0, 5.0, 9.0]))
    print(space_out([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
