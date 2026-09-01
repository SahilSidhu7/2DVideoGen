"""AniSVG <-> a hybrid token/value sequence, so numbers stop being digit strings.

Our measured failure (see ATTEMPTS.md, Attempt 15) was that colour and structure
were learned and geometry was not. The literature calls this coordinate
hallucination and traces it to the same cause: a coordinate written as text is
shredded into digit tokens, which destroys the numeric relationship between
"12" and "13" and spends many tokens doing it.

This encodes a clip as two parallel arrays:

    ids   [... 'S' '<NUM>' '<NUM>' '<NUM>' ...]   token ids, one per atom
    vals  [... 0.0   0.0     102.0   144.0  ...]  the number at that position

A number becomes ONE token whose value rides alongside it, following xVal
(Golkar et al., arXiv:2310.02989). Two things follow. The sequence gets much
shorter, because a three-digit coordinate stops costing three tokens. And the
model can be given a scalar head trained with a regression loss, so 12 and 13
are near-misses instead of unrelated symbols.

This module is the data half only - it is deliberately usable, and measurable,
before any model changes.
"""
import re

# Structural atoms of the grammar. Everything else in a clip is a number.
STRUCT = ["H", "P", "S", "@", "t", "r", "s", "o", "v", "h", "w", "\n"]
NUM = "<NUM>"

_NUMRE = re.compile(r"^-?\d+$")


def atoms(text):
    """Split a clip into atoms, preserving line breaks as their own atom."""
    out = []
    for line in text.split("\n"):
        f = line.split()
        if not f:
            continue
        out.extend(f)
        out.append("\n")
    return out


def encode(text, scale=100.0):
    """Clip -> (atom list, values), numbers replaced by NUM.

    `scale` divides values so the model sees roughly unit-magnitude numbers;
    canvas coordinates live in 0..256 and deltas in about -30..30.
    """
    toks, vals = [], []
    for a in atoms(text):
        if _NUMRE.match(a):
            toks.append(NUM)
            vals.append(float(a) / scale)
        elif a.startswith("#"):
            toks.append(a)                 # palette entries stay symbolic
            vals.append(0.0)
        else:
            toks.append(a)
            vals.append(0.0)
    return toks, vals


def decode(toks, vals, scale=100.0):
    """(atoms, values) -> clip text. Inverse of `encode`."""
    parts, line = [], []
    for t, v in zip(toks, vals):
        if t == "\n":
            if line:
                parts.append(" ".join(line))
            line = []
        elif t == NUM:
            line.append(str(int(round(v * scale))))
        else:
            line.append(t)
    if line:
        parts.append(" ".join(line))
    return "\n".join(parts)


def vocab():
    """The complete atom vocabulary: structure, NUM, and hex colours are open."""
    return list(STRUCT) + [NUM]


def stats(text, tok, scale=100.0):
    """Token cost under the current text encoding vs the hybrid encoding."""
    baseline = len(tok(text, add_special_tokens=False)["input_ids"])
    toks, _ = encode(text, scale)
    # One id per atom; hex colours are the only atoms that may need more than
    # one id, and there are at most a handful per clip.
    hybrid = sum(len(tok(t, add_special_tokens=False)["input_ids"])
                 if t.startswith("#") else 1 for t in toks)
    return baseline, hybrid


if __name__ == "__main__":
    import argparse, glob, gzip, json, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="svg/data/train_anime/train.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = []
    op = gzip.open if args.data.endswith(".gz") else open
    with op(args.data, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.n:
                break
            rows.append(json.loads(line))

    base_tot = hyb_tot = 0
    ok = 0
    for r in rows:
        b, h = stats(r["text"], tok)
        base_tot += b
        hyb_tot += h
        t2, v2 = encode(r["text"])
        if decode(t2, v2).strip() == r["text"].strip():
            ok += 1

    n = len(rows)
    print("%d clips" % n)
    print("text tokens      median-equivalent %8.0f  (mean %.0f/clip)"
          % (base_tot, base_tot / n))
    print("hybrid atoms                       %8.0f  (mean %.0f/clip)"
          % (hyb_tot, hyb_tot / n))
    print("reduction        %.2fx" % (base_tot / max(hyb_tot, 1)))
    print("round-trip exact %d/%d" % (ok, n))
