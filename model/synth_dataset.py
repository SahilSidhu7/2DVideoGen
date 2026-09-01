"""
synth_dataset.py - build a synthetic (prompt -> DSL) dataset from the grammar.

There is no public dataset mapping English to math-animation specs, so we
generate one from geovid's own scene grammar. Each example samples an archetype
and a random subset of slots, renders a natural-language prompt with heavy
paraphrase variety, and pairs it with the canonical DSL target the model learns.

    out: data/train.jsonl , data/val.jsonl   ({"prompt":..., "dsl":...})
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import grammar as G

# reverse maps: canonical slot value -> list of prompt words that mean it
_COLOR_SAY = {}
for word, canon in G.COLOR_WORDS.items():
    _COLOR_SAY.setdefault(canon, []).append(word)
_CMAP_SAY = {}
for word, canon in G.CMAPS.items():
    _CMAP_SAY.setdefault(canon, []).append(word)
_SPEED_SAY = {}
for word, canon in G.SPEED_WORDS.items():
    _SPEED_SAY.setdefault(canon, []).append(word)
_MOOD_SAY = {}
for word, canon in G.MOOD_WORDS.items():
    _MOOD_SAY.setdefault(canon, []).append(word)

COUNT_NOUNS = {
    "rose": ["petals", "leaves", "lobes"],
    "mandala": ["fold", "points", "spikes"],
    "superformula": ["sides", "points", "arms"],
    "lissajous": ["lobes", "loops"],
    "spiral": ["arms"],
}

VERBS = ["generate", "create", "make", "render", "draw", "produce", "show me",
         "give me", "animate", "i want", "build", "design"]
ARTICLES = ["a", "an", "the", ""]
CLOSERS = ["", "", "video", "animation", "clip", "scene", "on a graph",
           "plotted on a graph", "loop"]
MOTION_TAIL = {
    "slow": ["that slowly morphs", "drifting gently", "evolving slowly",
             "that changes calmly", "with a gentle motion"],
    "med": ["that animates", "in motion", "that moves", "steadily shifting"],
    "fast": ["spinning fast", "that rapidly changes", "with energetic motion",
             "pulsing quickly", "moving fast"],
}


def _pick(lst):
    return random.choice(lst)


def sample_spec():
    arch = _pick(list(G.ARCH_NAMES))
    spec = {"arch": arch}
    is_field = arch in G.FIELD_ARCH

    if random.random() < 0.75:
        if is_field:
            spec["cmap"] = _pick(list(G.CMAPS.values()))
        else:
            spec["color"] = _pick(list(G.COLORS))
    if random.random() < 0.6:
        spec["speed"] = _pick(["slow", "med", "fast"])
    if random.random() < 0.4:
        spec["mood"] = _pick(["glow", "dark", "minimal"])
    if not is_field and arch in COUNT_NOUNS and random.random() < 0.5:
        spec["sym"] = random.randint(3, 9)
    elif arch == "mandala" and random.random() < 0.5:
        spec["sym"] = random.randint(4, 9)
    return G.validate_spec(spec)


def render_prompt(spec):
    """Turn a spec into a natural-language prompt with random surface form."""
    arch = spec["arch"]
    name = _pick(G.ARCH_NAMES[arch])

    adjs = []          # pre-noun descriptors
    tails = []         # post-noun phrases

    # color / cmap
    if "color" in spec and spec["color"] in _COLOR_SAY:
        adjs.append(_pick(_COLOR_SAY[spec["color"]]))
    if "cmap" in spec and spec["cmap"] in _CMAP_SAY:
        w = _pick(_CMAP_SAY[spec["cmap"]])
        (adjs if random.random() < 0.5 else tails).append(
            w if random.random() < 0.5 else f"in {w} colors")

    # mood
    if "mood" in spec and spec["mood"] in _MOOD_SAY:
        adjs.append(_pick(_MOOD_SAY[spec["mood"]]))

    # symmetry / count
    if "sym" in spec:
        noun = _pick(COUNT_NOUNS.get(arch, ["fold", "points"]))
        if random.random() < 0.5:
            adjs.append(f"{spec['sym']}-{noun.rstrip('s')}"
                        if noun in ("fold",) else f"{spec['sym']} {noun}")
        else:
            tails.append(f"with {spec['sym']} {noun}")

    # speed -> motion tail
    if "speed" in spec and random.random() < 0.85:
        tails.append(_pick(MOTION_TAIL[spec["speed"]]))
    elif "speed" in spec:
        adjs.insert(0, _pick(_SPEED_SAY[spec["speed"]]))

    random.shuffle(adjs)
    adj_str = " ".join(adjs)
    core = f"{adj_str} {name}".strip()

    verb = _pick(VERBS)
    art = _pick(ARTICLES)
    closer = _pick(CLOSERS)
    tail = " ".join(tails)

    parts = [verb, art, core, tail, closer]
    prompt = " ".join(x for x in parts if x).strip()
    prompt = " ".join(prompt.split())  # collapse whitespace
    # occasionally drop the leading verb for terse prompts
    if random.random() < 0.25 and prompt.split()[0] in VERBS:
        prompt = prompt.split(" ", 1)[1]
    return prompt[0].lower() + prompt[1:] if prompt else prompt


def build(n, seed):
    random.seed(seed)
    seen = set()
    rows = []
    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        spec = sample_spec()
        prompt = render_prompt(spec)
        key = (prompt, G.spec_to_dsl(spec))
        if key in seen or len(prompt) < 4:
            continue
        seen.add(key)
        rows.append({"prompt": prompt, "dsl": G.spec_to_dsl(spec)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    train = build(args.n, args.seed)
    val = build(args.val, args.seed + 999)

    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train), encoding="utf-8")
    (out / "val.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val), encoding="utf-8")
    print(f"wrote {len(train)} train, {len(val)} val -> {out}/")
    print("samples:")
    for r in random.sample(train, 8):
        print(f"  {r['prompt']!r}\n      -> {r['dsl']}")


if __name__ == "__main__":
    main()
