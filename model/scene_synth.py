"""
scene_synth.py - paired (natural-language prompt, scene-script DSL) data.

The synthesis samples a *scene* first (scene_grammar.sample_spec), then writes
an English prompt that describes it, and records - separately - an `intent`
dict holding ONLY the facts the prompt actually stated. The intent is what the
evaluation scores the model against; nothing the prompt never said is ever
counted for or against the model.

Surface-form variety is the point. Attempt 2's failure mode was one phrasing
per archetype, which lets a model pattern-match a template instead of reading
a sentence. Here every fact has 4-8 paraphrases, cast count can be a digit, a
word or a quantifier, clause order follows the scene's own time order, and the
connectives, the framing verb and the sentence template are all resampled.

    python scene_synth.py --n 6000 --val 600 --outdir data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import scene_grammar as SG

NUM_SAY = {
    1: ["a", "one", "a single", "a lone", "1"],
    2: ["two", "2", "a pair of", "a couple of", "two"],
    3: ["three", "3", "three", "a trio of"],
    4: ["four", "4", "four"],
    5: ["five", "5", "five"],
    6: ["six", "6", "six", "half a dozen"],
    7: ["seven", "7", "seven"],
    8: ["eight", "8", "eight"],
}
CAST_NOUN = ["friends", "people", "characters", "figures", "stick figures",
             "kids", "players", "strangers", "guys"]
CAST_NOUN_1 = ["friend", "person", "character", "figure", "stick figure",
               "kid", "player", "stranger", "guy"]
BG_SAY = {
    "park": ["in the park", "at the park", "in a park", "on the grass",
             "in a green park", "outside in a park", "in a sunny park"],
    "night": ["on a night street", "at night", "on a dark street",
              "in the dark", "on a street at night", "under a night sky"],
    "studio": ["in a studio", "on a plain stage", "against a plain backdrop",
               "in an empty studio", "on a bare set", "in a blank studio"],
}
ACT_SAY = {
    "walk": ["walks across", "strolls over", "walks to the other side",
             "wanders across", "walks", "goes for a walk"],
    "run": ["runs across", "sprints over", "dashes across", "runs",
            "runs off to the side", "bolts across"],
    "jump": ["jumps", "leaps into the air", "hops", "does a jump",
             "jumps up", "springs up"],
    "wave": ["waves", "waves hello", "greets the others", "waves at them",
             "says hi", "raises a hand and waves"],
    "dance": ["dances", "starts dancing", "does a little dance",
              "dances around", "breaks into a dance"],
    "idle": ["just stands there", "waits", "stands still", "does nothing",
             "hangs back"],
}
KICK_SAY_MANY = ["kick a ball around", "pass a ball back and forth",
                 "knock a ball between them", "play with a ball",
                 "kick a football to each other", "pass a ball to each other"]
KICK_SAY_ONE = ["kicks a ball", "kicks a ball over", "boots the ball away",
                "kicks the ball across", "passes the ball"]
PROP_SAY = {
    "tree": ["a tree", "a big tree", "a tree in shot"],
    "bush": ["a bush", "some bushes"],
    "rock": ["a rock", "a boulder"],
    "box": ["a crate", "a box", "a wooden crate"],
    "cloud": ["a cloud", "clouds overhead", "a cloud in the sky"],
    "sun": ["the sun", "the sun up in the sky"],
}
ORD = ["one", "another", "the third", "the fourth", "the fifth",
       "the sixth", "the seventh", "the eighth"]
ORD_ALT = ["the first", "the second", "the third one", "the next one",
           "the fifth one", "the sixth one", "the seventh one",
           "the last one"]

# ---- Attempt 23: the spatial vocabulary. Every phrase here is only ever used
# when the fact it states is TRUE of the sampled scene, so the position slot
# gains real supervision instead of the model guessing.
LAYOUT_SAY = {
    "leftmost":  ["on the left", "over on the left", "on the left-hand side",
                  "at the left of the frame", "off to the left"],
    "rightmost": ["on the right", "over on the right", "on the right-hand side",
                  "at the right of the frame", "off to the right"],
    "middle":    ["in the middle", "between the others", "in between them",
                  "in the centre", "stood between them"],
    "back":      ["further back", "in the background", "further away",
                  "at the back", "off in the distance"],
    "front":     ["up front", "closest to us", "in the foreground",
                  "nearest the camera"],
}
GROUP_LAYOUT_SAY = {
    "spread": ["spread out across the frame", "spread out",
               "standing well apart", "at opposite ends of the frame",
               "spaced out across the shot", "facing each other from either side"],
    "close":  ["standing close together", "huddled together",
               "bunched up together", "right next to each other",
               "clustered in one spot"],
}
JOIN = [", ", ", then ", " and then ", ", while ", ", after that ",
        " and ", ", meanwhile "]
VERB = ["make", "create", "generate", "render", "animate", "show me",
        "i want", "draw", "give me", "build", ""]
COLOR_SAY = {"cyan": ["cyan", "aqua"], "orange": ["orange"],
             "pink": ["pink"], "lime": ["lime", "lime green"],
             "violet": ["violet", "purple"], "amber": ["amber", "golden"],
             "white": ["white"], "green": ["green"], "blue": ["blue"],
             "red": ["red"]}


def _per_character(sp):
    """[(cast_index, [actions in time order])] excluding kicks."""
    order = {c["id"]: i for i, c in enumerate(sp["cast"])}
    out = {i: [] for i in range(len(sp["cast"]))}
    for e in sorted(sp["timeline"], key=lambda e: e["s"]):
        if e["act"] == "kick":
            continue
        out[order[e["who"]]].append(e["act"])
    return out


GROUP_SAY = ["all of them", "everyone", "all {n} of them", "every one of them",
             "the whole group", "they all"]


def force_group_action(sp, rng):
    """Give every character one common action, so the prompt can state the
    cast size ONCE and describe the cast ONCE. Without examples like this the
    dataset makes 'count the clauses' and 'read the numeral' the same
    function, and the model learns the cheaper one - which is exactly the
    Attempt 22 v1 failure this exists to remove."""
    act = rng.choice(["dance", "wave", "jump", "idle"])
    for c in sp["cast"]:
        evs = [e for e in sp["timeline"]
               if e["who"] == c["id"] and e["act"] != "kick"]
        if evs:
            for e in evs:
                e.pop("to", None)
                e["act"] = act
    return act


def make_prompt(sp, rng, group_act=None):
    """Write an English prompt for `sp` and return (prompt, intent)."""
    n = len(sp["cast"])
    intent = {"n_cast": None, "bg": None, "acts": {}, "kick": False,
              "props": [], "colors": {}, "layout": []}

    # ---- Attempt 23: which true spatial facts will this prompt state?
    # Per-character facts are attached to that character's own action clause,
    # so the reference is unambiguous; the group fact goes on the cast phrase.
    per_layout, group_layout = {}, None
    if STAGE and n >= 2:
        facts = SG.layout_facts(sp)
        percs = [f for f in facts if "i" in f]
        grps = [f for f in facts if "i" not in f]
        rng.shuffle(percs)
        taken = set()
        for f in percs:
            if f["i"] in taken:
                continue
            if rng.random() < (0.34 if f["kind"] in ("leftmost", "rightmost")
                               else 0.22):
                per_layout[f["i"]] = f
                taken.add(f["i"])
        # a clustered scene ALWAYS says so - otherwise "close together" would
        # be an unstated fact the model is scored on, which is the mistake
        # this whole attempt is about.
        for f in grps:
            if f["kind"] == "close" or rng.random() < 0.25:
                group_layout = f
                break

    bits = []
    # ---- who
    say_n = rng.random() < 0.95
    noun = rng.choice(CAST_NOUN_1 if n == 1 else CAST_NOUN)
    if say_n:
        intent["n_cast"] = n
        who = "%s %s" % (rng.choice(NUM_SAY[n]), noun)
    else:
        who = "some %s" % noun

    # ---- colours (only sometimes, and then for every character)
    if rng.random() < 0.35 and n <= 3:
        cols = [rng.choice(COLOR_SAY[c["color"]]) for c in sp["cast"]]
        intent["colors"] = {i: sp["cast"][i]["color"] for i in range(n)}
        who = "%s (%s)" % (who, " and ".join("a %s one" % c for c in cols)) \
            if rng.random() < 0.4 else "%s %s" % (" and ".join(cols), who)

    if group_layout is not None:
        intent["layout"].append(group_layout)
        who = "%s %s" % (who, rng.choice(GROUP_LAYOUT_SAY[group_layout["kind"]]))

    # ---- where
    where = ""
    if rng.random() < 0.85:
        intent["bg"] = sp["bg"]
        where = rng.choice(BG_SAY[sp["bg"]])

    # ---- what each of them does
    per = _per_character(sp)
    clauses = []
    n_described = 0
    use_ord = rng.random() < 0.85
    described = list(range(n))
    if group_act is not None:
        # one clause for the entire cast
        g = rng.choice(GROUP_SAY).replace("{n}", str(n))
        clauses.append("%s %s" % (g, rng.choice(ACT_SAY[group_act])))
        n_described = 1
        for i in range(n):
            if per[i]:
                intent["acts"][i] = [group_act]
        described = []
    elif n >= 2 and DECONFOUND and rng.random() < 0.45:
        # state the count, describe only SOME of them
        k = rng.randint(1, max(1, n - 1))
        described = sorted(rng.sample(range(n), k))
    for i in described:
        acts = per[i]
        if not acts:
            continue
        acts = acts[:3]
        intent["acts"][i] = list(acts)
        if n == 1:
            subj = rng.choice(["it", "they", "the %s" % noun, "he", "she"])
        else:
            pool = ORD if use_ord else ORD_ALT
            subj = pool[min(i, len(pool) - 1)]
        phr = [rng.choice(ACT_SAY[a]) for a in acts]
        if len(phr) == 1:
            cl = "%s %s" % (subj, phr[0])
        else:
            cl = "%s %s" % (subj, rng.choice([" then ", " and then ",
                                              ", then "]).join(phr))
        f = per_layout.get(i)
        if f is not None:
            intent["layout"].append(f)
            cl = (("%s %s %s" % (subj, rng.choice(LAYOUT_SAY[f["kind"]]),
                                 cl[len(subj) + 1:]))
                  if rng.random() < 0.5
                  else ("%s %s" % (cl, rng.choice(LAYOUT_SAY[f["kind"]]))))
        clauses.append(cl)
        n_described += 1
    rng_join = lambda: rng.choice(JOIN)
    body = clauses[0] if clauses else ""
    for c in clauses[1:]:
        body += rng_join() + c

    # ---- the ball
    kicks = [e for e in sp["timeline"] if e["act"] == "kick"]
    if kicks:
        intent["kick"] = True
        if len(kicks) == 1:
            body += rng_join() + "%s %s" % (
                rng.choice(["one", "one of them", "the first"]),
                rng.choice(KICK_SAY_ONE))
        else:
            body += rng_join() + "%s %s" % (
                rng.choice(["they", "the two of them", "they both"]),
                rng.choice(KICK_SAY_MANY))

    # ---- scenery
    extras = []
    for p in sp["props"]:
        if p["kind"] == "ball" or rng.random() > 0.45:
            continue
        intent["props"].append(p["kind"])
        extras.append(rng.choice(PROP_SAY[p["kind"]]))
    tail = ""
    if extras:
        tail = " %s %s" % (rng.choice(["with", "there is", "and there's"]),
                           " and ".join(extras))
        if tail.strip().startswith("there"):
            tail = "." + tail

    # ---- assemble
    v = rng.choice(VERB)
    t = rng.random()
    if t < 0.35:
        s = "%s %s %s: %s" % (v, who, where, body)
    elif t < 0.6:
        s = "%s %s, %s" % (who, where, body)
    elif t < 0.8:
        s = "%s a clip %s with %s. %s" % (v, where, who, body)
    else:
        s = "%s %s where %s %s" % (v, who, where, body)
    s = (s + tail).strip()
    s = " ".join(s.split()).replace(" ,", ",").replace(" :", ":")
    s = s.replace(" .", ".").replace("::", ":")
    if s.startswith(":"):
        s = s[1:].strip()
    intent["n_described"] = n_described
    return s.lower(), intent


DECONFOUND = False
STAGE = False          # Attempt 23: staged positions + spatial language
MAX_CAST = SG.MAX_CAST


def make_pair(rng):
    sp = SG.sample_spec(rng, staged=STAGE, max_cast=MAX_CAST)
    g = None
    if DECONFOUND and len(sp["cast"]) >= 2 and rng.random() < 0.22:
        g = force_group_action(sp, rng)
    prompt, intent = make_prompt(sp, rng, group_act=g)
    return {"prompt": prompt, "dsl": SG.spec_to_dsl(sp), "intent": intent,
            "n_cast": len(sp["cast"]),
            "n_described": intent.pop("n_described")}


def build(n, seed):
    rng = random.Random(seed)
    rows, seen, guard = [], set(), 0
    while len(rows) < n and guard < n * 30:
        guard += 1
        r = make_pair(rng)
        if len(r["prompt"]) < 12 or r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        rows.append(r)
    return rows


def main():
    global DECONFOUND, STAGE, MAX_CAST
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--prefix", default="scene")
    ap.add_argument("--deconfound", action="store_true",
                    help="break the 'one clause per character' correlation")
    ap.add_argument("--stage", action="store_true",
                    help="Attempt 23: jointly-sampled staged positions, plus "
                         "prompts that state true spatial facts")
    ap.add_argument("--max-cast", type=int, default=SG.MAX_CAST,
                    help="largest cast size the sampler may draw (default 8)")
    a = ap.parse_args()
    DECONFOUND = a.deconfound
    STAGE = a.stage
    MAX_CAST = a.max_cast
    out = Path(a.outdir)
    out.mkdir(exist_ok=True)
    tr, va = build(a.n, a.seed), build(a.val, a.seed + 4321)
    trp = set(r["prompt"] for r in tr)
    va = [r for r in va if r["prompt"] not in trp]
    (out / ("%s_train.jsonl" % a.prefix)).write_text(
        "\n".join(json.dumps(r) for r in tr), encoding="utf-8")
    (out / ("%s_val.jsonl" % a.prefix)).write_text(
        "\n".join(json.dumps(r) for r in va), encoding="utf-8")
    print("wrote %d train, %d val (val prompts disjoint from train) -> %s/"
          % (len(tr), len(va), out))
    for r in random.Random(1).sample(tr, 6):
        print("\n  %r\n    -> %s" % (r["prompt"], r["dsl"]))


if __name__ == "__main__":
    main()
