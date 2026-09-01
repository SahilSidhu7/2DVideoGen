"""
scene_grammar.py - the *scene script* output space, as a thing a small seq2seq
model can emit, be validated against, and be scored on.

Attempt 2's grammar mapped a prompt to ONE archetype with a handful of slots:
nine possible outputs, and 100% slot accuracy on that is not generation.
Attempt 21's `scenescript.py` can render a cast, a timeline, props and prop
interactions, but nothing wrote those scripts. This module is the bridge: it
defines what a scene script may say, in a flat one-line DSL, plus

    sample_spec()      a random *valid* scene, sampled over combinations
    spec_to_dsl()      spec  -> the model's target string
    dsl_to_spec()      the model's string -> spec        (strict, may raise)
    validate()         spec  -> list of structural problems
    repair()           spec  -> renderable spec + what was changed
    spec_to_scene()    spec  -> a .scene file scenescript.py already parses

The DSL:

  bg park dur 12.0 | cast ana cyan 1.7 1.0 ; bo orange 9.4 0.9 |
  prop tree 0.6 1.1 ; ball 6.4 1.0 | tl 0.0 3.2 ana walk 3.9 ; 5.2 7.2 bo kick ball ana

Four `|` sections: header, cast, props, timeline. Sections 3 and 4 may be
empty. Everything is positional, so the model never has to emit a key name.
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------- vocabulary #

BACKGROUNDS = ("park", "night", "studio")
# names are canonical and positional: cast member i is always NAMES[i]. This is
# a deliberate collapse of a meaningless degree of freedom (a name carries no
# render semantics) and is reported as such - it is NOT a claim about how hard
# the rest of the problem is.
NAMES = ("ana", "bo", "cy", "dee", "ell", "fay", "gus", "hal")
CAST_COLORS = ("cyan", "orange", "pink", "lime", "violet", "amber",
               "white", "green", "blue", "red")
DECOR = ("tree", "bush", "rock", "box", "cloud", "sun")
SOLO_ACTIONS = ("walk", "run", "jump", "wave", "dance", "idle")
MOVE_ACTIONS = ("walk", "run")
ACTIONS = SOLO_ACTIONS + ("kick",)

WORLD_W = 10.0           # (854/480)*5.62, the frame width in scenescript units
XMIN, XMAX = 0.4, 9.6
USABLE = XMAX - XMIN     # 9.2
# Attempt 23, per paper/RESEARCH2.md §1 item 3: the trained cast range is
# extended 1-5 -> 1-8, so that "in range" and "true extrapolation" become two
# separable questions instead of one. Six was the wall in Attempt 22; with 1-8
# trained, six is inside the range and nine is the new outside.
MAX_CAST = 8
MAX_EVENTS = 18          # was 12; 8 characters each need at least one event


def _r1(v):
    return round(float(v), 1)


# ------------------------------------------------------------------- staging #
# Attempt 22's positions were drawn independently (rng.sample over a grid),
# which makes P(x_i | x_<i) nearly uniform in the training distribution. The
# scenes that came out were well staged (measured: mean 0.865 on the Attempt 23
# metric) but the *conditional structure between the slots was absent*, so a
# beam search - which looks for the most likely sequence, not a typical one -
# had nothing to tell it that 9.2 is a bad answer after 9.6. These two
# functions put the dependency in.

def target_span(n):
    """A sensible cast span as a fraction of the usable width, by cast size.
    Same curve as model/scene_staging.py, deliberately: the data is generated
    against the metric it will be judged by, and that is stated rather than
    hidden."""
    return min(0.75, 0.45 + 0.10 * (n - 2))


def sample_positions(rng, n, clustered=False):
    """n x-positions that are staged by construction: a span at or above the
    target for that cast size, split into gaps that are all at least a
    silhouette apart, then SHUFFLED so cast order is not x order (otherwise the
    model learns 'ana is always leftmost', which is just a different confound).

    `clustered=True` deliberately produces a tight group - the prompt always
    says so when it does, so 'close together' has training signal and the
    spread is not unconditional."""
    if n == 1:
        return [_r1(rng.uniform(2.0, 8.0))]
    if clustered:
        span = rng.uniform(0.9, 1.5) * (n - 1) * 0.9
        span = min(span, USABLE * 0.9)
    else:
        span = rng.uniform(target_span(n), 1.0) * USABLE
    left = rng.uniform(XMIN, XMAX - span)
    min_gap = min(1.25, span / (n - 1) * 0.55)
    extra = span - (n - 1) * min_gap
    w = [rng.random() + 1e-6 for _ in range(n - 1)]
    tot = sum(w)
    xs = [left]
    for wi in w:
        xs.append(xs[-1] + min_gap + extra * wi / tot)
    xs = [_r1(min(XMAX, max(XMIN, x))) for x in xs]
    rng.shuffle(xs)
    return xs


# ------------------------------------------------------------------ sampling #

def _sched(rng, busy, span, k, pool):
    """Fill k non-overlapping action windows into the free gaps of [0, span]."""
    out, cursor = [], _r1(rng.uniform(0.0, 1.2))
    for _ in range(k):
        dur = _r1(rng.uniform(1.4, 2.8))
        moved = True
        while moved:                      # push past any busy window
            moved = False
            for (bs, be) in busy:
                if cursor < be and bs < cursor + dur:
                    cursor, moved = _r1(be + rng.uniform(0.1, 0.6)), True
        if cursor + dur > span:
            break
        out.append((cursor, _r1(cursor + dur), rng.choice(pool)))
        cursor = _r1(cursor + dur + rng.uniform(0.1, 1.1))
    return out


def sample_spec(rng=None, staged=True, max_cast=MAX_CAST):
    """One random scene, sampled over *combinations* rather than templates.

    `staged=True` (Attempt 23) draws the cast positions jointly instead of
    independently - see sample_positions(). `staged=False` reproduces the
    Attempt 22 sampler exactly, so the two can be compared."""
    rng = rng or random
    sizes = [1, 2, 3, 4, 5, 6, 7, 8][:max_cast]
    wts = [8, 26, 30, 24, 12, 9, 7, 5][:max_cast]
    n = rng.choices(sizes, weights=wts)[0]
    colors = rng.sample(CAST_COLORS, n)
    clustered = staged and n >= 2 and rng.random() < 0.12
    if staged:
        xs = sample_positions(rng, n, clustered=clustered)
    else:
        grid = [_r1(0.6 + i * 0.45) for i in range(21)]
        xs = rng.sample(grid, n)
    cast = []
    for i in range(n):
        cast.append({"id": NAMES[i], "color": colors[i], "x": xs[i],
                     "scale": _r1(rng.choice([0.55, 0.6, 0.7, 0.8, 0.85,
                                              0.9, 0.95, 1.0, 1.0]))})
    layout = {"clustered": clustered}

    n_kick = 0
    if n >= 2 and rng.random() < 0.55:
        n_kick = rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
    props = []
    if n_kick or rng.random() < 0.15:
        props.append({"kind": "ball", "x": _r1(rng.uniform(1.5, 8.5)),
                      "scale": 1.0})
    for kind in rng.sample(DECOR, rng.choices([0, 1, 2, 3],
                                              weights=[3, 5, 5, 3])[0]):
        props.append({"kind": kind, "x": _r1(rng.uniform(0.4, 9.6)),
                      "scale": _r1(rng.uniform(0.6, 1.2))})

    # the ball relay: a chain of kicks, spaced so the ball is at rest when the
    # next character strikes it (contact at s+0.7d, flight 1.1, roll 0.6).
    tl, busy = [], {c["id"]: [] for c in cast}
    if n_kick:
        pair = [c["id"] for c in rng.sample(cast, min(n, 2))]
        chain = [pair[0]] if n_kick == 1 else \
                (pair if n_kick == 2 else [pair[0], pair[1], pair[0]])
        t = _r1(rng.uniform(0.5, 4.0))
        for i, who in enumerate(chain):
            d = _r1(rng.uniform(1.6, 2.2))
            if i + 1 < len(chain):
                tgt = chain[i + 1]
            else:
                tgt = rng.choice([c["id"] for c in cast if c["id"] != who])
            tl.append({"s": t, "e": _r1(t + d), "who": who, "act": "kick",
                       "prop": "ball", "target": tgt})
            busy[who].append((t, t + d))
            t = _r1(t + d + rng.uniform(1.6, 2.4))
    span = _r1(max([e["e"] for e in tl] + [6.0]) + rng.uniform(1.5, 4.5))

    for c in cast:
        k = rng.choices([1, 2, 3], weights=[4, 5, 3])[0]
        for (s, e, act) in _sched(rng, busy[c["id"]], span, k,
                                  list(SOLO_ACTIONS)):
            ev = {"s": s, "e": e, "who": c["id"], "act": act}
            if act in MOVE_ACTIONS:
                lo, hi = XMIN + 0.4, XMAX - 0.4
                others = [o["x"] for o in cast if o["id"] != c["id"]]
                best = None
                # a walk that ends on top of somebody undoes the staging the
                # positions were sampled for, so try a few destinations and
                # keep the one with the most clearance.
                for _ in range(6 if staged else 1):
                    cand = _r1(min(hi, max(lo, c["x"] + rng.choice([-1, 1]) *
                                           rng.uniform(1.2, 4.5))))
                    clr = min([abs(cand - o) for o in others] or [9.9])
                    if best is None or clr > best[1]:
                        best = (cand, clr)
                    if clr >= 1.25:
                        break
                ev["to"] = best[0]
            tl.append(ev)
    tl.sort(key=lambda e: (e["s"], e["who"]))
    tl = _cap_events(tl, [c["id"] for c in cast])
    dur = _r1(max([e["e"] for e in tl] + [4.0]) + rng.choice([0.3, 0.5, 1.0]))
    return {"bg": rng.choice(BACKGROUNDS), "dur": dur, "cast": cast,
            "props": props, "timeline": tl, "layout": layout}


def _cap_events(tl, ids):
    """Truncate to MAX_EVENTS while keeping every character's FIRST event.
    A flat tl[:MAX_EVENTS] was safe with a cast of 5 and is not with a cast of
    8: it silently dropped a character's only event and turned a valid sample
    into an invalid one (a cast member with no timeline event = a mannequin)."""
    if len(tl) <= MAX_EVENTS:
        return tl
    keep, seen = [], set()
    for e in tl:
        if e["who"] not in seen:
            seen.add(e["who"])
            keep.append(e)
    kept = set(id(e) for e in keep)
    rest = [e for e in tl if id(e) not in kept]
    keep += rest[:max(0, MAX_EVENTS - len(keep))]
    keep.sort(key=lambda e: (e["s"], e["who"]))
    return keep


def layout_facts(sp):
    """Spatial facts that are TRUE of this scene, as checkable constraints.

    Attempt 23's second half of the staging fix: the position slot has no
    supervision because the prompt never mentions it. These are the facts a
    prompt is allowed to state about where people stand, and each one is
    machine-checkable against a generated spec by scene_eval.match(), so
    "the model obeyed the spatial instruction" becomes a rate rather than an
    impression.

      leftmost / rightmost   character i is at the smallest / largest x
      middle                 character i is strictly between two others
      back / front           character i has the smallest / largest scale
      spread                 the cast spans at least target_span(n)
      close                  the cast spans less than half of that
    """
    cast = sp["cast"]
    n = len(cast)
    out = []
    if n < 2:
        return out
    xs = [c["x"] for c in cast]
    ss = [c["scale"] for c in cast]
    lo, hi = xs.index(min(xs)), xs.index(max(xs))
    out.append({"kind": "leftmost", "i": lo})
    out.append({"kind": "rightmost", "i": hi})
    if n >= 3:
        for i in range(n):
            if i not in (lo, hi):
                out.append({"kind": "middle", "i": i})
                break
    if len(set(ss)) > 1:
        out.append({"kind": "back", "i": ss.index(min(ss))})
        out.append({"kind": "front", "i": ss.index(max(ss))})
    span = (max(xs) - min(xs)) / USABLE
    if span >= target_span(n):
        out.append({"kind": "spread"})
    elif span < 0.5 * target_span(n):
        out.append({"kind": "close"})
    return out


def check_layout_fact(sp, fact):
    """Is `fact` true of `sp`? Used by the evaluation, not by the synthesis."""
    cast = sp["cast"]
    n = len(cast)
    k, i = fact["kind"], fact.get("i")
    if k in ("leftmost", "rightmost", "middle", "back", "front"):
        if i is None or i >= n:
            return False
    xs = [c["x"] for c in cast]
    ss = [c["scale"] for c in cast]
    if k == "leftmost":
        return xs[i] == min(xs)
    if k == "rightmost":
        return xs[i] == max(xs)
    if k == "middle":
        return min(xs) < xs[i] < max(xs)
    if k == "back":
        return ss[i] == min(ss)
    if k == "front":
        return ss[i] == max(ss)
    if n < 2:
        return False
    span = (max(xs) - min(xs)) / USABLE
    if k == "spread":
        return span >= target_span(n)
    if k == "close":
        return span < target_span(n)
    return False


# ------------------------------------------------------------ serialisation #

def _d(v):
    """Tenths, as an integer. The tokenizer spends three tokens on '3.3'
    ('3', '.', '3') and one or two on '33'; a scene carries ~40 numbers, so
    this is worth ~60 decoder tokens per example - a third of the sequence."""
    return str(int(round(float(v) * 10)))


def _c(v):
    return str(int(round(float(v) * 100)))


def spec_to_dsl(sp):
    head = "bg %s dur %s" % (sp["bg"], _d(sp["dur"]))
    cast = " ; ".join("%s %s %s %s" % (c["id"], c["color"], _d(c["x"]),
                                       _c(c["scale"])) for c in sp["cast"])
    props = " ; ".join("%s %s %s" % (p["kind"], _d(p["x"]), _c(p["scale"]))
                       for p in sp["props"])
    ev = []
    for e in sp["timeline"]:
        s = "%s %s %s %s" % (_d(e["s"]), _d(e["e"]), e["who"], e["act"])
        if e["act"] in MOVE_ACTIONS and "to" in e:
            s += " %s" % _d(e["to"])
        elif e["act"] == "kick":
            s += (" %s %s" % (e.get("prop", "ball"), e.get("target", ""))).rstrip()
        ev.append(s)
    return "%s | cast %s | prop %s | tl %s" % (head, cast, props, " ; ".join(ev))


class DSLError(ValueError):
    pass


def _num(tok):
    try:
        return float(tok)
    except ValueError:
        raise DSLError("not a number: %r" % (tok,))


def dsl_to_spec(text):
    """Strict parse. Anything malformed raises - the parse RATE is a headline
    metric, so this must not quietly succeed on garbage."""
    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 2:
        raise DSLError("fewer than 2 sections")
    h = parts[0].split()
    if len(h) < 4 or h[0] != "bg" or h[2] != "dur":
        raise DSLError("bad header %r" % (parts[0],))
    sp = {"bg": h[1], "dur": _num(h[3]) / 10.0, "cast": [], "props": [],
          "timeline": []}
    for sec in parts[1:]:
        tok = sec.split()
        if not tok:
            continue
        kind, body = tok[0], " ".join(tok[1:])
        items = [i.strip().split() for i in body.split(";") if i.strip()]
        if kind == "cast":
            for it in items:
                if len(it) != 4:
                    raise DSLError("bad cast entry %r" % (it,))
                sp["cast"].append({"id": it[0], "color": it[1],
                                   "x": _num(it[2]) / 10.0,
                                   "scale": _num(it[3]) / 100.0})
        elif kind == "prop":
            for it in items:
                if len(it) != 3:
                    raise DSLError("bad prop entry %r" % (it,))
                sp["props"].append({"kind": it[0], "x": _num(it[1]) / 10.0,
                                    "scale": _num(it[2]) / 100.0})
        elif kind == "tl":
            for it in items:
                if len(it) < 4:
                    raise DSLError("bad event %r" % (it,))
                e = {"s": _num(it[0]) / 10.0, "e": _num(it[1]) / 10.0,
                     "who": it[2], "act": it[3]}
                if e["act"] in MOVE_ACTIONS:
                    if len(it) > 4:
                        e["to"] = _num(it[4]) / 10.0
                elif e["act"] == "kick":
                    e["prop"] = it[4] if len(it) > 4 else "ball"
                    t = it[5] if len(it) > 5 else ""
                    try:                      # a bare number is a coordinate
                        t = str(_r1(float(t) / 10.0))
                    except ValueError:
                        pass
                    e["target"] = t
                sp["timeline"].append(e)
        else:
            raise DSLError("unknown section %r" % (kind,))
    if not sp["cast"]:
        raise DSLError("empty cast")
    return sp


# --------------------------------------------------------------- validation #

def validate(sp):
    """Everything that would make scenescript.py render nonsense. Returns a
    list of problem strings; empty means structurally clean."""
    bad = []
    if sp["bg"] not in BACKGROUNDS:
        bad.append("background %r" % (sp["bg"],))
    if not (2.0 <= sp["dur"] <= 40.0):
        bad.append("duration %s" % (sp["dur"],))
    ids = [c["id"] for c in sp["cast"]]
    if len(set(ids)) != len(ids):
        bad.append("duplicate cast id")
    if not (1 <= len(ids) <= MAX_CAST):
        bad.append("cast size %d" % len(ids))
    for c in sp["cast"]:
        if c["color"] not in CAST_COLORS:
            bad.append("colour %r" % (c["color"],))
        if not (XMIN <= c["x"] <= XMAX):
            bad.append("%s x=%s off frame" % (c["id"], c["x"]))
        if not (0.4 <= c["scale"] <= 1.2):
            bad.append("%s scale=%s" % (c["id"], c["scale"]))
    if len(set(c["color"] for c in sp["cast"])) != len(sp["cast"]):
        bad.append("two characters share a colour")
    for p in sp["props"]:
        if p["kind"] not in DECOR + ("ball",):
            bad.append("prop %r" % (p["kind"],))
        if not (0.0 <= p["x"] <= WORLD_W):
            bad.append("prop %s x=%s off frame" % (p["kind"], p["x"]))
    has_ball = any(p["kind"] == "ball" for p in sp["props"])
    for e in sp["timeline"]:
        if e["who"] not in ids:
            bad.append("event for unknown character %r" % (e["who"],))
        if e["act"] not in ACTIONS:
            bad.append("unknown action %r" % (e["act"],))
        if not (e["e"] > e["s"] >= 0.0):
            bad.append("bad window %s..%s" % (e["s"], e["e"]))
        if e["e"] > sp["dur"] + 1e-6:
            bad.append("event ends %s past duration %s" % (e["e"], sp["dur"]))
        if e["act"] in MOVE_ACTIONS and "to" in e and \
                not (XMIN <= e["to"] <= XMAX):
            bad.append("walk target %s off frame" % (e["to"],))
        if e["act"] == "kick":
            if not has_ball:
                bad.append("kick with no ball prop")
            tgt = e.get("target", "")
            if tgt and tgt not in ids:
                try:
                    float(tgt)
                except ValueError:
                    bad.append("kick target %r is neither cast nor coord"
                               % (tgt,))
    acted = set(e["who"] for e in sp["timeline"])
    for i in ids:
        if i not in acted:
            bad.append("%s has no timeline event" % i)      # a mannequin
    ks = sorted([(e["s"], e["e"]) for e in sp["timeline"] if e["act"] == "kick"])
    for a, b in zip(ks, ks[1:]):
        if b[0] < a[1]:
            bad.append("overlapping kicks on one ball")
    return bad


def repair(sp):
    """Make a spec renderable. Returns (spec, [what was changed]). The RAW
    validity rate is reported separately from the post-repair render rate."""
    fixed = []
    sp = {k: (list(v) if isinstance(v, list) else v) for k, v in sp.items()}
    if sp["bg"] not in BACKGROUNDS:
        fixed.append("bg %s->park" % sp["bg"])
        sp["bg"] = "park"
    seen, cast, used_col = set(), [], set()
    for c in [dict(c) for c in sp["cast"]][:MAX_CAST]:
        if c["id"] in seen:
            fixed.append("dropped duplicate cast id %s" % c["id"])
            continue
        seen.add(c["id"])
        if c["color"] not in CAST_COLORS or c["color"] in used_col:
            new = next(k for k in CAST_COLORS if k not in used_col)
            fixed.append("colour %s->%s" % (c["color"], new))
            c["color"] = new
        used_col.add(c["color"])
        c["x"] = min(XMAX, max(XMIN, c["x"]))
        c["scale"] = min(1.2, max(0.45, c["scale"]))
        cast.append(c)
    sp["cast"] = cast or [{"id": "ana", "color": "cyan", "x": 5.0,
                           "scale": 1.0}]
    ids = [c["id"] for c in sp["cast"]]
    sp["props"] = [dict(p) for p in sp["props"]
                   if p["kind"] in DECOR + ("ball",)]
    for p in sp["props"]:
        p["x"] = min(WORLD_W, max(0.0, p["x"]))
        p["scale"] = min(2.0, max(0.3, p["scale"]))
    tl = []
    for e in sp["timeline"]:
        e = dict(e)
        if e["who"] not in ids:
            fixed.append("dropped event for unknown %s" % e["who"])
            continue
        if e["act"] not in ACTIONS:
            fixed.append("action %s->idle" % e["act"])
            e["act"] = "idle"
        if e["e"] <= e["s"]:
            e["e"] = e["s"] + 1.5
            fixed.append("zero-length window widened")
        if e["act"] == "kick":
            if not any(p["kind"] == "ball" for p in sp["props"]):
                sp["props"].append({"kind": "ball", "x": 5.0, "scale": 1.0})
                fixed.append("added the missing ball")
            t = e.get("target", "")
            if t not in ids:
                try:
                    float(t)
                except ValueError:
                    alt = [i for i in ids if i != e["who"]]
                    e["target"] = alt[0] if alt else "5.0"
                    fixed.append("kick target %r->%s" % (t, e["target"]))
        if e["act"] in MOVE_ACTIONS and "to" in e:
            e["to"] = min(XMAX, max(XMIN, e["to"]))
        tl.append(e)
    tl.sort(key=lambda e: e["s"])
    sp["timeline"] = tl
    need = max([e["e"] for e in tl] + [3.0]) + 0.3
    if sp["dur"] < need:
        fixed.append("duration %s->%s" % (sp["dur"], _r1(need)))
        sp["dur"] = _r1(need)
    sp["dur"] = min(40.0, max(2.0, sp["dur"]))
    for i in ids:                       # no mannequins
        if not any(e["who"] == i for e in sp["timeline"]):
            sp["timeline"].append({"s": 0.0, "e": _r1(sp["dur"]), "who": i,
                                   "act": "idle"})
            fixed.append("%s had no event -> idle" % i)
    return sp, fixed


# -------------------------------------------------------- .scene generation #

def spec_to_scene(sp, title="generated scene"):
    """Emit the exact line format scenescript.parse_script already reads."""
    L = ["# generated by model/scene_infer.py",
         "title      %s" % title, "size       854 480", "fps        30",
         "seconds    %s" % _r1(sp["dur"]), "background %s" % sp["bg"], ""]
    seen = {}
    for p in sp["props"]:
        if p["kind"] == "ball":
            pid = "ball"
        else:
            seen[p["kind"]] = seen.get(p["kind"], 0) + 1
            pid = "%s%d" % (p["kind"], seen[p["kind"]])
        extra = " y 4.5" if p["kind"] in ("cloud", "sun") else ""
        L.append("prop %s %s at %s scale %s%s"
                 % (p["kind"], pid, _r1(p["x"]), _r1(p["scale"]), extra))
    L.append("")
    for c in sp["cast"]:
        L.append("cast %s color %s at %s scale %s"
                 % (c["id"], c["color"], _r1(c["x"]), _r1(c["scale"])))
    L.append("")
    for e in sorted(sp["timeline"], key=lambda e: e["s"]):
        line = "%s %s %s %s" % (_r1(e["s"]), _r1(e["e"]), e["who"], e["act"])
        if e["act"] in MOVE_ACTIONS and "to" in e:
            line += " to %s" % _r1(e["to"])
        elif e["act"] == "kick":
            line += " ball at %s" % e.get("target", "")
        L.append(line.strip())
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rng = random.Random(3)
    for _ in range(3):
        sp = sample_spec(rng)
        d = spec_to_dsl(sp)
        print(d)
        rt = dsl_to_spec(d)
        assert spec_to_dsl(rt) == d, "round trip broken"
        print("  problems:", validate(rt))
    print("round trip OK")
