"""
scene_staging.py - a number for how a scene is *blocked*.

Attempt 21 recorded that "staging is not composition" and that nothing in the
pipeline checks blocking. Attempt 22 then made it worse: the position slots are
unsupervised by the prompt, so the decoder answers an unasked question with its
prior, and its prior is a mode - the headline generated clip puts two characters
at x=9.6 and x=9.2 on a stage ten units wide. That defect was visible and
unmeasured. This module measures it.

What it scores, per scene, over the whole clip (not just the declared
positions - characters move):

  min_sep     minimum pairwise horizontal distance between two characters
  mean_sep    mean pairwise horizontal distance
  span_frac   how much of the usable stage width the cast spans
  occl_frac   fraction of (sample, pair) observations where one silhouette
              badly covers another ON THE SAME DEPTH PLANE
  even_score  how far the arrangement is from a sensible spread

The composite `staging` in [0, 1] is the mean of THREE of those:
sep_score, span_score, occl_score. Every part is reported, because a composite
is only useful if you can see which term killed it.

Two further terms were built, measured, and **demoted to diagnostics because
they did not separate the hand-written scenes from the generated ones** - one
of them inverted the separation outright:

* `edge_score` (clearance from the frame edge) reads **0.00 on BOTH**
  `park_meet.scene` and `street_relay.scene`, because a character walking to
  x=9.6 puts one hand a centimetre past the frame edge for a few frames. It
  punished the two clips that look good, so it is not evidence of bad staging.
* `even_score` (distance from a uniform spread) reads 0.637 on `park_meet` -
  two characters left, one far right is a *composition*, not a fault. Evenly
  spaced is a picket fence, and a degenerate arrangement is already caught by
  span and sep. Kept as a diagnostic, not scored.

Both are still computed and written out; they are simply not allowed to move
the number.

`staging` is NaN for a one-character scene. This metric measures where bodies
are *relative to each other*, and with one body there is nothing to measure;
scoring it 1.0 would silently reward the model for emitting a cast of one.

Three things this deliberately does NOT do:

* It does not treat every overlap as a fault. A near character crossing in
  front of a far one is depth, and it is what Attempt 21's matte was built for.
  A pair is only counted as occluding when the silhouettes overlap *and* the
  two stand on ground planes within DEPTH_SEP of each other, i.e. they read as
  being on one line rather than as near and far.
* It does not guess the silhouette width. HALF_W is measured off
  stickman.StickFigure at import, per action, by posing the rig - a kicking
  figure is nearly three times wider than an idle one and pretending otherwise
  would make the occlusion term fiction.
* It does not score how good the scene looks. It scores where the bodies are.

    USE_TF=0 python model/scene_staging.py scenes/park_meet.scene ...
    USE_TF=0 python model/scene_staging.py --dir scenes --glob "a22_*.scene"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import scenescript as SS
import stickman

# ------------------------------------------------------------------ constants #

WORLD_W = 10.0
XMIN, XMAX = 0.4, 9.6
USABLE = XMAX - XMIN                 # 9.2
MATTE_PAD = 4.0 / 85.4               # the compositor dilates the matte by 4 px
DEPTH_SEP = 0.20                     # ground_y difference that reads as depth
GAP_MARGIN = 0.25                    # clear air a pair should have between them
EDGE_TARGET = 0.50                   # edge clearance that earns full marks
N_SAMPLES = 61


def _measure_half_widths():
    """Half-width of the rig silhouette about its own hip, at scale 1.0, per
    action. Measured, not assumed: idle is 0.34 and kick is 0.92."""
    fig = stickman.StickFigure(854, 480, ground=SS.GROUND)
    out = {}
    for act in ("walk", "run", "jump", "wave", "dance", "idle", "kick"):
        ext = []
        for i in range(24):
            j, _ = fig.pose(act, i / 24.0)
            hip = j["hip"][0]
            xs = [v[0] for v in j.values()]
            ext.append(max(max(xs) - hip, hip - min(xs)))
        out[act] = sum(ext) / len(ext)
    return out


HALF_W = _measure_half_widths()
HALF_W_DEFAULT = HALF_W["idle"]

# half-widths of the opaque ground props, read off _prop_layers geometry
PROP_HALF_W = {"tree": 0.76, "bush": 0.50, "rock": 0.36, "box": 0.40}


def half_width(action, scale):
    return HALF_W.get(action, HALF_W_DEFAULT) * float(scale) + MATTE_PAD


# --------------------------------------------------------------------- score #

def _clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _target_span(n):
    """A sensible cast span as a fraction of the usable width. Two characters
    do not have to fill the frame; five do."""
    return min(0.75, 0.45 + 0.10 * (n - 2))


def stage_scene(scene, n_samples=N_SAMPLES):
    """`scene` is an SS.Scene, a path to a .scene, or .scene text."""
    if isinstance(scene, Path):
        sc = SS.Scene(SS.parse_script(scene.read_text(encoding="utf-8")))
    elif isinstance(scene, str):
        txt = scene if "\n" in scene else Path(scene).read_text(encoding="utf-8")
        sc = SS.Scene(SS.parse_script(txt))
    else:
        sc = scene
    cast = sc.cast
    n = len(cast)
    dur = sc.seconds
    ts = [dur * i / max(1, n_samples - 1) for i in range(n_samples)]

    def hw(c, t):
        clip = c.clip_at(t)
        return half_width(clip[2] if clip else "idle", c.scale)

    seps, clears, spans = [], [], []
    occl_hits = occl_n = 0
    edge_clear = 1e9
    per_t_positions = []
    for t in ts:
        xs = [c.x_at(t) for c in cast]
        ws = [hw(c, t) for c in cast]
        per_t_positions.append(xs)
        spans.append((max(xs) - min(xs)) / USABLE if n >= 2 else 0.0)
        for i in range(n):
            edge_clear = min(edge_clear, xs[i] - ws[i], WORLD_W - (xs[i] + ws[i]))
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(xs[i] - xs[j])
                seps.append(d)
                clears.append(d - (ws[i] + ws[j]))
                occl_n += 1
                same_plane = abs(cast[i].ground_y - cast[j].ground_y) < DEPTH_SEP
                # "badly": centres closer than half the summed silhouette, so
                # more than half of the narrower figure is behind the other
                if same_plane and d < 0.5 * (ws[i] + ws[j]):
                    occl_hits += 1

    # evenness: compare the sorted neighbour gaps to the uniform ones they
    # would have at the same span, averaged over the clip.
    even_vals = []
    if n >= 3:
        for xs in per_t_positions:
            s = sorted(xs)
            gaps = [b - a for a, b in zip(s, s[1:])]
            span = s[-1] - s[0]
            if span <= 1e-6:
                even_vals.append(0.0)
                continue
            uni = span / (n - 1)
            rmse = math.sqrt(sum((g - uni) ** 2 for g in gaps) / len(gaps))
            even_vals.append(_clamp01(1.0 - rmse / uni))
    even = sum(even_vals) / len(even_vals) if even_vals else 1.0

    min_sep = min(seps) if seps else float("nan")
    mean_sep = sum(seps) / len(seps) if seps else float("nan")
    min_clear = min(clears) if clears else float("nan")
    span = sum(spans) / len(spans) if spans else 0.0
    occl_frac = occl_hits / occl_n if occl_n else 0.0

    if n >= 2:
        need = []
        for k, t in enumerate(ts):
            xs = per_t_positions[k]
            ws = [hw(c, t) for c in cast]
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(cast[i].ground_y - cast[j].ground_y) >= DEPTH_SEP:
                        continue          # different plane: distance is free
                    req = ws[i] + ws[j] + GAP_MARGIN
                    need.append(_clamp01(abs(xs[i] - xs[j]) / req))
        # the MEAN adequacy over the clip, not the worst instant. Two
        # characters passing each other for a third of a second is staging;
        # two characters standing on top of each other for nine seconds is the
        # defect. A min() cannot tell those apart and scored the hand-written
        # street_relay.scene at 0.016 for one frame of a ball hand-off.
        sep_score = sum(need) / len(need) if need else 1.0
        span_score = _clamp01(span / _target_span(n))
        occl_score = _clamp01(1.0 - 3.0 * occl_frac)      # a third of the clip = 0
        even_score = even
    else:
        sep_score = span_score = occl_score = even_score = float("nan")
    edge_score = _clamp01(edge_clear / EDGE_TARGET)

    # The composite is the three terms that measure the named defect. `even`
    # and `edge` are reported and NOT scored - see the module note below.
    staging = float("nan") if n < 2 else \
        (sep_score + span_score + occl_score) / 3.0

    # diagnostic, not scored: a character standing behind a nearer opaque prop
    hidden = []
    for c in cast:
        for kind, p in sc.props:
            if kind not in PROP_HALF_W:
                continue
            ps, px = float(p.get("scale", 1.0)), float(p.get("at", 5.0))
            if ps <= c.scale:
                continue              # drawn behind the character
            pw = PROP_HALF_W[kind] * ps
            covered = sum(1 for t in ts
                          if abs(c.x_at(t) - px) < 0.5 * (pw + hw(c, t)))
            if covered > 0.5 * len(ts):
                hidden.append([c.id, kind, round(covered / len(ts), 2)])

    return {"n_cast": n, "seconds": round(dur, 2),
            "min_sep": round(min_sep, 3), "mean_sep": round(mean_sep, 3),
            "min_clearance": round(min_clear, 3),
            "span_frac": round(span, 3),
            "span_target": round(_target_span(n), 3) if n >= 2 else None,
            "occl_frac": round(occl_frac, 3),
            "edge_clearance": round(edge_clear, 3),
            "sep_score": round(sep_score, 3), "span_score": round(span_score, 3),
            "occl_score": round(occl_score, 3), "even_score": round(even_score, 3),
            "edge_score": round(edge_score, 3),
            "staging": round(staging, 3),
            "hidden_behind_prop": hidden,
            "declared_x": [round(c.x0, 2) for c in cast],
            "declared_scale": [round(c.scale, 2) for c in cast]}


def stage_spec(spec, **kw):
    """A scene_grammar spec -> staging metrics, through the .scene text."""
    import scene_grammar as SG
    return stage_scene(SS.Scene(SS.parse_script(SG.spec_to_scene(spec))), **kw)


# ----------------------------------------------------------------------- cli #

COLS = ["n_cast", "min_sep", "mean_sep", "span_frac", "occl_frac",
        "sep_score", "span_score", "occl_score", "even_score", "edge_score",
        "staging"]


def table(rows):
    head = "%-26s " % "scene" + " ".join("%9s" % c for c in COLS)
    out = [head, "-" * len(head)]
    for name, m in rows:
        out.append("%-26s " % name[:26] + " ".join("%9s" % m[c] for c in COLS))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--glob", default="*.scene")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    paths = [Path(p) for p in a.scenes]
    if a.dir:
        paths += sorted(Path(a.dir).glob(a.glob))
    rows = []
    for p in paths:
        try:
            rows.append((p.name, stage_scene(p)))
        except Exception as e:
            print("!! %s: %s: %s" % (p.name, type(e).__name__, e))
    print(table(rows))
    for name, m in rows:
        if m["hidden_behind_prop"]:
            print("  %s hidden behind a prop: %s" % (name, m["hidden_behind_prop"]))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(rows), indent=1), encoding="utf-8")
        print("-> %s" % a.json)


if __name__ == "__main__":
    main()
