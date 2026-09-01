"""
grammar.py - the bounded scene grammar shared by the dataset synthesizer,
the trained model's output parser, and the rule-based fallback.

Pipeline:  prompt  ->  DSL spec  ->  full geovid scene dict

The DSL spec is a compact, flat, order-free string the small model learns to
emit, e.g.:

    arch=rose sym=5 color=orange speed=slow mood=glow

It is deliberately tiny and regular so a 60M-param seq2seq learns it reliably,
and so it can always be validated and repaired. `build_scene(spec)` expands a
spec into a full scene dict that geovid.Scene renders.
"""
from __future__ import annotations

import math
import random

# --------------------------------------------------------------------------- #
# Slot vocabularies
# --------------------------------------------------------------------------- #

# named line colors -> RGB
COLORS = {
    "blue": (90, 210, 255), "cyan": (80, 240, 240), "teal": (60, 220, 200),
    "red": (255, 80, 80), "orange": (255, 150, 60), "amber": (255, 190, 70),
    "yellow": (255, 220, 90), "green": (140, 255, 150), "lime": (170, 255, 90),
    "purple": (180, 120, 255), "violet": (160, 100, 255), "pink": (255, 110, 180),
    "magenta": (255, 90, 200), "white": (240, 245, 255), "gold": (255, 200, 120),
}
COLOR_WORDS = {  # prompt synonyms -> canonical color slot
    "blue": "blue", "azure": "blue", "sky": "blue",
    "cyan": "cyan", "aqua": "cyan", "turquoise": "teal", "teal": "teal",
    "red": "red", "crimson": "red", "scarlet": "red",
    "orange": "orange", "tangerine": "orange", "amber": "amber",
    "yellow": "yellow", "gold": "gold", "golden": "gold",
    "green": "green", "emerald": "green", "lime": "lime",
    "purple": "purple", "violet": "violet", "indigo": "violet",
    "pink": "pink", "rose": "pink", "magenta": "magenta",
    "white": "white", "silver": "white",
}

# field colormaps + prompt synonyms
CMAPS = {"fire": "fire", "flame": "fire", "heat": "fire", "lava": "fire",
         "ice": "ice", "frost": "ice", "cold": "ice", "ocean": "ice",
         "inferno": "inferno", "warm": "inferno", "sunset": "inferno",
         "magma": "magma", "purple": "magma", "cosmic": "magma", "galaxy": "magma",
         "viridis": "viridis", "rainbow": "viridis", "green": "viridis",
         "neon": "viridis"}

SPEED_WORDS = {"slow": "slow", "gentle": "slow", "calm": "slow", "lazy": "slow",
               "slowly": "slow", "relaxed": "slow", "smooth": "slow",
               "medium": "med", "moderate": "med", "steady": "med",
               "fast": "fast", "quick": "fast", "rapid": "fast", "energetic": "fast",
               "frantic": "fast", "quickly": "fast", "hyper": "fast"}
SPEED_COEF = {"slow": 0.2, "med": 0.55, "fast": 1.2}
SPEED_DUR = {"slow": 12.0, "med": 9.0, "fast": 7.0}

MOOD_WORDS = {"glow": "glow", "glowing": "glow", "neon": "glow", "bright": "glow",
              "vibrant": "glow", "luminous": "glow",
              "dark": "dark", "moody": "dark", "deep": "dark", "night": "dark",
              "minimal": "minimal", "clean": "minimal", "simple": "minimal"}
MOOD_BG = {"glow": (4, 4, 12), "dark": (2, 2, 6), "minimal": (14, 14, 20)}

# count/symmetry words for "N petals / N-fold / N arms"
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}

# stickman actions: prompt word -> canonical action
ACTION_WORDS = {
    "walk": "walk", "walking": "walk", "walks": "walk", "stroll": "walk",
    "strolling": "walk", "marching": "walk", "march": "walk",
    "run": "run", "running": "run", "runs": "run", "sprint": "run",
    "sprinting": "run", "jog": "run", "jogging": "run", "dash": "run",
    "jump": "jump", "jumping": "jump", "jumps": "jump", "hop": "jump",
    "hopping": "jump", "leap": "jump", "leaping": "jump", "bounce": "jump",
    "kick": "kick", "kicking": "kick", "kicks": "kick", "football": "kick",
    "soccer": "kick", "ball": "kick",
    "wave": "wave", "waving": "wave", "waves": "wave", "hello": "wave",
    "hi": "wave", "greeting": "wave", "greet": "wave",
    "dance": "dance", "dancing": "dance", "dances": "dance", "dancer": "dance",
}
ACTIONS = {"walk", "run", "jump", "kick", "wave", "dance"}

# words that imply a human/character (so "a man running" -> stickman)
CHARACTER_WORDS = {"man", "woman", "person", "people", "figure", "guy", "dude",
                   "character", "player", "someone", "human", "boy", "girl",
                   "he", "she", "they", "kid", "child"}

# --------------------------------------------------------------------------- #
# Archetypes: each knows its prompt names and how to build geovid layers
# --------------------------------------------------------------------------- #

# name -> list of natural-language nouns used in prompts
ARCH_NAMES = {
    "lissajous": ["lissajous curve", "lissajous figure", "harmonic curve",
                  "crossing loops", "oscilloscope pattern", "xy figure"],
    "rose": ["rose curve", "flower", "rose pattern", "petal pattern",
             "rhodonea curve", "blooming flower"],
    "spirograph": ["spirograph", "guilloche pattern", "hypotrochoid",
                   "spiro pattern", "geometric web"],
    "superformula": ["superformula shape", "supershape", "organic shape",
                     "morphing polygon", "gielis curve"],
    "harmonograph": ["harmonograph", "pendulum drawing", "damped lissajous",
                     "swinging curve", "phase spiral"],
    "spiral": ["spiral", "logarithmic spiral", "swirl", "vortex", "helix"],
    "interference": ["wave interference", "ripple pattern", "interference field",
                     "two wave sources", "water ripples"],
    "mandala": ["mandala", "kaleidoscope", "radial pattern", "symmetric bloom",
                "sacred geometry"],
    "waves": ["sine waves", "wave plot", "oscillating waves", "signal waves",
              "traveling wave"],
    "stickman": ["stickman", "stick figure", "stick man", "walking stickman",
                 "walking man", "walking person", "stick figure walking",
                 "man walking", "person walking"],
}

FIELD_ARCH = {"interference", "mandala"}
SPECIAL_ARCH = {"stickman"}   # rendered per-frame, not via a single geovid scene


def _color(spec, default="blue"):
    return COLORS.get(spec.get("color", default), COLORS[default])


def _speed(spec):
    return spec.get("speed", "med")


def build_scene(spec: dict) -> dict:
    """Expand a validated DSL spec dict into a full geovid scene dict."""
    arch = spec.get("arch", "lissajous")
    speed = _speed(spec)
    k = SPEED_COEF[speed]
    dur = SPEED_DUR[speed]
    mood = spec.get("mood", "dark")
    bg = MOOD_BG.get(mood, (6, 6, 14))
    sym = int(spec.get("sym", 0)) or None
    col = _color(spec)
    cmap = spec.get("cmap", "magma")
    width = 3 if mood == "glow" else 2

    base = {"fps": 30, "duration": round(dur, 1), "background": list(bg)}

    if arch == "lissajous":
        a = sym or random.choice([3, 3, 5])
        return {**base, "width": 854, "height": 480, "supersample": 2,
                "view": {"xmin": -1.4, "xmax": 1.4, "ymin": -0.8, "ymax": 0.8},
                "layers": [
                    {"type": "parametric", "t": [0, 6.283185, 3000],
                     "x": f"sin({a}*t + {k}*time)", "y": "sin(2*t)",
                     "color": list(col), "width": width},
                    {"type": "parametric", "t": [0, 6.283185, 3000],
                     "x": f"sin({a+1}*t)", "y": f"sin(3*t + {k*0.6:.3f}*time)",
                     "color": list(_lighter(col)), "width": max(1, width - 1)},
                ]}

    if arch == "rose":
        n = sym or random.choice([3, 4, 5, 6, 7])
        return {**base, "width": 720, "height": 720, "supersample": 2,
                "view": {"xmin": -1.2, "xmax": 1.2, "ymin": -1.2, "ymax": 1.2},
                "layers": [
                    {"type": "polar", "theta": [0, 25.13274, 4000],
                     "r": f"cos({n}*theta + {k}*time)",
                     "color": list(col), "width": width}]}

    if arch == "spirograph":
        return {**base, "width": 720, "height": 720, "supersample": 2,
                "view": {"xmin": -1.3, "xmax": 1.3, "ymin": -1.3, "ymax": 1.3},
                "layers": [
                    {"type": "parametric", "t": [0, 125.66, 8000],
                     "params": {"R": 0.7, "r": 0.23, "d": 0.55},
                     "x": f"(R-r)*cos(t) + d*cos(((R-r)/r)*t + {k}*time)",
                     "y": f"(R-r)*sin(t) - d*sin(((R-r)/r)*t + {k}*time)",
                     "color": list(col), "width": 1}]}

    if arch == "superformula":
        m = sym or random.choice([5, 6, 7, 8])
        return {**base, "width": 720, "height": 720, "supersample": 2,
                "view": {"xmin": -1.4, "xmax": 1.4, "ymin": -1.4, "ymax": 1.4},
                "layers": [
                    {"type": "parametric", "t": [0, 6.283185, 4000],
                     "params": {"m": m, "n1": 0.3, "n2": 1.7, "n3": 1.7},
                     "x": f"cos(t)*((abs(cos(m*(t+{k}*time)/4))**n2 + abs(sin(m*(t+{k}*time)/4))**n3)**(-1/n1))",
                     "y": f"sin(t)*((abs(cos(m*(t+{k}*time)/4))**n2 + abs(sin(m*(t+{k}*time)/4))**n3)**(-1/n1))",
                     "color": list(col), "width": width}]}

    if arch == "harmonograph":
        return {**base, "width": 720, "height": 720, "supersample": 2,
                "view": {"xmin": -1.2, "xmax": 1.2, "ymin": -1.2, "ymax": 1.2},
                "layers": [
                    {"type": "parametric", "t": [0, 62.83, 8000],
                     "x": f"exp(-0.01*t)*sin(2*t + {k}*time)*0.5 + exp(-0.008*t)*sin(3*t)*0.5",
                     "y": f"exp(-0.01*t)*sin(3*t)*0.5 + exp(-0.008*t)*sin(2*t + {k*0.7:.3f}*time)*0.5",
                     "color": list(col), "width": 1}]}

    if arch == "spiral":
        arms = sym or 1
        return {**base, "width": 720, "height": 720, "supersample": 2,
                "view": {"xmin": -1.3, "xmax": 1.3, "ymin": -1.3, "ymax": 1.3},
                "layers": [
                    {"type": "parametric", "t": [0, 37.7, 5000],
                     "x": f"0.03*t*cos({arms}*t + {k}*time)",
                     "y": f"0.03*t*sin({arms}*t + {k}*time)",
                     "color": list(col), "width": 2}]}

    if arch == "waves":
        return {**base, "width": 854, "height": 480, "axes": True,
                "view": {"xmin": -6.283, "xmax": 6.283, "ymin": -2.5, "ymax": 2.5},
                "layers": [
                    {"type": "explicit", "y": f"sin(x - {k}*time)*exp(-0.05*x*x)",
                     "color": list(col), "width": width},
                    {"type": "explicit", "y": f"0.6*sin(2*x + {k*1.5:.3f}*time)",
                     "color": list(_lighter(col)), "width": 1}]}

    if arch == "interference":
        return {**base, "width": 854, "height": 480,
                "view": {"xmin": -6, "xmax": 6, "ymin": -3.37, "ymax": 3.37},
                "layers": [
                    {"type": "field",
                     "f": f"sin(4*hypot(x-2*cos({k}*time), y-2*sin({k}*time))) + "
                          f"sin(4*hypot(x+2*cos({k}*time), y+2*sin({k}*time)))",
                     "min": -2, "max": 2, "cmap": cmap}]}

    if arch == "mandala":
        s = sym or random.choice([5, 6, 7, 8])
        return {**base, "width": 720, "height": 720,
                "view": {"xmin": -3.14, "xmax": 3.14, "ymin": -3.14, "ymax": 3.14},
                "layers": [
                    {"type": "field",
                     "f": f"sin({s}*theta + {k*3:.3f}*time)*sin(5*r - {k}*time) + cos(4*r + theta)",
                     "min": -2, "max": 2, "cmap": cmap}]}

    if arch == "stickman":
        # animated walk is rendered per-frame by stickman.render_from_spec;
        # this static mid-stride pose is only a defensive fallback.
        import stickman
        return stickman.static_scene(0.5, col, bg, 854, 480, steps=6.0)

    # unknown arch -> safe default
    return build_scene({**spec, "arch": "lissajous"})


def _lighter(rgb, f=0.6):
    return tuple(int(c + (255 - c) * f) for c in rgb)


# --------------------------------------------------------------------------- #
# DSL <-> string
# --------------------------------------------------------------------------- #

DSL_KEYS = ["arch", "action", "color", "cmap", "speed", "sym", "mood"]


def spec_to_dsl(spec: dict) -> str:
    parts = []
    for key in DSL_KEYS:
        if key in spec and spec[key] not in (None, "", 0):
            parts.append(f"{key}={spec[key]}")
    return " ".join(parts)


def dsl_to_spec(text: str) -> dict:
    """Parse a DSL string (possibly noisy model output) into a validated spec."""
    spec = {}
    for tok in text.replace(",", " ").split():
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        k = k.strip().lower()
        v = v.strip().lower()
        if k in DSL_KEYS and v:
            spec[k] = v
    return validate_spec(spec)


def validate_spec(spec: dict) -> dict:
    """Coerce a raw spec to a valid, renderable one."""
    out = {}
    out["arch"] = spec.get("arch") if spec.get("arch") in ARCH_NAMES else "lissajous"
    if spec.get("color") in COLORS:
        out["color"] = spec["color"]
    if spec.get("cmap") in set(CMAPS.values()):
        out["cmap"] = spec["cmap"]
    out["speed"] = spec.get("speed") if spec.get("speed") in SPEED_COEF else "med"
    out["mood"] = spec.get("mood") if spec.get("mood") in MOOD_BG else "dark"
    try:
        s = int(spec.get("sym", 0))
        if 2 <= s <= 16:
            out["sym"] = s
    except (ValueError, TypeError):
        pass
    # action only meaningful for stickman; default to walk
    if out["arch"] == "stickman":
        out["action"] = spec.get("action") if spec.get("action") in ACTIONS \
            else "walk"
    return out


# --------------------------------------------------------------------------- #
# Rule-based fallback: prompt -> spec (used when the model output is garbage)
# --------------------------------------------------------------------------- #

def parse_prompt_rules(prompt: str) -> dict:
    p = prompt.lower()
    spec = {}

    # archetype: match any known noun
    best = None
    for arch, names in ARCH_NAMES.items():
        for nm in names:
            if nm in p:
                best = arch
                break
        if best:
            break
    if not best:  # single-word hints
        for arch in ARCH_NAMES:
            if arch in p:
                best = arch
                break
    spec["arch"] = best or "lissajous"

    # detect an action word; an action + a character noun (or a stick word)
    # means the user wants the stickman, doing that action.
    import re as _re
    action = None
    for w, a in ACTION_WORDS.items():
        if _re.search(rf"\b{w}\b", p):
            action = a
            break
    if action:
        spec["action"] = action
        has_char = any(_re.search(rf"\b{c}\b", p) for c in CHARACTER_WORDS)
        if spec["arch"] == "stickman" or has_char or "stick" in p:
            spec["arch"] = "stickman"

    for w, c in COLOR_WORDS.items():
        if w in p:
            spec["color"] = c
            break
    for w, cm in CMAPS.items():
        if w in p:
            spec["cmap"] = cm
            break
    for w, s in SPEED_WORDS.items():
        if w in p:
            spec["speed"] = s
            break
    for w, m in MOOD_WORDS.items():
        if w in p:
            spec["mood"] = m
            break

    # symmetry: digit or number-word before a count noun, else bare digit
    import re
    m = re.search(r"(\d+)\s*(?:petal|point|fold|arm|side|leaf|lobe|spike)", p)
    if not m:
        for w, n in COUNT_WORDS.items():
            if re.search(rf"\b{w}\b\s*(?:petal|point|fold|arm|side|leaf|lobe|spike)", p):
                spec["sym"] = n
                break
    if m:
        spec["sym"] = int(m.group(1))

    return validate_spec(spec)
