# scenes/ — the `.scene` script format

A `.scene` file is the project's intermediate representation: a short, plain-text
description of a shot (who is in it, where they stand, what is in the scenery,
and who does what when). It is the thing a human or a model writes, and the thing
the compositor renders deterministically. The same script always gives the same
frames.

```
natural language  →  model/scene_infer.py  →  scenes/*.scene  →  scenescript.py  →  .mp4 / PNG frames  →  critic/evaluate.py
```

- [`model/scene_infer.py`](../model/scene_infer.py) writes `.scene` files from a prompt.
- [`scenescript.py`](../scenescript.py) parses (`parse_script`) and renders them.
- [`critic/evaluate.py`](../critic/evaluate.py) scores the rendered frames.

Everything on this page describes what `scenescript.py` accepts today. Where the
parser takes something the renderer then ignores, the page says so.

---

## Grammar

The format is line based.

- Everything after `#` on a line is a comment. That includes `#` inside a title.
- Blank lines are ignored.
- Tokens are separated by whitespace. Keywords are matched as written, except the
  first word of a line and a timeline action, which are lower-cased.
- A line is a **header directive**, a **`prop`** line, a **`cast`** line, or (if
  it starts with anything else) a **timeline** line. A line that cannot be
  parsed raises `ValueError: scene script line N: ...`.

### Header directives

| directive | form | default | meaning |
|---|---|---|---|
| `title` | `title <free text>` | `untitled` | Shot name. Printed when rendering; not drawn. |
| `size` | `size <width> <height>` | `854 480` | Output resolution in pixels (integers). |
| `fps` | `fps <n>` | `30` | Frame rate (integer). |
| `seconds` | `seconds <s>` | `8.0` | Clip length. The CLI flag `--seconds` overrides it. |
| `background` | `background <name>` | `park` | One of `park`, `night`, `studio`. An unknown name falls back to `park`. |

The three backgrounds are the same layout (sky, two hill silhouettes, ground,
ground line) in different palettes (`BACKGROUNDS` in `scenescript.py`).

### `cast` lines

```
cast <id> [color <name>] [at <x>] [scale <s>] [y <y>] [facing <1|-1>] [depth <d>]
```

| option | default | meaning |
|---|---|---|
| `<id>` | (required) | Name used by timeline lines. |
| `color` | `blue` | A colour name from the table below. An unknown name renders as a light blue `(120, 220, 255)`. |
| `at` | `5.0` | Starting x position, world units (see [Coordinates](#coordinates)). |
| `scale` | `1.0` | Body size. Smaller reads as further away: unless `y` is given, the figure stands higher up the frame, and it is drawn behind larger figures. |
| `y` | `1.15 + (1 − scale) × 1.05` | Ground line the feet stand on, world units. Overrides the scale-derived value. |
| `facing` | `1` | Initial facing, `1` = right, `-1` = left. Walking, running and kicking change it. |
| `depth` | equal to `scale` | Draw order. Lower is drawn first, so it sits behind. |

### `prop` lines

```
prop <kind> [<id>] [at <x>] [scale <s>] [y <y>] [depth <d>]
```

If the word after the kind is not an option keyword, it is the prop's id;
otherwise the id is the kind itself (`prop ball at 6.2` makes a ball with id `ball`).

| kind | notes |
|---|---|
| `tree` | Trunk and three-blob canopy. |
| `bush` | Half-ellipse. |
| `rock` | Five-point polygon. |
| `box` | Outlined crate. |
| `sun` | Disc in the sky. Default `y` is `4.5`. |
| `cloud` | Three blobs. Default `y` is `4.4`. |
| `ball` | The only **stateful** prop. Characters can kick it, and it remembers where it landed. |

Other options:

- `at` defaults to `5.0` (a ball defaults to the middle of the frame).
- `scale` defaults to `1.0`.
- `y` is the prop's base line (for `sun`/`cloud`, its centre height). The default is the ground, `1.15`.
- `depth` sets draw order. The default is the prop's `scale`, or `-1` for `sun`
  and `cloud` so they always sit behind everything. Balls are drawn at `1.02`.
- An unknown kind parses fine and draws nothing.
- `color` is accepted by the parser but **ignored by the renderer**. Prop colours are fixed.

### Timeline lines

```
<start> <end> <who> <action> [<prop>] [to <x>] [at <target>]
```

`<start>` and `<end>` are seconds and `<who>` is a cast id. Events are compiled in
start-time order. If two events for the same character overlap, the one that
started later wins. A character with no event running stands idle.

| action | modifiers | what happens |
|---|---|---|
| `walk` | `to <x>` | Walks from its current x to `x`. The step count comes from the distance, so feet do not slide. Without `to`, it walks on the spot. |
| `run` | `to <x>` | Same as `walk`, with a longer stride. |
| `jump` | (none) | Jumps in place. |
| `kick` | `[<prop>] [at <target>]` | Walks to where the ball is *at that moment*, faces it, and kicks. `<prop>` is the ball's id (default `ball`; `prop <id>` also works). `at` takes a cast id (the ball flies to where that character will be) or a number (an x position). Without `at`, the ball travels 3 units forward. It lands inside `[0.4, W − 0.4]` and rolls a little further. |
| `wave` | (none) | Waves. |
| `dance` | (none) | Dances. |
| `idle` | (none) | Stands with a slight idle motion. |

Problems in the timeline do not stop the render. `Scene.notes` records them and
the CLI prints them as `note: ...` lines:

- an unknown cast id skips the event;
- an unknown action becomes `idle`;
- a `kick` with no matching ball is noted and plays without a ball;
- a kick that starts while the ball is still moving is noted, and the ball is struck in mid-flight.

`validate()` does not catch a character kicking the ball at itself
(`fay kick ball at fay`, in `a24_six.scene`). It renders.

### Colours

Taken from `model.grammar.COLORS`:

`blue` · `cyan` · `teal` · `red` · `orange` · `amber` · `yellow` · `green` · `lime` · `purple` · `violet` · `pink` · `magenta` · `white` · `gold`

### Coordinates

The world is measured in units, not pixels.

- **Height** is fixed at `5.62` units (`WORLD_H`). `y = 0` is the bottom of the frame.
- **Width** is `W = (width / height) × 5.62`. At the default `854 × 480` that is
  **x from 0 to 10.0**. This is the range `at` and `to` use.
- **The ground line** is at `y = 1.15` (`GROUND`). A `scale 1.0` character stands
  on it. A `scale 0.6` character stands at `1.15 + 0.4 × 1.05 = 1.57`, so it looks
  further back.
- **`scale`** multiplies the rig's limb lengths and head radius, not just the drawing.

---

## Walk-through: `park_meet.scene`

This is the hand-written reference scene from Attempt 21.

```
title      Two friends meet in the park
size       854 480          # 854x480 -> x runs 0 .. 10.0
fps        30
seconds    12.5             # 375 frames
background park

prop tree  bigtree  at 0.55 scale 1.15            # foreground tree, left edge
prop tree  fartree  at 9.30 scale 0.70 y 1.55     # small and raised -> reads as distant
prop bush  bush1    at 3.40 scale 0.95
prop rock  rock1    at 7.80 scale 0.80
prop cloud cloud1   at 2.60 scale 1.10 y 4.60     # sky props: always drawn behind
prop sun   sun1     at 8.70 y 4.70
prop ball  ball     at 6.40                       # the one prop the cast can act on

cast ana color cyan   at 1.70 scale 1.00          # front
cast bo  color orange at 9.40 scale 0.94
cast cy  color pink   at 0.90 scale 0.60          # background walker, stands at y≈1.57

0.0  3.2  ana walk to 3.90      # the two friends close the gap ...
0.2  3.4  bo  walk to 6.60
0.0  7.0  cy  walk to 9.60      # ... while cy crosses the whole frame behind them
3.4  5.0  ana wave              # greet
3.6  5.0  bo  wave
5.2  7.2  bo  kick ball at ana  # bo walks to the ball at 6.40 and kicks it to ana
8.4 10.2  ana kick ball at bo   # ana walks to wherever it landed and kicks it back
7.2 12.5  cy  dance
10.6 12.4 bo  jump
```

The two kicks are the reason the ball is stateful. Ana's kick at 8.4 s starts from
wherever bo's kick left the ball, not from the ball's `at`.

## Files in this directory

| file | origin | attempt |
|---|---|---|
| `park_meet.scene` | hand-written | 21: the reference scene ([`out/a21_park.mp4`](../out/a21_park.mp4)) |
| `street_relay.scene` | hand-written | 21: a fresh script written after the engine was frozen, to test that it generalises (night, 4 cast, three-way relay) |
| `a22_generated.scene` | model (`model/scene_infer.py`) | 22: the first end-to-end generation, from the park prompt |
| `a22_generated3.scene` | model | 22: three-character generation |
| `a22_batch_ind0-4.scene` | model | 22: in-distribution prompt batch |
| `a22_batch_ood0-4.scene` | model | 22: out-of-distribution prompt batch (`ood1` scored NOT-AN-ANIMATION) |
| `a23_generated.scene` | model, v3, sampling + rerank | 23: the same park prompt as Attempt 22, for comparison |
| `a23_six.scene` | model, v3, sampling + rerank | 23: six characters on a night street |
| `a24_six.scene` | model, v4 (t5-base, 220M) | 24: the same prompt and seed as `a23_six` |
| `my_scene.json` | hand-written | **Not a scene script.** It is a raw [`geovid.py`](../geovid.py) equation scene (`python geovid.py render scenes/my_scene.json -o out.mp4`). |

The model writes a `# generated by model/scene_infer.py` comment on the first
line, and cuts the prompt to 60 characters for the `title`.

## JSON form

`scenescript.py` also accepts JSON. A file is read as JSON if its name ends in
`.json` or its text starts with `{`. The JSON is the dict that `parse_script`
returns, so the keys differ slightly from the text form: `width`/`height`
instead of `size`, and arrays of objects for `cast`, `props` and `timeline`.

```json
{
  "title": "two figures", "width": 854, "height": 480, "fps": 30,
  "seconds": 6, "background": "studio",
  "props":    [{"kind": "ball", "id": "ball", "at": 5.0}],
  "cast":     [{"id": "a", "color": "cyan", "at": 2.0, "scale": 1.0},
               {"id": "b", "color": "orange", "at": 8.0, "scale": 0.8}],
  "timeline": [{"start": 0.0, "end": 2.0, "who": "a", "action": "walk", "to": 4.0},
               {"start": 2.2, "end": 4.0, "who": "a", "action": "kick", "prop": "ball", "at": "b"}]
}
```

To see the JSON for any text script:
`python -c "import scenescript,json;print(json.dumps(scenescript.parse_script(open('scenes/park_meet.scene').read()),indent=1))"`.

## Commands

All of these run on CPU only.

```bash
# render to mp4, and optionally keep the PNG frames
python scenescript.py scenes/park_meet.scene -o out/park.mp4 --frames out/park_frames

# override the script's length
python scenescript.py scenes/park_meet.scene -o out/park_short.mp4 --seconds 4

# built-in checks: static backdrop, independent timelines, feet on the ground, ball kicked and landed
python scenescript.py --selftest

# per-character presence/occlusion audit of rendered frames (needs opencv)
python scenescript.py scenes/park_meet.scene --audit out/park_frames
```

`--audit` checks a declared character's colour on every 5th frame. It reports
how often each character is visible and how many separate blobs its ink forms.
It measures presence and occlusion, not whether the figure looks like a person.
It exists because the critic's pose detector sees no people in this style at all.
Writing `.mp4` needs `ffmpeg` on PATH.
