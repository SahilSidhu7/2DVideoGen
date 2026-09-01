"""
2DVideoGen — prompt -> scene script -> rendered animation, entirely on CPU.

    prompt  ->  t5-small (v3-staged)  ->  .scene DSL  ->  scenescript.py  ->  .mp4

Nothing here touches a GPU. The symbolic intermediate is shown next to the
video on purpose: the DSL is the interesting part of the project.
"""
import os

# transformers 4.57 tries a Keras 3 backend and dies unless this is set BEFORE
# transformers is imported anywhere. Keep this above every other import.
os.environ["USE_TF"] = "0"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "model"))

import scenescript                                   # noqa: E402
from model.scene_infer import SceneWriter            # noqa: E402

CKPT = os.environ.get("SCENE_CKPT", "SahilSidhu/2dvideogen-scene-writer")
OUTDIR = Path(tempfile.gettempdir()) / "2dvideogen_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

NUM_SAY = {1: "one", 2: "two", 3: "three", 4: "four",
           5: "five", 6: "six", 7: "seven", 8: "eight"}

_writers = {}


def get_writer(unique_colours, temperature, seed):
    """SceneWriter builds its logits processor at construction time, so cache
    one per (colour-processor) setting and re-seed per call instead."""
    key = bool(unique_colours)
    w = _writers.get(key)
    if w is None:
        w = SceneWriter(CKPT, max_out=256, max_in=160,
                        unique_colours=unique_colours, sample=True,
                        top_p=0.95, temperature=float(temperature),
                        seed=int(seed))
        _writers[key] = w
    w.temperature = float(temperature)
    w.sample = True
    w.torch.manual_seed(int(seed))
    return w


def generate(prompt, cast_hint, temperature, seed, unique_colours, max_seconds=6.0):
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "", "Type a prompt first."
    if cast_hint and int(cast_hint) > 0:
        prompt = "%s characters: %s" % (NUM_SAY[int(cast_hint)], prompt)

    t0 = time.time()
    try:
        w = get_writer(unique_colours, temperature, seed)
        r = w.write(prompt)
    except Exception:
        return None, "", "Model failed:\n" + traceback.format_exc(limit=3)
    t_gen = time.time() - t0

    if r["parse_error"]:
        return None, r["raw"], (
            "The model emitted a string the strict parser rejected:\n  %s\n\n"
            "This happens on about 1%% of prompts under sampling. Change the "
            "seed and try again." % r["parse_error"])

    scene_text = r["scene"]
    t1 = time.time()
    try:
        spec = scenescript.parse_script(scene_text)
        # The free CPU tier renders roughly 5 frames a second, so cap the clip
        # rather than making someone wait three minutes for a 12-second scene.
        if max_seconds and spec.get("seconds", 0) > float(max_seconds):
            spec["seconds"] = float(max_seconds)
        sc = scenescript.Scene(spec)
        out = OUTDIR / ("scene_%d.mp4" % int(t1 * 1000))
        n = scenescript.render(sc, out=str(out), verbose=False)
    except Exception:
        return None, scene_text, "Render failed:\n" + traceback.format_exc(limit=3)
    t_render = time.time() - t1

    if not out.exists() or out.stat().st_size < 2048:
        return None, scene_text, (
            "ffmpeg produced no usable file (%s bytes). ffmpeg on PATH: %s"
            % (out.stat().st_size if out.exists() else 0, shutil.which("ffmpeg")))

    notes = "; ".join(sc.notes) if sc.notes else "none"
    status = (
        "**Rendered.** %d characters, %d props, %d timeline events, "
        "%d frames.\n\n"
        "- structural problems in the raw output: **%s**\n"
        "- repairs applied before rendering: **%s**\n"
        "- compositor notes: %s\n"
        "- generation %.1f s, render %.1f s (CPU)\n"
        % (len(sc.cast), len(sc.props), len(scenescript.spec_timeline(spec)), n,
           r["problems"] or "none", r["repairs"] or "none", notes,
           t_gen, t_render))
    return str(out), scene_text, status


EXAMPLES = [
    ["two friends meet in the park, one waves, then they kick a ball around", 0, 1.0, 23, True, 6.0],
    ["six stick figures in the park, all of them dancing", 0, 1.0, 23, True, 6.0],
    ["a long clip in the park where four people take turns: first one waves, then the next runs across, then the third jumps, then the fourth dances", 0, 1.0, 7, True, 8.0],
    ["scene: nightfall on a city street. a trio of strangers wander past each other without stopping", 0, 1.0, 23, True, 6.0],
    ["a park with a tree, a bush, a rock, a crate, a cloud and the sun, and two people waving under it all", 0, 1.0, 23, True, 6.0],
    ["a green figure, a pink figure and a white figure in the park, all three waving", 0, 1.0, 23, True, 6.0],
]

DESCRIPTION = """
# 2DVideoGen

A **60M-parameter t5-small** writes a symbolic **scene script**; a pure-Python compositor renders
that script to an `.mp4`. The whole path is CPU-only — the model trained in 548.5 seconds on a
laptop CPU, and this Space runs on the free CPU tier.

```
prompt  ->  t5-small  ->  scene-script DSL  ->  compositor  ->  video
```

The DSL is shown next to the video because it is the point: the model is not generating pixels,
it is writing a program that a deterministic renderer executes.

**Decoding is sampled, not beam search, and that is deliberate.** On identical weights, beam
search stages a cast at 0.188 with 3 distinct positions across 30 scenes; sampling stages at
0.862 with 55. Beam search reports the mode of a wide distribution and makes the model look as
though it never learned staging. First render takes a little longer while the model downloads.
"""

FOOTER = """
### What it can do

Casts of **1 to 8** characters, three backgrounds (park, night street, studio), a fixed action
set (walk, run, jump, kick, wave, dance, idle), a ball that characters can pass between them, and
a small prop list (tree, bush, rock, crate, cloud, sun). Out-of-distribution prompts parse and
render **100%** of the time, and satisfy every stated field **75.0%** of the time under sampling.

### What it cannot do

- **Cast size saturates at the trained ceiling.** Ask for nine and it gives you six, every time.
  The wall was moved from 6 to 9 by widening the training range; it was not removed.
- **Spatial language is not obeyed above chance.** "On the left", "facing each other" and similar
  gave about +2.7 points over a mismatched-pair control. That was a stated prediction, and it was
  falsified.
- **Sampling costs obedience.** In distribution it buys 0.693 of staging for roughly 13-20 points
  of action and layout fidelity. Occasionally it emits a token outside the closed DSL vocabulary
  and the strict parser rejects the whole scene — change the seed if that happens.
- Stick figures, not styled animation. Styled output was ruled out by measurement at this
  hardware budget, not by opinion.

### More

[GitHub repository](https://github.com/SahilSidhu7/2DVideoGen) ·
[model card](https://huggingface.co/SahilSidhu/2dvideogen-scene-writer) ·
the paper and the full log of 23 attempts, including the failures and their real causes, are in
`paper/PAPER.md` and `ATTEMPTS.md` in the repo.
"""

with gr.Blocks(title="2DVideoGen") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="two friends meet in the park, one waves, then they kick a ball around",
                lines=3)
            with gr.Row():
                cast_hint = gr.Slider(
                    0, 8, value=0, step=1, label="Cast size hint",
                    info="0 = let the prompt decide. Anything else restates the "
                         "number in words, because the model reads the numeral.")
                temperature = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.05, label="Temperature",
                    info="Sampling temperature. 1.0 is the measured configuration.")
            with gr.Row():
                seed = gr.Number(value=23, precision=0, label="Seed")
                unique_colours = gr.Checkbox(
                    value=True, label="Unique cast colours",
                    info="Constrained decoding. Takes duplicate-colour "
                         "violations from 71/150 to 0/150.")
            max_seconds = gr.Slider(
                3.0, 12.0, value=6.0, step=0.5, label="Clip length cap (seconds)",
                info="The renderer manages about 5 frames a second on this free "
                     "CPU box. Longer clips work, they just take longer.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=4):
            video = gr.Video(label="Rendered animation", autoplay=True)
            status = gr.Markdown()
    scene_out = gr.Code(label="Generated scene script (the model's actual output)",
                        language=None, lines=8)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt, cast_hint, temperature, seed, unique_colours, max_seconds],
        outputs=[video, scene_out, status],
        fn=generate,
        cache_examples=False,
        label="Prompts from the project's out-of-distribution set")

    gr.Markdown(FOOTER)

    go.click(generate,
             inputs=[prompt, cast_hint, temperature, seed, unique_colours, max_seconds],
             outputs=[video, scene_out, status])
    prompt.submit(generate,
                  inputs=[prompt, cast_hint, temperature, seed, unique_colours, max_seconds],
                  outputs=[video, scene_out, status])

if __name__ == "__main__":
    demo.queue(max_size=12).launch()
