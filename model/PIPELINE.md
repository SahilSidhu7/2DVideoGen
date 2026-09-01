# Two-stage learned equation generator

Turns a prompt into a video where **the equations for every frame are produced
by a trained neural model**, not by a hand-written script.

```
 prompt
   │
   ▼  Stage A — planner (PromptModel: rules + t5-small)
 scene plan  {subject, action, style, duration}  ->  per-frame descriptors
   │
   ▼  Stage B — PoseNet (trained MLP)
 for each frame:  descriptor -> 14-d pose vector -> that frame's equations
   │
   ▼  render (geovid)  ->  mp4  +  model-produced equations.jsonl
```

## Why it's built this way (honest scope)

There is **no dataset that maps real anime frames to equations**, and learning
that from raw pixels needs GPUs and a large image-vectorization effort — not
feasible on medium hardware. So the equation engine (`stickman.py`) is used as a
**teacher**: render it to get perfectly-labeled `(image, tags, pose, equations)`
data, then train small models to reproduce and generalize it. This is standard
self-distillation and it's what makes the whole thing trainable on a CPU in
minutes.

Consequence: a distilled model imitates the teacher — it won't invent motion the
engine can't express. Its value is (a) a continuous, blendable pose manifold
(smoother interpolation, mixing actions) and (b) a real, extensible ML pipeline:
drop in **real** `(image → pose)` labels later and the same `ImgPoseNet` trains
on them with zero code change.

## The trained models

| model | file | maps | size | accuracy vs engine |
|-------|------|------|------|--------------------|
| PoseNet | `pose_model.py` | frame descriptor → 14-d pose vector | 81 k | **< 0.5° / joint** |
| ImgPoseNet | `pose_model.py` | 80×80 frame image → pose vector | 70 k | ~6° / joint |
| Planner | `infer.py` (t5-small) | prompt → scene/action tags | 60 M | 100% action acc |

The **pose vector** (14 numbers: hip x/y, 9 joint angles, ball x/y + flag) maps
one-to-one to a frame's equations via `StickFigure.pose_vector` /
`equations_from_vector` (exact round-trip verified). Learning this low-dim vector
— instead of generating equation *text* — is what keeps the numbers stable.

## Reproduce

```bash
cd model
python frames_dataset.py --poses 20000 --images 3500   # build corpus from engine
python pose_model.py train-pose                         # -> checkpoint/posenet.pt
python pose_model.py train-image                        # -> checkpoint/imgposenet.pt
python pose_model.py eval                               # per-action degree error

# generate with the learned pipeline
python pipeline.py "a stickman playing football" -o ../out/nn_kick.mp4
python pipeline.py "a green stickman running"    -o ../out/nn_run.mp4 --seconds 4
```

`pipeline.image_to_equations(png, out_png)` runs the image→equations model on any
frame image and re-renders it from the recovered equations.

## Using the neural path in the web app

The web app renders stickman via the engine by default (stable). To drive it with
the trained PoseNet instead, start the server with `EQV_NEURAL=1`:

```bash
EQV_NEURAL=1 USE_TF=0 python webapp/server.py --port 5057
```

`render.render_spec` then routes stickman prompts through
`pipeline.Pipeline.render_from_spec` (Stage B = PoseNet). Non-stickman prompts
are unaffected.

## Extending with real data

To make it learn from actual anime / video (beyond the engine):
1. Get frames + a pose label per frame (e.g. an off-the-shelf 2D pose estimator,
   or manual joint annotation). Store as `(image, pose_vector)`.
2. Append to `data/images.npz` and retrain `ImgPoseNet` — same code.
3. Widen the pose vector / add subjects & props in `stickman.py` +
   `frames_dataset.py` to grow expressivity, then regenerate + retrain.
```
