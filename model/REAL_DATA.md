# Real-data pipeline (GPU)

This turns **real photos and videos into our equation skeleton**, and trains our
own models on real, GPU-accelerated data — no synthetic-only limits.

## Hardware / stack

- GPU: **RTX 4060 Laptop (8 GB)**, driver 592.82
- `torch 2.13.0+cu126` (CUDA build), `torchvision`, `ultralytics 8.4` (YOLO-pose)
- Enable with a CUDA torch install:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

## Data

Real images: **COCO val2017** (5 000 photos) auto-downloaded from cocodataset.org.
Real pose labels: **YOLO11-pose** (trained on real COCO keypoints) run on the GPU
over every image → **5 051 real person poses** mapped to our 14-d pose vector.

```
real photo ──YOLO-pose──▶ 17 COCO keypoints ──pose_map──▶ our 14-d vector
          ──equations_from_vector──▶ stick-figure equations
```

## Modules

| file | role |
|------|------|
| `pose_map.py` | COCO-17 keypoints → our pose vector (legs/torso recover **exactly**; validated) |
| `vectorize.py` | real image / video → equations (+ short clip, + equations.jsonl). Also `build_dataset` over a folder |
| `build_real_dataset.py` | YOLO-pose labels a folder of photos → `data/real_imgs.npz` (crop + vector) |
| `pose_model.py train-real` | trains `ImgPoseNet` on real crops, **on GPU**, with horizontal-flip augmentation + noise |

## What's trained

- **ImgPoseNet_real** (`checkpoint/imgposenet_real.pt`, 70 k params): real person
  crop → pose vector → equations. Trained on 5 051 real + synthetic on the 4060.
- The **production** image/video→equations path uses **YOLO-pose directly** (most
  accurate). ImgPoseNet_real is our own compact distilled model — a small net on
  messy real poses, so it's lower-fidelity than YOLO; use it when you want a tiny
  dependency-free vectorizer.

## Use

```bash
# real image -> equations (reconstructed stick figure)
python model/vectorize.py image path/to/photo.jpg -o out/vec.png
# real video -> per-frame equations + reconstructed animation
python model/vectorize.py video path/to/clip.mp4  -o out/vec.mp4
# build a real dataset from any folder of media
python model/vectorize.py dataset my_videos/ -o data/real_poses.npz
# (re)train the compact real model on GPU
python model/pose_model.py train-real
```

In the **web app**: the 📎 button uploads a real image or video; the server runs
YOLO-pose, reconstructs the figure from equations, and returns the clip plus a
**∑ equations** download. (`POST /api/chats/<id>/vectorize`.)

## Real-video MOTION training

Beyond single frames, we learn a **motion prior from real video clips**.

```
real clip ──YOLO-pose per frame──▶ pose sequence ──▶ RealMotionNet(action,phase)
         ──equations per frame──▶ smooth generated animation
```

| file | role |
|------|------|
| `motion_dataset.py` | video folder → per-clip real pose sequences (`data/real_motion.npz`) |
| `motion_model.py` | **RealMotionNet**: `(action, phase)→pose` — smooth loop, one motion per action |
| `motion_ar.py` | **MotionAR**: GRU, `history → next pose` — non-looping, evolving; handles multiple differing clips per action |

Data: **side-view Muybridge motion studies** from Wikimedia Commons (public
domain) — 6 clips (3 walk, 1 jump/somersault, 2 dance), **3 746 frames**.
Side-view gait is clean, unlike the earlier frontal clips.

Two motion models:

- **RealMotionNet** (phase→pose): a smooth function of phase, so it can never
  drift or collapse — but it *averages* multiple clips of an action into one loop
  (blurs differing walks). Good for a single clean clip.
- **MotionAR** (autoregressive GRU, **default**): predicts the next pose from a
  window of recent poses, seeded with real frames and rolled out. Non-looping and
  evolving, and it does **not** blur differing clips. Trained with input noise
  (scheduled sampling) for stable rollout — verified coherent over 5 s. Much lower
  next-step error (val ~0.0006 vs 0.045 for the phase model).

```bash
# fetch side-view clips -> datasets/videos_side/ , convert gif->mp4, then:
python model/motion_dataset.py datasets/videos_side/mp4 -o data/real_motion.npz
python model/motion_ar.py train                 # autoregressive (default)
python model/motion_model.py train              # phase model (alt)
python model/motion_ar.py generate walk  -o out/ar_walk.mp4
python model/motion_ar.py generate jump  -o out/ar_jump.mp4
```

Enable in the pipeline / web app with `EQV_REALMOTION=1`; pick the backend with
`EQV_MOTION=ar` (default) or `EQV_MOTION=phase`. Stickman prompts whose action was
learned (walk, jump, dance) render from real motion; other actions fall back to
the hand-scripted engine.

To add actions or improve quality, drop more clean side-view clips in a folder and
rerun `motion_dataset.py` + `motion_ar.py train`.

## Honest scope

- YOLO-pose is single-frame; long videos get light temporal smoothing to reduce
  jitter, but fast motion / occlusion still causes wobble.
- Our skeleton is a single stick figure (one person, no props from photos).
- To go further: fine-tune YOLO-pose on more sports footage, widen the pose
  vector (hands, feet orientation), add multi-person, and add a learned temporal
  model over sequences (needs real video clips — the tool builds that dataset for
  you via `vectorize.py dataset`).
