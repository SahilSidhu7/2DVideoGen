"""The six metrics. Each function takes numpy RGB frames (and sometimes a
YOLO pose model, or a CLIP model) and returns a plain dict — numbers only, no
verdicts. Verdicts are verdict.py's job; this file never decides pass/fail.

Everything here runs on CPU except the YOLO pose calls, which are the only GPU
use in the critic and are gated with tools.gpuguard when a model is supplied.
CLIPSIM (metric 6) is CPU-only by construction — see content_correctness.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "svg"))


def gray(frame):
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. Temporal stability
# --------------------------------------------------------------------------- #

def mean_abs_diff(frames):
    """Mean absolute inter-frame pixel change, normalised to [0, 1]."""
    if len(frames) < 2:
        return 0.0
    grays = [gray(f) for f in frames]
    diffs = [np.abs(grays[i + 1] - grays[i]).mean() / 255.0
             for i in range(len(grays) - 1)]
    return float(np.mean(diffs))


def warp_error(frames, max_pairs=60):
    """Optical-flow warp residual: warp frame t forward to t+1, measure what's
    left over. Low residual means the change between frames is *explained by
    motion* (legitimate movement); high residual means content is popping in
    and out that no motion field can account for (the real flicker signal —
    raw pixel diff alone can't tell a fast-moving character from a flickering
    one, this can).
    """
    import cv2
    if len(frames) < 2:
        return 0.0
    grays = [gray(f) for f in frames]
    idx = list(range(len(grays) - 1))
    if len(idx) > max_pairs:
        idx = np.linspace(0, len(idx) - 1, max_pairs).astype(int).tolist()
    residuals = []
    h, w = grays[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    for i in idx:
        a, b = grays[i], grays[i + 1]
        if a.shape != b.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            a.astype(np.uint8), b.astype(np.uint8), None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped = cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        residuals.append(float(np.abs(warped - b).mean() / 255.0))
    return float(np.mean(residuals)) if residuals else 0.0


def flicker_ratio(out_frames, guide_frames):
    """output inter-frame change / input-guide inter-frame change.

    Attempt 18 (ATTEMPTS.md) measured 23.6x on the SD1.5+ControlNet hybrid:
    the guide moved by 1.29 (grayscale mean abs diff), the styled output by
    30.44. Returns None when no guide is supplied — the ratio is undefined
    without a reference for "the motion we asked for."
    """
    if not guide_frames:
        return None
    g_out = mean_abs_diff(out_frames)
    g_in = mean_abs_diff(guide_frames)
    return float(g_out / max(g_in, 1e-6))


def temporal_stability(out_frames, guide_frames=None):
    return dict(
        mean_abs_diff=mean_abs_diff(out_frames),
        warp_error=warp_error(out_frames),
        flicker_ratio=flicker_ratio(out_frames, guide_frames),
    )


# --------------------------------------------------------------------------- #
# 2. Identity preservation
# --------------------------------------------------------------------------- #

def _center_crop_box(frame, frac=0.7):
    h, w = frame.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


def character_boxes(frames, yolo_model=None, guard=None):
    """One bounding box per frame around the character — either ALL from
    YOLO detection or ALL from the fallback centre crop, never a mix.

    BUG THIS FIXES (found by the scene lane, Attempt 21, ATTEMPTS.md ~line
    1180): the previous version chose per-frame between a YOLO box and a
    centre-crop fallback independently for every frame. On
    `out/a21_street.mp4` YOLO fired on exactly 1 of 24 sampled frames,
    producing a 48x135 detection box wildly different in scale and position
    from the identical centre box used by the other 23. `identity_preservation`
    then compared that one outlier crop against the rest and reported
    worst_similarity 0.4368 — below the 0.45 pass bar — on a clip the scene
    lane's own ink-presence audit confirmed had all 4 characters visible in
    84/84 sampled frames. Forcing the fallback crop for every frame gives
    0.9972. The clip never changed; the *measurement basis* changed mid-clip,
    and the metric reported that as an identity change. A single stochastic
    detection was worth more than half the metric's range.

    The fix: decide ONE crop regime for the whole clip, using the same
    "refuse to score on noise" bar `pose_fidelity` already applies
    (MEASURABLE_DETECTION_RATE = 0.50 confident-detection rate; see metric 4).
    Below that bar, or with no detector at all, every frame uses the centre
    crop. At or above it, every frame uses a YOLO-derived box: frames with a
    confident detection use it directly, and a frame without one reuses the
    nearest confident detection in time (held forward, or backward for any
    leading frames before the first one) rather than falling through to the
    centre-crop heuristic — so a "detection regime" clip never contains a
    single non-detection-derived box, and a "fallback regime" clip never
    contains a single detection-derived one. Returns (boxes, meta) — meta
    records which regime was used and why, so a report never has to guess.
    """
    n = len(frames)
    if yolo_model is None or n == 0:
        return [_center_crop_box(f) for f in frames], dict(
            regime="fallback", reason="no pose model", n_frames=n,
            n_confident=0, confident_rate=0.0)

    dets = []          # per-frame: box or None, and its confidence
    n_confident = 0
    confidences = []
    for f in frames:
        box, conf = None, 0.0
        r = yolo_model.predict(f[:, :, ::-1], verbose=False)[0]
        if guard is not None:
            guard.step()
        if len(r.boxes):
            areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * \
                    (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
            best = int(areas.argmax())
            conf = float(r.boxes.conf[best]) if r.boxes.conf is not None else 1.0
            x0, y0, x1, y1 = r.boxes.xyxy[best].tolist()
            box = (max(0, int(x0)), max(0, int(y0)),
                   min(f.shape[1], int(x1)), min(f.shape[0], int(y1)))
            if box[2] <= box[0] or box[3] <= box[1]:
                box = None
        dets.append((box, conf))
        if box is not None and conf >= DETECTION_CONF_FLOOR:
            n_confident += 1
            confidences.append(conf)

    confident_rate = n_confident / n
    meta = dict(n_frames=n, n_confident=n_confident, confident_rate=confident_rate,
                mean_confidence=float(np.mean(confidences)) if confidences else 0.0)

    if confident_rate < MEASURABLE_DETECTION_RATE:
        meta["regime"] = "fallback"
        meta["reason"] = ("confident detection rate %.0f%% < %.0f%% bar - "
                          "using the centre crop for every frame so no "
                          "detection-derived box is ever compared against a "
                          "fallback one within this clip"
                          % (confident_rate * 100, MEASURABLE_DETECTION_RATE * 100))
        return [_center_crop_box(f) for f in frames], meta

    meta["regime"] = "detection"
    meta["reason"] = ("confident detection rate %.0f%% >= %.0f%% bar - every "
                      "box is YOLO-derived (held over from the nearest "
                      "confident frame where none fired directly)"
                      % (confident_rate * 100, MEASURABLE_DETECTION_RATE * 100))
    boxes = [None] * n
    last = None
    for i, (box, conf) in enumerate(dets):
        if box is not None and conf >= DETECTION_CONF_FLOOR:
            last = box
        boxes[i] = last
    # backfill any leading frames before the first confident detection
    first = next((b for b in boxes if b is not None), None)
    boxes = [b if b is not None else first for b in boxes]
    return boxes, meta


def _embed(frame, box):
    """Colour-histogram embedding: HSV joint histogram, L2-normalised.

    Chosen as the default (over a learned CLIP embedding) because it needs no
    download and no VRAM — the project's hardware note explicitly allows this
    fallback. It is deliberately crude: it will not catch a same-silhouette
    swap to a different character wearing the same palette, only a change of
    colour scheme / composition. That limitation is stated up front rather
    than papered over with a fancier-sounding number.
    """
    import cv2
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float64)
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else hist


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identity_preservation(frames, yolo_model=None, guard=None, max_frames=60):
    """Mean and worst-case pairwise similarity of the character crop across
    the clip. Subsampled to max_frames to keep the O(n^2) comparison cheap.

    All compared crops come from ONE regime — see character_boxes(). The
    regime and the detection stats behind it are always reported
    (`crop_regime`, `crop_regime_reason`, `confident_detection_rate`) so a
    low-detection clip's result is never presented without the context that
    explains it.
    """
    idx = list(range(len(frames)))
    if len(idx) > max_frames:
        idx = np.linspace(0, len(idx) - 1, max_frames).astype(int).tolist()
    sub = [frames[i] for i in idx]
    boxes, crop_meta = character_boxes(sub, yolo_model, guard)
    embeds = [_embed(f, b) for f, b in zip(sub, boxes)]
    if len(embeds) < 2:
        return dict(mean_similarity=1.0, worst_similarity=1.0, n_compared=len(embeds),
                    crop_regime=crop_meta["regime"], crop_regime_reason=crop_meta["reason"],
                    confident_detection_rate=crop_meta["confident_rate"])
    sims = []
    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            sims.append(_cosine(embeds[i], embeds[j]))
    return dict(mean_similarity=float(np.mean(sims)),
                worst_similarity=float(np.min(sims)),
                n_compared=len(embeds),
                crop_regime=crop_meta["regime"],
                crop_regime_reason=crop_meta["reason"],
                confident_detection_rate=crop_meta["confident_rate"])


# --------------------------------------------------------------------------- #
# 3. Motion presence
# --------------------------------------------------------------------------- #

def motion_presence(frames, max_pairs=60, noise_floor_px=0.5):
    """Motion energy among the pixels that actually moved, normalised by the
    frame diagonal — plus the fraction of the frame those pixels cover.

    Averaging flow magnitude over *every* pixel (background included) would
    dilute a thin line-art figure moving against a large flat background to
    near zero even when the figure's own motion is large — exactly the kind
    of sparse content this project's stick-figure and vector renders produce.
    Restricting the mean to pixels whose flow clears a small noise floor
    (0.5 px — below that, Farneback's own quantisation noise on a static
    background is indistinguishable from motion) measures "how much did the
    part that moved actually move," and `motion_coverage` catches the
    genuinely-static case separately: near-zero coverage means nothing
    cleared the noise floor at all, which is what a frozen clip looks like.
    """
    import cv2
    if len(frames) < 2:
        return dict(motion_energy=0.0, motion_coverage=0.0)
    grays = [gray(f) for f in frames]
    diag = float(np.hypot(*grays[0].shape))
    idx = list(range(len(grays) - 1))
    if len(idx) > max_pairs:
        idx = np.linspace(0, len(idx) - 1, max_pairs).astype(int).tolist()
    mags, coverages = [], []
    for i in idx:
        a, b = grays[i], grays[i + 1]
        if a.shape != b.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            a.astype(np.uint8), b.astype(np.uint8), None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        moving = mag > noise_floor_px
        coverages.append(float(moving.mean()))
        if moving.any():
            mags.append(float(mag[moving].mean()))
    energy = float(np.mean(mags)) / diag if mags else 0.0
    coverage = float(np.mean(coverages)) if coverages else 0.0
    return dict(motion_energy=energy, motion_coverage=coverage)


# --------------------------------------------------------------------------- #
# 4. Pose fidelity (only when pose guides are supplied)
# --------------------------------------------------------------------------- #

# COCO-17 indices used by ultralytics yolo11n-pose.
COCO = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4,
            l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
            l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
            l_knee=13, r_knee=14, l_ankle=15, r_ankle=16)

# guide limb-colour name -> (proximal joint, distal joint) in our naming.
# Proximal = nearer the torso centroid; resolved per-image by distance, since
# an axis-aligned "end" from PCA has no inherent anatomical direction.
LIMB_JOINTS = {
    "uarm_R": ("r_shoulder", "r_elbow"), "farm_R": ("r_elbow", "r_wrist"),
    "uarm_L": ("l_shoulder", "l_elbow"), "farm_L": ("l_elbow", "l_wrist"),
    "thigh_R": ("r_hip", "r_knee"), "shin_R": ("r_knee", "r_ankle"),
    "thigh_L": ("l_hip", "l_knee"), "shin_L": ("l_knee", "l_ankle"),
}


def _guide_joint_points(guide_img):
    """Recover named joint positions from an exact-colour pose guide (see
    svg/guides.py: each limb is a solid OpenPose-palette line, drawn from an
    exact rig — nothing is detected, so the guide's joints are ground truth
    by construction).
    """
    from guides import POSE_COLOURS  # svg/guides.py
    arr = np.asarray(guide_img)
    limb_ends = {}
    all_pts = []
    for name, colour in POSE_COLOURS.items():
        mask = np.all(np.abs(arr[..., :3].astype(int) - np.array(colour)) <= 8, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        c = pts.mean(0)
        q = pts - c
        cov = q.T @ q
        w, v = np.linalg.eigh(cov)
        axis = v[:, int(np.argmax(w))]
        t = q @ axis
        a, b = c + axis * t.min(), c + axis * t.max()
        limb_ends[name] = (a, b)
        all_pts.append(pts)
    if not all_pts:
        return {}
    body_centroid = np.concatenate(all_pts, axis=0).mean(0)
    joints = {}
    for limb, (prox_name, dist_name) in LIMB_JOINTS.items():
        if limb not in limb_ends:
            continue
        a, b = limb_ends[limb]
        prox, dist = (a, b) if np.hypot(*(a - body_centroid)) < np.hypot(*(b - body_centroid)) else (b, a)
        joints.setdefault(prox_name, []).append(prox)
        joints.setdefault(dist_name, []).append(dist)
    return {k: np.mean(v, axis=0) for k, v in joints.items()}


# Cross-lane finding (Attempt 19, builder lane): CMU OpenPose/controlnet_aux
# returns NO PERSON at all on our styled/stylised frames and on our rendered
# skeletons, on real photos it works fine (12/12 keypoints on zidane.jpg) - the
# detector is not broken, our art is off its training distribution. YOLO11-pose
# does somewhat better but found the cel figure in only 2 of 6 tested clips, at
# confidence 0.41 and 0.22 (~100 px error even where it fired). The builder
# traces this to the same root cause as ControlNet ignoring our guides: our rig
# has an ear span 1.65x its shoulder width against ~0.52 for a real human -
# every pose estimator here (ours and ControlNet's) is trained on photographs
# of human proportions, and our deliberately non-human cel proportions are
# simply outside that distribution. One cause, two symptoms.
#
# Consequence: a "no confident detection" result on a stylised clip is not
# evidence of good OR bad pose fidelity - it is evidence the detector could not
# parse the art style at all, and must never silently inherit a pass or a
# fail. DETECTION_CONF_FLOOR and MEASURABLE_DETECTION_RATE below decide when
# there is enough signal to trust a PCK number; below that, the result is
# UNMEASURABLE and reported as its own separate outcome, with the detection
# rate and mean confidence attached so the failure itself is loggable.
DETECTION_CONF_FLOOR = 0.40      # matches the builder's "0.41 = real, 0.22 = noise" split
MEASURABLE_DETECTION_RATE = 0.50  # need a confident person in at least half the sampled frames


def pose_fidelity(out_frames, guide_pose_files, yolo_model, guard=None,
                   pck_thresh=0.10):
    """Per-joint error between the output's detected pose and the exact guide
    pose it was supposed to follow. This is what catches Attempt 18's
    specific failure: guides showed a raised arm, renders showed hands on
    hips — a plausible-looking pose that is simply the wrong one.

    pck_thresh is a fraction of the frame diagonal; a joint counts as correct
    if it falls within that radius of the guide joint (a PCK@radius metric,
    the standard "percentage of correct keypoints" convention from pose-
    estimation benchmarks, e.g. Yang & Ramanan 2011 / MPII).

    Detection is only trusted above DETECTION_CONF_FLOOR box confidence, and
    only reported as a real PCK score when at least MEASURABLE_DETECTION_RATE
    of the sampled guide frames clear that floor — otherwise `available` is
    False and `measurable` is explicitly False, distinct from "no guides were
    supplied" (`measurable` absent). See the block comment above.
    """
    if not guide_pose_files or yolo_model is None:
        return dict(available=False,
                    reason="no pose guides supplied" if not guide_pose_files
                    else "no pose model")
    from PIL import Image
    n_out, n_guide = len(out_frames), len(guide_pose_files)
    if n_out == 0 or n_guide == 0:
        return dict(available=False, reason="empty frame set")
    errs, matched = [], 0
    per_joint = {}
    n_with_guide = 0
    n_detected = 0            # any person box at all, any confidence
    n_confident = 0           # box confidence >= DETECTION_CONF_FLOOR
    det_confidences = []
    for gi, gpath in enumerate(guide_pose_files):
        oi = min(n_out - 1, round(gi * (n_out - 1) / max(n_guide - 1, 1)))
        guide_joints = _guide_joint_points(Image.open(gpath).convert("RGB"))
        if not guide_joints:
            continue
        n_with_guide += 1
        frame = out_frames[oi]
        r = yolo_model.predict(frame[:, :, ::-1], verbose=False)[0]
        if guard is not None:
            guard.step()
        if not len(r.boxes) or r.keypoints is None:
            continue
        n_detected += 1
        areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * \
                (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
        best = int(areas.argmax())
        box_conf = float(r.boxes.conf[best]) if r.boxes.conf is not None else 1.0
        det_confidences.append(box_conf)
        if box_conf < DETECTION_CONF_FLOOR:
            continue
        n_confident += 1
        kxy = r.keypoints.xy[best].cpu().numpy()
        kconf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None \
            else np.ones(len(kxy))
        diag = float(np.hypot(*frame.shape[:2]))
        matched += 1
        for name, gp in guide_joints.items():
            ci = COCO.get(name)
            if ci is None or kconf[ci] < 0.3:
                continue
            d = float(np.hypot(*(kxy[ci] - gp))) / diag
            errs.append(d)
            per_joint.setdefault(name, []).append(d)

    detection_rate = (n_detected / n_with_guide) if n_with_guide else 0.0
    confident_rate = (n_confident / n_with_guide) if n_with_guide else 0.0
    mean_confidence = float(np.mean(det_confidences)) if det_confidences else 0.0
    base = dict(n_guide_frames=n_with_guide, n_detected=n_detected,
                detection_rate=detection_rate, n_confident=n_confident,
                confident_rate=confident_rate, mean_confidence=mean_confidence)

    if n_with_guide == 0:
        return dict(available=False, measurable=False,
                    reason="no valid guide skeletons decoded", **base)
    if confident_rate < MEASURABLE_DETECTION_RATE or not errs:
        return dict(available=False, measurable=False,
                    reason=("UNMEASURABLE: pose estimator found a confident "
                            "(>= %.2f conf) person in only %d/%d frames "
                            "(mean conf %.2f where detected at all in %d/%d) "
                            "- this is detector failure on the art style, not "
                            "a pose-accuracy result"
                            % (DETECTION_CONF_FLOOR, n_confident, n_with_guide,
                               mean_confidence, n_detected, n_with_guide)),
                    **base)
    errs = np.array(errs)
    return dict(available=True, measurable=True,
                mean_norm_error=float(errs.mean()),
                pck=float((errs < pck_thresh).mean()),
                pck_thresh=pck_thresh, n_frames_matched=matched,
                n_joint_samples=len(errs),
                worst_joint=max(per_joint, key=lambda k: np.mean(per_joint[k])),
                **base)


# --------------------------------------------------------------------------- #
# 5. Scene complexity
# --------------------------------------------------------------------------- #

def scene_complexity(frames, yolo_model, guard=None, max_frames=60):
    """Distinct persons detected per frame, and whether the cast size is
    stable. Honest limitation: yolo11n-pose only detects the `person` class,
    so this counts characters, not props or background elements — the
    project's stated goal (multiple characters + background + props) is only
    partially measurable with the tools available on this hardware, and that
    gap is reported rather than silently assumed away.

    Correction (Attempt 21, scene lane, ATTEMPTS.md ~line 1150): measured
    directly that `yolo11n-pose.pt` finds no person at all in EITHER of this
    project's hand-written single-figure controls (`control_wave`,
    `control_kick`) and returns a modal count of 0 on a hand-authored scene
    that a colour-based audit confirmed has 3-4 characters visible in
    essentially every sampled frame. A "modal_count: 0" from this metric was
    therefore being reported indistinguishably from "this clip genuinely has
    no characters" and "this detector cannot see this art style at all" —
    the second is true of every clip this project has ever produced, and
    reporting a bare 0 as if it were a measurement is worse than declining.
    Uses the same MEASURABLE_DETECTION_RATE / DETECTION_CONF_FLOOR bar
    pose_fidelity already applies: below it, the count is reported as
    unmeasurable rather than zero, and (like pose_fidelity) it is never
    hard-gated either way.
    """
    if yolo_model is None:
        return dict(available=False, measurable=False, reason="no pose model")
    idx = list(range(len(frames)))
    if len(idx) > max_frames:
        idx = np.linspace(0, len(idx) - 1, max_frames).astype(int).tolist()
    counts = []
    n_detected = 0        # frames with >=1 box at any confidence
    n_confident = 0        # frames with >=1 box at >= DETECTION_CONF_FLOOR
    confidences = []
    for i in idx:
        r = yolo_model.predict(frames[i][:, :, ::-1], verbose=False)[0]
        if guard is not None:
            guard.step()
        counts.append(len(r.boxes))
        if len(r.boxes):
            n_detected += 1
            conf = r.boxes.conf
            if conf is not None and len(conf):
                c = float(conf.max())
                confidences.append(c)
                if c >= DETECTION_CONF_FLOOR:
                    n_confident += 1
    counts = np.array(counts)
    n = len(counts)
    if n == 0:
        return dict(available=False, measurable=False, reason="no frames")

    detection_rate = n_detected / n
    confident_rate = n_confident / n
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    base = dict(n_frames=n, n_detected=n_detected, detection_rate=detection_rate,
                n_confident=n_confident, confident_rate=confident_rate,
                mean_confidence=mean_confidence,
                note="counts `person` detections only; props/background are "
                     "not measured on this hardware")

    if confident_rate < MEASURABLE_DETECTION_RATE:
        return dict(available=False, measurable=False,
                    reason=("UNMEASURABLE: confident (>= %.2f conf) person "
                            "detection in only %d/%d frames (mean conf %.2f "
                            "where detected at all in %d/%d) - this reports "
                            "detector failure on the art style, never a cast "
                            "count of zero"
                            % (DETECTION_CONF_FLOOR, n_confident, n,
                               mean_confidence, n_detected, n)),
                    **base)

    vals, freq = np.unique(counts, return_counts=True)
    modal = int(vals[int(freq.argmax())])
    return dict(available=True, measurable=True, mean_count=float(counts.mean()),
                modal_count=modal,
                cast_stability=float((counts == modal).mean()),
                min_count=int(counts.min()), max_count=int(counts.max()),
                **base)


# --------------------------------------------------------------------------- #
# 6. Content correctness / prompt alignment (CLIPSIM)
# --------------------------------------------------------------------------- #

_CLIP_CACHE = {}


def _load_clip(model_name="openai/clip-vit-base-patch32"):
    """CPU-only by construction: this metric does not touch the GPU at all,
    per the hardware note (damaged fan, GPU work must go through gpuguard) —
    CLIP ViT-B/32 is ~600 MB and runs adequately on CPU for a few dozen
    frames, so there is no reason to put it on the card.
    """
    if model_name in _CLIP_CACHE:
        return _CLIP_CACHE[model_name]
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    model.to("cpu").eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    _CLIP_CACHE[model_name] = (model, proc)
    return model, proc


def content_correctness(frames, caption, max_frames=30):
    """CLIPSIM: mean cosine similarity between the clip's caption and each
    frame's CLIP image embedding (paper/RESEARCH.md §6 — CLIPScore/CLIPSIM,
    Hessel et al. 2021 EMNLP; first applied to video in GODIVA, arXiv 2021).

    Reports mean, min (worst single frame) and the standard deviation across
    frames — mean catches a clip that is uniformly wrong, std catches one
    that starts on-topic and drifts, which is a distinct failure mode.

    Returns available=False (not a score of 0) when no caption is supplied —
    CLIPSIM is undefined without one, and reporting a fabricated 0 would be
    worse than declining.
    """
    if not caption:
        return dict(available=False, reason="no caption supplied")
    import torch
    from PIL import Image
    model, proc = _load_clip()
    idx = list(range(len(frames)))
    if len(idx) > max_frames:
        idx = np.linspace(0, len(idx) - 1, max_frames).astype(int).tolist()
    imgs = [Image.fromarray(frames[i]) for i in idx]
    with torch.no_grad():
        text_in = proc(text=[caption], return_tensors="pt", padding=True,
                       truncation=True)
        text_feat = model.get_text_features(**text_in)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        img_in = proc(images=imgs, return_tensors="pt")
        img_feat = model.get_image_features(**img_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_feat.T).squeeze(-1).cpu().numpy()
    return dict(available=True, mean=float(sims.mean()), min=float(sims.min()),
                std=float(sims.std()), n_frames=len(idx), caption=caption)


# --------------------------------------------------------------------------- #
# 7. Structural correctness (AniSVG source, not pixels)
# --------------------------------------------------------------------------- #
#
# CLIPSIM (metric 6) measured that perceptual embedding similarity cannot
# separate "drew what was asked" from "drew a colour-matched blob" in this
# project's vector-art domain. But the AniSVG generation path has something a
# normal video-generation project does not: the output IS the source. A
# generated clip is text with a fixed cast of shapes and a fixed paint order
# (svg/generate.py's own scorer already checks it parses / declares shapes /
# moves), and per ATTEMPTS.md, corpus captions are emitted from the
# generating parameters, so they are true by construction. That means content
# correctness can be checked exactly, by parsing the declaration, instead of
# approximately, by embedding the render.
#
# Scope, stated up front: this metric only applies to the symbolic AniSVG
# path (svg/generate.py output, and the procedural chars.py/anime_chars.py
# corpus). It has no opinion on raster or diffusion output (svg/out/styled) -
# there is no declarative source to parse for a diffusion image, only pixels.
# See verdict.py and critic/README.md for what that asymmetry means for the
# project as a whole.

import re as _re

# Fixed, in paint order - chars.py's own LIMBS list plus the head circle it
# prepends (see clip_frames() in svg/chars.py: shapes[0] is always the head,
# shapes[1:] are LIMBS in declaration order). anime_chars.py exports its own
# PART_ORDER directly, imported lazily below to avoid paying for its palette/
# caption tables when this metric isn't used.
STICK_PART_ORDER = ["head", "spine", "thigh_R", "shin_R", "thigh_L", "shin_L",
                     "uarm_R", "farm_R", "uarm_L", "farm_L"]

# Measured, not assumed: every shape in every sampled corpus-stickman AND
# corpus-anime clip (12 clips, both schemas, 6 each) resamples to exactly 12
# points, regardless of whether it is a 12-point head circle or a limb -
# lottie_to_anisvg's arc-length resampling targets one fixed vertex count for
# the whole cast. See critic/README.md for the raw per-clip counts.
POINTS_PER_SHAPE = 12


def _anime_part_order_len():
    from anime_chars import PART_ORDER
    return len(PART_ORDER)


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
              "twelve": 12}
_NUM_ALT = "|".join(_NUM_WORDS)

# A number word (or digit) followed, within a short window, by a plural noun
# this project's caption vocabulary actually uses for countable cast members
# (svg/chars.py SUBJECT / stream_corpus.py caption text / gen17 captions).
_COUNT_NOUN_RE = _re.compile(
    r"\b(" + _NUM_ALT + r"|\d+)\b[^.]{0,40}?\b"
    r"(stars?|figures?|characters?|circles?|shapes?|bubbles?|spheres?|"
    r"balls?|dots?|arrows?|icons?)\b", _re.I)

# "four-pointed star" / "4-pointed stars" - the one caption pattern this pass
# can verify exactly, because a star has a determinable vertex count and
# radius alternation (see star_score below).
_STAR_RE = _re.compile(r"\b(" + _NUM_ALT + r"|\d+)-pointed\s+stars?", _re.I)


def _num(tok):
    tok = tok.lower()
    if tok in _NUM_WORDS:
        return _NUM_WORDS[tok]
    return int(tok) if tok.isdigit() else None


def parse_expected_count(caption):
    """First '<count> ... <countable noun>' match in the caption, or None."""
    if not caption:
        return None
    m = _COUNT_NOUN_RE.search(caption)
    return _num(m.group(1)) if m else None


def parse_star_request(caption):
    """The k in '<k>-pointed star(s)', or None if the caption asks for none."""
    if not caption:
        return None
    m = _STAR_RE.search(caption)
    return _num(m.group(1)) if m else None


def star_score(pts, k):
    """How strongly a polygon's radius (from its own centroid) oscillates at
    exactly k cycles per revolution - the discrete Fourier coefficient of the
    (mean-normalised) radius sequence at frequency k, indexing vertices as if
    evenly spaced in angle (true for chars.py/Lottie shapes, which trace their
    outline in vertex order).

    Chosen over "exactly 2k vertices" (the textbook star-polygon definition)
    because this corpus's converter resamples every outline to a roughly
    fixed vertex count regardless of the underlying primitive (measured: a
    real ground-truth 4-pointed star and a generated 4-shape blob both have
    16 vertices - see critic/README.md) - vertex count carries no signal here,
    only the shape the outline actually traces does. A true k-pointed star
    has k radius peaks and k troughs; a circle or blob has none, regardless
    of how many points its outline is resampled to.
    """
    n = len(pts)
    if n < 6:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(0)
    r = np.linalg.norm(pts - c, axis=1)
    if r.mean() < 1e-6:
        return 0.0
    rn = r / r.mean()
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coef = np.abs(np.sum((rn - 1.0) * np.exp(-1j * k * theta))) / n
    return float(coef)


# Calibrated against the one apples-to-apples pair available: the exact same
# caption ("Two four-pointed star-shaped figures...") as both real
# ground-truth AniSVG (svg/data/train/val.jsonl id 10887279) and a trained
# model's generation of it (svg/out/gen17/g002, ATTEMPTS.md #15's own "neither
# is a star" example). Measured star_score(k=4), best over all frames, per
# declared shape:
#   ground truth : three shapes score 0.135, 0.154, 0.140 (the rest ~0.0 -
#                  non-star decoration/shadow shapes; a caption of "two
#                  figures" does not mean exactly two declared shapes even in
#                  real corpus data, see structural_correctness below)
#   g002 (blobs) : all four shapes score 0.0048-0.0052 - roughly 27x lower
#                  than the weakest genuine star shape
# 0.05 sits in the wide gap between those two clusters.
STAR_SCORE_THRESHOLD = 0.05


def _shape_star_best(anim, sh, k):
    from anisvg import _transform
    best = 0.0
    for st in anim.states():
        s = st.get(sh.id)
        if s is None or not s.visible:
            continue
        best = max(best, star_score(_transform(sh, s), k))
    return best


def structural_correctness(anisvg_text, caption=None, schema=None):
    """Parse-level content correctness on the AniSVG source itself.

    anisvg_text: the raw generated (or procedural) document. If it is a raw
    model completion that may carry preamble/trailing drift, pass it through
    svg/generate.py's extract() first - this function does that internally,
    reusing svg/generate.py's own extract()/score() rather than re-parsing.

    Returns available=False when no source text is supplied at all - this
    metric has nothing to say about raster/diffusion output.
    """
    if not anisvg_text:
        return dict(available=False,
                    reason="no AniSVG source supplied - this metric applies "
                    "only to the symbolic AniSVG generation path, not "
                    "raster/diffusion output (no declarative source to parse)")

    from anisvg import Anim
    import generate as _gen  # svg/generate.py - reuse, don't reimplement

    out = dict(available=True, domain="anisvg")
    verdict = _gen.score(anisvg_text)
    out["parses"] = bool(verdict.get("ok"))
    if not out["parses"]:
        out["reason"] = verdict.get("why", "parse failed")
        return out

    anim = Anim.from_text(_gen.extract(anisvg_text))
    n = len(anim.shapes)
    out["declared_shapes"] = n
    out["frames"] = len(anim.frames)

    # schema is NOT auto-inferred from shape count. Measured reason: a
    # LottieAnimation-660K-domain generated clip (release/v1-8b-icons/
    # clip00.anisvg) happens to declare exactly 10 shapes - the same count as
    # the STICK_PART_ORDER rig - despite having nothing to do with it (its
    # shapes are 16 points each, not the rig's 12; it is an ordinary icon).
    # Guessing schema from a bare count would have flagged that icon as a
    # broken stick figure. Schema is only checked when the caller states it
    # explicitly (`schema="stick"|"anime"`) - which evaluate.py only does for
    # clips it knows came from chars.py/anime_chars.py or their trained
    # models, never for the generic Lottie-icon domain.
    out["schema"] = schema

    # Paint order: for this format, shape index IS the part/paint-order role,
    # so "declared parts in the required paint order" reduces to "ids are
    # exactly 0..n-1, contiguous, no gaps or duplicates."
    ids = [sh.id for sh in anim.shapes]
    out["ids_contiguous"] = (sorted(ids) == list(range(n)))

    if schema in ("stick", "anime"):
        expected_n = (len(STICK_PART_ORDER) if schema == "stick"
                     else _anime_part_order_len())
        out["schema_shape_count_ok"] = (n == expected_n)

        # Per-part geometry: every shape in a schema clip is resampled to the
        # SAME point count by this project's own converter, verified directly
        # rather than guessed - measured 12/12 across all 12 sampled
        # corpus-stickman and corpus-anime clips (both schemas), zero
        # exceptions (critic/README.md has the raw counts). A generated clip
        # that declares a schema-sized cast but with the wrong per-shape
        # point count did not actually reproduce the rig's geometry.
        counts = set(len(sh.pts) for sh in anim.shapes)
        out["points_per_shape"] = sorted(counts)
        out["points_per_shape_ok"] = (counts == {POINTS_PER_SHAPE})

        # Cast stability: chars.py/anime_chars.py never hide a declared body
        # part, so on a schema-matched clip any frame where a part is
        # invisible is a defect the pixel metrics cannot see - to optical
        # flow, a limb popping out of existence looks like ordinary motion,
        # not a missing part.
        hidden_frames = sum(1 for st in anim.states()
                            if any(not s.visible for s in st.values()))
        out["cast_stable"] = (hidden_frames == 0)
        out["hidden_frame_count"] = hidden_frames

    # Caption-driven checks.
    expected_count = parse_expected_count(caption)
    out["expected_object_count"] = expected_count
    if expected_count is not None:
        # NOT gated - see the docstring above star_score: even real
        # ground-truth AniSVG for "two star figures" declares 11 shapes, not
        # 2 (main shapes plus shading/decoration). Raw shape-count-vs-caption-
        # count is informational only; it does not separate good from bad.
        out["declared_vs_expected_count_note"] = (
            "informational only, not gated - shape count is not 1:1 with "
            "described objects even in real ground-truth corpus data")

    k = parse_star_request(caption)
    out["requested_star_points"] = k
    if k:
        scores = {sh.id: _shape_star_best(anim, sh, k) for sh in anim.shapes}
        star_like = [sid for sid, sc in scores.items() if sc >= STAR_SCORE_THRESHOLD]
        need = expected_count or 1
        out["star_scores"] = {str(sid): round(sc, 4) for sid, sc in scores.items()}
        out["star_shapes_found"] = len(star_like)
        out["star_shapes_needed"] = need
        out["geometry_verified"] = len(star_like) >= need

    return out


# --------------------------------------------------------------------------- #
# 8. Scene presence audit (colour + connected components, detector-free)
# --------------------------------------------------------------------------- #
#
# scene_complexity (metric 5) is a photo-trained person detector and, per its
# own docstring, cannot see this project's art style at all — confirmed on
# every stick figure this project has produced, one character or several
# (Attempt 21, scene lane). It reports UNMEASURABLE rather than a false zero,
# which is honest, but it leaves the project's actual question - is a
# multi-character scene actually multi-character - with no working answer.
#
# The scene lane's own workaround (scenescript.py --audit) answers it without
# any detector: a declared character has a declared colour, so presence is
# "is enough of that colour on screen, in a connected shape" - the same
# measurement principle guides.py already uses for pose (declare the ground
# truth, don't detect it). That function is coupled to `.scene` script
# parsing and lives outside critic/, so the technique - not the code - is
# reimplemented here, independently, so critic/ has no dependency on
# scenescript.py's file format. Same tolerance (45), same minimum-ink floor
# (40 px), same 8-connectivity, so numbers from the two tools are comparable.
#
# This requires the caller to supply the cast's actual colours - it is a
# verification tool, not a detector, and has nothing to measure without
# knowing what it is looking for. Because of that it can never replace
# scene_complexity in general (arbitrary/unlabelled input has no colours to
# check), only stand in for it on exactly this project's own scripted,
# colour-cast material - stated as a scope limit, not hidden.

def scene_presence_audit(frames, colors, tol=45, min_px=40, max_frames=90):
    """Per-declared-character ink presence and blob count, colour-matched.

    colors: list of (r, g, b) tuples, one per declared cast member, in any
    caller-meaningful order (e.g. scene-script cast order).

    Measures presence and occlusion, not personhood - explicitly the same
    framing scenescript.py's own audit uses. Never gated (see verdict.py):
    it depends on caller-supplied ground truth, so a missing/wrong colour
    list would silently produce a wrong "0 present" reading if this were
    trusted the way a detector-free ground-truth check should not be trusted
    blindly - reported for the record, not treated as an infallible oracle.
    """
    if not colors:
        return dict(available=False, reason="no character colours supplied")
    import cv2
    idx = list(range(len(frames)))
    if len(idx) > max_frames:
        idx = np.linspace(0, len(idx) - 1, max_frames).astype(int).tolist()
    sub = [frames[i] for i in idx]
    n = len(sub)

    per_char = []
    all_present = np.ones(n, dtype=bool)
    for ci, colour in enumerate(colors):
        col = np.array(colour, dtype=np.int16)
        present = np.zeros(n, dtype=bool)
        px_counts, blob_counts = [], []
        for i, f in enumerate(sub):
            d = np.abs(f.astype(np.int16) - col).sum(-1)
            m = (d < tol).astype(np.uint8)
            px = int(m.sum())
            px_counts.append(px)
            if px > min_px:
                present[i] = True
                nlab, _ = cv2.connectedComponents(m, connectivity=8)
                blob_counts.append(nlab - 1)
        all_present &= present
        per_char.append(dict(
            colour=list(int(c) for c in colour),
            frames_visible=int(present.sum()), n_frames=n,
            presence_rate=float(present.mean()),
            mean_ink_px=float(np.mean(px_counts)), min_ink_px=int(min(px_counts)),
            mean_blobs=float(np.mean(blob_counts)) if blob_counts else 0.0))

    return dict(available=True, n_characters=len(colors), n_frames=n,
                tol=tol, min_px=min_px,
                frames_all_visible=int(all_present.sum()),
                frames_all_visible_rate=float(all_present.mean()),
                per_character=per_char,
                note="measures presence and occlusion via caller-supplied "
                     "colour, not detected personhood - see metrics.py "
                     "scene_presence_audit docstring")
