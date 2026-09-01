"""One gate: given the metrics.json a run of evaluate.py produced, answer
"is this a usable animation video?" as USABLE / PROMISING / NOT-AN-ANIMATION,
naming the specific failing metric.

Every threshold below is a deliberate engineering choice, defended in its own
comment. None of them come from a published benchmark for this exact task —
none exists, because "is a stick-figure/vector-animation clip usable" is not a
standard CV benchmark — so each is defended by reasoning from first principles
or from a citeable convention (PCK, optical-flow warp-residual practice), and
by being checked against this project's own known-good and known-bad clips
(see critic/README.md for the calibration run). Be strict: ATTEMPTS.md is
explicit that the project's best output to date is 100% hand-written, so a
critic tuned to pass everything is worthless.

Usage:
    python critic/verdict.py path/to/report.json
"""
import argparse
import json
import sys

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

# --- temporal stability ---
# Warp-residual is measured as mean |warp(frame_t) - frame_t+1| / 255 after
# optical-flow compensation, i.e. the fraction of the clip's dynamic range
# that motion *cannot* explain. Video-prediction/generation work that reports
# a warping-error metric (e.g. first-order motion / video-to-video literature)
# treats residuals under roughly 5% of dynamic range as well-aligned, with
# larger residuals read as popping/ghosting rather than motion. We set the
# pass bar at 0.06 (slightly above that, to tolerate this project's own
# rasteriser aliasing) and a hard-fail bar at 0.15 (2.5x that) — past that
# point optical flow is not explaining the frame-to-frame change at all, which
# means content is appearing/disappearing rather than moving.
WARP_ERROR_PASS = 0.06
WARP_ERROR_HARDFAIL = 0.15

# Flicker ratio = output inter-frame change / guide inter-frame change.
# Attempt 18 (ATTEMPTS.md) measured 23.6x and every frame was visibly a
# different character — an order-of-magnitude blowup, not stylistic
# embellishment. Real secondary motion a diffusion model legitimately adds on
# top of the guide (hair sway, shading, cloth) plausibly doubles or triples
# the guide's own inter-frame change; we set the pass bar at 3x to give that
# room, and treat anything past 8x (roughly a third of Attempt 18's measured
# failure) as a hard fail — no plausible amount of secondary motion accounts
# for it.
FLICKER_RATIO_PASS = 3.0
FLICKER_RATIO_HARDFAIL = 8.0

# --- identity preservation ---
# Colour-histogram cosine similarity of the character crop, pairwise across
# the clip. This is a crude, download-free proxy (see metrics.py::_embed) —
# it measures palette/composition stability, not "is it the same drawn
# character" in any deep sense. Because it is crude, the bars are set loose:
# a mean similarity below 0.75 means the *average* frame pair already looks
# like a different colour scheme, and a worst-case below 0.45 means at least
# one point in the clip swapped almost completely (Attempt 18: hair, face and
# background all reorganised every frame — its worst-case similarity should
# sit well under this).
IDENTITY_MEAN_PASS = 0.75
IDENTITY_WORST_PASS = 0.45
IDENTITY_WORST_HARDFAIL = 0.30

# --- motion presence ---
# Gated on motion_coverage: the fraction of pixels per frame pair whose
# optical flow clears a 0.5 px noise floor (see metrics.py::motion_presence).
# motion_energy alone is not gated here — it is computed only over pixels that
# already cleared the noise floor, so it says how fast the moving part moved,
# not whether anything moved at all. Coverage is a stricter, resolution- and
# content-density-independent version of "did the clip do anything": encoder/
# rasteriser dither on a genuinely frozen frame touches well under 0.1% of
# pixels, while even a small waving hand against a static background covers
# noticeably more. The floor is set at 0.2%.
MOTION_COVERAGE_FLOOR = 0.002

# --- pose fidelity ---
# PCK@0.10: a joint counts as correct within 10% of the frame diagonal of its
# guide position — a standard "percentage of correct keypoints" convention
# from pose-estimation benchmarks (Yang & Ramanan 2011 uses a comparable
# fraction of a reference body segment). We ask that at least half the sampled
# joints across the clip clear that bar to pass; below 20% correct means the
# render is doing a materially different pose than the guide asked for
# (Attempt 18: guide = raised arm wave, render = hands on hips — most joints
# would miss even a loose 10%-of-diagonal bar).
POSE_PCK_PASS = 0.50
POSE_PCK_HARDFAIL = 0.20
# Only applied when metrics.py reports measurable=True (a confident
# detection in >=50% of sampled frames, see metrics.py:DETECTION_CONF_FLOOR).
# On stylised output the pose estimator itself frequently fails to find a
# person at all (Attempt 19, builder lane: CMU OpenPose 0/N on our styled
# frames; YOLO11-pose 2/6 clips at 0.41/0.22 confidence) - our cel rig's
# proportions (ear span 1.65x shoulder width vs ~0.52 for a real human) sit
# outside what any of these detectors were trained on. An unmeasurable clip
# is reported, never hard-gated - see judge() below.

# --- scene complexity ---
# Informational only — see metrics.py::scene_complexity. No existing output in
# this project (per ATTEMPTS.md) achieves a stable multi-character cast with
# background and props, so gating on it would mark every clip in the repo
# NOT-AN-ANIMATION regardless of animation quality, conflating "not yet at the
# project's ultimate goal" with "not an animation at all." It is reported so
# the gap is visible, not to decide the verdict.

# --- content correctness / prompt alignment (CLIPSIM) ---
# CLIPSIM = mean cosine similarity between the clip's caption and each frame's
# CLIP ViT-B/32 image embedding (paper/RESEARCH.md §6; Hessel et al. 2021).
#
# THIS THRESHOLD WAS CALIBRATED AND THE RESULT IS A NEGATIVE FINDING, STATED
# PLAINLY RATHER THAN TUNED AWAY. The brief was: score the hand-written
# controls (known-good content) against release/v2-1.7b-icons' known-bad
# "two four-pointed stars" clip (svg/out/gen17/g002 — it draws two blobs,
# per ATTEMPTS.md #15) and pick a threshold that separates them.
#
# Measured (openai/clip-vit-base-patch32, CPU, mean over 20 frames):
#   control_wave.mp4  vs its own correct caption          -> 0.329
#   control_kick.mp4  vs its own correct caption          -> 0.233
#   g002 (blobs)      vs its own correct caption          -> 0.306   <- higher
#                                                                        than the
#                                                                        KICK CONTROL
#   control_kick.mp4  vs the WAVE caption (wrong action)  -> 0.235   <- indistinguishable
#                                                                        from its own
#                                                                        correct caption (0.233)
#   control_wave.mp4  vs an unrelated caption ("a car")   -> 0.163   <- CLIP *can*
#                                                                        separate a
#                                                                        wholly different
#                                                                        category
#
# The populations DO NOT SEPARATE: known-good hand-drawn content (0.233-0.329)
# and known-bad model output (0.306, g002) overlap completely, and swapping in
# the wrong caption for a *correct* clip barely moves the score (0.233 -> 0.235
# for kick/wave). CLIP ViT-B/32 was trained on natural photographs; flat vector
# renders and stick-figure line art sit far off that training distribution, and
# its embedding evidently has enough resolution to reject a category-level
# mismatch (stick figure vs car: 0.163) but not enough to tell "kicking" from
# "waving," or "two stars" from "two blobs," within that domain.
#
# Consequence for the gate: CONTENT_CORRECTNESS_FLOOR is set at 0.18 — the
# midpoint between the lowest in-domain-but-plausible score measured (0.203,
# a real gen17 clip against its own caption) and the one clearly-wrong-category
# score measured (0.163). This is defensible only as a "did CLIP land on a
# completely different concept" tripwire. IT DOES NOT, AND CANNOT AT THIS
# THRESHOLD, CATCH THE FAILURE IT WAS ADDED TO CATCH (a model drawing blobs
# instead of stars) — g002 scores 0.306, comfortably above 0.18. That gap is
# the actual finding: temporal smoothness is measurable and was gated
# successfully (see warp_error / flicker_ratio above, which reproduced Attempt
# 18's number to two decimal places); content correctness, with a CPU-budget
# CLIP model and no fine-tuning on this project's vector-art domain, was not.
# Closing it for real would need either a domain-adapted alignment model or a
# stronger VQA-style scorer (VQAScore, paper/RESEARCH.md §6) — both out of
# scope for an 8 GB, fan-limited box in this pass.
CONTENT_CORRECTNESS_FLOOR = 0.18

# --- structural correctness (AniSVG source) ---
# Scope, stated plainly: this metric, and everything gated on it below,
# applies ONLY to the symbolic AniSVG generation path (svg/generate.py output
# and the procedural chars.py/anime_chars.py corpus). It has no source to
# parse for raster or diffusion output (svg/out/styled) and is always
# reported as unavailable there — see metrics.py:structural_correctness and
# critic/README.md for what that asymmetry means for comparing the two halves
# of the hybrid.
#
# Where CLIPSIM (metric 6) could not separate known-good from known-bad
# content, this metric was calibrated on the same apples-to-apples pair and
# DID separate cleanly — measured on the exact same caption ("Two four-
# pointed star-shaped figures...") as both real ground-truth AniSVG
# (svg/data/train/val.jsonl id 10887279) and a trained model's generation of
# it (svg/out/gen17/g002, ATTEMPTS.md #15's own "neither is a star" clip):
#   ground truth : geometry_verified = True  (3 shapes score 0.135-0.154 on
#                   star_score(k=4), >= the 2 the caption requires)
#   g002 (blobs) : geometry_verified = False (all 4 shapes score 0.0048-0.0052,
#                   ~27x below the weakest genuine star shape)
# Full numbers and the exact command are in critic/README.md.
#
# A SECOND check on the same calibration pair did NOT separate and is NOT
# gated: raw declared-shape-count vs the caption's object count. Ground truth
# for "two star figures" declares 11 shapes, not 2 (main bodies plus shading/
# decoration shapes) — so "declared_shapes == expected_object_count" would
# fail on real ground truth and is reported as informational only
# (structural_correctness.declared_vs_expected_count_note), never hard-gated.
# This is the second honest null this pass found, not tuned away either.
#
# Hard-fail conditions, applied only when structural_correctness.available:
#   - does not parse at all
#   - a declared schema (stick/anime) has the wrong shape count for it, wrong
#     per-shape point count, non-contiguous ids, or a body part that goes
#     invisible mid-clip (cast_stable is False) — all measured, calibrated
#     regularities of the real corpus (critic/README.md has the raw counts),
#     not assumed ones (an earlier version of this check assumed "each limb
#     is a 2-point line" from skimming chars.py's *pre-encoding* joint pairs;
#     the actual encoder resamples every shape to 12 points uniformly, and
#     the wrong assumption was caught and fixed before shipping — see git
#     history / README for the correction)
#   - the caption requests a k-pointed star and geometry_verified is False —
#     the flagship check, and the one that catches "asked for two stars, drew
#     two blobs" by parsing the declaration rather than looking at pixels


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def judge(report):
    """report: the dict evaluate.py writes (report['metrics'] holds the five
    sections). Returns (verdict, reasons: list[str]).
    """
    m = report.get("metrics", {})
    hard_fails, soft_fails, notes = [], [], []

    # motion presence
    coverage = _get(m, "motion_presence", "motion_coverage", default=0.0)
    if coverage < MOTION_COVERAGE_FLOOR:
        hard_fails.append(
            "motion_presence: coverage %.4f%% < floor %.2f%% — stable "
            "because nothing moved"
            % (coverage * 100, MOTION_COVERAGE_FLOOR * 100))

    # temporal stability
    warp = _get(m, "temporal_stability", "warp_error", default=0.0)
    if warp >= WARP_ERROR_HARDFAIL:
        hard_fails.append("temporal_stability.warp_error: %.4f >= hard-fail "
                          "%.2f — motion doesn't explain the frame changes"
                          % (warp, WARP_ERROR_HARDFAIL))
    elif warp > WARP_ERROR_PASS:
        soft_fails.append("temporal_stability.warp_error: %.4f > pass bar %.2f"
                          % (warp, WARP_ERROR_PASS))

    ratio = _get(m, "temporal_stability", "flicker_ratio")
    if ratio is not None:
        if ratio >= FLICKER_RATIO_HARDFAIL:
            hard_fails.append("temporal_stability.flicker_ratio: %.1fx >= "
                              "hard-fail %.0fx" % (ratio, FLICKER_RATIO_HARDFAIL))
        elif ratio > FLICKER_RATIO_PASS:
            soft_fails.append("temporal_stability.flicker_ratio: %.1fx > pass "
                              "bar %.0fx" % (ratio, FLICKER_RATIO_PASS))
    else:
        notes.append("flicker_ratio: N/A (no guide frames supplied)")

    # identity preservation
    mean_sim = _get(m, "identity_preservation", "mean_similarity")
    worst_sim = _get(m, "identity_preservation", "worst_similarity")
    if worst_sim is not None:
        if worst_sim < IDENTITY_WORST_HARDFAIL:
            hard_fails.append("identity_preservation.worst_similarity: %.2f "
                              "< hard-fail %.2f — character changes entirely "
                              "at some point in the clip"
                              % (worst_sim, IDENTITY_WORST_HARDFAIL))
        elif worst_sim < IDENTITY_WORST_PASS:
            soft_fails.append("identity_preservation.worst_similarity: %.2f "
                              "< pass bar %.2f" % (worst_sim, IDENTITY_WORST_PASS))
        if mean_sim is not None and mean_sim < IDENTITY_MEAN_PASS:
            soft_fails.append("identity_preservation.mean_similarity: %.2f "
                              "< pass bar %.2f" % (mean_sim, IDENTITY_MEAN_PASS))
        regime = _get(m, "identity_preservation", "crop_regime")
        if regime is not None:
            notes.append("identity_preservation: crop_regime=%s — %s"
                         % (regime, _get(m, "identity_preservation", "crop_regime_reason", default="")))

    # pose fidelity
    pose = m.get("pose_fidelity", {})
    if pose.get("available"):
        pck = pose["pck"]
        if pck < POSE_PCK_HARDFAIL:
            hard_fails.append("pose_fidelity.pck: %.0f%% < hard-fail %.0f%% "
                              "— render follows a different pose than the "
                              "guide (worst joint: %s)"
                              % (pck * 100, POSE_PCK_HARDFAIL * 100,
                                 pose.get("worst_joint", "?")))
        elif pck < POSE_PCK_PASS:
            soft_fails.append("pose_fidelity.pck: %.0f%% < pass bar %.0f%%"
                              % (pck * 100, POSE_PCK_PASS * 100))
    elif pose.get("measurable") is False:
        notes.append("pose_fidelity: %s" % pose.get("reason", "UNMEASURABLE"))
    else:
        notes.append("pose_fidelity: N/A (%s)" % pose.get("reason", "unavailable"))

    scene = m.get("scene_complexity", {})
    if scene.get("available"):
        notes.append("scene_complexity: modal cast %d, stable %.0f%% of "
                     "frames (informational, not gated — see verdict.py)"
                     % (scene["modal_count"], scene["cast_stability"] * 100))
    elif scene.get("measurable") is False and "reason" in scene:
        notes.append("scene_complexity: %s" % scene["reason"])

    spa = m.get("scene_presence_audit", {})
    if spa.get("available"):
        notes.append(
            "scene_presence_audit: all %d character(s) visible in %d/%d "
            "sampled frames (%.0f%%) — detector-free, colour-based, "
            "informational only, not gated"
            % (spa["n_characters"], spa["frames_all_visible"], spa["n_frames"],
               spa["frames_all_visible_rate"] * 100))

    # content correctness (CLIPSIM) — see the calibration note above.
    cc = m.get("content_correctness", {})
    if cc.get("available"):
        if cc["mean"] < CONTENT_CORRECTNESS_FLOOR:
            hard_fails.append(
                "content_correctness.mean: %.3f < floor %.2f — the clip "
                "does not match its caption even at this metric's loose, "
                "category-level resolution (caption: %r)"
                % (cc["mean"], CONTENT_CORRECTNESS_FLOOR, cc.get("caption", "")[:60]))
        else:
            notes.append(
                "content_correctness.mean: %.3f (floor %.2f) — cleared the "
                "floor, but this floor only catches gross topic mismatch, "
                "not wrong geometry/count/shape; see verdict.py calibration "
                "note" % (cc["mean"], CONTENT_CORRECTNESS_FLOOR))
    else:
        notes.append("content_correctness: N/A (%s)"
                     % cc.get("reason", "unavailable"))

    # structural correctness (AniSVG source) — see the calibration note above.
    sc = m.get("structural_correctness", {})
    if sc.get("available"):
        if not sc.get("parses"):
            hard_fails.append("structural_correctness: does not parse (%s)"
                              % sc.get("reason", "?"))
        else:
            schema = sc.get("schema")
            if schema in ("stick", "anime"):
                if not sc.get("ids_contiguous"):
                    hard_fails.append("structural_correctness: shape ids not "
                                      "contiguous 0..n-1 — paint order/part "
                                      "identity is broken")
                if sc.get("schema_shape_count_ok") is False:
                    hard_fails.append(
                        "structural_correctness: declared %d shapes, schema "
                        "'%s' requires a different count"
                        % (sc.get("declared_shapes", -1), schema))
                if sc.get("points_per_shape_ok") is False:
                    hard_fails.append(
                        "structural_correctness: per-shape point counts %s "
                        "!= the schema's uniform %d — declared parts do not "
                        "reproduce the rig's geometry"
                        % (sc.get("points_per_shape"), 12))
                if sc.get("cast_stable") is False:
                    hard_fails.append(
                        "structural_correctness: a declared body part is "
                        "invisible in %d frame(s) — a part vanished mid-clip, "
                        "which optical flow cannot see"
                        % sc.get("hidden_frame_count", -1))
            if sc.get("requested_star_points") and sc.get("geometry_verified") is False:
                hard_fails.append(
                    "structural_correctness: caption requests %d %d-pointed "
                    "star shape(s), only %d declared shape(s) verify as one "
                    "— the file does not declare the geometry it claims to"
                    % (sc.get("star_shapes_needed", 1),
                       sc["requested_star_points"], sc.get("star_shapes_found", 0)))
            elif sc.get("requested_star_points"):
                notes.append(
                    "structural_correctness: %d/%d shapes verify as the "
                    "requested %d-pointed star (geometry_verified=True)"
                    % (sc.get("star_shapes_found", 0), sc.get("star_shapes_needed", 1),
                       sc["requested_star_points"]))
            if sc.get("declared_vs_expected_count_note"):
                notes.append("structural_correctness.declared_shapes=%d vs "
                             "expected_object_count=%s: %s"
                             % (sc.get("declared_shapes", -1),
                                sc.get("expected_object_count"),
                                sc["declared_vs_expected_count_note"]))
    else:
        notes.append("structural_correctness: N/A (%s)"
                     % sc.get("reason", "unavailable — applies only to the "
                                        "symbolic AniSVG path"))

    if hard_fails:
        verdict = "NOT-AN-ANIMATION"
    elif soft_fails:
        verdict = "PROMISING"
    else:
        verdict = "USABLE"
    return verdict, hard_fails, soft_fails, notes


def format_report(name, report):
    verdict, hard, soft, notes = judge(report)
    lines = ["=" * 70, "VERDICT for %s: %s" % (name, verdict), "=" * 70]
    if hard:
        lines.append("\nDISQUALIFYING (hard fail):")
        lines += ["  - " + h for h in hard]
    if soft:
        lines.append("\nBELOW BAR (soft fail -> capped at PROMISING):")
        lines += ["  - " + s for s in soft]
    if notes:
        lines.append("\nNotes:")
        lines += ["  - " + n for n in notes]
    if verdict == "USABLE":
        lines.append("\nAll gated metrics cleared their pass bar.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="path to a report.json from evaluate.py")
    args = ap.parse_args()
    report = json.load(open(args.report, encoding="utf-8"))
    name = report.get("input", args.report)
    print(format_report(name, report))
    verdict, hard, _, _ = judge(report)
    sys.exit(0 if verdict != "NOT-AN-ANIMATION" else 1)


if __name__ == "__main__":
    main()
