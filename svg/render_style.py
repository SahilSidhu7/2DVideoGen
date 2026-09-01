"""Paint AniSVG guide frames with a diffusion model.

The hybrid's second half. `guides.py` turns a clip into per-frame conditioning
that is *exact* rather than detected; this feeds those frames to ControlNet and
gets appearance the token budget could never buy.

Temporal stability is the whole question, so the sampler is pinned as hard as it
can be: one seed for the entire clip, and identical prompt and settings on every
frame. Whatever flicker survives that comes from the sampler responding to small
changes in the conditioning image, not from the run being sloppy - which is the
measurement we actually want.

    python svg/render_style.py --clip release/corpus-anime/clip00.anisvg \\
        --prompt "anime girl, cel shaded, clean lineart" --frames 12

`--guide coco18` is the default since Attempt 19. Attempt 18 conditioned on a
skeleton that used the OpenPose *palette* but not the OpenPose *topology* and
the pose was ignored; `--guide pose` still selects that older skeleton, because
the comparison between the two is the measurement.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_NEG = ("photo, realistic, 3d render, blurry, lowres, watermark, "
               "text, extra limbs, deformed")


def load(model, controlnet, dtype="fp16"):
    import torch
    from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                           UniPCMultistepScheduler)
    td = torch.float16 if dtype == "fp16" else torch.float32
    net = ControlNetModel.from_pretrained(controlnet, torch_dtype=td)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model, controlnet=net, torch_dtype=td, safety_checker=None,
        requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # 8 GB has room for this at 512px, but attention slicing costs almost
    # nothing here and keeps headroom for the VAE decode.
    pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def flicker(frames):
    """Mean absolute inter-frame change, ignoring the motion we asked for.

    Not a perfect separation - real motion also changes pixels - but the same
    measure over the *guide* frames gives a baseline, so the ratio is
    informative: styled >> guide means the sampler is adding instability.
    """
    import numpy as np
    a = np.stack([np.asarray(f.convert("L"), dtype=np.float32) for f in frames])
    if len(a) < 2:
        return 0.0
    return float(np.abs(np.diff(a, axis=0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help=".anisvg file")
    ap.add_argument("--guide", choices=["coco18", "pose", "lineart"],
                    default="coco18",
                    help="coco18 = true COCO-18 OpenPose topology (svg/coco18.py); "
                         "pose = the Attempt-18 limb-axis skeleton, kept as the "
                         "baseline it is compared against")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default=None,
                    help="defaults to the ControlNet matching --guide")
    ap.add_argument("--prompt", default="anime style character, cel shaded, "
                                        "clean line art, flat colours, white background")
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--frames", type=int, default=12, help="0 = all")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=7.0)
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("-o", "--out", default="svg/out/styled")
    args = ap.parse_args()

    if args.controlnet is None:
        args.controlnet = ("lllyasviel/control_v11p_sd15_openpose"
                           if args.guide in ("pose", "coco18")
                           else "lllyasviel/control_v11p_sd15_lineart")

    import torch
    from guides import build

    text = open(args.clip, encoding="utf-8").read()
    gdir = os.path.join(args.out, "guides")
    anim, made = build(text, gdir, args.width, kinds=(args.guide,))
    if made.get(args.guide, 0) == 0:
        raise SystemExit("no %s guides for this clip (schema mismatch?)" % args.guide)

    from PIL import Image
    files = sorted(os.listdir(os.path.join(gdir, args.guide)))
    if args.frames:
        files = files[:args.frames]
    conds = [Image.open(os.path.join(gdir, args.guide, f)).convert("RGB")
             for f in files]
    print("clip: %d shapes, %d frames; styling %d with %s"
          % (len(anim.shapes), len(anim.frames), len(conds), args.guide))

    pipe = load(args.model, args.controlnet)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    # The fan on this machine is damaged; every GPU loop on this project runs
    # at ~50% duty. One frame is one unit of work.
    from tools.gpuguard import Guard
    guard = Guard()

    out = []
    t0 = time.time()
    for i, cond in enumerate(conds):
        # One fixed seed for the whole clip: the only thing that varies between
        # frames is the conditioning image.
        gen = torch.Generator(device=pipe.device).manual_seed(args.seed)
        img = pipe(args.prompt, image=cond, negative_prompt=args.negative,
                   num_inference_steps=args.steps, guidance_scale=args.scale,
                   controlnet_conditioning_scale=args.cond_scale,
                   generator=gen).images[0]
        img.save(os.path.join(args.out, "frames", "s%04d.png" % i))
        out.append(img)
        guard.step()
        print("  frame %2d/%d  %5.1fs" % (i + 1, len(conds), time.time() - t0))

    gif = os.path.join(args.out, "styled.gif")
    out[0].save(gif, save_all=True, append_images=out[1:],
                duration=int(1000 / (anim.fps or 15)), loop=0)

    fs, fg = flicker(out), flicker(conds)
    print("\nstyled  -> %s" % gif)
    print("flicker  styled %.2f   guide %.2f   ratio %.1fx"
          % (fs, fg, fs / max(fg, 1e-6)))
    print("(guide flicker is the motion we asked for; the ratio is what the "
          "sampler added)")
    print(guard.report())
    guard.release()


if __name__ == "__main__":
    main()
