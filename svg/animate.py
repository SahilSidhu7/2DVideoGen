"""AnimateDiff + ControlNet-OpenPose on the exact COCO-18 guides.

Attempt 19 measured that better pose conditioning does nothing for flicker -
the ratio went *up*, 23.6x -> 36.2x, as the guide got more correct. Pose says
where the limbs go and nothing about hair, shading or background, which is
where the flicker lives. So temporal stability needs an explicit temporal
mechanism, and this is it: AnimateDiff's motion module (Guo et al., arXiv
2307.04725) inserted into the same frozen SD1.5 UNet, run at inference only.

The comparison is held as tight as Attempt 18's was. Same clip, same seed, same
prompt, same negative, same steps, same guidance, same conditioning scale, same
COCO-18 guides. The only difference is that the frames are now denoised jointly
through temporal attention instead of independently.

The 8 GB card is the binding constraint and the point of `--sweep`: AnimateDiff
denoises every frame of the clip in one batched pass, so peak VRAM scales with
frame count and with latent area, not just with model size. `--sweep` walks
(resolution, frames) configurations from the largest down, records peak
allocated and reserved VRAM for each, and reports OOM as a measurement rather
than as a failure to run.

    USE_TF=0 python svg/animate.py --sweep
    USE_TF=0 python svg/animate.py --frames 8 --width 384
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_PROMPT = ("anime style character, cel shaded, clean line art, "
                  "flat colours, white background")
DEFAULT_NEG = ("photo, realistic, 3d render, blurry, lowres, watermark, "
               "text, extra limbs, deformed")

# Largest first: the first configuration that fits is the one worth reporting,
# and the ones that do not fit are still a measurement.
SWEEP = [(512, 16), (448, 16), (384, 16), (320, 16), (256, 16), (512, 8)]


def guides_for(clip, width, n_frames, out_dir):
    """COCO-18 conditioning frames for a clip, at a given render width."""
    from PIL import Image
    from guides import build
    text = open(clip, encoding="utf-8").read()
    anim, made = build(text, out_dir, width, kinds=("coco18",))
    if made.get("coco18", 0) == 0:
        raise SystemExit("clip does not match the 16-part character schema")
    d = os.path.join(out_dir, "coco18")
    files = sorted(os.listdir(d))[:n_frames]
    return anim, [Image.open(os.path.join(d, f)).convert("RGB") for f in files]


def build_pipe(model, controlnet, adapter, lightning_ckpt=None,
               offload=False):
    import torch
    from diffusers import (AnimateDiffControlNetPipeline, ControlNetModel,
                           MotionAdapter, LCMScheduler, DDIMScheduler)
    td = torch.float16
    net = ControlNetModel.from_pretrained(controlnet, torch_dtype=td)
    ada = MotionAdapter.from_pretrained(adapter, torch_dtype=td)
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        model, motion_adapter=ada, controlnet=net, torch_dtype=td,
        safety_checker=None, requires_safety_checker=False)
    # AnimateDiff's own recommended scheduler for the v1.5-2 adapter: linear
    # beta schedule with the trailing timestep spacing, clipping off.
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear",
        timestep_spacing="linspace", clip_sample=False,
        steps_offset=1)
    if lightning_ckpt:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("ByteDance/AnimateDiff-Lightning", lightning_ckpt)
        pipe.unet.load_state_dict(load_file(p, device="cpu"), strict=False)
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config, beta_schedule="linear",
            timestep_spacing="trailing")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    if not offload:
        pipe.to("cuda")
    # NB when offloading: the hooks must be installed *after* every module
    # exists, so `enable_model_cpu_offload` is called by the caller once any
    # IP-Adapter image encoder has been loaded. Installing it first leaves the
    # encoder unhooked and on the CPU, and the run dies with a half-tensor
    # dtype mismatch several minutes in.
    return pipe


def run(pipe, conds, prompt, negative, steps, scale, cond_scale, seed, guard,
        size=512, ip_image=None):
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def on_step(p, i, t, kw):
        guard.step()                              # ~50% duty, damaged fan
        return kw

    t0 = time.time()
    out = pipe(prompt=prompt, negative_prompt=negative,
               conditioning_frames=conds, num_frames=len(conds),
               height=size, width=size,
               num_inference_steps=steps, guidance_scale=scale,
               controlnet_conditioning_scale=cond_scale,
               generator=gen, callback_on_step_end=on_step,
               **({"ip_adapter_image": ip_image} if ip_image is not None else {})
               ).frames[0]
    return out, time.time() - t0, (torch.cuda.max_memory_allocated() / 2 ** 30,
                                   torch.cuda.max_memory_reserved() / 2 ** 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="release/corpus-anime/clip00.anisvg")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet",
                    default="lllyasviel/control_v11p_sd15_openpose")
    ap.add_argument("--adapter",
                    default="guoyww/animatediff-motion-adapter-v1-5-2")
    ap.add_argument("--lightning", default=None,
                    help="e.g. animatediff_lightning_4step_diffusers.safetensors")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=7.0)
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--ip-adapter", action="store_true",
                    help="anchor identity on the clip's own flat colour render "
                         "of frame 0 via IP-Adapter (h94/IP-Adapter, sd15)")
    ap.add_argument("--ip-scale", type=float, default=0.6)
    ap.add_argument("--sweep", action="store_true",
                    help="walk resolutions/frame counts and record peak VRAM")
    ap.add_argument("-o", "--out", default="svg/out/a20")
    args = ap.parse_args()

    import torch
    from render_style import flicker
    from tools.gpuguard import Guard

    print("loading SD1.5 + ControlNet-OpenPose + motion adapter ...")
    pipe = build_pipe(args.model, args.controlnet, args.adapter,
                      args.lightning, args.offload)
    weights = torch.cuda.memory_allocated() / 2 ** 30
    print("weights resident: %.2f GiB" % weights)

    ip_image = None
    if args.ip_adapter:
        # The anchor is the character we already drew - the clip's own flat
        # colour render. Nothing external is introduced; the appearance the
        # vector side produced becomes the appearance the diffusion side has
        # to hold on to.
        import raster
        from anisvg import Anim
        from generate import extract
        a = Anim.from_text(extract(open(args.clip, encoding="utf-8").read()))
        ip_image = raster.render(a.to_svgs()[0], width=512).convert("RGB")
        os.makedirs(args.out, exist_ok=True)
        ip_image.save(os.path.join(args.out, "ip_reference.png"))
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                             weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(args.ip_scale)
        print("IP-Adapter loaded, scale %.2f; weights now %.2f GiB"
              % (args.ip_scale, torch.cuda.memory_allocated() / 2 ** 30))

    if args.offload:
        # Keeps only the module in use resident; the trade is host<->device
        # copies per step, which is time rather than memory.
        pipe.enable_model_cpu_offload()

    guard = Guard()
    configs = SWEEP if args.sweep else [(args.width, args.frames)]
    rows = []
    for width, n in configs:
        gdir = os.path.join(args.out, "guides_w%d" % width)
        anim, conds = guides_for(args.clip, width, n, gdir)
        if len(conds) < n:
            print("  (clip has only %d frames; using those)" % len(conds))
        tag = "w%d_f%d" % (width, len(conds))
        try:
            frames, secs, (alloc, res) = run(
                pipe, conds, args.prompt, args.negative, args.steps,
                args.scale, args.cond_scale, args.seed, guard, size=width,
                ip_image=ip_image)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_reserved() / 2 ** 30
            print("%-12s OOM at %.2f GiB reserved  (%s)"
                  % (tag, peak, str(exc).split(".")[0][:60]))
            rows.append((tag, None, None, peak, None, None))
            continue
        d = os.path.join(args.out, tag)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(os.path.join(d, "s%04d.png" % i))
        frames[0].save(os.path.join(d, "styled.gif"), save_all=True,
                       append_images=frames[1:],
                       duration=int(1000 / (anim.fps or 15)), loop=0)
        fs, fg = flicker(frames), flicker(conds)
        print("%-12s ok  %5.1fs  peak alloc %.2f / reserved %.2f GiB  "
              "flicker styled %.2f guide %.2f ratio %.1fx"
              % (tag, secs, alloc, res, fs, fg, fs / max(fg, 1e-6)))
        rows.append((tag, secs, alloc, res, fs, fg))

    print("\n| config | s | peak alloc GiB | peak reserved GiB | guide | styled | ratio |")
    print("|---|---|---|---|---|---|---|")
    for tag, secs, alloc, res, fs, fg in rows:
        if fs is None:
            print("| %s | OOM | - | %.2f | - | - | - |" % (tag, res))
        else:
            print("| %s | %.0f | %.2f | %.2f | %.2f | %.2f | %.1fx |"
                  % (tag, secs, alloc, res, fg, fs, fs / max(fg, 1e-6)))
    print(guard.report())
    guard.release()


if __name__ == "__main__":
    main()
