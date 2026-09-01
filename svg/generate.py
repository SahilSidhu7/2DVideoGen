"""Caption -> AniSVG -> frames, and the metric that actually matters.

Loss says nothing useful about a structured format: a model can sit at a
tempting loss and still emit text that will not parse, or that parses into a
cast nothing ever moves. So generation is scored on what the format demands -
does it parse, does it declare shapes, does anything move - and the samples are
rendered so they can be looked at.

    python svg/generate.py --adapter svg/data/soup_pilot -c "a cat waving"
    python svg/generate.py --adapter svg/data/soup_pilot --eval 32
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = "### animation\n%s\n\n### anisvg\n"
ALPACA = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n%s\n\n### Response:\n")


def extract(text):
    """Trim a generation down to the AniSVG document inside it.

    A base model warms up before it locks onto the format, emitting a few
    unrelated tokens first. The document itself starts at the `H` header, so
    anything before that is preamble, and anything after the last recognised
    line is trailing drift. Salvaging is legitimate here - the question is
    whether a usable clip was produced, not whether the very first token was.
    """
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("H ")), None)
    if start is None:
        return text.strip()
    kept = []
    for ln in lines[start:]:
        if ln[:2] in ("H ", "P ", "S ", "@ "):
            kept.append(ln)
        elif ln.strip():
            break                       # drifted off the grammar; stop here
    return "\n".join(kept)


def score(text):
    """Structural verdict for one generated clip."""
    from anisvg import Anim
    try:
        anim = Anim.from_text(extract(text))
    except Exception as exc:
        return dict(ok=False, why="parse: %s" % str(exc)[:60])
    if not anim.shapes:
        return dict(ok=False, why="no shapes")
    moving = sum(1 for f in anim.frames if f)
    if not anim.frames or not moving:
        return dict(ok=False, why="static")
    try:
        # Parsing is not enough - replay resolves every op against the cast, and
        # that is where a malformed clip actually shows up. "Valid" has to mean
        # "renderable", or the metric flatters the model.
        anim.to_svgs()
    except Exception as exc:
        return dict(ok=False, why="replay: %s" % str(exc)[:60])
    from anisvg import valid_op
    bad = sum(1 for f in anim.frames for o in f if not valid_op(o))
    return dict(ok=True, shapes=len(anim.shapes), frames=len(anim.frames),
                moving=moving, ops=sum(len(f) for f in anim.frames),
                dropped_ops=bad)


def render(text, out_dir, stem, width=256, every=1):
    """Write PNG frames and an animated GIF; returns the frame count."""
    from anisvg import Anim
    import raster
    anim = Anim.from_text(extract(text))
    svgs = anim.to_svgs()
    os.makedirs(out_dir, exist_ok=True)
    imgs = []
    for i, svg in enumerate(svgs):
        if i % every:
            continue
        img = raster.render(svg, width=width)
        img.save(os.path.join(out_dir, "%s_f%03d.png" % (stem, i)))
        imgs.append(img)
    if len(imgs) > 1:
        imgs[0].save(os.path.join(out_dir, "%s.gif" % stem), save_all=True,
                     append_images=imgs[1:],
                     duration=int(1000 / max(anim.fps, 1)), loop=0)
    return len(svgs)


def load(base, adapter, four_bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def build_prompt(tok, caption, style):
    """Match how the checkpoint was trained, or the model sees noise.

    Soup's alpaca loader rewrites each row into user/assistant chat messages and
    applies the tokenizer's chat template (`data/formats.py::_convert_alpaca`) -
    it never emits the "Below is an instruction" text. So a Soup-trained adapter
    must be prompted through the same template.
    """
    if style == "chat":
        # `add_generation_prompt=True` is NOT the prefix the model was trained
        # against: Qwen3's template also emits an empty `<think></think>` block
        # before the answer, so training text ran
        #   <|im_start|>assistant\n<think>\n\n</think>\n\nH 256 ...
        # Stopping at the assistant marker leaves the model to invent that block
        # itself, and greedy decoding derails there. Rendering a sentinel answer
        # and cutting at it recovers the exact prefix, whatever the template does.
        mark = " ANSWER "
        full = tok.apply_chat_template(
            [{"role": "user", "content": caption},
             {"role": "assistant", "content": mark}], tokenize=False)
        if mark in full:
            return full.split(mark)[0]
        return tok.apply_chat_template(
            [{"role": "user", "content": caption}],
            tokenize=False, add_generation_prompt=True)
    return (ALPACA if style == "alpaca" else PROMPT) % caption


# A character-set filter is too weak: the hex digits (a-f) and the op letters
# (t r s o v h w) share an alphabet that spells real words, so tokens like
# "edores", "errar" and ".Dao" pass a per-character test and then land inside a
# number. Whole tokens are matched instead - a run of digits, one structural
# character, or a short hex group for the palette.
_TOKEN_OK = re.compile(r"^(\d+|-|#|[0-9A-Fa-f]{1,6}|[HPS@trsovhw]|\s+)$")


def grammar_mask(tok):
    """Token ids whose text stays inside the AniSVG alphabet.

    The two decoders fail in opposite ways: greedy collapses into repeated
    zero-extent shapes, while sampling produces good geometry and then drops a
    stray token *inside a number* ("-2<thai>.didReceiveMe") and the clip will not
    parse. The grammar only ever needs about twenty characters, so rather than
    make bad tokens unlikely, make them unreachable - then sampling can supply
    variety without being able to corrupt the syntax.
    """
    import torch
    keep = torch.zeros(len(tok), dtype=torch.bool)
    for text, tid in tok.get_vocab().items():
        if tid >= len(keep):
            continue
        s = tok.convert_tokens_to_string([text])
        # Test the raw token too: a pure-whitespace token lstrips to "" and
        # would be rejected, which would leave the model unable to emit the
        # spaces the grammar is built from.
        if s and (_TOKEN_OK.match(s) or _TOKEN_OK.match(s.lstrip(" "))):
            keep[tid] = True
    for special in (tok.eos_token_id, tok.pad_token_id):
        if special is not None:
            keep[special] = True
    return keep


def generate(tok, model, caption, style, max_new, temp, top_p, keep=None):
    import torch
    prompt = build_prompt(tok, caption, style)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    kw = {}
    if keep is not None:
        from transformers import LogitsProcessorList
        # The model's output layer is wider than the tokenizer (151936 vs
        # 151669): the tail ids are padding with no text, so they are banned by
        # construction. Pad rather than slice, or the mask silently misaligns.
        banned = (~keep).to(model.device)

        def mask(_input_ids, scores):
            n = scores.shape[1]
            b = banned
            if b.numel() < n:
                b = torch.cat([b, torch.ones(n - b.numel(), dtype=torch.bool,
                                             device=b.device)])
            elif b.numel() > n:
                b = b[:n]
            return scores.masked_fill(b, float("-inf"))

        kw["logits_processor"] = LogitsProcessorList([mask])
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=temp > 0, temperature=max(temp, 1e-5),
                             top_p=top_p, pad_token_id=tok.pad_token_id, **kw)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--style", choices=["chat", "alpaca", "plain"], default="chat",
                    help="chat matches Soup (it applies the tokenizer template)")
    ap.add_argument("-c", "--caption", action="append", default=[])
    ap.add_argument("--eval", type=int, default=0, metavar="N",
                    help="score N held-out captions instead")
    ap.add_argument("--val", default="svg/data/train/val.jsonl")
    ap.add_argument("--out", default="svg/out/gen")
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--constrain", action="store_true",
                    help="mask the vocabulary to the AniSVG alphabet")
    args = ap.parse_args()

    captions = list(args.caption)
    if args.eval:
        rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][:args.eval]
        captions = [r["caption"] for r in rows]
    if not captions:
        captions = ["a black line-drawn cat waving its paw",
                    "a blue circle bouncing up and down",
                    "a stick figure walking to the right"]

    tok, model = load(args.base, args.adapter, args.four_bit)
    keep = grammar_mask(tok) if args.constrain else None
    if keep is not None:
        print("grammar mask: %d of %d tokens allowed" % (int(keep.sum()), len(keep)))
    os.makedirs(args.out, exist_ok=True)
    good = 0
    report = []
    for i, cap in enumerate(captions):
        text = generate(tok, model, cap, args.style, args.max_new,
                        args.temp, args.top_p, keep)
        verdict = score(text)
        stem = "g%03d" % i
        with open(os.path.join(args.out, stem + ".anisvg"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)
        if verdict["ok"]:
            good += 1
            try:
                render(text, args.out, stem)
            except Exception as exc:
                verdict = dict(ok=False, why="render: %s" % str(exc)[:60])
                good -= 1
        report.append((cap, verdict))
        print("%-3d %s | %s" % (i, "ok " if verdict["ok"] else "BAD",
                                cap[:64]))
        print("      %s" % json.dumps(verdict))

    print("\nvalid %d/%d = %.0f%%" % (good, len(captions),
                                      100.0 * good / max(len(captions), 1)))
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(caption=c, **v) for c, v in report], fh, indent=1)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
