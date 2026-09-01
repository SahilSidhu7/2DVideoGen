"""LoRA fine-tune Qwen3-0.6B-Base: caption -> AniSVG.

Era 1 used t5-small (60M, 512 context) to emit a nine-archetype DSL. AniSVG is
structured markup running ~1800 tokens a clip, which is both longer than
t5-small's context and a different job - so the base model changes with the
representation. Qwen3-0.6B-Base is a decoder-only model with a long context
that has already seen a great deal of markup, and at 0.6B a LoRA fits an 8 GB
card.

Loss is masked over the prompt: the model is graded on the animation it writes,
not on echoing the caption back.
"""
import argparse
import json
import math
import os

# Long sequences against a 151k vocab churn large short-lived logit buffers;
# expandable segments keep that from fragmenting the 8 GB card. Must be set
# before torch initialises its allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset

PROMPT = "### animation\n%s\n\n### anisvg\n"


class Clips(Dataset):
    """Caption + AniSVG body, with the prompt masked out of the loss."""

    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        head = self.tok(PROMPT % r["caption"], add_special_tokens=False)["input_ids"]
        body = self.tok(r["text"], add_special_tokens=False)["input_ids"]
        ids = (head + body + [self.tok.eos_token_id])[:self.max_len]
        labels = list(ids)
        for j in range(min(len(head), len(labels))):
            labels[j] = -100
        return dict(input_ids=ids, labels=labels)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def chunked_ce(hidden, lm_head, labels, chunk):
    """Cross-entropy without ever materialising the full logit tensor.

    `Qwen3ForCausalLM` upcasts logits to fp32 for the loss, so one 2k-token
    sample against a 151936 vocab costs ~1.05 GB, and its gradient the same
    again - that buffer, not the weights, is what puts a 1.7B or 4B base over
    an 8 GB card. Projecting and scoring one slice of the sequence at a time
    caps the cost at `chunk` rows instead of the whole sequence, at the price
    of a second lm_head pass in backward.

    Returns (summed loss, supervised token count) so the caller can normalise.
    Which divisor is correct depends on the caller: Trainer hands
    `num_items_in_batch` down when it accumulates, and then does *not* divide by
    the accumulation steps itself.
    """
    import torch.nn.functional as F
    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_y = labels.reshape(-1)
    for i in range(0, flat_h.shape[0], chunk):
        h, y = flat_h[i:i + chunk], flat_y[i:i + chunk]
        keep = y != -100
        if not bool(keep.any()):
            continue
        logits = lm_head(h[keep]).float()
        total = total + F.cross_entropy(logits, y[keep], reduction="sum")
        count += int(keep.sum())
    return total, count


def make_trainer_class(Trainer, chunk):
    class ChunkedTrainer(Trainer):
        """Trainer that scores the sequence in slices - see `chunked_ce`."""

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            out = base.model(input_ids=inputs["input_ids"],
                             attention_mask=inputs.get("attention_mask"))
            hidden = out.last_hidden_state[:, :-1, :]
            total, count = chunked_ce(hidden, base.lm_head, labels[:, 1:], chunk)
            # When Trainer supplies num_items_in_batch it has already decided the
            # denominator for the whole accumulated batch and will not divide by
            # the accumulation steps afterwards. Returning a plain mean here
            # inflates every gradient by exactly that factor.
            loss = total / (num_items_in_batch if num_items_in_batch else max(count, 1))
            return (loss, out) if return_outputs else loss

    return ChunkedTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--data", default="svg/data/train")
    ap.add_argument("--out", default="svg/data/lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="peak VRAM is set by the longest clip: the loss "
                         "materialises seq x 151936 logits in fp32, plus grad")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train rows")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4-quantise the frozen base (QLoRA)")
    ap.add_argument("--loss-chunk", type=int, default=512,
                    help="tokens scored per slice; 0 uses the built-in loss")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train = Clips(os.path.join(args.data, "train.jsonl"), tok, args.max_len)
    val = Clips(os.path.join(args.data, "val.jsonl"), tok, args.max_len)
    if args.limit:
        train.rows = train.rows[:args.limit]
        val.rows = val.rows[:max(8, args.limit // 20)]
    print("train %d clips, val %d clips" % (len(train), len(val)))

    load = dict(torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu")
    if args.load_4bit:
        # 4-bit NF4 roughly quarters the weight footprint, which is what makes a
        # 4B base fit beside the logit buffers on an 8 GB card.
        from transformers import BitsAndBytesConfig
        load["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **load)
    model.config.use_cache = False              # incompatible with checkpointing
    # With every base weight frozen, the embedding output does not require grad,
    # so a checkpointed block has nothing to differentiate and backward dies with
    # "element 0 of tensors does not require grad". Forcing the input to require
    # grad gives the recomputed graph something to hang off.
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps = max(1, math.ceil(len(train) / (args.batch * args.accum) * args.epochs))
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(50, steps // 10),
        save_steps=max(50, steps // 10),
        save_total_limit=3,
        report_to=[],
    )
    cls = make_trainer_class(Trainer, args.loss_chunk) if args.loss_chunk else Trainer
    cls(model=model, args=targs, train_dataset=train, eval_dataset=val,
        data_collator=lambda b: collate(b, tok.pad_token_id)).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if torch.cuda.is_available():
        print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 2**30))
    print("adapter -> %s" % args.out)


if __name__ == "__main__":
    main()
