"""
train.py - fine-tune t5-small to map: user prompt -> DSL spec.

Small model (60M params), short sequences, CPU-friendly. On a 16-core CPU a few
epochs over ~9k synthetic pairs takes on the order of minutes.

    python train.py --epochs 3 --batch 32
    -> model/checkpoint/   (load with infer.py)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class GuardCallback(TrainerCallback):
    """Hold the GPU to the project's 50% duty budget, one step at a time.

    Every GPU loop in this project runs through tools.gpuguard.Guard because
    the development machine's fan is damaged. Trainer owns the loop, so the
    guard has to enter it as a callback rather than as a for-loop call.
    """

    def __init__(self):
        from tools.gpuguard import Guard
        self.guard = Guard()

    def on_step_end(self, args, state, control, **kw):
        self.guard.step()

    def on_train_end(self, args, state, control, **kw):
        print(self.guard.report(), flush=True)
        self.guard.release()

BASE = "t5-small"   # Attempts 2-23 default; overridable on the CLI
PREFIX = "prompt2scene: "
MAX_IN, MAX_OUT = 64, 48   # Attempt 2 defaults; overridable on the CLI


class PairDS(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in Path(path).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(PREFIX + r["prompt"], max_length=MAX_IN,
                     truncation=True, padding=False)
        y = self.tok(text_target=r["dsl"], max_length=MAX_OUT,
                     truncation=True, padding=False)
        x["labels"] = y["input_ids"]
        return x


def main():
    global MAX_IN, MAX_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="checkpoint")
    # Attempt 22: a scene script is a much longer target than Attempt 2's
    # one-line archetype spec, so the sequence caps have to be settable.
    ap.add_argument("--max-in", type=int, default=MAX_IN)
    ap.add_argument("--max-out", type=int, default=MAX_OUT)
    # generate-during-eval is O(beam x max_out) per val row; on a long
    # structured target that costs more than the epoch it reports on.
    ap.add_argument("--gen-eval", action="store_true")
    ap.add_argument("--val-cap", type=int, default=0)
    # Attempt 24: the capacity ablation needs the base checkpoint to be a
    # variable, so that a run's provenance records which one it started from.
    ap.add_argument("--base", default=BASE)
    # Attempt 24: t5-base does not fit on the 8 GB card at batch 16 / 256
    # tokens, and this path is CPU-only by design anyway - any GPU work in
    # this project has to go through tools/gpuguard.py because the fan on the
    # development machine is damaged. Trainer will silently claim the GPU if
    # one is visible, so CPU has to be the explicit default, not the accident.
    ap.add_argument("--gpu", action="store_true",
                    help="allow CUDA (only ever under tools/gpuguard.py)")
    # Peak CPU memory scales with the microbatch, not the effective batch.
    # Accumulating keeps the optimisation identical while halving the
    # activation footprint.
    ap.add_argument("--grad-accum", type=int, default=1)
    # Activation memory is what puts t5-base over the 8 GB card; recomputing
    # activations trades ~30% step time for most of that footprint back.
    ap.add_argument("--grad-checkpoint", action="store_true")
    # A run on this machine can be interrupted (thermal, or the operator);
    # per-epoch checkpoints are useless if the harness cannot pick them up.
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume from --out, or from a named checkpoint dir")
    args = ap.parse_args()
    MAX_IN, MAX_OUT = args.max_in, args.max_out
    print(f"base={args.base} max_in={MAX_IN} max_out={MAX_OUT} "
          f"device={'cuda' if args.gpu else 'cpu'} "
          f"batch={args.batch}x{args.grad_accum}")

    torch.set_num_threads(torch.get_num_threads())  # use all cores
    print(f"threads: {torch.get_num_threads()}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e6:.1f}M")

    train_ds, val_ds = PairDS(args.train, tok), PairDS(args.val, tok)
    if args.val_cap:
        val_ds.rows = val_ds.rows[:args.val_cap]
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    collator = DataCollatorForSeq2Seq(tok, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        use_cpu=not args.gpu,
        gradient_checkpointing=args.grad_checkpoint,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        # 2, not 1: a run killed mid-save leaves a truncated checkpoint, and
        # with a limit of 1 the previous good one is already gone.
        save_total_limit=2,
        predict_with_generate=args.gen_eval,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tok)
    if args.gpu:
        trainer.add_callback(GuardCallback())
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # Provenance travels with the weights: a checkpoint directory alone does
    # not otherwise say which base it was fine-tuned from.
    Path(args.out, "run_provenance.json").write_text(json.dumps({
        "base": args.base, "params": n_par, "train": args.train,
        "val": args.val, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "max_in": MAX_IN, "max_out": MAX_OUT,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "device": "cuda" if args.gpu else "cpu",
        "grad_checkpoint": args.grad_checkpoint,
    }, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}/")


if __name__ == "__main__":
    main()
