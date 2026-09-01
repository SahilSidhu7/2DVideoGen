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
                          Seq2SeqTrainingArguments)

BASE = "t5-small"
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
    args = ap.parse_args()
    MAX_IN, MAX_OUT = args.max_in, args.max_out
    print(f"max_in={MAX_IN} max_out={MAX_OUT}")

    torch.set_num_threads(torch.get_num_threads())  # use all cores
    print(f"threads: {torch.get_num_threads()}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    train_ds, val_ds = PairDS(args.train, tok), PairDS(args.val, tok)
    if args.val_cap:
        val_ds.rows = val_ds.rows[:args.val_cap]
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    collator = DataCollatorForSeq2Seq(tok, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=args.gen_eval,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}/")


if __name__ == "__main__":
    main()
