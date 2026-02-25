#!/usr/bin/env python3
"""Fine-tune a transformer model for clinical NER with BIO tagging.

Supports train/eval split and per-entity-type metrics during training.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.models.tokenization import tokenize_with_offsets
from clinical_nlp.training.labeling import LABELS, token_level_labels


@dataclass
class NERExample:
    tokens: list[str]
    labels: list[int]


def build_examples(path: str, limit: Optional[int] = None) -> list[NERExample]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    examples = []
    label2id = {l: i for i, l in enumerate(LABELS)}
    for row in rows:
        toks = [t for t, _, _ in tokenize_with_offsets(row["text"])]
        labs = token_level_labels(toks)
        examples.append(NERExample(tokens=toks, labels=[label2id[l] for l in labs]))
    return examples


def tokenize_and_align(tokenizer, examples: list[NERExample], max_length: int = 256):
    all_inputs = []
    for ex in examples:
        enc = tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = enc.word_ids()
        labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(ex.labels[wid] if wid < len(ex.labels) else 0)
            else:
                labels.append(-100)
            prev_word = wid
        enc["labels"] = labels
        all_inputs.append(enc)
    return all_inputs


class ListDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def compute_metrics(eval_pred):
    """Compute token accuracy and per-entity-type accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100

    correct = (preds == labels) & mask
    total = mask.sum()
    acc = float(correct.sum() / total) if total else 0.0

    metrics = {"token_accuracy": acc}
    for i, label_name in enumerate(LABELS):
        if label_name == "O":
            continue
        label_mask = (labels == i) & mask
        label_total = label_mask.sum()
        if label_total > 0:
            label_correct = ((preds == i) & label_mask).sum()
            metrics[f"acc_{label_name}"] = float(label_correct / label_total)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune transformer for clinical NER")
    parser.add_argument("--train", required=True, help="Training JSONL path")
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data for evaluation")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    examples = build_examples(args.train, limit=args.max_samples)
    print(f"Built {len(examples)} examples with {len(LABELS)} BIO labels: {LABELS}")

    # Train/eval split
    split_idx = max(1, int(len(examples) * (1 - args.eval_split)))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_enc = tokenize_and_align(tokenizer, train_examples)
    eval_enc = tokenize_and_align(tokenizer, eval_examples)
    train_dataset = ListDataset(train_enc)
    eval_dataset = ListDataset(eval_enc)

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for i, l in enumerate(LABELS)}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    train_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved transformer NER model to {args.save_dir}")


if __name__ == "__main__":
    main()
