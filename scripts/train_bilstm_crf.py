#!/usr/bin/env python3
"""Train BiLSTM-CRF model for clinical NER.

Uses the real CRF loss (negative log-likelihood) instead of cross-entropy.
Supports CUDA, Apple MPS, and CPU devices.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.models.tokenization import tokenize_with_offsets
from clinical_nlp.models.bilstm_crf import BiLSTMCRFTagger
from clinical_nlp.training.labeling import LABELS, token_level_labels


MAX_LEN = 128


def prepare(records: list[dict], tagger: BiLSTMCRFTagger, max_len: int):
    """Tokenize records and build padded tensors for training."""
    token_seqs = []
    label_ids = []
    for row in records:
        toks = [t for t, _, _ in tokenize_with_offsets(row["text"])]
        labs = token_level_labels(toks)
        token_seqs.append(toks)
        clipped = labs[:max_len]
        label_ids.append(
            [tagger.label2id[l] for l in clipped]
            + [0] * max(0, max_len - len(clipped))
        )
    tagger.fit_vocab(token_seqs)
    X = [tagger.encode(toks, max_len=max_len) for toks in token_seqs]
    return torch.tensor(X, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF for clinical NER")
    parser.add_argument("--train", required=True, help="Training JSONL path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Cap training rows to avoid OOM; use 0 for all.")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--save-dir", required=True, help="Output directory for model checkpoint")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_len <= 0:
        raise ValueError("--max-len must be > 0")

    limit = args.max_samples if args.max_samples > 0 else None
    records = read_jsonl(args.train, limit=limit)
    if not records:
        raise ValueError("No training records found")

    print(f"Loaded {len(records)} records")
    print(f"Labels ({len(LABELS)}): {LABELS}")
    tagger = BiLSTMCRFTagger(labels=LABELS)
    X, y = prepare(records, tagger, max_len=args.max_len)
    tagger.init_model()

    model = tagger.model
    assert model is not None
    device = select_device()
    model.to(device)
    print(f"Using device: {device}")
    print(f"Vocab size: {len(tagger.vocab)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * ((len(X) + args.batch_size - 1) // args.batch_size)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # CRF loss (negative log-likelihood)
            loss = model(x_batch, labels=y_batch)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        lr = scheduler.get_last_lr()[0]
        print(f"epoch={epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": tagger.labels,
            "vocab": tagger.vocab,
        },
        out / "bilstm_crf.pt",
    )
    print(f"Saved model to {out / 'bilstm_crf.pt'} (best_loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
