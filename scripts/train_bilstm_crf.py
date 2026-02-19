#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.models.tokenization import tokenize_with_offsets
from clinical_nlp.models.bilstm_crf import BiLSTMCRFTagger
from clinical_nlp.training.labeling import LABELS, token_level_labels


MAX_LEN = 128


def prepare(records: list[dict], tagger: BiLSTMCRFTagger, max_len: int):
    token_seqs = []
    label_ids = []
    for row in records:
        toks = [t for t, _, _ in tokenize_with_offsets(row["text"])]
        labs = token_level_labels(toks)
        token_seqs.append(toks)
        clipped = labs[:max_len]
        label_ids.append([tagger.label2id[l] for l in clipped] + [0] * max(0, max_len - len(clipped)))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=20000, help="Cap training rows to avoid OOM; use 0 for all.")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--save-dir", required=True)
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
    tagger = BiLSTMCRFTagger(labels=LABELS)
    X, y = prepare(records, tagger, max_len=args.max_len)
    tagger.init_model()

    model = tagger.model
    assert model is not None
    device = select_device()
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, len(LABELS)), y_batch.view(-1))
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1
        avg_loss = running_loss / max(1, num_batches)
        print(f"epoch={epoch+1} loss={avg_loss:.4f}")

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
    print(f"Saved model to {out / 'bilstm_crf.pt'}")


if __name__ == "__main__":
    main()
