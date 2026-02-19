#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.models.tokenization import tokenize_with_offsets
from clinical_nlp.models.bilstm_crf import BiLSTMCRFTagger
from clinical_nlp.training.labeling import LABELS, token_level_labels


MAX_LEN = 128


def prepare(records: list[dict], tagger: BiLSTMCRFTagger):
    token_seqs = []
    label_ids = []
    for row in records:
        toks = [t for t, _, _ in tokenize_with_offsets(row["text"])]
        labs = token_level_labels(toks)
        token_seqs.append(toks)
        label_ids.append([tagger.label2id[l] for l in labs[:MAX_LEN]] + [0] * max(0, MAX_LEN - len(labs[:MAX_LEN])))
    tagger.fit_vocab(token_seqs)
    X = [tagger.encode(toks, max_len=MAX_LEN) for toks in token_seqs]
    return torch.tensor(X, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    records = read_jsonl(args.train)
    tagger = BiLSTMCRFTagger(labels=LABELS)
    X, y = prepare(records, tagger)
    tagger.init_model()

    model = tagger.model
    assert model is not None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits.view(-1, len(LABELS)), y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch+1} loss={loss.item():.4f}")

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
