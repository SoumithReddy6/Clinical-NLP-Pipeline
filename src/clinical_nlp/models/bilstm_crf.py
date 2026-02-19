from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        return logits


class BiLSTMCRFTagger:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for i, l in enumerate(labels)}
        self.vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.model: Optional[BiLSTMCRF] = None

    def fit_vocab(self, token_sequences: list[list[str]]) -> None:
        for seq in token_sequences:
            for token in seq:
                key = token.lower()
                if key not in self.vocab:
                    self.vocab[key] = len(self.vocab)

    def encode(self, tokens: list[str], max_len: int = 256) -> list[int]:
        ids = [self.vocab.get(t.lower(), 1) for t in tokens[:max_len]]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def init_model(self) -> None:
        self.model = BiLSTMCRF(vocab_size=len(self.vocab), num_labels=len(self.labels))

    def predict_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)
            return logits.argmax(-1)
