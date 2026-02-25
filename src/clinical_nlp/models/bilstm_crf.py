from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CRFLayer(nn.Module):
    """Linear-chain Conditional Random Field for sequence labeling."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        # Transition scores: transitions[i][j] = score of j -> i
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))

    def forward(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood of the label sequence.

        Args:
            emissions: (batch, seq_len, num_labels)
            labels: (batch, seq_len) — ground truth label IDs
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            Scalar loss (negative log-likelihood, mean over batch).
        """
        gold_score = self._score_sentence(emissions, labels, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        nll = (forward_score - gold_score)
        return nll.mean()

    def _score_sentence(
        self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, num_labels = emissions.shape
        score = self.start_transitions[labels[:, 0]] + emissions[:, 0].gather(1, labels[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_len):
            cur_mask = mask[:, t]
            emit = emissions[:, t].gather(1, labels[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[labels[:, t], labels[:, t - 1]]
            score = score + (emit + trans) * cur_mask
        # Find last valid position for each sequence
        lengths = mask.long().sum(dim=1) - 1
        last_labels = labels.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_labels]
        return score

    def _forward_algorithm(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, num_labels = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch, num_labels)
        for t in range(1, seq_len):
            cur_mask = mask[:, t].unsqueeze(1)  # (batch, 1)
            # score: (batch, num_labels, 1) + transitions: (num_labels, num_labels) + emit: (batch, 1, num_labels)
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)  # (batch, num_labels)
            score = torch.where(cur_mask.bool(), next_score, score)
        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)  # (batch,)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding to find the best label sequence."""
        batch, seq_len, num_labels = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []

        for t in range(1, seq_len):
            cur_mask = mask[:, t].unsqueeze(1)
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            next_score, indices = next_score.max(dim=1)
            score = torch.where(cur_mask.bool(), next_score, score)
            history.append(indices)

        score = score + self.end_transitions.unsqueeze(0)
        _, best_last = score.max(dim=1)

        best_paths: list[list[int]] = []
        lengths = mask.long().sum(dim=1)
        for b in range(batch):
            best_path = [best_last[b].item()]
            for t in range(len(history) - 1, -1, -1):
                if t + 1 < lengths[b]:
                    best_path.append(history[t][b][best_path[-1]].item())
            best_path.reverse()
            best_paths.append(best_path[: lengths[b].item()])

        return best_paths


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for clinical NER sequence labeling."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRFLayer(num_labels)

    def _emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Returns CRF loss if labels provided, else emissions."""
        emissions = self._emissions(input_ids)
        if mask is None:
            mask = (input_ids != 0).float()
        if labels is not None:
            return self.crf(emissions, labels, mask)
        return emissions

    def decode(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """Viterbi decode the best label sequence."""
        emissions = self._emissions(input_ids)
        if mask is None:
            mask = (input_ids != 0).float()
        return self.crf.decode(emissions, mask)


class BiLSTMCRFTagger:
    """High-level wrapper for training and inference with BiLSTM-CRF."""

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

    def predict_ids(self, input_ids: torch.Tensor) -> list[list[int]]:
        """Decode using Viterbi (CRF) instead of argmax."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.eval()
        with torch.no_grad():
            return self.model.decode(input_ids)
