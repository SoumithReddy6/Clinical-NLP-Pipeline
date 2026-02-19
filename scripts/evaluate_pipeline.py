#!/usr/bin/env python3
from __future__ import annotations

import argparse

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.pipeline.engine import ClinicalNLPPipeline


def score(pred: list[tuple[int, int, str]], gold: list[tuple[int, int, str]]) -> tuple[float, float, float]:
    pset = set(pred)
    gset = set(gold)
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    rows = read_jsonl(args.data, limit=args.limit if args.limit > 0 else None)
    pipe = ClinicalNLPPipeline()

    all_pred = []
    all_gold = []
    for row in rows:
        out = pipe.process(row["text"])
        all_pred.extend([(e.start, e.end, e.label) for e in out.entities])
        all_gold.extend([(e["start"], e["end"], e["label"]) for e in row.get("entities", [])])

    p, r, f1 = score(all_pred, all_gold)
    print(f"precision={p:.4f} recall={r:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
