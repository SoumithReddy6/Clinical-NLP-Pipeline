#!/usr/bin/env python3
"""Evaluate the Clinical NLP pipeline with per-entity-type metrics.

Reports micro/macro precision, recall, F1 overall and per entity type.
Supports both exact span matching and relaxed (overlap) matching.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from clinical_nlp.data.io import read_jsonl
from clinical_nlp.pipeline.engine import ClinicalNLPPipeline


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(
    pred_spans: list[tuple[int, int, str]],
    gold_spans: list[tuple[int, int, str]],
    labels: list[str] | None = None,
) -> dict:
    """Compute per-entity and overall micro/macro metrics (exact span match).

    Returns dict with keys: 'per_entity', 'micro', 'macro'.
    """
    if labels is None:
        labels = sorted({s[2] for s in pred_spans} | {s[2] for s in gold_spans})

    pset = set(pred_spans)
    gset = set(gold_spans)

    # Per-entity counts
    per_entity: dict[str, dict] = {}
    micro_tp = micro_fp = micro_fn = 0

    for label in labels:
        p_label = {s for s in pset if s[2] == label}
        g_label = {s for s in gset if s[2] == label}
        tp = len(p_label & g_label)
        fp = len(p_label - g_label)
        fn = len(g_label - p_label)
        prec, rec, f1 = _prf(tp, fp, fn)
        per_entity[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": len(g_label),
            "predicted": len(p_label),
            "tp": tp, "fp": fp, "fn": fn,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_p, micro_r, micro_f1 = _prf(micro_tp, micro_fp, micro_fn)

    # Macro averages (only over labels with support > 0)
    labels_with_support = [l for l in labels if per_entity[l]["support"] > 0]
    macro_p = sum(per_entity[l]["precision"] for l in labels_with_support) / max(len(labels_with_support), 1)
    macro_r = sum(per_entity[l]["recall"] for l in labels_with_support) / max(len(labels_with_support), 1)
    macro_f1 = sum(per_entity[l]["f1"] for l in labels_with_support) / max(len(labels_with_support), 1)

    return {
        "per_entity": per_entity,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "total_predicted": micro_tp + micro_fp,
        "total_gold": micro_tp + micro_fn,
    }


def relaxed_match(
    pred_spans: list[tuple[int, int, str]],
    gold_spans: list[tuple[int, int, str]],
) -> dict:
    """Relaxed matching — count a prediction correct if it overlaps a gold span with the same label."""
    tp = fp = fn = 0
    matched_gold: set[int] = set()

    for ps, pe, pl in pred_spans:
        found = False
        for gi, (gs, ge, gl) in enumerate(gold_spans):
            if gl == pl and ps < ge and pe > gs and gi not in matched_gold:
                tp += 1
                matched_gold.add(gi)
                found = True
                break
        if not found:
            fp += 1

    fn = len(gold_spans) - len(matched_gold)
    prec, rec, f1 = _prf(tp, fp, fn)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def print_report(results: dict) -> None:
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 72)
    print(f"{'Entity Type':<25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>9}")
    print("-" * 72)
    for label, m in sorted(results["per_entity"].items()):
        print(f"  {label:<23} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:9d}")
    print("-" * 72)
    micro = results["micro"]
    macro = results["macro"]
    print(f"  {'MICRO avg':<23} {micro['precision']:8.4f} {micro['recall']:8.4f} {micro['f1']:8.4f} {results['total_gold']:9d}")
    print(f"  {'MACRO avg':<23} {macro['precision']:8.4f} {macro['recall']:8.4f} {macro['f1']:8.4f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clinical NER pipeline")
    parser.add_argument("--data", required=True, help="JSONL evaluation file")
    parser.add_argument("--limit", type=int, default=1000, help="Max evaluation records (0 = all)")
    parser.add_argument("--relaxed", action="store_true", help="Also report relaxed overlap matching")
    args = parser.parse_args()

    rows = read_jsonl(args.data, limit=args.limit if args.limit > 0 else None)
    pipe = ClinicalNLPPipeline()

    all_pred: list[tuple[int, int, str]] = []
    all_gold: list[tuple[int, int, str]] = []

    print(f"Evaluating on {len(rows)} records...")
    for i, row in enumerate(rows):
        out = pipe.process(row["text"])
        all_pred.extend([(e.start, e.end, e.label) for e in out.entities])
        all_gold.extend([(e["start"], e["end"], e["label"]) for e in row.get("entities", [])])
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(rows)}...")

    # Exact match evaluation
    results = evaluate(all_pred, all_gold, labels=["DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"])
    print("\n--- EXACT SPAN MATCH ---")
    print_report(results)

    # Relaxed match
    if args.relaxed:
        relaxed = relaxed_match(all_pred, all_gold)
        print("\n--- RELAXED OVERLAP MATCH ---")
        print(f"  precision={relaxed['precision']:.4f}  recall={relaxed['recall']:.4f}  f1={relaxed['f1']:.4f}")
        print(f"  (tp={relaxed['tp']} fp={relaxed['fp']} fn={relaxed['fn']})")


if __name__ == "__main__":
    main()
