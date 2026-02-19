#!/usr/bin/env python3
from __future__ import annotations

import argparse

from clinical_nlp.data.synthetic import generate_dataset
from clinical_nlp.data.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-notes", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = generate_dataset(args.num_notes, seed=args.seed)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
