#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_ann(ann_path: Path) -> list[dict]:
    entities = []
    if not ann_path.exists():
        return entities
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected simplified BRAT-like: T1\tLABEL start end\ttext
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        meta = parts[1].split()
        if len(meta) < 3:
            continue
        label = meta[0].upper()
        start = int(meta[1])
        end = int(meta[2])
        entities.append({"start": start, "end": end, "text": parts[2], "label": label})
    return entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt and .ann files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    root = Path(args.input_dir)
    rows = []
    for txt in sorted(root.glob("*.txt")):
        text = txt.read_text(encoding="utf-8")
        ann = txt.with_suffix(".ann")
        entities = parse_ann(ann)
        rows.append({"id": txt.stem, "text": text, "entities": entities})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Converted {len(rows)} documents to {out}")


if __name__ == "__main__":
    main()
