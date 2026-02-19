#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from clinical_nlp.pipeline.engine import ClinicalNLPPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    pipe = ClinicalNLPPipeline()
    text = "Patient John Doe with diabetes started metformin 500 mg daily on 01/03/2024."

    t0 = time.time()
    for _ in range(args.n):
        pipe.process(text)
    elapsed = time.time() - t0
    per_doc_ms = (elapsed / args.n) * 1000
    throughput = args.n / elapsed

    print(f"docs={args.n} elapsed={elapsed:.2f}s")
    print(f"latency_ms={per_doc_ms:.2f} throughput_docs_per_sec={throughput:.2f}")


if __name__ == "__main__":
    main()
