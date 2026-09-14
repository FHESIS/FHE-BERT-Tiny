#!/usr/bin/env python3
"""
Benchmarks the optimized, in-process plaintext circuit (src/python/PlainCircuit.py)
and records results for comparison against the pre-optimization baseline in
sentiment_comparison_results.json (see PLAINTEXT_OPTIMIZATION_REPORT.md /
PLAINTEXT_OPTIMIZATION_COMPARISON_REPORT.md).

Unlike run_sentiment_comparison.py (which subprocess-execs a fresh
`python3 PlainCircuit.py <sentence>` per sentence, paying the ~8s
import+model-load cost every time), this script loads the tokenizer/model
ONCE via PlainCircuit.load_model() and classifies every sentence in the same
process, timing only the classify() call itself -- this is optimization #1
from PLAINTEXT_OPTIMIZATION_REPORT.md.

Usage (run from the repository root):
    python3 src/python/run_plaintext_benchmark_v2.py [--limit N] [--out-json FILE]
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "python"))
os.chdir(REPO_ROOT)  # PlainCircuit.py loads weights via paths relative to the repo root

import PlainCircuit  # noqa: E402
from run_sentiment_comparison import SENTENCES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=len(SENTENCES), help="only run the first N sentences")
    parser.add_argument("--out-json", default="plaintext_optimized_results.json")
    args = parser.parse_args()

    sentences = SENTENCES[: args.limit]

    load_start = time.perf_counter()
    tokenizer, model, device = PlainCircuit.load_model()
    load_elapsed = time.perf_counter() - load_start
    print(f"Model+tokenizer loaded once in {load_elapsed:.3f}s (device={device})", flush=True)

    results = []
    total_start = time.perf_counter()
    for i, (label, sentence) in enumerate(sentences, 1):
        t0 = time.perf_counter()
        r = PlainCircuit.classify(tokenizer, model, device, sentence)
        elapsed = time.perf_counter() - t0

        logits = [float(r["precomputed_logits"][0]), float(r["precomputed_logits"][1])]
        pytorch_logits = [float(r["pytorch_logits"][0]), float(r["pytorch_logits"][1])]
        predicted = PlainCircuit.sentiment(logits)
        pytorch_predicted = PlainCircuit.sentiment(pytorch_logits)

        print(
            f"[{i}/{len(sentences)}] ({label}) {sentence!r} -> {predicted} ({elapsed:.4f}s)",
            flush=True,
        )

        results.append(
            {
                "sentence": sentence,
                "expected_sentiment": label,
                "elapsed_sec": round(elapsed, 4),
                "logits": logits,
                "predicted_sentiment": predicted,
                "pytorch_predicted_sentiment": pytorch_predicted,
            }
        )
    total_classify_elapsed = time.perf_counter() - total_start

    out = {
        "model_load_sec": round(load_elapsed, 3),
        "n_sentences": len(results),
        "total_classify_sec": round(total_classify_elapsed, 3),
        "total_wall_sec_including_load": round(load_elapsed + total_classify_elapsed, 3),
        "mean_classify_sec": round(total_classify_elapsed / len(results), 4) if results else None,
        "results": results,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"\nDone. model_load={load_elapsed:.2f}s total_classify={total_classify_elapsed:.2f}s "
        f"({len(results)} sentences, mean={out['mean_classify_sec']:.4f}s/sentence)"
    )
    print(f"Raw results: {args.out_json}")


if __name__ == "__main__":
    main()
