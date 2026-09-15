#!/usr/bin/env python3
"""
Benchmarks plaintext vs. encrypted (FHE) BERT-tiny sentiment inference

Usage (run from the repository root):
    python3 src/python/run_benchmark.py [--repeat N] [--out results.json]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENTENCES = [
    ("negative", "This was a complete waste of my time and money"),
    ("negative", "The plot was boring and the acting felt wooden"),
    ("positive", "This is one of the best performances I have seen all year"),
    ("positive", "A brilliant, funny, and touching film that exceeded my expectations"),
]

PLAIN_LOGITS_RE = re.compile(
    r"Output logits Plain-Precomputed:\s*\[\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\]"
)
ENC_LOGITS_RE = re.compile(
    r"Output logits.*?\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]"
)
ENC_TIMING_RE = re.compile(
    r"The evaluation of the FHE circuit took:\s*([\d.]+)\s*seconds"
)
ENC_RETRY_RE = re.compile(r"retrying\.\.\.")


def run_plaintext(sentence):
    start = time.perf_counter()
    proc = subprocess.run(
        ["python3", "src/python/PlainCircuit.py", sentence],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    logits = None
    m = PLAIN_LOGITS_RE.search(proc.stdout)
    if m:
        logits = [float(m.group(1)), float(m.group(2))]

    return {
        "elapsed_sec": round(elapsed, 3),
        "logits": logits,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run_encrypted(sentence):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")

    start = time.perf_counter()
    proc = subprocess.run(
        ["./build/FHE-BERT-Tiny", sentence, "--verbose"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - start

    logits = None
    m = ENC_LOGITS_RE.search(proc.stdout)
    if m:
        logits = [float(m.group(1)), float(m.group(2))]

    reported_timing = None
    m2 = ENC_TIMING_RE.search(proc.stdout)
    if m2:
        reported_timing = float(m2.group(1))

    retries = len(ENC_RETRY_RE.findall(proc.stdout)) + len(ENC_RETRY_RE.findall(proc.stderr))

    return {
        "elapsed_sec": round(elapsed, 3),
        "reported_circuit_time_sec": reported_timing,
        "logits": logits,
        "retries": retries,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sentiment_from_logits(logits):
    if logits is None:
        return None
    return "negative" if logits[0] > logits[1] else "positive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark_results.json")
    parser.add_argument(
        "--skip-encrypted", action="store_true", help="only run plaintext inference"
    )
    args = parser.parse_args()

    results = []
    for i, (label, sentence) in enumerate(SENTENCES, 1):
        print(f"[{i}/{len(SENTENCES)}] ({label}) {sentence!r}", flush=True)

        print("  running plaintext...", flush=True)
        plain = run_plaintext(sentence)
        print(
            f"    logits={plain['logits']} time={plain['elapsed_sec']}s "
            f"predicted={sentiment_from_logits(plain['logits'])}",
            flush=True,
        )

        entry = {
            "sentence": sentence,
            "expected_sentiment": label,
            "plaintext": plain,
        }

        if not args.skip_encrypted:
            print("  running encrypted (this can take 30-90s)...", flush=True)
            enc = run_encrypted(sentence)
            print(
                f"    logits={enc['logits']} time={enc['elapsed_sec']}s "
                f"retries={enc['retries']} predicted={sentiment_from_logits(enc['logits'])} "
                f"returncode={enc['returncode']}",
                flush=True,
            )
            entry["encrypted"] = enc

        results.append(entry)

        # Persist incrementally so partial progress survives a crash/timeout.
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
