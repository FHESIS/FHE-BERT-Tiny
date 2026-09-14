#!/usr/bin/env python3
"""
Full SST-2 validation-set run: evaluates BOTH the optimized in-process
plaintext circuit (src/python/PlainCircuit.py) AND the encrypted FHE circuit
(./build/FHE-BERT-Tiny) over all 872 sentences of the real SST-2 validation
split (notebooks/SST-2-val.parquet), scoring both against the actual SST-2
gold labels (not a hand-assigned sanity-check label).

This mirrors run_ciphertext_vs_optimized_plaintext.py's approach (plaintext
model loaded once in-process; encrypted circuit run as a fresh
`./build/FHE-BERT-Tiny <sentence> --verbose` subprocess per sentence) but
over the full validation set instead of a 50/100-sentence hand-picked
sample, and reports real classification accuracy/precision/recall/F1
against SST-2 gold labels.

The encrypted side reloads ~9GB of GPU bootstrap/rotation-key material on
every subprocess invocation (~35-40s of fixed overhead) on top of the
~5-6s actual FHE circuit evaluation, so a full 872-sentence run takes
roughly 10-11 hours. Progress (raw JSON + Markdown report) is persisted
after every sentence, so an interrupted run can always be resumed with
--resume.

Usage (run from the repository root):
    python3 src/python/run_full_sst2_validation.py                  # full 872, both circuits
    python3 src/python/run_full_sst2_validation.py --limit 10        # smoke test
    python3 src/python/run_full_sst2_validation.py --resume          # continue a previous run
    python3 src/python/run_full_sst2_validation.py --skip-encrypted  # plaintext only (fast)
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "python"))

import PlainCircuit  # noqa: E402

SST2_VAL_PARQUET = os.path.join(REPO_ROOT, "notebooks", "SST-2-val.parquet")

ENC_LOGITS_RE = re.compile(r"Output logits.*?\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]")
ENC_TIMING_RE = re.compile(r"The evaluation of the FHE circuit took:\s*([\d.]+)\s*seconds")
ENC_RETRY_RE = re.compile(r"retrying\.\.\.")

# SST-2 (GLUE) label convention: 0 = negative, 1 = positive.
GOLD_LABEL_MAP = {0: "negative", 1: "positive"}


def load_sst2_val():
    """Load the 872-sentence SST-2 validation split (sentence, gold label)."""
    import pyarrow.parquet as pq

    table = pq.read_table(SST2_VAL_PARQUET)
    rows = table.to_pylist()
    sentences = [(GOLD_LABEL_MAP[row["label"]], row["sentence"].strip()) for row in rows]
    return sentences


def sentiment_from_logits(logits):
    if logits is None:
        return None
    return "negative" if logits[0] > logits[1] else "positive"


def run_encrypted(sentence, timeout):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["./build/FHE-BERT-Tiny", sentence, "--verbose"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        timed_out = False
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = None
        stdout = e.stdout or ""
        stderr = e.stderr or ""
    elapsed = time.perf_counter() - start

    logits = None
    m = ENC_LOGITS_RE.search(stdout)
    if m:
        logits = [float(m.group(1)), float(m.group(2))]

    reported_timing = None
    m2 = ENC_TIMING_RE.search(stdout)
    if m2:
        reported_timing = float(m2.group(1))

    retries = len(ENC_RETRY_RE.findall(stdout)) + len(ENC_RETRY_RE.findall(stderr))

    return {
        "elapsed_sec": round(elapsed, 3),
        "reported_circuit_time_sec": reported_timing,
        "logits": logits,
        "predicted_sentiment": sentiment_from_logits(logits),
        "retries": retries,
        "timed_out": timed_out,
        "returncode": returncode,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


def load_existing_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_logits(logits):
    if logits is None:
        return "n/a"
    return f"[{logits[0]:.4f}, {logits[1]:.4f}]"


def truncate(s, n=60):
    return s if len(s) <= n else s[: n - 1] + "…"


def classification_metrics(preds_labels):
    """preds_labels: list of (predicted, gold) pairs, both 'positive'/'negative', gold non-None."""
    n = len(preds_labels)
    if n == 0:
        return None
    correct = sum(1 for p, g in preds_labels if p == g)

    tp = sum(1 for p, g in preds_labels if p == "positive" and g == "positive")
    fp = sum(1 for p, g in preds_labels if p == "positive" and g == "negative")
    fn = sum(1 for p, g in preds_labels if p == "negative" and g == "positive")
    tn = sum(1 for p, g in preds_labels if p == "negative" and g == "negative")

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    return {
        "n": n,
        "accuracy": correct / n,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision_positive": precision,
        "recall_positive": recall,
        "f1_positive": f1,
    }


def compute_aggregates(results, skip_encrypted):
    n = len(results)
    plain_times = [r["plaintext"]["elapsed_sec"] for r in results if r["plaintext"].get("logits")]
    plain_pairs = [
        (r["plaintext"]["predicted_sentiment"], r["gold_sentiment"])
        for r in results
        if r["plaintext"].get("predicted_sentiment") is not None
    ]
    plain_ok = sum(1 for r in results if r["plaintext"].get("logits"))

    agg = {
        "n_sentences": n,
        "plaintext": {
            "n_ok": plain_ok,
            "metrics": classification_metrics(plain_pairs),
            "mean_time_sec": statistics.mean(plain_times) if plain_times else None,
            "median_time_sec": statistics.median(plain_times) if plain_times else None,
            "min_time_sec": min(plain_times) if plain_times else None,
            "max_time_sec": max(plain_times) if plain_times else None,
        },
    }

    if not skip_encrypted:
        enc_results = [r["encrypted"] for r in results if "encrypted" in r]
        enc_times = [r["elapsed_sec"] for r in enc_results if r.get("logits")]
        enc_ok = sum(1 for r in enc_results if r.get("logits"))
        enc_failed = sum(1 for r in enc_results if not r.get("logits"))
        total_retries = sum(r.get("retries", 0) for r in enc_results)
        enc_pairs = [
            (r["encrypted"]["predicted_sentiment"], r["gold_sentiment"])
            for r in results
            if r.get("encrypted", {}).get("predicted_sentiment") is not None
        ]

        agree = 0
        diffs = []
        for r in results:
            enc = r.get("encrypted", {})
            p_pred, e_pred = r["plaintext"].get("predicted_sentiment"), enc.get("predicted_sentiment")
            if p_pred is not None and e_pred is not None and p_pred == e_pred:
                agree += 1
            p_log, e_log = r["plaintext"].get("logits"), enc.get("logits")
            if p_log and e_log:
                diffs.append((abs(p_log[0] - e_log[0]) + abs(p_log[1] - e_log[1])) / 2)

        both_ok = sum(
            1 for r in results
            if r["plaintext"].get("logits") and r.get("encrypted", {}).get("logits")
        )

        agg["encrypted"] = {
            "n_ok": enc_ok,
            "n_failed_or_timed_out": enc_failed,
            "total_retries": total_retries,
            "metrics": classification_metrics(enc_pairs),
            "mean_time_sec": statistics.mean(enc_times) if enc_times else None,
            "median_time_sec": statistics.median(enc_times) if enc_times else None,
            "min_time_sec": min(enc_times) if enc_times else None,
            "max_time_sec": max(enc_times) if enc_times else None,
        }
        agg["comparison"] = {
            "both_succeeded": both_ok,
            "prediction_agreement": agree / both_ok if both_ok else None,
            "mean_abs_logit_diff": statistics.mean(diffs) if diffs else None,
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "encrypted_over_plaintext_slowdown": (
                agg["encrypted"]["mean_time_sec"] / agg["plaintext"]["mean_time_sec"]
                if agg["encrypted"]["mean_time_sec"] and agg["plaintext"]["mean_time_sec"]
                else None
            ),
        }

    return agg


def metrics_row(m):
    if m is None:
        return ["n/a"] * 6
    cm = m["confusion_matrix"]
    return [
        fmt(m["accuracy"], 4),
        fmt(m["precision_positive"], 4),
        fmt(m["recall_positive"], 4),
        fmt(m["f1_positive"], 4),
        f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}",
    ]


def write_report(report_path, results, skip_encrypted, total_target):
    agg = compute_aggregates(results, skip_encrypted)
    n = len(results)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    p, e = agg["plaintext"], agg.get("encrypted")

    lines = []
    lines.append("# Full SST-2 Validation Set — Plaintext vs Encrypted (FHE) Accuracy Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Progress: {n}/{total_target} sentences completed")
    lines.append("")
    lines.append(
        "Dataset: the real SST-2 (GLUE) validation split, 872 sentences, `notebooks/SST-2-val.parquet` "
        "(`sentence`, gold `label`: 0=negative, 1=positive). Plaintext circuit: the optimized, "
        "in-process `src/python/PlainCircuit.py` (`load_model()` once, `classify()` timed per "
        "sentence). Encrypted circuit: `./build/FHE-BERT-Tiny <sentence> --verbose`, one fresh "
        "subprocess per sentence (CKKS evaluation over the trained BERT-tiny/SST-2 weights, GPU-"
        "accelerated via FIDESlib). Both implement the same precomputed-LayerNorm approximation "
        "of BERT-tiny/SST-2. Unlike earlier reports in this repo, accuracy here is measured "
        "against the **actual SST-2 gold labels**, not a hand-assigned sanity-check label."
    )
    lines.append("")
    lines.append("## Classification metrics vs SST-2 gold labels")
    lines.append("")
    lines.append("| Circuit | N scored | Accuracy | Precision (pos) | Recall (pos) | F1 (pos) | Confusion matrix |")
    lines.append("|---|---|---|---|---|---|---|")
    p_row = metrics_row(p["metrics"])
    lines.append(
        f"| Optimized plaintext | {p['metrics']['n'] if p['metrics'] else 0} | " + " | ".join(p_row) + " |"
    )
    if e:
        e_row = metrics_row(e["metrics"])
        lines.append(
            f"| Encrypted (FHE) | {e['metrics']['n'] if e['metrics'] else 0} | " + " | ".join(e_row) + " |"
        )
    lines.append("")
    lines.append("## Runtime & agreement")
    lines.append("")
    lines.append("| Metric | Optimized plaintext | Encrypted (FHE) |")
    lines.append("|---|---|---|")
    lines.append(f"| Successful runs | {p['n_ok']}/{n} | {(e['n_ok'] if e else 'n/a')}/{n} |")
    if e:
        lines.append(f"| Failed / timed out | 0 | {e['n_failed_or_timed_out']} |")
        lines.append(f"| Total internal retries | n/a | {e['total_retries']} |")
    lines.append(f"| Mean time (s) | {fmt(p['mean_time_sec'], 4)} | {fmt(e['mean_time_sec']) if e else 'n/a'} |")
    lines.append(f"| Median time (s) | {fmt(p['median_time_sec'], 4)} | {fmt(e['median_time_sec']) if e else 'n/a'} |")
    lines.append(
        f"| Min / Max time (s) | {fmt(p['min_time_sec'], 4)} / {fmt(p['max_time_sec'], 4)} | "
        + (f"{fmt(e['min_time_sec'])} / {fmt(e['max_time_sec'])}" if e else "n/a")
        + " |"
    )

    if e:
        c = agg["comparison"]
        lines.append("")
        lines.append(
            f"**Prediction agreement (optimized plaintext vs encrypted):** "
            f"{c['both_succeeded']} pairs compared, {fmt(c['prediction_agreement'], 4)} agreement rate"
        )
        lines.append(f"**Mean absolute logit difference (avg of both logits):** {fmt(c['mean_abs_logit_diff'], 4)}")
        lines.append(f"**Max absolute logit difference:** {fmt(c['max_abs_logit_diff'], 4)}")
        lines.append(
            f"**Encrypted / optimized-plaintext time ratio (slowdown):** "
            f"{fmt(c['encrypted_over_plaintext_slowdown'], 1)}x"
        )

    lines.append("")
    lines.append("## Per-sentence results")
    lines.append("")
    if skip_encrypted:
        lines.append("| # | Gold | Sentence | Plain pred | Plain logits | Plain time (s) |")
        lines.append("|---|---|---|---|---|---|")
    else:
        lines.append(
            "| # | Gold | Sentence | Plain pred | Plain logits | Plain time (s) | "
            "Enc pred | Enc logits | Enc time (s) | Retries |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for i, r in enumerate(results, 1):
        pr = r["plaintext"]
        row = [
            str(i),
            r["gold_sentiment"],
            truncate(r["sentence"]),
            pr.get("predicted_sentiment") or "FAIL",
            fmt_logits(pr.get("logits")),
            fmt(pr["elapsed_sec"], 4),
        ]
        if not skip_encrypted:
            enc = r.get("encrypted", {})
            row += [
                enc.get("predicted_sentiment") or "FAIL",
                fmt_logits(enc.get("logits")),
                fmt(enc.get("elapsed_sec"), 2),
                str(enc.get("retries", 0)),
            ]
        lines.append("| " + " | ".join(row) + " |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-json", default="sst2_full_validation_results.json")
    parser.add_argument("--out-report", default="SST2_FULL_VALIDATION_REPORT.md")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N sentences (default: all 872)")
    parser.add_argument("--report-every", type=int, default=10, help="(re)write the report every N sentences")
    parser.add_argument("--skip-encrypted", action="store_true", help="only run plaintext inference")
    parser.add_argument("--plain-timeout", type=float, default=120.0)
    parser.add_argument("--encrypted-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true", help="continue from an existing --out-json file")
    args = parser.parse_args()

    all_sentences = load_sst2_val()
    sentences = all_sentences[: args.limit] if args.limit else all_sentences
    print(f"Loaded {len(all_sentences)} SST-2 validation sentences, running {len(sentences)}", flush=True)

    results = load_existing_results(args.out_json) if args.resume else []
    if results:
        print(f"Resuming: {len(results)} sentences already completed, continuing from #{len(results) + 1}", flush=True)

    load_start = time.perf_counter()
    tokenizer, model, device = PlainCircuit.load_model()
    load_elapsed = time.perf_counter() - load_start
    print(f"Plaintext model+tokenizer loaded once in {load_elapsed:.3f}s (device={device})", flush=True)

    for i in range(len(results), len(sentences)):
        gold, sentence = sentences[i]
        print(f"[{i + 1}/{len(sentences)}] (gold={gold}) {sentence!r}", flush=True)

        t0 = time.perf_counter()
        r = PlainCircuit.classify(tokenizer, model, device, sentence)
        p_elapsed = time.perf_counter() - t0
        p_logits = [float(r["precomputed_logits"][0]), float(r["precomputed_logits"][1])]
        p_pred = sentiment_from_logits(p_logits)
        print(f"    plaintext: logits={p_logits} time={p_elapsed:.4f}s predicted={p_pred}", flush=True)

        entry = {
            "sentence": sentence,
            "gold_sentiment": gold,
            "plaintext": {
                "elapsed_sec": round(p_elapsed, 4),
                "logits": p_logits,
                "predicted_sentiment": p_pred,
            },
        }

        if not args.skip_encrypted:
            print("  running encrypted (this can take ~40-60s)...", flush=True)
            enc = run_encrypted(sentence, args.encrypted_timeout)
            print(
                f"    encrypted: logits={enc['logits']} time={enc['elapsed_sec']}s "
                f"retries={enc['retries']} predicted={enc['predicted_sentiment']} returncode={enc['returncode']}",
                flush=True,
            )
            entry["encrypted"] = enc

        results.append(entry)

        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

        if len(results) % args.report_every == 0 or len(results) == len(sentences):
            write_report(args.out_report, results, args.skip_encrypted, len(sentences))
            print(f"  -> wrote report to {args.out_report} ({len(results)} sentences so far)", flush=True)

    print(f"\nDone. {len(results)} sentences processed.")
    print(f"Raw results: {args.out_json}")
    print(f"Report: {args.out_report}")


if __name__ == "__main__":
    main()
