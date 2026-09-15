#!/usr/bin/env python3
"""
Compares the encrypted (FHE, ./build/FHE-BERT-Tiny) circuit against the
OPTIMIZED, in-process plaintext circuit (src/python/PlainCircuit.py, loaded
once via PlainCircuit.load_model()/classify() -- see
PLAINTEXT_OPTIMIZATION_REPORT.md / PLAINTEXT_OPTIMIZATION_COMPARISON_REPORT.md)
over the same N sentences, recording wall-clock time and logits for both.

Unlike run_sentiment_comparison.py (which subprocess-execs a fresh
`python3 PlainCircuit.py <sentence>` per sentence, paying ~8-10s of
import+model-load overhead every call), the plaintext side here loads the
tokenizer/model ONCE and only times the classify() call itself, so the
plaintext numbers reflect steady-state inference cost rather than process
startup. The encrypted side is unchanged: one `./build/FHE-BERT-Tiny
<sentence> --verbose` subprocess per sentence (the FHE circuit has no
in-process/batched API).

Usage (run from the repository root):
    python3 src/python/run_ciphertext_vs_optimized_plaintext.py [--limit 50]
"""

import argparse
import json
import os
import re
import statistics
import sys
import subprocess
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "python"))

import PlainCircuit  # noqa: E402
from run_sentiment_comparison import SENTENCES  # noqa: E402

ENC_LOGITS_RE = re.compile(r"Output logits.*?\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]")
ENC_TIMING_RE = re.compile(r"The evaluation of the FHE circuit took:\s*([\d.]+)\s*seconds")
ENC_RETRY_RE = re.compile(r"retrying\.\.\.")


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


def truncate(s, n=55):
    return s if len(s) <= n else s[: n - 1] + "…"


def compute_aggregates(results):
    n = len(results)
    plain_times = [r["plaintext"]["elapsed_sec"] for r in results if r["plaintext"].get("logits")]
    plain_correct = sum(
        1 for r in results if r["plaintext"].get("predicted_sentiment") == r["expected_sentiment"]
    )
    plain_ok = sum(1 for r in results if r["plaintext"].get("logits"))

    enc_results = [r["encrypted"] for r in results]
    enc_times = [r["elapsed_sec"] for r in enc_results if r.get("logits")]
    enc_ok = sum(1 for r in enc_results if r.get("logits"))
    enc_correct = sum(
        1 for r in results if r["encrypted"].get("predicted_sentiment") == r["expected_sentiment"]
    )
    enc_failed = sum(1 for r in enc_results if not r.get("logits"))
    total_retries = sum(r.get("retries", 0) for r in enc_results)

    agree = 0
    diffs = []
    diffs0 = []
    diffs1 = []
    for r in results:
        enc = r["encrypted"]
        p_pred, e_pred = r["plaintext"].get("predicted_sentiment"), enc.get("predicted_sentiment")
        if p_pred is not None and e_pred is not None and p_pred == e_pred:
            agree += 1
        p_log, e_log = r["plaintext"].get("logits"), enc.get("logits")
        if p_log and e_log:
            diffs0.append(abs(p_log[0] - e_log[0]))
            diffs1.append(abs(p_log[1] - e_log[1]))
            diffs.append((abs(p_log[0] - e_log[0]) + abs(p_log[1] - e_log[1])) / 2)

    both_ok = sum(1 for r in results if r["plaintext"].get("logits") and r["encrypted"].get("logits"))

    agg = {
        "n_sentences": n,
        "plaintext": {
            "n_ok": plain_ok,
            "accuracy_vs_expected": plain_correct / n if n else None,
            "mean_time_sec": statistics.mean(plain_times) if plain_times else None,
            "median_time_sec": statistics.median(plain_times) if plain_times else None,
            "min_time_sec": min(plain_times) if plain_times else None,
            "max_time_sec": max(plain_times) if plain_times else None,
        },
        "encrypted": {
            "n_ok": enc_ok,
            "n_failed_or_timed_out": enc_failed,
            "total_retries": total_retries,
            "accuracy_vs_expected": enc_correct / n if n else None,
            "mean_time_sec": statistics.mean(enc_times) if enc_times else None,
            "median_time_sec": statistics.median(enc_times) if enc_times else None,
            "min_time_sec": min(enc_times) if enc_times else None,
            "max_time_sec": max(enc_times) if enc_times else None,
        },
        "comparison": {
            "both_succeeded": both_ok,
            "prediction_agreement": agree / both_ok if both_ok else None,
            "mean_abs_logit_diff": statistics.mean(diffs) if diffs else None,
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "mean_abs_logit0_diff": statistics.mean(diffs0) if diffs0 else None,
            "mean_abs_logit1_diff": statistics.mean(diffs1) if diffs1 else None,
        },
    }
    e_mean = agg["encrypted"]["mean_time_sec"]
    p_mean = agg["plaintext"]["mean_time_sec"]
    agg["comparison"]["encrypted_over_plaintext_slowdown"] = (
        e_mean / p_mean if e_mean and p_mean else None
    )
    return agg


def write_report(report_path, results, total_target):
    agg = compute_aggregates(results)
    n = len(results)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    p, e, c = agg["plaintext"], agg["encrypted"], agg["comparison"]

    lines = []
    lines.append("# Ciphertext (FHE) vs Optimized Plaintext Circuit — Inference Comparison")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Progress: {n}/{total_target} sentences completed")
    lines.append("")
    lines.append(
        "Encrypted circuit: `./build/FHE-BERT-Tiny <sentence> --verbose` (one fresh subprocess "
        "per sentence, CKKS evaluation over the trained BERT-tiny/SST-2 weights). Plaintext "
        "circuit: the **optimized** `src/python/PlainCircuit.py` "
        "(`PlainCircuit.load_model()` called once, `classify()` timed per sentence in the same "
        "process — see `PLAINTEXT_OPTIMIZATION_REPORT.md`). Both implement the same "
        "precomputed-LayerNorm approximation of BERT-tiny/SST-2; `expected_sentiment` is a "
        "hand-assigned sanity-check label, not an SST-2 gold label."
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Optimized plaintext | Encrypted (FHE) |")
    lines.append("|---|---|---|")
    lines.append(f"| Successful runs | {p['n_ok']}/{n} | {e['n_ok']}/{n} |")
    lines.append(f"| Failed / timed out | 0 | {e['n_failed_or_timed_out']} |")
    lines.append(f"| Total internal retries | n/a | {e['total_retries']} |")
    lines.append(
        f"| Accuracy vs expected label | {fmt(p['accuracy_vs_expected'], 3)} | {fmt(e['accuracy_vs_expected'], 3)} |"
    )
    lines.append(f"| Mean time (s) | {fmt(p['mean_time_sec'], 4)} | {fmt(e['mean_time_sec'])} |")
    lines.append(f"| Median time (s) | {fmt(p['median_time_sec'], 4)} | {fmt(e['median_time_sec'])} |")
    lines.append(
        f"| Min / Max time (s) | {fmt(p['min_time_sec'], 4)} / {fmt(p['max_time_sec'], 4)} | "
        f"{fmt(e['min_time_sec'])} / {fmt(e['max_time_sec'])} |"
    )
    lines.append("")
    lines.append(
        f"**Prediction agreement (optimized plaintext vs encrypted):** "
        f"{c['both_succeeded']} pairs compared, {fmt(c['prediction_agreement'], 3)} agreement rate"
    )
    lines.append(f"**Mean absolute logit difference (avg of both logits):** {fmt(c['mean_abs_logit_diff'], 4)}")
    lines.append(
        f"**Mean absolute diff, logit₀ / logit₁:** {fmt(c['mean_abs_logit0_diff'], 4)} / "
        f"{fmt(c['mean_abs_logit1_diff'], 4)}"
    )
    lines.append(f"**Max absolute logit difference:** {fmt(c['max_abs_logit_diff'], 4)}")
    lines.append(
        f"**Encrypted / optimized-plaintext time ratio (slowdown):** "
        f"{fmt(c['encrypted_over_plaintext_slowdown'], 1)}x"
    )

    lines.append("")
    lines.append("## Per-sentence results")
    lines.append("")
    lines.append(
        "| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | "
        "Enc pred | Enc logits | Enc time (s) | Retries | Agree | |Δlogit| avg |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for i, r in enumerate(results, 1):
        p_r = r["plaintext"]
        e_r = r["encrypted"]
        p_pred, e_pred = p_r.get("predicted_sentiment"), e_r.get("predicted_sentiment")
        agree = "-" if (p_pred is None or e_pred is None) else ("✓" if p_pred == e_pred else "✗")
        p_log, e_log = p_r.get("logits"), e_r.get("logits")
        diff = "n/a"
        if p_log and e_log:
            diff = f"{(abs(p_log[0]-e_log[0]) + abs(p_log[1]-e_log[1])) / 2:.4f}"
        row = [
            str(i),
            r["expected_sentiment"],
            truncate(r["sentence"]),
            p_pred or "FAIL",
            fmt_logits(p_log),
            fmt(p_r["elapsed_sec"], 4),
            e_pred or "FAIL",
            fmt_logits(e_log),
            fmt(e_r.get("elapsed_sec"), 2),
            str(e_r.get("retries", 0)),
            agree,
            diff,
        ]
        lines.append("| " + " | ".join(row) + " |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-json", default="ciphertext_vs_optimized_plaintext_results.json")
    parser.add_argument("--out-report", default="CIPHERTEXT_VS_OPTIMIZED_PLAINTEXT_REPORT.md")
    parser.add_argument("--limit", type=int, default=50, help="only run the first N sentences")
    parser.add_argument("--report-every", type=int, default=5, help="(re)write the report every N sentences")
    parser.add_argument("--encrypted-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true", help="continue from an existing --out-json file")
    args = parser.parse_args()

    sentences = SENTENCES[: args.limit]

    results = load_existing_results(args.out_json) if args.resume else []
    if results:
        print(f"Resuming: {len(results)} sentences already completed, continuing from #{len(results) + 1}", flush=True)

    load_start = time.perf_counter()
    tokenizer, model, device = PlainCircuit.load_model()
    load_elapsed = time.perf_counter() - load_start
    print(f"Plaintext model+tokenizer loaded once in {load_elapsed:.3f}s (device={device})", flush=True)

    for i in range(len(results), len(sentences)):
        label, sentence = sentences[i]
        print(f"[{i + 1}/{len(sentences)}] ({label}) {sentence!r}", flush=True)

        t0 = time.perf_counter()
        r = PlainCircuit.classify(tokenizer, model, device, sentence)
        p_elapsed = time.perf_counter() - t0
        p_logits = [float(r["precomputed_logits"][0]), float(r["precomputed_logits"][1])]
        p_pred = sentiment_from_logits(p_logits)
        print(f"    plaintext: logits={p_logits} time={p_elapsed:.4f}s predicted={p_pred}", flush=True)

        print("  running encrypted (this can take 40-60s)...", flush=True)
        enc = run_encrypted(sentence, args.encrypted_timeout)
        print(
            f"    encrypted: logits={enc['logits']} time={enc['elapsed_sec']}s "
            f"retries={enc['retries']} predicted={enc['predicted_sentiment']} returncode={enc['returncode']}",
            flush=True,
        )

        entry = {
            "sentence": sentence,
            "expected_sentiment": label,
            "plaintext": {
                "elapsed_sec": round(p_elapsed, 4),
                "logits": p_logits,
                "predicted_sentiment": p_pred,
            },
            "encrypted": enc,
        }
        results.append(entry)

        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

        if len(results) % args.report_every == 0 or len(results) == len(sentences):
            write_report(args.out_report, results, len(sentences))
            print(f"  -> wrote report to {args.out_report} ({len(results)} sentences so far)", flush=True)

    print(f"\nDone. {len(results)} sentences processed.")
    print(f"Raw results: {args.out_json}")
    print(f"Report: {args.out_report}")


if __name__ == "__main__":
    main()
