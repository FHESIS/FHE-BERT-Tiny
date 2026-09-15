#!/usr/bin/env python3
"""
Compares plaintext vs. encrypted (FHE) BERT-tiny sentiment inference over a
set of sentences with varied sentiment (strength, domain, length, negation).

For every sentence it records, for both the plaintext and the encrypted
circuit: wall-clock running time, output logits, and predicted sentiment.
A Markdown comparison report (plus the raw JSON results) is (re)written
every `--report-every` sentences (default 10), so progress is never lost
even if a run is interrupted or a sentence crashes the FHE circuit.

Usage (run from the repository root):
    python3 src/python/run_sentiment_comparison.py
    python3 src/python/run_sentiment_comparison.py --limit 10          # smoke test
    python3 src/python/run_sentiment_comparison.py --skip-encrypted    # plaintext only
    python3 src/python/run_sentiment_comparison.py --resume            # continue a previous run
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

# 100 sentences with varied sentiment: strong/mild, positive/negative, short/long,
# straightforward and negation-based, across movies/food/products/service/general.
SENTENCES = [
    ("positive", "This is one of the best performances I have seen all year"),
    ("positive", "A brilliant, funny, and touching film that exceeded my expectations"),
    ("positive", "I absolutely loved this film, it was a masterpiece"),
    ("positive", "What a delightful and heartwarming movie experience"),
    ("positive", "The acting was superb and the story kept me engaged the whole time"),
    ("positive", "A truly inspiring story told with grace and humor"),
    ("positive", "The cinematography was stunning from start to finish"),
    ("positive", "This film restored my faith in modern storytelling"),
    ("positive", "An absolute triumph of direction and performance"),
    ("positive", "The soundtrack alone is worth the price of admission"),
    ("positive", "A charming little film that grows on you"),
    ("positive", "I laughed out loud more than once, a genuinely funny movie"),
    ("positive", "The chemistry between the leads made the whole film work"),
    ("positive", "A beautifully written script with real emotional depth"),
    ("positive", "This exceeded every expectation I had going in"),
    ("positive", "The special effects were seamless and genuinely thrilling"),
    ("positive", "A masterclass in pacing and tension"),
    ("positive", "I would happily watch this again tomorrow"),
    ("positive", "The ending brought tears to my eyes, in the best way"),
    ("positive", "Every scene felt purposeful and beautifully shot"),
    ("positive", "A wonderfully clever plot with a satisfying payoff"),
    ("positive", "The dialogue crackles with wit and energy"),
    ("positive", "This is family entertainment done exactly right"),
    ("positive", "A gripping thriller that never lets up"),
    ("positive", "The lead actress gives a career-defining performance"),
    ("positive", "Genuinely one of the most enjoyable films of the decade"),
    ("positive", "The food at this restaurant was absolutely delicious"),
    ("positive", "Customer service was fast, friendly, and helpful"),
    ("positive", "This phone works flawlessly and the battery lasts all day"),
    ("positive", "The hotel room was spotless and the staff were wonderful"),
    ("positive", "Great value for the price, highly recommended"),
    ("positive", "The book kept me hooked until the very last page"),
    ("positive", "A refreshing and uplifting comedy the whole family enjoyed"),
    ("positive", "The concert was electric, easily the best show I have seen"),
    ("positive", "This laptop is fast, quiet, and beautifully designed"),
    ("positive", "The weather was perfect for our trip to the coast"),
    ("positive", "Our vacation exceeded every expectation we had"),
    ("positive", "The coffee here is consistently excellent"),
    ("positive", "I am thoroughly impressed with the quality of this product"),
    ("positive", "The team played with incredible energy and won deservedly"),
    ("positive", "This app makes managing my schedule so much easier"),
    ("positive", "The garden looked absolutely gorgeous in the spring sunshine"),
    ("positive", "Their new album is a joyful, inventive piece of work"),
    ("positive", "This is not a bad film at all, it is actually quite good"),
    ("positive", "I never expected to enjoy this movie as much as I did"),
    ("positive", "Despite a slow start, the film wins you over completely"),
    ("positive", "It is hard not to smile the entire way through this movie"),
    ("positive", "A small film with a surprisingly big heart"),
    ("positive", "The sequel actually improves on the original in every way"),
    ("positive", "Well worth the wait, this delivers on every level"),
    ("negative", "This was a complete waste of my time and money"),
    ("negative", "The plot was boring and the acting felt wooden"),
    ("negative", "I hated every minute of this dull and pointless movie"),
    ("negative", "A disappointing mess with no redeeming qualities"),
    ("negative", "The worst film I have seen in years, truly terrible"),
    ("negative", "This movie is a tedious, incoherent slog"),
    ("negative", "The dialogue was cringeworthy and the pacing glacial"),
    ("negative", "A lazy, uninspired sequel that adds nothing new"),
    ("negative", "I walked out halfway through, it was that bad"),
    ("negative", "The special effects looked cheap and unconvincing"),
    ("negative", "A forgettable film with a plot full of holes"),
    ("negative", "The lead performance was flat and utterly unconvincing"),
    ("negative", "This film insults the intelligence of its audience"),
    ("negative", "A bloated, self-indulgent film that never earns its runtime"),
    ("negative", "The jokes fell flat and the story made no sense"),
    ("negative", "An empty, soulless cash grab of a sequel"),
    ("negative", "This is easily the worst movie of the year"),
    ("negative", "The editing was choppy and the story dragged endlessly"),
    ("negative", "A painfully dull film with nothing to say"),
    ("negative", "The characters were flat and impossible to care about"),
    ("negative", "A confused mess that never finds its footing"),
    ("negative", "The score was intrusive and the script painfully cliched"),
    ("negative", "I regret spending an evening on this dreadful movie"),
    ("negative", "A shallow, forgettable film with no depth whatsoever"),
    ("negative", "The pacing kills any tension the story might have had"),
    ("negative", "This sequel betrays everything that made the original good"),
    ("negative", "The food arrived cold and tasted terrible"),
    ("negative", "Customer service was rude and unhelpful throughout"),
    ("negative", "This phone constantly freezes and the battery is awful"),
    ("negative", "The hotel room was filthy and the staff were dismissive"),
    ("negative", "Overpriced and poorly made, do not waste your money"),
    ("negative", "The book was tedious and I never finished it"),
    ("negative", "A joyless comedy that never once made me laugh"),
    ("negative", "The concert was a disorganized, disappointing mess"),
    ("negative", "This laptop overheats constantly and the fan is deafening"),
    ("negative", "The weather ruined what could have been a nice trip"),
    ("negative", "Our vacation was plagued with delays and bad service"),
    ("negative", "The coffee here tastes burnt and watered down"),
    ("negative", "I am thoroughly disappointed with the quality of this product"),
    ("negative", "The team played sloppily and lost embarrassingly"),
    ("negative", "This app crashes constantly and loses my data"),
    ("negative", "The garden was overgrown and poorly maintained"),
    ("negative", "Their new album is a tired, uninspired retread"),
    ("negative", "This is not a good film, it is a genuine chore to watch"),
    ("negative", "I never expected to dislike this movie as much as I did"),
    ("negative", "Despite a promising start, the film collapses completely"),
    ("negative", "It is hard to sit through this movie without checking the time"),
    ("negative", "A big-budget film with a surprisingly hollow core"),
    ("negative", "The sequel manages to be worse than the original in every way"),
    ("negative", "Not worth the wait, this fails on almost every level"),
]

assert len(SENTENCES) == 100, f"expected 100 sentences, found {len(SENTENCES)}"

PLAIN_LOGITS_RE = re.compile(
    r"Output logits Plain-Precomputed:\s*\[\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\]"
)
PLAIN_PRECOMPUTED_PRED_RE = re.compile(r"Plain-Precomputed:\s*(negative|positive) sentiment!")
PLAIN_PYTORCH_PRED_RE = re.compile(r"Plain-PyTorch\s*:\s*(negative|positive) sentiment!")

ENC_LOGITS_RE = re.compile(r"Output logits.*?\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]")
ENC_TIMING_RE = re.compile(r"The evaluation of the FHE circuit took:\s*([\d.]+)\s*seconds")
ENC_RETRY_RE = re.compile(r"retrying\.\.\.")


def sentiment_from_logits(logits):
    if logits is None:
        return None
    return "negative" if logits[0] > logits[1] else "positive"


def run_plaintext(sentence, timeout):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python3", "src/python/PlainCircuit.py", sentence],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
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
    m = PLAIN_LOGITS_RE.search(stdout)
    if m:
        logits = [float(m.group(1)), float(m.group(2))]

    pytorch_pred = None
    m2 = PLAIN_PYTORCH_PRED_RE.search(stdout)
    if m2:
        pytorch_pred = m2.group(1)

    return {
        "elapsed_sec": round(elapsed, 3),
        "logits": logits,
        "predicted_sentiment": sentiment_from_logits(logits),
        "pytorch_predicted_sentiment": pytorch_pred,
        "timed_out": timed_out,
        "returncode": returncode,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


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


def compute_aggregates(results, skip_encrypted):
    n = len(results)
    plain_times = [r["plaintext"]["elapsed_sec"] for r in results if r["plaintext"].get("logits")]
    plain_correct = sum(
        1 for r in results
        if r["plaintext"].get("predicted_sentiment") == r["expected_sentiment"]
    )
    plain_ok = sum(1 for r in results if r["plaintext"].get("logits"))

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
    }

    if not skip_encrypted:
        enc_results = [r["encrypted"] for r in results if "encrypted" in r]
        enc_times = [r["elapsed_sec"] for r in enc_results if r.get("logits")]
        enc_ok = sum(1 for r in enc_results if r.get("logits"))
        enc_correct = sum(
            1 for r in results
            if r.get("encrypted", {}).get("predicted_sentiment") == r["expected_sentiment"]
        )
        enc_failed = sum(1 for r in enc_results if not r.get("logits"))
        total_retries = sum(r.get("retries", 0) for r in enc_results)

        agree = 0
        diffs = []
        for r in results:
            enc = r.get("encrypted", {})
            p_pred, e_pred = r["plaintext"].get("predicted_sentiment"), enc.get("predicted_sentiment")
            if p_pred is not None and e_pred is not None:
                if p_pred == e_pred:
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
            "accuracy_vs_expected": enc_correct / n if n else None,
            "mean_time_sec": statistics.mean(enc_times) if enc_times else None,
            "median_time_sec": statistics.median(enc_times) if enc_times else None,
            "min_time_sec": min(enc_times) if enc_times else None,
            "max_time_sec": max(enc_times) if enc_times else None,
        }
        agg["comparison"] = {
            "both_succeeded": both_ok,
            "prediction_agreement": agree / both_ok if both_ok else None,
            "mean_abs_logit_diff": statistics.mean(diffs) if diffs else None,
            "speedup_encrypted_over_plaintext": (
                agg["encrypted"]["mean_time_sec"] / agg["plaintext"]["mean_time_sec"]
                if agg["encrypted"]["mean_time_sec"] and agg["plaintext"]["mean_time_sec"]
                else None
            ),
        }

    return agg


def write_report(report_path, results, skip_encrypted, total_target):
    agg = compute_aggregates(results, skip_encrypted)
    n = len(results)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("# Plaintext vs Encrypted Sentiment Inference Comparison")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Progress: {n}/{total_target} sentences completed")
    lines.append("")
    lines.append(
        "`expected_sentiment` is a hand-assigned sentiment label used only as a rough "
        "sanity check (not an SST-2 gold label)."
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Plaintext | Encrypted (FHE) |")
    lines.append("|---|---|---|")
    p, e = agg["plaintext"], agg.get("encrypted")
    lines.append(f"| Successful runs | {p['n_ok']}/{n} | {(e['n_ok'] if e else 'n/a')}/{n} |")
    if e:
        lines.append(f"| Failed / timed out | 0 | {e['n_failed_or_timed_out']} |")
        lines.append(f"| Total internal retries | n/a | {e['total_retries']} |")
    lines.append(
        f"| Accuracy vs expected label | {fmt(p['accuracy_vs_expected'], 3)} | "
        f"{fmt(e['accuracy_vs_expected'], 3) if e else 'n/a'} |"
    )
    lines.append(
        f"| Mean time (s) | {fmt(p['mean_time_sec'])} | {fmt(e['mean_time_sec']) if e else 'n/a'} |"
    )
    lines.append(
        f"| Median time (s) | {fmt(p['median_time_sec'])} | {fmt(e['median_time_sec']) if e else 'n/a'} |"
    )
    lines.append(
        f"| Min / Max time (s) | {fmt(p['min_time_sec'])} / {fmt(p['max_time_sec'])} | "
        + (f"{fmt(e['min_time_sec'])} / {fmt(e['max_time_sec'])}" if e else "n/a")
        + " |"
    )

    if not skip_encrypted:
        c = agg["comparison"]
        lines.append("")
        lines.append(
            f"**Prediction agreement (plaintext vs encrypted):** "
            f"{c['both_succeeded']} pairs compared, "
            f"{fmt(c['prediction_agreement'], 3)} agreement rate"
        )
        lines.append(f"**Mean absolute logit difference:** {fmt(c['mean_abs_logit_diff'], 4)}")
        lines.append(
            f"**Encrypted / plaintext time ratio (slowdown):** "
            f"{fmt(c['speedup_encrypted_over_plaintext'], 1)}x"
        )

    lines.append("")
    lines.append("## Per-sentence results")
    lines.append("")
    if skip_encrypted:
        lines.append("| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) |")
        lines.append("|---|---|---|---|---|---|")
    else:
        lines.append(
            "| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | "
            "Enc pred | Enc logits | Enc time (s) | Retries | Agree |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for i, r in enumerate(results, 1):
        p = r["plaintext"]
        row = [
            str(i),
            r["expected_sentiment"],
            truncate(r["sentence"]),
            p.get("predicted_sentiment") or "FAIL",
            fmt_logits(p.get("logits")),
            fmt(p["elapsed_sec"], 2),
        ]
        if not skip_encrypted:
            enc = r.get("encrypted", {})
            p_pred, e_pred = p.get("predicted_sentiment"), enc.get("predicted_sentiment")
            agree = "-" if (p_pred is None or e_pred is None) else ("✓" if p_pred == e_pred else "✗")
            row += [
                enc.get("predicted_sentiment") or "FAIL",
                fmt_logits(enc.get("logits")),
                fmt(enc.get("elapsed_sec"), 2),
                str(enc.get("retries", 0)),
                agree,
            ]
        lines.append("| " + " | ".join(row) + " |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-json", default="sentiment_comparison_results.json")
    parser.add_argument("--out-report", default="sentiment_comparison_report.md")
    parser.add_argument("--limit", type=int, default=len(SENTENCES), help="only run the first N sentences")
    parser.add_argument("--report-every", type=int, default=10, help="(re)write the report every N sentences")
    parser.add_argument("--skip-encrypted", action="store_true", help="only run plaintext inference")
    parser.add_argument("--plain-timeout", type=float, default=120.0)
    parser.add_argument("--encrypted-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true", help="continue from an existing --out-json file")
    args = parser.parse_args()

    sentences = SENTENCES[: args.limit]

    results = load_existing_results(args.out_json) if args.resume else []
    if results:
        print(f"Resuming: {len(results)} sentences already completed, continuing from #{len(results) + 1}", flush=True)

    for i in range(len(results), len(sentences)):
        label, sentence = sentences[i]
        print(f"[{i + 1}/{len(sentences)}] ({label}) {sentence!r}", flush=True)

        print("  running plaintext...", flush=True)
        plain = run_plaintext(sentence, args.plain_timeout)
        print(
            f"    logits={plain['logits']} time={plain['elapsed_sec']}s "
            f"predicted={plain['predicted_sentiment']}",
            flush=True,
        )

        entry = {
            "sentence": sentence,
            "expected_sentiment": label,
            "plaintext": plain,
        }

        if not args.skip_encrypted:
            print("  running encrypted (this can take 1-4 minutes)...", flush=True)
            enc = run_encrypted(sentence, args.encrypted_timeout)
            print(
                f"    logits={enc['logits']} time={enc['elapsed_sec']}s "
                f"retries={enc['retries']} predicted={enc['predicted_sentiment']} "
                f"returncode={enc['returncode']}",
                flush=True,
            )
            entry["encrypted"] = enc

        results.append(entry)

        # Persist raw results after every sentence so progress always survives a crash.
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
