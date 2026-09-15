#!/usr/bin/env python3
"""
Chosen-plaintext side-channel test harness for FHE-BERT-Tiny.

Runs the real client/server binary (./build/FHE-BERT-Tiny) once per sentence in a chosen corpus,
and records:
  - "server-observable" metadata: wall-clock time, per-stage timings and ciphertext levels (from
    the JSONL server log), retry/attempt count, and the number of input ciphertexts -- everything
    an attacker who can watch the server process (timing, logs, resource usage) but NOT decrypt
    anything could plausibly observe.
  - "ground truth" (known only to us as the experimenters, via the client's own decrypted output
    and the corpus metadata): the sentence text, its token count, and which "arm" of a
    length-matched positive/negative pair it belongs to.

The corpus is built as length-matched positive/negative sentence pairs (e.g. "Good movie" /
"Bad movie") specifically so that any correlation found between server-observable metadata and
sentiment cannot be trivially explained by sentence length -- length itself is analyzed
separately, since the *number of ciphertexts submitted* already equals the token count by
construction (see FHEBERT_SECURITY_REPORT.md).

Usage (from the repository root):
    ./src/python/.venv/bin/python3 security_tests/run_cpa_test.py [--repeats N] [--out FILE]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO_ROOT, "build", "FHE-BERT-Tiny")
SERVER_LOG = os.path.join(REPO_ROOT, "logs", "server_processing.jsonl")

# Length-matched (positive, negative) pairs, grouped by bucket. Pairs are built by swapping the
# sentiment-bearing word(s) so token counts match almost exactly within a pair.
CORPUS = {
    "short": [
        ("Good movie", "Bad movie"),
        ("I loved it", "I hated it"),
        ("Great acting here", "Terrible acting here"),
        ("Really enjoyed this", "Really disliked this"),
        ("A wonderful film", "A terrible film"),
    ],
    "medium": [
        ("This was a truly wonderful and delightful experience",
         "This was a truly awful and disappointing experience"),
        ("The film was funny charming and well acted",
         "The film was boring dull and badly acted"),
        ("A brilliant heartwarming story I really enjoyed",
         "A terrible depressing story I really hated"),
        ("Such a fantastic and beautifully shot movie",
         "Such a dreadful and poorly shot movie"),
        ("The actors gave a genuinely moving performance",
         "The actors gave a genuinely dull performance"),
    ],
    "long": [
        ("I absolutely loved every single minute of this wonderfully crafted and deeply moving film",
         "I absolutely hated every single minute of this poorly crafted and deeply boring film"),
        ("This is one of the best and most inspiring movies I have seen in years",
         "This is one of the worst and most tedious movies I have seen in years"),
        ("The director created a beautiful touching and unforgettable story about hope",
         "The director created a clumsy tedious and forgettable story about nothing"),
        ("A remarkable and joyful film that everyone in the family will truly love",
         "A miserable and joyless film that everyone in the family will truly hate"),
    ],
}

OUTPUT_LOGITS_RE = re.compile(r"Output logits.*?\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
CIRCUIT_TIME_RE = re.compile(r"The evaluation of the FHE circuit took:\s*([\d.]+)\s*seconds\s*\((\d+)\s*attempt")
CLIENT_ATTEMPT_RE = re.compile(r"\[client attempt (\d+)/(\d+)\]")
OUTCOME_RE = re.compile(r"Outcome:.*?\b(positive|negative)\b")


def run_one(sentence, bucket, pair_label, arm, repeat_idx):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [BINARY, sentence, "--verbose"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wall_time = time.perf_counter() - t0

    stdout = proc.stdout
    logits = None
    m = OUTPUT_LOGITS_RE.search(stdout)
    if m:
        logits = (float(m.group(1)), float(m.group(2)))

    # A request can involve multiple CLIENT-level attempts (see client_main.cpp: a decode
    # failure -- only detectable client-side -- makes the client ask the server to recompute
    # from scratch). Each attempt prints its own "circuit took" line; take the total across all
    # of them (what a real observer waiting for a response would see end-to-end) as well as the
    # final (successful) one's own figures.
    circuit_time_matches = list(CIRCUIT_TIME_RE.finditer(stdout))
    circuit_time_s, attempts, total_circuit_time_s = None, None, None
    if circuit_time_matches:
        total_circuit_time_s = sum(float(m.group(1)) for m in circuit_time_matches)
        last = circuit_time_matches[-1]
        circuit_time_s = float(last.group(1))
        attempts = int(last.group(2))

    client_attempts_used = None
    client_attempt_matches = list(CLIENT_ATTEMPT_RE.finditer(stdout))
    if client_attempt_matches:
        client_attempts_used = int(client_attempt_matches[-1].group(1))

    outcome = None
    m2 = re.search(r"Outcome:\s*\x1b?\[?[\d;]*m?(positive|negative)", stdout)
    if m2:
        outcome = m2.group(1)

    return {
        "sentence": sentence,
        "bucket": bucket,
        "pair_label": pair_label,
        "arm": arm,  # "positive" or "negative" half of the pair
        "repeat_idx": repeat_idx,
        "returncode": proc.returncode,
        "wall_time_s": wall_time,
        "circuit_time_s": circuit_time_s,
        "total_circuit_time_s": total_circuit_time_s,
        "attempts": attempts,
        "client_attempts_used": client_attempts_used,
        "logit0": logits[0] if logits else None,
        "logit1": logits[1] if logits else None,
        "predicted_sentiment": outcome,
        "stderr_tail": proc.stderr[-500:] if proc.returncode != 0 else "",
    }


def read_server_log_since(offset_bytes):
    """Read newly-appended JSONL server-log lines since the given byte offset."""
    if not os.path.exists(SERVER_LOG):
        return [], offset_bytes
    with open(SERVER_LOG, "rb") as f:
        f.seek(offset_bytes)
        data = f.read()
    new_offset = offset_bytes + len(data)
    lines = []
    for l in data.decode("utf-8", "replace").splitlines():
        if not l.strip():
            continue
        try:
            lines.append(json.loads(l))
        except json.JSONDecodeError:
            print(f"    WARNING: skipping malformed server-log line: {l[:200]!r}")
    return lines, new_offset


def summarize_stage_timings(lines):
    """Collapse one request's JSONL events into per-stage elapsed_ms and final metadata."""
    stages = {}
    num_input_ciphertexts = None
    output_level = None
    for ev in lines:
        stage = ev.get("stage", "")
        if stage == "request.received":
            num_input_ciphertexts = ev.get("num_input_ciphertexts")
        if stage.endswith(".end") and "elapsed_ms" in ev:
            name = stage[: -len(".end")]
            stages.setdefault(name, []).append(ev["elapsed_ms"])
        if stage == "request.completed":
            output_level = ev.get("output_level")
    # Sum durations per named stage (a stage can repeat, e.g. circuit.attempt on retry)
    agg = {name: sum(vals) for name, vals in stages.items()}
    return {
        "num_input_ciphertexts": num_input_ciphertexts,
        "output_level": output_level,
        "stage_ms": agg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1, help="repeats per sentence (for noise-floor estimation)")
    ap.add_argument("--noise-floor-repeats", type=int, default=4,
                     help="extra repeats for one fixed sentence per bucket, to estimate run-to-run timing noise")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "security_tests", "cpa_results.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)

    # Track server log offset so we only attribute newly-appended lines to each run.
    log_offset = os.path.getsize(SERVER_LOG) if os.path.exists(SERVER_LOG) else 0

    rows = []
    plan = []
    for bucket, pairs in CORPUS.items():
        for i, (pos, neg) in enumerate(pairs):
            pair_label = f"{bucket}_{i}"
            for arm, sentence in (("positive", pos), ("negative", neg)):
                for r in range(args.repeats):
                    plan.append((sentence, bucket, pair_label, arm, r))
        # noise-floor: repeat the first pair's positive sentence a few extra times
        if pairs and args.noise_floor_repeats > 0:
            sentence = pairs[0][0]
            for r in range(args.noise_floor_repeats):
                plan.append((sentence, bucket, f"{bucket}_0", "positive", args.repeats + r))

    print(f"Planned {len(plan)} runs across {len(CORPUS)} buckets.")

    for idx, (sentence, bucket, pair_label, arm, r) in enumerate(plan):
        print(f"[{idx + 1}/{len(plan)}] bucket={bucket} arm={arm} repeat={r}: {sentence!r}")
        result = run_one(sentence, bucket, pair_label, arm, r)

        log_lines, log_offset = read_server_log_since(log_offset)
        server_meta = summarize_stage_timings(log_lines)
        result.update({
            "num_input_ciphertexts": server_meta["num_input_ciphertexts"],
            "output_level": server_meta["output_level"],
        })
        for stage_name, ms in server_meta["stage_ms"].items():
            result[f"stage_ms__{stage_name}"] = ms

        rows.append(result)
        print(f"    -> rc={result['returncode']} wall={result['wall_time_s']:.1f}s "
              f"circuit={result['circuit_time_s']} attempts={result['attempts']} "
              f"predicted={result['predicted_sentiment']} n_ctxt={result['num_input_ciphertexts']}")

    # Union of all columns across rows (stage names can vary run to run, e.g. on retries)
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
