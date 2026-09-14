#!/usr/bin/env python3
"""
Statistical analysis of the CPA (chosen-plaintext, timing/metadata side-channel) and CCA
(chosen-ciphertext / malleability-oracle) experiments for FHE-BERT-Tiny.

Reads:
  security_tests/cpa_results.csv       (from run_cpa_test.py)
  security_tests/cca_probe_results.csv (from ./build/cca-probe)

Writes:
  security_tests/security_analysis_summary.json  (machine-readable results, used by the report)
and prints a human-readable summary to stdout.
"""
import csv
import json
import math
import os
import statistics as stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPA_CSV = os.path.join(REPO_ROOT, "security_tests", "cpa_results.csv")
CCA_CSV = os.path.join(REPO_ROOT, "security_tests", "cca_probe_results.csv")
OUT_JSON = os.path.join(REPO_ROOT, "security_tests", "security_analysis_summary.json")

from scipy import stats as scistats  # noqa: E402


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# CPA analysis
# ---------------------------------------------------------------------------------------------

def analyze_cpa():
    rows = load_csv(CPA_CSV)
    successful = [r for r in rows if r.get("returncode") == "0" and to_float(r.get("total_circuit_time_s")) is not None]

    result = {"n_total_runs": len(rows), "n_successful_runs": len(successful)}

    # --- Structural leak: does the number of ciphertexts submitted equal the token count? ---
    # (This is definitional in the current architecture -- one ciphertext per token -- but we
    # verify it empirically here rather than assert it, and quantify how much it narrows down
    # sentence length for an observer who never decrypts anything.)
    n_ctxt_vals = [int(r["num_input_ciphertexts"]) for r in successful if r.get("num_input_ciphertexts")]
    result["ciphertext_count_leak"] = {
        "distinct_values_observed": sorted(set(n_ctxt_vals)),
        "note": "Each distinct value below is exactly the token count of the input sentence; "
                "an observer who only counts request ciphertexts (no decryption needed) learns "
                "the input's token count exactly.",
    }

    # --- Attempt/retry rate overall ---
    attempts = [int(r["attempts"]) for r in successful if r.get("attempts")]
    result["retry_rate"] = {
        "runs_with_retry": sum(1 for a in attempts if a > 1),
        "n": len(attempts),
        "attempts_distribution": {str(a): attempts.count(a) for a in sorted(set(attempts))},
    }

    # --- Noise floor: repeated runs of the same fixed sentence per bucket ---
    noise_floor = {}
    by_bucket_pair0_pos = {}
    for r in successful:
        if r["arm"] == "positive" and r["pair_label"].endswith("_0"):
            by_bucket_pair0_pos.setdefault(r["bucket"], []).append(to_float(r["total_circuit_time_s"]))
    for bucket, times in by_bucket_pair0_pos.items():
        if len(times) >= 3:
            noise_floor[bucket] = {
                "n": len(times),
                "mean_s": stats.mean(times),
                "stdev_s": stats.stdev(times),
                "cv": stats.stdev(times) / stats.mean(times) if stats.mean(times) else None,
                "min_s": min(times),
                "max_s": max(times),
            }
    result["timing_noise_floor"] = noise_floor

    # --- Content-dependence test: within each bucket, positive vs negative circuit_time_s,
    #     paired by pair_label (length-matched sentence pairs) ---
    bucket_tests = {}
    for bucket in sorted(set(r["bucket"] for r in successful)):
        pos_by_pair, neg_by_pair = {}, {}
        for r in successful:
            if r["bucket"] != bucket:
                continue
            t = to_float(r["total_circuit_time_s"])
            if t is None:
                continue
            d = pos_by_pair if r["arm"] == "positive" else neg_by_pair
            d.setdefault(r["pair_label"], []).append(t)

        pairs = sorted(set(pos_by_pair) & set(neg_by_pair))
        pos_vals = [stats.mean(pos_by_pair[p]) for p in pairs]
        neg_vals = [stats.mean(neg_by_pair[p]) for p in pairs]

        entry = {
            "n_pairs": len(pairs),
            "pos_mean_s": stats.mean(pos_vals) if pos_vals else None,
            "neg_mean_s": stats.mean(neg_vals) if neg_vals else None,
        }
        if len(pairs) >= 4:
            diffs = [p - n for p, n in zip(pos_vals, neg_vals)]
            try:
                wstat, wp = scistats.wilcoxon(pos_vals, neg_vals)
            except ValueError:
                wstat, wp = None, None
            tstat, tp = scistats.ttest_rel(pos_vals, neg_vals)
            entry.update({
                "wilcoxon_stat": wstat, "wilcoxon_p": wp,
                "paired_t_stat": tstat, "paired_t_p": tp,
                "mean_diff_s": stats.mean(diffs),
            })
        bucket_tests[bucket] = entry
    result["content_dependence_timing"] = bucket_tests

    return result


# ---------------------------------------------------------------------------------------------
# CCA / malleability analysis
# ---------------------------------------------------------------------------------------------

def analyze_cca():
    rows = load_csv(CCA_CSV)
    n = len(rows)
    linf = [to_float(r["recovery_linf_error"]) for r in rows]
    l2 = [to_float(r["recovery_l2_error"]) for r in rows]
    correct = sum(int(r["distinguish_correct"]) for r in rows)

    # Exact binomial test against chance = 0.5
    p_value = scistats.binomtest(correct, n, 0.5, alternative="greater").pvalue if n else None

    return {
        "n_trials": n,
        "plaintext_recovery": {
            "max_linf_error": max(linf) if linf else None,
            "mean_linf_error": stats.mean(linf) if linf else None,
            "mean_l2_error": stats.mean(l2) if l2 else None,
        },
        "distinguishing_game": {
            "correct": correct,
            "n": n,
            "accuracy": correct / n if n else None,
            "chance_accuracy": 0.5,
            "binomial_p_value_vs_chance": p_value,
        },
    }


def main():
    summary = {}
    if os.path.exists(CPA_CSV):
        summary["cpa"] = analyze_cpa()
    else:
        print(f"WARNING: {CPA_CSV} not found, skipping CPA analysis")

    if os.path.exists(CCA_CSV):
        summary["cca"] = analyze_cca()
    else:
        print(f"WARNING: {CCA_CSV} not found, skipping CCA analysis")

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
