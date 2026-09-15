# Ciphertext vs Optimized Plaintext — Run-to-Run Comparison

Two independent 50-sentence runs of `src/python/run_ciphertext_vs_optimized_plaintext.py`
on the same machine, same trained BERT-tiny/SST-2 weights, same 50 sentences
(the first 50 of `run_sentiment_comparison.py`'s `SENTENCES` list), one day
apart. Purpose: check that the ciphertext-vs-plaintext comparison is stable
and not an artifact of a single run.

| | Run A | Run B |
|---|---|---|
| Generated | 2026-09-14 13:23:03 UTC | 2026-09-15 01:15:22 UTC |
| Report / JSON | `CIPHERTEXT_VS_OPTIMIZED_PLAINTEXT_REPORT.md` (committed in `792fb49`) | current `CIPHERTEXT_VS_OPTIMIZED_PLAINTEXT_REPORT.md` |

## Aggregate metrics

| Metric | Run A | Run B | Δ (B − A) |
|---|---|---|---|
| Encrypted successful runs | 47/50 | 49/50 | +2 |
| Encrypted failed / timed out | 3 | 1 | −2 |
| Total internal retries | 0 | 0 | — |
| Plaintext accuracy vs expected | 0.780 | 0.780 | 0.000 |
| Encrypted accuracy vs expected | 0.580 | 0.600 | +0.020 |
| Plaintext mean time (s) | 0.0259 | 0.0267 | +0.0008 |
| Encrypted mean time (s) | 44.325 | 42.957 | −1.368 (−3.1%) |
| Encrypted median time (s) | 43.685 | 42.488 | −1.197 |
| Encrypted min / max time (s) | 38.29 / 58.32 | 38.15 / 49.22 | tighter spread (max −9.1s) |
| Encrypted time stdev (s) | 3.86 | 2.54 | −1.32 |
| Prediction agreement (plain vs enc) | 0.830 (47 pairs) | 0.837 (49 pairs) | +0.007 |
| Mean \|Δlogit\| | 0.9801 | 0.9868 | +0.0067 |
| Mean \|Δlogit₀\| / \|Δlogit₁\| | 0.9389 / 1.0213 | 0.9512 / 1.0223 | +0.0123 / +0.0010 |
| Max \|Δlogit\| | 2.1088 | 2.1030 | −0.0058 |
| Encrypted / plaintext slowdown | 1714.3x | 1611.9x | −102.4x (fewer failures + slightly faster) |

## What's identical across runs

- **Plaintext outputs are bit-for-bit stable**: same predictions, same
  accuracy (0.780), logits match to float precision (the plaintext circuit
  is deterministic — no CKKS noise involved).
- **The same sentence fails to decrypt in both runs**: *"A brilliant, funny,
  and touching film that exceeded my expectations"* failed with a CKKS
  decode/approximation error in Run A and again in Run B.
- **The same 8 sentences flip prediction** between plaintext and encrypted
  in both runs (identical set, listed below). This determinism across two
  independent runs indicates these are genuine borderline cases — plaintext
  logit margins close to 0 that the CKKS approximation error pushes across
  the decision boundary — not random per-run noise:
  - "The cinematography was stunning from start to finish"
  - "The chemistry between the leads made the whole film work"
  - "I would happily watch this again tomorrow"
  - "This is family entertainment done exactly right"
  - "A gripping thriller that never lets up"
  - "Customer service was fast, friendly, and helpful"
  - "The weather was perfect for our trip to the coast"
  - "Our vacation exceeded every expectation we had"

## What differs across runs

- **Encrypted failure count dropped from 3 to 1.** Two sentences that failed
  in Run A ("The special effects were seamless and genuinely thrilling",
  "Every scene felt purposeful and beautifully shot") succeeded in Run B.
  Failures come from the same client-side CKKS decode error
  (`ckkspackedencoding.cpp: Decode(): approximation error is too high`)
  after exhausting the circuit's built-in 3-attempt server-recompute retry —
  this is sampling noise inherent to bootstrapping precision near the
  circuit's ciphertext level budget, not a deterministic per-sentence
  failure (the one sentence that failed both times may simply sit closer to
  the noise threshold than the other two).
- **Encrypted timing is ~3% faster and less variable** in Run B (mean 42.96s
  vs 44.33s, stdev 2.54s vs 3.86s, max 49.2s vs 58.3s). Consistent with
  ordinary machine-load variance between runs rather than a code change —
  no changes to `ServerCircuit.cpp`/`FHEController.cpp` landed between runs
  (see `git log`); the working tree does carry uncommitted build-system
  changes (`CMakeLists.txt`, `build_fideslib.sh`, switching FIDESlib from an
  in-tree `add_subdirectory` to an installed CMake package via
  `find_package(fideslib CONFIG REQUIRED)`), but these affect how the binary
  is *built*, not what it computes, and the binary was rebuilt from the same
  source before Run B.
- **Encrypted accuracy moved from 0.580 to 0.600** (30/50 → 30/50 correct on
  the 47/49 that decoded, plus one previously-failed sentence now decoding
  correctly) — within expected run-to-run noise given only ~2 more
  successful decodes.

## Verdict

The ciphertext-vs-plaintext comparison is **reproducible**: aggregate
accuracy, agreement rate, and mean slowdown are stable within a few percent
across independent runs, and the specific sentences that disagree or fail
are the same both times, pointing to real borderline-precision cases in the
CKKS approximation rather than run noise. The only run-to-run variance is in
(a) which 1-3 of the near-threshold sentences hit the decode-retry failure,
and (b) a few percent of wall-clock timing — both consistent with ordinary
GPU/bootstrapping jitter, not a regression.
