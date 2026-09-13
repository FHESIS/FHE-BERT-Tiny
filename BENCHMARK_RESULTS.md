# Sentiment Benchmark: Plaintext vs. Encrypted Inference

Automated run of `src/python/run_benchmark.py` across 5 negative and 5 positive
sentences, comparing the plaintext reference circuit (`PlainCircuit.py`,
"Plain-Precomputed" — the same precomputed-LayerNorm approximation the FHE
circuit itself uses) against the encrypted `FHE-BERT-Tiny` binary
(`--verbose`, GPU/FIDESlib backend). Raw output is in `benchmark_results.json`.

## Results

| # | Sentence | Expected | Plain logits [neg, pos] | Plain time | Plain pred. | Encrypted logits [neg, pos] | Encrypted time | Retries | Encrypted pred. | Agree w/ plain |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | This was a complete waste of my time and money | negative | `[2.610, -2.291]` | 9.4s | negative | `[1.117, -0.909]` | 170.5s | 0 | negative | ✅ |
| 2 | The plot was boring and the acting felt wooden | negative | `[2.653, -2.405]` | 9.5s | negative | `[0.988, -0.772]` | 165.4s | 0 | negative | ✅ |
| 3 | I hated every minute of this dull and pointless movie | negative | `[2.494, -2.141]` | 9.9s | negative | `[0.561, -0.354]` | 172.6s | 0 | negative | ✅ |
| 4 | A disappointing mess with no redeeming qualities | negative | `[1.916, -1.563]` | 9.5s | negative | `[-0.038, 0.316]` | 171.4s | 0 | **positive** | ❌ |
| 5 | The worst film I have seen in years, truly terrible | negative | `[2.116, -1.676]` | 9.4s | negative | — | 218.8s | 3 | **CRASH** | — |
| 6 | The acting was superb and the story kept me engaged the whole time | positive | `[0.622, -0.306]` | 9.4s | **negative** | `[0.022, 0.044]` | 174.8s | 0 | positive | ❌ |
| 7 | I absolutely loved this film, it was a masterpiece | positive | `[-2.616, 2.533]` | 9.3s | positive | — | 214.8s | 3 | **CRASH** | — |
| 8 | What a delightful and heartwarming movie experience | positive | `[-2.774, 2.651]` | 9.4s | positive | `[-0.220, 0.274]` | 166.7s | 0 | positive | ✅ |
| 9 | This is one of the best performances I have seen all year | positive | `[-1.918, 1.943]` | 9.3s | positive | `[-0.717, 0.652]` | 173.7s | 0 | positive | ✅ |
| 10 | A brilliant, funny, and touching film that exceeded my expectations | positive | `[-2.702, 2.601]` | 9.6s | positive | — | 225.3s | 3 | **CRASH** | — |

`CRASH` = the binary exhausted all 3 internal retry attempts and aborted with
`OpenFHEException: The decryption failed because the approximation error is
too high` (uncaught, `SIGABRT`/returncode -6) — no logits recoverable.

## Observations

**Accuracy vs. expected label.** Plaintext (Plain-Precomputed) matched the
intended label in 9/10 cases; it mis-predicted #6 ("The acting was superb...")
as negative. Encrypted inference completed successfully for 7/10 sentences,
of which 6/7 matched the *expected* label — notably, encrypted got #6 right
(positive) even though plaintext got it wrong, and encrypted got #4 wrong
(predicted positive for a negative sentence) despite plaintext getting it
right. This is consistent with the CKKS circuit's compressed/lossy logits
occasionally flipping a marginal case in either direction.

**Reliability.** 3/10 encrypted runs (30%) crashed after exhausting all 3
retries — a higher failure rate than the 1/5 (20%) observed in the earlier
manual `TIMING_COMPARISON.md` sample, though the sample sizes are small and
this is consistent with the documented intermittent GPU-bootstrap noise bug
(see `FIDESLIB_GPU_MIGRATION.md` §5/§6), not a new regression.

**Timing.** Plaintext: ~9.3–9.9s per sentence, dominated by fixed model-load
overhead (`PlainCircuit.py` reloads weights every invocation); actual
plaintext compute is sub-second. Encrypted: successful runs took 165–175s;
crashed runs (which still burn through 3 full retry attempts before aborting)
took 215–225s. This is markedly slower — roughly 2–3x — than the 37–78s
range recorded in the earlier manual comparison; the delta is worth
investigating (e.g. GPU contention/thermal state, or accumulated state across
repeated invocations in the same session) but wasn't diagnosed further here.

**Logit magnitude.** As previously observed, encrypted logits are
consistently compressed relative to plaintext (e.g. `±0.3–0.9` vs. `±1.5–2.8`),
expected given the circuit's polynomial approximations of softmax/GELU/tanh.

## Reproducing

```bash
python3 src/python/run_benchmark.py --out benchmark_results.json
# add --skip-encrypted to only run the fast plaintext half
```
