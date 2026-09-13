# Plaintext vs. Encrypted Inference: Timing Comparison

Comparison of `PlainCircuit.py` (plaintext) and `FHE-BERT-Tiny` (encrypted/FHE) inference across five sentences spanning positive and negative sentiment.

## Results

| # | Sentence | Sentiment | Plain logits | Plain time | Encrypted logits | Encrypted time | Match | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | "Dune was a quite interesting movie" | positive | `[-0.744, 0.974]` | 9.4s | `[-0.002, 0.151]` | 78s | ✅ | clean run |
| 2 | "This film was absolutely fantastic and I loved every second of it" | positive | `[-2.320, 2.297]` | 9.4s | `[-0.183, 0.321]` | 78s | ✅ | 1 retry (noise) |
| 3 | "This was a terrible movie and a complete waste of time" | negative | `[2.558, -2.217]` | 10.7s | `[0.852, -0.616]` | 37s | ✅ | clean run, fastest |
| 4 | "I hated the ending, it was boring and disappointing" | negative | `[2.436, -2.091]` | 9.4s | — | — | ❌ | **failed after 3 retries** |
| 5 | "The acting was brilliant and the story was truly inspiring" | positive | `[-2.509, 2.442]` | 10.4s | `[-0.746, 0.720]` | 69s | ✅ | 1 retry (noise) |

## Observations

**Timing.** Plaintext runs are dominated by fixed model-load overhead (~9s each), since `PlainCircuit.py` reloads weights on every invocation — actual plaintext compute is sub-second. Encrypted runs ranged from 37–78 seconds, roughly 4–8x the plaintext wall-clock time as measured, and far more if the plaintext load overhead is subtracted out (true plaintext inference is near-instant).

**Sentiment agreement.** All four successful encrypted runs agree with plaintext on the predicted class. Encrypted logit magnitudes are consistently smaller/compressed relative to plaintext, which is expected given the FHE circuit's polynomial approximations of nonlinearities (e.g. softmax, GELU).

**Reliability issue.** 3 of 5 encrypted runs hit:

```
The decryption failed because the approximation error is too high. Check the parameters.
```

during self-attention or pooling. Two recovered via the circuit's automatic retry (adding roughly 15–40s of extra evaluation time), but sentence 4 exhausted all 3 retry attempts and crashed with an uncaught `OpenFHEException`, producing no output. This suggests the bootstrapping/noise-budget parameters are marginal for some inputs and may warrant tuning.

## Commands used

```bash
python3 ./src/python/PlainCircuit.py "<sentence>" --verbose
./build/FHE-BERT-Tiny "<sentence>" --verbose
```
