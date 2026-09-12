# Experiment Results: Plaintext vs. Encrypted Inference

Raw data from all sentences run through both `src/python/PlainCircuit.py` (plaintext-precomputed
reference) and `build/FHE-BERT-Tiny --plain` (actual encrypted circuit) during this investigation.
"Ratio" = plaintext logit gap / FHE logit gap (compression factor). All negative-sentiment sentences
below are true negatives per SST-2-style labeling; all positive-sentiment sentences are true positives.

## Negative-sentiment sentences — pre-`eval_exp`-refit (original Taylor-based softmax)

| Sentence | Plain logits | FHE logits | Compression | Retries | Result |
|---|---|---|---|---|---|
| "Dune was a bad movie" | `[2.464, -2.045]` | `[0.611, -0.540]` | 3.92x | 0 | correct |
| "The acting was terrible" | `[2.269, -1.905]` | `[0.808, -0.670]` | 2.82x | 0 | correct |
| "The plot was boring and predictable" | `[2.654, -2.436]` | `[0.919, -0.793]` | 2.97x | 0 | correct |
| "I hated every minute of this film" | `[1.741, -1.354]` | `[0.587, -0.382]` | 3.19x | 0 | correct |
| "The acting was wooden and unconvincing" | `[2.560, -2.329]` | `[0.652, -0.383]` | 4.72x | 0 | correct |
| "This was a complete waste of time" | `[2.556, -2.220]` | `[0.876, -0.680]` | 3.07x | 1 | correct |
| "The movie was dull and lifeless" | — | — | — | 3/3 failed | **crash** (deterministic decode failure) |
| "Such a disappointing sequel" | `[2.057, -1.639]` | `[0.614, -0.488]` | 3.35x | 0 | correct |
| "The script was weak and uninspired" | `[2.545, -2.266]` | `[0.751, -0.547]` | 3.71x | 2 | correct |

**9/9 attempted, 8/8 completed runs correctly classified, 1 fatal crash.**

## Positive-sentiment sentences — pre- and post-refit (all runs)

| Sentence | Plain logits | FHE logits | Retries | Result |
|---|---|---|---|---|
| "This movie was fantastic" (pre-refit, run 1) | `[-2.112, 2.170]` | — | 3/3 failed | **crash** |
| "This movie was fantastic" (pre-refit, run 2, independent retry) | `[-2.112, 2.170]` | — | 3/3 failed | **crash**, identical failure point both times |
| "I loved this film" (pre-refit) | `[-2.048, 2.243]` | `[0.532, -0.440]` | 0 | **misclassified negative** |
| "This movie was fantastic" (post-refit) | `[-2.112, 2.170]` | `[0.612, -0.586]` | 0 | **misclassified negative** (crash fixed, still wrong) |
| "I loved this film" (post-refit) | `[-2.048, 2.243]` | `[0.649, -0.598]` | 1 | **misclassified negative** |
| "The acting was brilliant" | `[-2.623, 2.569]` | `[0.612, -0.576]` | 0 | **misclassified negative** |
| "I really enjoyed this movie" | `[-2.741, 2.611]` | `[0.624, -0.588]` | 0 | **misclassified negative** |
| "This film was absolutely wonderful" | `[-2.842, 2.702]` | `[0.630, -0.596]` | 0 | **misclassified negative** |
| "The plot was engaging and clever" | `[-1.941, 1.835]` | `[0.622, -0.591]` | 0 | **misclassified negative** |
| "A truly great performance by the lead" | `[-2.372, 2.338]` | `[0.637, -0.601]` | 1 | **misclassified negative** |

**7/7 completed positive-sentiment runs misclassified as negative; 2/2 attempts on one sentence crashed
outright before the `eval_exp` refit.** Note how tightly the FHE logits cluster (`~[0.61-0.65, -0.58 to
-0.60]`) regardless of which positive sentence was run — see `04-attention-signal-collapse.md` for why.

## Negative-sentiment sentences — post-refit, with wall-clock timing

| Sentence | Plain logits | FHE logits | Compression | Plain time | FHE time | Retries |
|---|---|---|---|---|---|---|
| "The special effects looked cheap" | `[2.212, -1.754]` | `[0.635, -0.590]` | 3.24x | 14.9s | 167.8s | 0 |
| "This sequel ruined the original" | `[2.546, -2.269]` | `[0.629, -0.589]` | 3.95x | 9.3s | 256.9s | 1 |
| "The story made absolutely no sense" | `[1.156, -0.700]` | `[0.625, -0.585]` | 1.53x | 7.0s | 173.4s | 0 |
| "I found this movie tedious and forgettable" | `[1.934, -1.582]` | `[0.647, -0.593]` | 2.84x | 7.3s | 197.3s | 0 |

Plaintext inference: ~7-15s (CPU, single-threaded PyTorch, dominated by model/tokenizer load).
Encrypted inference: ~168-257s, i.e. **roughly 20-30x slower**, before accounting for the compression
in output fidelity. A retry (full circuit re-evaluation) costs roughly one extra clean-run's worth of
time (~90s here).

## Totals across the whole investigation

- **13 negative-sentiment attempts**: 12 correctly classified, 1 crashed (deterministic, "The movie was
  dull and lifeless").
- **9 positive-sentiment attempts**: 7 misclassified as negative, 2 crashed (same sentence, both
  deterministic, pre-refit).
- **0/9 positive-sentiment attempts produced a correct classification**, across two different `eval_exp`
  implementations and two different circuit-depth configurations (level 15 and level 16).
- Compression on correctly-classified negative sentences ranged **1.53x-4.72x** (mean ≈ 3.3x) — see
  `01-precision-investigation.md` for what was and wasn't found to explain this, and
  `04-attention-signal-collapse.md` for the actual root cause.
