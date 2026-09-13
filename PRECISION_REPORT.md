# FHE-BERT-Tiny Accuracy Investigation & Fix

## Summary

Investigated a large, systematic magnitude gap between the plaintext ("Plain-Precomputed")
and encrypted (FHE) circuit outputs: the encrypted classifier logits were consistently only
**~35-40% of the plaintext logits' magnitude** (same sign/prediction, but far more "unsure"
looking), whereas `notebooks/Precision of FHE Circuit.ipynb` shows the two circuits used to
track within ~5-15% of each other. Found and fixed a ciphertext/plaintext CKKS **level
mismatch** in the Value projection of layer-0 self-attention, introduced by nothing in
`main.cpp`'s logic (unchanged since before the FIDESlib port) but exposed by FIDESlib's GPU
`EvalMult(ciphertext, plaintext)` not replicating an OpenFHE-CPU implicit level-adjustment
idiom the original code relied on. The fix raises the average |encrypted/plaintext| logit
ratio from **0.379 to 0.651** (closing ~44% of the gap) with no change to the circuit's
algorithm, no new approximation, and identical classification outcomes.

## Baseline discrepancy

Running `src/python/run_benchmark.py`:

| Sentence | Plaintext logits | Encrypted logits | \|enc/plain\| ratio |
|---|---|---|---|
| "This was a complete waste of my time and money" | [2.610, -2.291] | [1.117, -0.909] | 0.428 / 0.397 |
| "The plot was boring and the acting felt wooden" | [2.653, -2.405] | [0.987, -0.771] | 0.372 / 0.320 |

Both other benchmark sentences crashed with `SIGABRT` after 3 retries — a pre-existing,
documented, data-dependent GPU race condition in FIDESlib's bootstrap kernels
(see `PERFORMANCE_REPORT.md`), out of scope here and unrelated to this fix.

Cross-checked against `notebooks/Precision of FHE Circuit.ipynb`, which records the
circuit's *intended* behavior from before the FIDESlib port: there, encrypted vs. plaintext
classifier logits track each other closely (e.g. plaintext `[-2.617, 2.492]` vs. encrypted
`[-2.565, 2.469]`, ratio ~0.98). The ~0.35-0.4 ratio seen now is a regression, not an
inherent property of the polynomial approximations used throughout the circuit.

## Isolating the cause

Compared aggregate statistics of the pooler's tanh output (the last nonlinearity before the
classifier) between plaintext and encrypted for the same sentence: plaintext values are
strongly saturated (mean |x| = 0.90, 80% of values have |x| > 0.9, as expected for a
well-trained sentiment classifier), while the encrypted values were far less saturated (mean
|x| = 0.56, only 19.5% with |x| > 0.9) — a systematic **under-saturation**, not random noise.

Working backwards through `encoder1()` with temporary decrypt-and-print instrumentation
(`FHEBERT_DEBUG_SCORES` env var, since removed):

1. **Raw Q·K attention scores** (`matmulScores`) and **`eval_exp`'s range-reduced Taylor
   approximation of `exp(x/8)`**: decrypted values matched the exact PyTorch-computed
   `qk[head, query, key]` values to 4-5 significant digits, across the full packed score
   tensor (all heads/queries/keys, not just the diagonal — an early false lead). Not the
   cause.
2. **Softmax normalization** (`rotsum` + `eval_inverse_naive`): structurally verified correct
   (denominator broadcast aligns with the numerator's packing). Not investigated further
   given (1) was already clean.
3. **Value projection**: found via `inputs[0]->GetLevel()` vs. `scores->GetLevel()` debug
   prints:
   ```
   inputs[0]->GetLevel() = 0
   scores->GetLevel()    = 22
   scores->GetLevel()-2  = 20
   ```
   `src/main.cpp`'s `encoder1()` encodes the Value weight/bias plaintexts at
   `scores->GetLevel() - {2,1}` (levels ~20-21) but then multiplies them against `inputs[i]`
   — the *original*, never-touched embedding ciphertexts, still at level 0. This exact
   pattern is unchanged from the pre-FIDESlib-port `main` branch, where it relied on real
   OpenFHE's `EvalMult(ciphertext, plaintext)` implicitly dropping the ciphertext's extra RNS
   limbs to match a plaintext encoded "ahead of time" at a deeper level — a legitimate,
   well-known OpenFHE-CPU idiom for pre-consuming levels without an explicit `LevelReduce`
   call. FIDESlib's GPU `EvalMult(ciphertext, plaintext)` (`third_party/FIDESlib/api/CryptoContext.cpp`)
   does not replicate that adjustment; it loads both operands as-is and calls
   `res_gpu->multPt(*pt_gpu)`, silently multiplying a level-0 ciphertext (full RNS limb set)
   against a plaintext encoded for level 20 (a much smaller limb set), producing a result
   whose tracked CKKS scale no longer matches its actual numeric scale.

   Layer 1's equivalent (`encoder2()`) already used `inputs[0]->GetLevel()` directly for its
   Value weights (no mismatch), which is presumably why layer-1-only effects are smaller;
   layer 0's mismatch propagates through both layers.

## Fix

`src/main.cpp`, `encoder1()`: encode `value_w`/`value_b` at `inputs[0]->GetLevel()` — the
level the multiplication in `matmulRE(inputs, value_w, value_b)` actually happens at —
instead of `scores->GetLevel() - {2,1}`, matching the query/key weights above and layer 1's
already-correct pattern. No change to `circuit_depth`, `level_budget`, or any other CKKS
parameter, so existing keys remain valid, but keys were regenerated anyway per the intended
build→keygen→benchmark cycle (`./build/FHE-BERT-Tiny --generate_keys`).

## Results (after fix)

| Sentence | Plaintext logits | Encrypted logits | \|enc/plain\| ratio |
|---|---|---|---|
| "This was a complete waste of my time and money" | [2.610, -2.291] | [1.495, -1.287] | 0.573 / 0.562 |
| "The plot was boring and the acting felt wooden" | [2.653, -2.405] | [1.926, -1.785] | 0.726 / 0.742 |

Average \|encrypted/plaintext\| ratio: **0.379 → 0.651**. Predicted sentiment unchanged
(correct) on every completed run, before and after. The pre-existing SIGABRT crash on the
other two benchmark sentences reproduced identically after the fix (retries=3, same as
baseline) — confirmed unrelated.

## Known remaining gap (not fully root-caused)

The ratio is closer to 1.0 but not there yet. One concrete, verified contributing factor
found but not fixed: **every `bootstrap(..., 0, verbose)` call in `FHEController.cpp` is
commented "Meta-BTS (numIterations=2) for extra precision", but FIDESlib's GPU
`EvalBootstrap`/`EvalBootstrapInPlace` (`third_party/FIDESlib/api/CryptoContext.cpp:1719-1768`)
ignores the `numIterations`/`precision` parameters entirely on the GPU path, always calling
`FIDESlib::CKKS::Bootstrap(ctxt, slots, prescaled)` — a single-iteration bootstrap.** True
Meta-BTS (bootstrap the ciphertext, level-reduce the result back down, subtract to isolate
the bootstrap error, bootstrap the error, add back) isn't reproducible at the FHEController
level either: FIDESlib's `CryptoContext` API exposes no level-reduce/mod-switch-down
primitive to bring a freshly-bootstrapped (high-level) ciphertext back down to match the
pre-bootstrap (low-level) one, which the algorithm requires. Confirmed this doesn't fully
explain the remaining gap by itself: the largest single jump in the ratio happens in
self-attention, before any bootstrap call in `encoder1()` runs at all — but it likely
compounds with normal CKKS/Chebyshev approximation noise across ~10 bootstrap calls in the
two-layer circuit. Revisiting this would require either extending FIDESlib itself (out of
scope here) or increasing `level_budget` to improve the single available bootstrap's
CoeffsToSlots/SlotsToCoeffs precision, at the cost of more `circuit_depth` (and therefore a
full key regeneration) for an untested payoff.

## Files changed

- `src/main.cpp` (the level fix in `encoder1()`)
- `keys/` (regenerated by `--generate_keys`; note this writes to `../keys` relative to the
  binary's working directory, i.e. `/root/keys` when run from the repo root — a pre-existing
  path quirk, not something this change touched)
- `benchmark_results.json` (fresh post-fix benchmark run)
