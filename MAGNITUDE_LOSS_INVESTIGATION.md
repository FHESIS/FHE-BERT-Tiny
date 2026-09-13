# Investigating Where Encrypted-Inference Magnitude Is Lost

`BENCHMARK_RESULTS.md` showed encrypted logits are consistently compressed
relative to plaintext (e.g. `±0.3–0.9` vs. `±1.5–2.8`, roughly a 2–5x
shrink). This report investigates *where in the circuit* that shrinkage
actually happens, using two standalone numerical scripts
(`src/python/analyze_attention_magnitude.py`,
`src/python/analyze_layernorm_magnitude.py`) that replicate the FHE
circuit's exact approximation math in plain NumPy/PyTorch against the real
BERT-tiny-SST2 weights — no GPU/FHE run required. Both scripts take a
sentence as `argv[1]` and must be run from the repo root.

The starting hypothesis, based on reading `FHEController.cpp`, was that the
self-attention softmax approximation was the likely culprit:

- `matmulScores` (:1049) — raw `Q·Kᵀ`, pre-scaled by `1/64` via `mask_heads`
- `eval_exp` (:1282) — softmax numerator: a degree-6 Taylor series of `e^x`
  evaluated at `x = raw_qk/64`, then cubed 3x (`EvalSquare` x3, i.e. raised
  to the 8th power) to reconstruct `e^(raw_qk/8)`, since FIDESlib has no
  `EvalPoly`/Paterson-Stockmeyer helper
- `eval_inverse_naive` / `eval_inverse_naive_2` (:1338/:1344) — softmax
  denominator: a Chebyshev-fit polynomial approximation of `1/x`
- `matmulRE` (ciphertext-weight overload, :943) — the `scores · V`
  aggregation via `mult` + `rotsum`

That hypothesis turned out to be **wrong**. Both suspects were tested in
isolation against real data and neither shrinks the output.

## 1. Self-attention softmax: not the source of shrinkage

Using real Q/K/V computed from actual BERT-tiny-SST2 weights for the
benchmark's test sentences, `analyze_attention_magnitude.py` compares the
circuit's exact approximation formulas against true `exp`/`softmax`:

| Stage | Layer 0 (`encoder1`) | Layer 1 (`encoder2`) |
|---|---|---|
| `eval_exp` (Taylor-6 + `^8`) relative error | — (shared code, ~0.00-0.01% mean, ~0.15% max) | same |
| Inverse domain / degree | `eval_inverse_naive(sum, 2, 5000)`, degree 119 | `eval_inverse_naive_2(sum, 3, 145000, 1)`, degree 200 |
| True `sum(exp)` range observed | `[4, 1506]` (9-30% of domain width) | `[19, 15803]` (3-11% of domain width) |
| Inverse relative error | mean 0.8-1.1%, max ~1.6% | **mean 19-21%, max ~31%** |
| Net attention-output norm ratio (approx/true) | ≈ 1.00 (unchanged) | **1.03-1.05 (inflated, not shrunk)** |

Findings:
- `eval_exp`'s Taylor+cubing trick is essentially exact in the operating
  range actually seen (`raw_qk/64` stays within roughly `[-0.5, 1.2]`) — not
  a meaningful error source at all.
- Layer 1's `eval_inverse_naive_2` domain, `[3, 145000]`, is wildly
  oversized relative to the sums it actually needs to invert (which never
  exceed ~16,000, and are usually under 5,000). A fixed-degree-200
  Chebyshev fit spread across a 145,000-wide domain has far less precision
  at the ~3-11% of that range where real data lands than layer 0's
  `[2, 5000]` fit, which is well-matched to its own `[4, 1506]` operating
  range (correspondingly ~25x lower error).
- Despite a genuinely large ~20% per-weight relative error, the layer-1
  inverse approximation **inflates** the attention output norm by 3-5%
  rather than shrinking it — its error isn't zero-mean in a
  magnitude-reducing direction.

**Conclusion: self-attention is not where the observed 2-5x compression
comes from.** It's a real precision bug (worth fixing — see §3) but not a
magnitude-loss bug.

## 2. Precomputed LayerNorm: also not the source of shrinkage

The circuit avoids computing true per-example mean/variance under
encryption by using `PrecomputedLayerNorm`: each of BERT-tiny's 4
LayerNorms is replaced with a fixed, **token-position-indexed**
dataset-average mean/inverse-std (`weights-sst2/layer{0,1}_{selfoutput,
output}_{mean,vy}.txt`), applied regardless of the actual content at that
position. `analyze_layernorm_magnitude.py` compares this against real
per-token statistics, again using actual model weights and real sentences,
with an otherwise-exact (PyTorch) self-attention/FFN pipeline so the
LayerNorm effect is isolated.

For "This was a complete waste of my time and money" (12 tokens):

| Stage | Norm ratio (precomputed-LN / true-LN) |
|---|---|
| Layer 0 attention-output LN | 1.71x |
| Layer 0 FFN-output LN | 1.17x |
| Layer 1 attention-output LN | 2.15x |
| Layer 1 FFN-output LN | 2.32x (cumulative) |
| **Final logits** (after pooler `tanh` + classifier) | **1.21x** |

The precomputed inverse-std is consistently *smaller* than each token's
true per-example std, so the normalization over-scales rather than
under-scales — every one of the 4 LayerNorm stages **inflates** hidden-state
magnitude (1.17-2.32x, compounding through the network). The pooler's
`tanh` saturates most of that back down, but the net effect at the logit
level is still a ~1.2x inflation, not a shrink.

**Conclusion: precomputed LayerNorm is not the source of shrinkage
either** — like the self-attention inverse approximation, its error pushes
in the wrong direction to explain what's observed.

## 3. Where the shrinkage most likely comes from instead

Since both candidate *mathematical* approximations amplify rather than
shrink signal magnitude, the 2-5x compression actually observed between
plaintext and encrypted logits is most likely **not** a polynomial/Chebyshev
approximation-error effect at all, but a genuine CKKS/bootstrapping
precision effect that a plain floating-point simulation can't reproduce:

- `encoder2`'s self-attention scores are explicitly rescaled around a
  bootstrap call — `mult(1/500) → bootstrap → mult(500)` — specifically to
  improve bootstrap precision at a smaller scale. Any *absolute* bootstrap
  noise introduced during that rescaled bootstrap gets amplified 500x when
  scaled back up.
- The circuit calls `Meta-BTS` bootstrap (`numIterations=2`) roughly 8
  times per forward pass (self-attention scores, softmax denominator,
  self-output, intermediate, output, pooler, final classifier, plus retry
  attempts). `FIDESLIB_GPU_MIGRATION.md` §5 already documents empirically
  observing bootstrap decode precision swinging from ~+23 bits to ~-4 bits
  between otherwise-identical runs on the same ciphertext/input — an
  intermittent, not purely deterministic, noise source.
  `FLEXIBLEAUTOEXT` rescaling behavior compounding across ~8 such calls in
  a ~30-level circuit is a plausible mechanism for consistent, systematic
  signal shrinkage that this report's pure-math simulations cannot capture.

Confirming this would require instrumenting the actual GPU binary (the
existing `controller.print(...)` calls, already gated behind `--verbose`,
could be extended to more intermediate points) and comparing the decrypted
intermediate ciphertext values directly against this report's simulated
reference values stage-by-stage — not yet done here.

## 4. Side finding: a `PlainCircuit.py` bug (fixed)

While tracing the layer-1 FFN-output LayerNorm's constants for §2, we found
`src/python/PlainCircuit.py` was reusing the layer-1 **attention-output**
LayerNorm's precomputed `mean`/`var` arrays for the layer-1 **FFN-output**
LayerNorm too — both loops referenced the same module-level `mean`/`var`
defined once, even though `weights-sst2/layer1_selfoutput_{mean,vy}.txt`
and `weights-sst2/layer1_output_{mean,vy}.txt` (which the real FHE circuit
correctly loads separately) are genuinely different values.
`src/python/PlainCircuitSimplified.py` already had the correct, distinct
`LAYER1_OUTPUT_MEAN`/`LAYER1_OUTPUT_VAR` constants defined (unused by
`PlainCircuit.py`), confirming this was a bug rather than an intentional
simplification.

This has been fixed: `PlainCircuit.py` now defines a second `mean`/`var`
array (full double precision, read directly from
`weights-sst2/layer1_output_{mean,vy}.txt`) before the FFN-output LayerNorm
loop. The fix changes plaintext logits slightly (e.g. `[2.610, -2.291]` →
`[2.661, -2.347]` for one benchmark sentence) but does not change any
sentiment predictions checked so far, and — consistent with §2's finding
that precomputed-LayerNorm effects inflate rather than shrink — this bug
was not a contributor to the encrypted/plaintext magnitude gap.

## Scripts

- `src/python/analyze_attention_magnitude.py "<sentence>"` — §1's analysis
  (matmulScores/eval_exp/eval_inverse_naive[_2], layers 0 and 1)
- `src/python/analyze_layernorm_magnitude.py "<sentence>"` — §2's analysis
  (all 4 PrecomputedLayerNorm instances, plaintext pipeline)

Both are read-only diagnostics: they load the real model/weights and print
a report to stdout, no files are written and no FHE/GPU dependency is
required.

## Suggested next steps

1. Instrument `FHEController.cpp`/`main.cpp` to decrypt and print the
   self-attention scores, softmax denominator, and post-LayerNorm
   ciphertexts at each layer, and diff those against
   `analyze_attention_magnitude.py`/`analyze_layernorm_magnitude.py`'s
   simulated values for the *same* sentence, to directly localize the
   bootstrap/rescale-noise contribution by elimination.
2. If §3's hypothesis is confirmed, consider whether the `mult(1/500) →
   bootstrap → mult(500)` trick in `encoder2` could use a rescale factor
   closer to 1 (less amplification of absolute bootstrap error) now that
   `level_budget` and `levelsUsedBeforeBootstrap` have already been raised
   per `FIDESLIB_GPU_MIGRATION.md` §4.
3. Right-size `eval_inverse_naive_2`'s Chebyshev domain for layer 1 (its
   `[3, 145000]` bound looks like an untuned worst-case guess) closer to
   the empirically observed `[19, 16000]` range, mirroring how layer 0's
   `[2, 5000]` bound is already reasonably matched to its own `[4, 1506]`
   range — this won't fix the shrinkage but would reduce the ~20%
   per-weight softmax error found in §1.
