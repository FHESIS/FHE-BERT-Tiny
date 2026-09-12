# Precision Investigation: Plaintext vs. Encrypted Logits

## Summary

Running the same sentence through the plaintext-precomputed reference (`src/python/PlainCircuit.py`,
which mirrors the FHE circuit's math including its fixed LayerNorm approximation) and through the
actual FHE circuit (`build/FHE-BERT-Tiny`) produces the same classification in most cases, but the
encrypted logit gap is consistently compressed **~3-4x** relative to the plaintext reference. Three
plausible causes were investigated and ruled out or downgraded before the true root cause (self-attention
signal collapse, see `04-attention-signal-collapse.md`) was found.

## Baseline observation

| Sentence | Plain logits | FHE logits | Compression | Classification |
|---|---|---|---|---|
| "Dune was a bad movie" | `[2.464, -2.045]` | `[0.611, -0.540]` | 3.92x | correct |
| "The acting was terrible" | `[2.269, -1.905]` | `[0.808, -0.670]` | 2.82x | correct |

## Hypothesis 1: Chebyshev/Taylor approximation domains are too narrow — RULED OUT

The circuit approximates several nonlinearities with polynomials fit over a fixed numeric domain,
calibrated at design time:

- GELU (`eval_gelu_function`, `FHEController.cpp:1357`): domain ±13.5 (layer 0) / ±17 (layer 1)
- Softmax denominator inverse: `eval_inverse_naive` domain `[2, 5000]` (layer 0),
  `eval_inverse_naive_2` domain `[3, 145000]` (layer 1)
- Pooler tanh (`eval_tanh_function`, `:1363`): domain ±30

If real activations exceed these domains, Chebyshev approximation error should blow up rapidly outside
the fitted interval — a plausible cause for both the compression and the deterministic decode failures
observed on some inputs (e.g. "This movie was fantastic" crashed 100% of the time, 6/6 attempts across
two independent runs, before the `eval_exp` refit).

**Test**: swept the entire 872-sentence SST-2 validation set (`notebooks/SST-2-val.parquet`) through the
plaintext-equivalent computation, recording the empirical min/max of every quantity that feeds one of
these approximations (script: scratchpad `sweep_ranges.py`, methodology mirrors
`notebooks/SST-2 BERT-Tiny Chebyshev Intervals.ipynb`, which the original hardcoded domains were derived from).

| Quantity | Empirical range (872 sentences) | Hardcoded domain |
|---|---|---|
| layer0 GELU input | `[-13.43, 10.09]` | `±13.5` |
| layer1 GELU input | `[-12.63, 17.31]` | `±17` |
| layer0 softmax sum | `[1.89, 4842.98]` | `[2, 5000]` |
| layer1 softmax sum | `[2.31, 1295320.83]` | `[3, 145000]` |
| tanh input | `[-30.92, 27.77]` | `±30` |

Checked all four test sentences (including "This movie was fantastic", the one that crashes
deterministically) individually against these domains — **every single one landed comfortably inside
every domain**. Domain range is not the bottleneck, and is not what causes the crash either.

## Hypothesis 2: Fixed/precomputed LayerNorm statistics — MINOR FACTOR (~10-15%)

LayerNorm is approximated using **fixed, precomputed per-token-position mean/variance constants**
(`weights-sst2/layer*_mean.txt` / `*_var.txt`), calibrated once from a reference set, rather than each
input's true per-sample statistics — this is present even in the plaintext "Plain-Precomputed" reference,
so it's a designed approximation, not an FHE artifact.

**Test**: compared "Plain-Precomputed" (fixed LayerNorm) output against the true PyTorch forward pass
(real, dynamic LayerNorm) for all 4 test sentences.

| Sentence | Real PyTorch gap | Plain-Precomputed gap | Ratio |
|---|---|---|---|
| "Dune was a bad movie" | 4.114 | 4.509 | 1.096 |
| "This movie was fantastic" | 4.956 | 4.282 | 0.864 |
| "I loved this film" | 4.419 | 4.291 | 0.971 |
| "The acting was terrible" | 3.704 | 4.174 | 1.127 |

Deviation is only ~10-15% in either direction — nowhere near the observed 3-4x compression. This is a
real but secondary contributor.

## Hypothesis 3: CKKS multiplicative-depth headroom — RULED OUT

`FHEController.cpp` bumps `levelsUsedBeforeBootstrap` (the circuit's per-bootstrap depth budget)
incrementally across its history (12→13→14→15) chasing intermittent decode failures.

**Test**: bumped it once more (15→16), rebuilt, and did a **full crypto context + key regeneration**
(new `circuit_depth = 32`, ~7.4GB key material) to test whether more headroom reduces compression or
fixes the deterministic crash.

| Test | Before (Lv. 15) | After (Lv. 16) |
|---|---|---|
| Dune FHE logits | `[0.6118, -0.5403]` | `[0.6118, -0.6013]` — effectively unchanged |
| "This movie was fantastic" crash | fails 3/3 attempts, deterministic | **still fails 3/3, identical failure point** |

No measurable effect on either symptom. Also independently confirmed via the fideslib source
(`/root/FIDESlib/api/CryptoContext.cpp:1719-1744`) that on the GPU execution path,
`EvalBootstrap`'s `numIterations`/`precision` arguments are never even passed through to the underlying
`FIDESlib::CKKS::Bootstrap` call — so the `bootstrap(c, precision, timing)` overload's `precision`
argument used throughout `main.cpp` (always called with `0`) is a dead parameter on this build.

## What actually explains the compression and the crash

See `04-attention-signal-collapse.md`: the real cause is that the self-attention mechanism's output is
being reduced to near-zero magnitude by the encrypted computation, at both encoder layers, for every
input tested. This single finding explains the compression, the misclassification pattern on
positive-sentiment inputs, and is a strong candidate explanation for the input-dependent decode failures
too (though that link wasn't separately confirmed).
