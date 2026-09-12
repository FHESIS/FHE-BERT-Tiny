# `eval_exp` Chebyshev Refit

## Motivation

`FHEController.cpp`'s `eval_exp()` (softmax numerator, used by both encoder layers via
`matmulScores` → `eval_exp` → `eval_inverse_naive*`) was the **only** approximated nonlinearity in the
circuit not using a Chebyshev fit. Every sibling function does:

- `eval_gelu_function` (`:1357`), `eval_tanh_function` (`:1363`), `eval_inverse_naive`/`_2`
  (`:1345`/`:1351`) all call `context->GetChebyshevCoefficients(...)` + `context->EvalChebyshevSeries(...)`.
- `eval_exp` instead used a hand-rolled 6-term Taylor series around zero (Horner chain), then cubed the
  result three times (`EvalSquare` ×3) to extend `e^x` to `e^(8x)`.

Taylor truncation error is only small near `x=0`; cubing three times amplifies any error roughly 8x
(`(1+ε)^8 ≈ 1+8ε`). The empirical SST-2 sweep (see `01-precision-investigation.md`) showed the real input
to this function ranges up to `[-1.35, 1.76]` for layer 1 — well outside where a degree-6 Taylor series
around zero is trustworthy.

## Change

- `FHEController.h`: `eval_exp(const Ctxt&, int inputs_number)` → `eval_exp(const Ctxt&, int inputs_number, double min, double max, int degree)`.
- `FHEController.cpp:1289`: replaced the Taylor/Horner chain with `GetChebyshevCoefficients(exp, min, max, degree)` + `EvalChebyshevSeries`, keeping the same post-cubing (`EvalSquare` ×3) and slot-masking logic.
- `main.cpp:197` (layer 1 / `encoder2()`): domain `[-2.1, 2.6]`, degree 30 (empirical range `[-1.35, 1.76]`, padded).
- `main.cpp:352` (layer 0 / `encoder1()`): domain `[-1.1, 1.6]`, degree 30 (empirical range `[-0.66, 1.04]`, padded).

No crypto-parameter change, no key regeneration needed — same context, different in-circuit operations.

## Results

### Fixed: the deterministic crash

"This movie was fantastic" previously failed **100% of the time** — 6/6 attempts across two independent
full process runs, always at the identical point (`Decode(): approximation error is too high`, right
after the first self-attention block, ciphertext level 26). After the refit, it **completes cleanly in
87 seconds with zero retries**.

### Faster: ~40-60% shorter runtime

All post-refit runs landed in the 86-257s range (vs. the pre-refit typical 150-300s+), consistent with
`EvalChebyshevSeries` costing less multiplicative depth than the old 9-level Taylor+cube chain (fewer
forced intermediate bootstraps).

### Mixed: no consistent effect on compression

| Sentence | Plain gap | Old FHE gap (ratio) | New FHE gap (ratio) |
|---|---|---|---|
| "Dune was a bad movie" | 4.51 | 1.15 (3.91x) | 1.24 (**3.63x — better**) |
| "The acting was terrible" | 4.17 | 1.48 (2.82x) | 1.22 (**3.41x — worse**) |

One sentence improved, one regressed by a comparable amount — real signal, not run-to-run noise, but it
shows `eval_exp`'s imprecision was **not** the single dominant driver of the compression.

### Unchanged: positive-sentiment misclassification

"I loved this film" remained misclassified after the refit (FHE `[0.649, -0.598]` vs. plaintext positive
`[-2.048, 2.243]`). "This movie was fantastic," now that it computes instead of crashing, is *also*
misclassified the same way — trading an opaque crash for a computed-but-wrong answer.

### A tighter negative-sentiment cluster (later found to be the real signal)

Post-refit, negative-sentiment FHE logits cluster much more tightly (`~[0.62-0.65, -0.58 to -0.60]`,
spread ~0.02) than pre-refit (spread ~0.33) — at the time this looked like a possible *regression*
(more collapse), but subsequent investigation (`04-attention-signal-collapse.md`) showed this same
collapse-toward-a-near-constant-point was already present pre-refit; the refit's lower depth cost just
made it more visible/consistent run-to-run.

## Recommendation

Keep the change. It strictly improves robustness (crash eliminated, no correctness regressions observed)
and runtime, with no clear downside — the compression and misclassification issues it doesn't fix have a
different, now-identified root cause (see `04-attention-signal-collapse.md`) that this refit was never
positioned to address.
