# Investigation Reports

Findings from comparing plaintext vs. encrypted (FHE) inference on this BERT-tiny SST-2 circuit.

1. **[01-precision-investigation.md](01-precision-investigation.md)** — the ~3-4x logit compression
   observed on the encrypted circuit; three hypotheses tested (Chebyshev/Taylor domain range, fixed
   LayerNorm statistics, CKKS depth headroom) and each ruled out or downgraded to a minor factor.
2. **[02-eval-exp-chebyshev-refit.md](02-eval-exp-chebyshev-refit.md)** — replaced the softmax
   numerator's ad-hoc Taylor-series approximation with a proper Chebyshev fit. Fixed a deterministic
   crash and improved runtime ~40-60%; mixed, inconclusive effect on compression.
3. **[03-experiment-results.md](03-experiment-results.md)** — raw data tables for every sentence run
   through both the plaintext reference and the encrypted circuit during this investigation, including
   wall-clock timing.
4. **[04-attention-signal-collapse.md](04-attention-signal-collapse.md)** — the actual root cause: the
   self-attention mechanism's output is destroyed to near-zero magnitude in the encrypted circuit,
   explaining the compression and the 7/7 misclassification rate on positive-sentiment inputs.

Read in order for the full narrative, or jump to (4) for the headline finding.
