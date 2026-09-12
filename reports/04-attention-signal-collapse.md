# Root Cause: Self-Attention Signal Collapse

## Summary

The encrypted circuit's self-attention mechanism produces near-zero output at **both** encoder layers,
for **every** input tested. This single finding explains the logit compression, the systematic
misclassification of every positive-sentiment sentence tested, and why the encrypted circuit's output
clusters so tightly around one point almost regardless of input. It was confirmed four independent ways.

## Background: why this was suspected

Earlier hypotheses (fixed Chebyshev/Taylor domains, fixed LayerNorm statistics, CKKS depth headroom —
see `01-precision-investigation.md`) were each tested and ruled out or downgraded to minor factors. But
one pattern kept recurring: across every positive-sentiment sentence tested (7 completed runs,
`03-experiment-results.md`), the FHE output logits clustered in an extremely narrow band
(`~[0.61-0.65, -0.58 to -0.60]`, spread ~0.02) even though the underlying plaintext signal strength
varied by 40%+ across those same sentences. That's the signature of the network losing access to the
actual input, not of diffuse rounding noise (which would show variation either direction).

## Evidence 1: the raw self-attention output is ~100-1000x too small

Ran `build/FHE-BERT-Tiny --verbose` on two sentences — one correctly-classified negative ("Dune was a
bad movie") and one misclassified positive ("I loved this film") — and captured the decrypted
"Self-Attention (Repeated)" ciphertext at both encoder layers (printed via
`controller.print(output[0], 128, "Self-Attention (Repeated)")`, `main.cpp:223`/`:367`).

A parallel plaintext script (`plain_checkpoints.py`, mirrors `PlainCircuit.py`'s fixed-LayerNorm math)
computed the same checkpoint — the CLS-token row of the post-softmax, post-value-weighting attention
output, before the output-dense projection.

| | Plaintext (real signal) | FHE (decrypted) |
|---|---|---|
| Layer 0, magnitude range | `[-2.50, 2.42]` | `~[-0.0015, -0.0010]` |
| Layer 1, magnitude range | `[-3.59, 3.79]` | `~[-0.02, 0.02]` |

The FHE-decrypted self-attention output is **100-1000x smaller** than the plaintext reference at both
layers, for both sentences.

## Evidence 2: two different sentences converge to a near-identical downstream vector

The next checkpoint, "Self-Output" (`main.cpp:261`/`:398`), equals
`attention_output_projected + residual(original_embedding) + LayerNorm`. If the attention term is
negligible, this stage should collapse toward the residual pass-through of the `[CLS]` token's own
embedding — which is **identical for every input**, since `[CLS]` is always the same token.

Compared the decrypted Self-Output vectors for "Dune was a bad movie" and "I loved this film" — two
sentences with opposite true sentiment and very different words:

```
mean absolute difference:  0.00077
typical magnitude:         0.957
relative difference:       0.08%
```

Two semantically opposite sentences produce a Self-Output vector that differs by **less than a tenth of
a percent**. This is only possible if the attention mechanism's contribution to that vector is
negligible.

## Evidence 2b: the pooler output — where the discriminative signal should live — collapses too

The pooler output (`tanh(dense(...))`, "Pooler (Repeated)", `main.cpp:176`) is the last checkpoint before
the classifier, and in the plaintext reference it's where sentiment is most visibly encoded: tanh
saturates almost every one of the 128 dimensions to ±1, and *which* dimensions land +1 vs. -1 differs
sharply between opposite-sentiment sentences.

| Comparison | Sign agreement | Correlation |
|---|---|---|
| Plaintext "Dune" vs. plaintext "I loved this film" | 24.2% | **-0.577** (properly anti-correlated — real signal) |
| FHE "Dune" vs. FHE "I loved this film" | **99.2%** | **0.999** (essentially the same vector) |
| FHE "Dune" vs. its own correct plaintext answer | 75.0% | — |
| FHE "I loved this film" vs. its own correct plaintext answer | **32.8%** | — (worse than chance) |

Two sentences engineered to be opposite in meaning are only 24.2% sign-agreeing in the correct
(plaintext) representation — as expected for a real, working discriminative signal — but **99.2%
sign-agreeing in the encrypted circuit**, i.e. functionally the same vector. And "I loved this film"'s
FHE pooler output agrees with its own *correct* answer's sign pattern only 32.8% of the time — worse
than a coin flip — meaning it's not merely losing signal, it's specifically drifting toward the wrong
(negative) pattern. This is the single clearest number in the investigation: the stage that should carry
the sentence's sentiment is instead carrying almost the same value for every sentence.

## Evidence 3: causal confirmation via a zero-attention plaintext simulation

Built a plaintext forward pass (same fixed-LayerNorm math as `PlainCircuit.py`) with the self-attention
output forced to exactly zero at both encoder layers, for both test sentences, and ran it through the
rest of the network unmodified (output-dense, GELU, second LayerNorm, pooler tanh, classifier).

```
"Dune was a bad movie"   -> [0.2957, -0.0449]
"I loved this film"      -> [0.2957, -0.0449]   (bit-for-bit identical)
```

This reproduces, by construction, the two defining properties of the observed FHE collapse:

1. **Sentence-independence** — both sentences produce identical output, since with zero attention
   contribution the model can only see the `[CLS]` token's own fixed embedding.
2. **Negative-leaning direction** — `logit[0] > logit[1]` (negative), matching the direction every
   observed FHE collapse leaned toward.

The magnitude differs from the real FHE collapse point (`[0.30, -0.04]` simulated vs. `~[0.6, -0.6]`
observed) because the real circuit doesn't have *exactly* zero attention, and other approximations
(GELU, tanh, CKKS rescale noise) contribute additional distortion on top — but the direction and the
mechanism are confirmed.

## Why the collapse point specifically leans negative

Not a designed behavior. This particular trained BERT-tiny SST-2 model's `[CLS]` token embedding,
pushed through its trained FFN/LayerNorm/pooler/classifier weights with no other-token context, happens
to land on the negative side of the decision boundary. It's an incidental property of this specific set
of trained weights, not something intrinsic to attention loss in general. A differently-fine-tuned model
could just as easily collapse toward "positive."

## Not introduced by this session's `eval_exp` change

Checked the original (pre-refit) `--verbose` trace captured earlier in this investigation, before any
source changes: it shows the identical near-zero, block-structured self-attention output pattern. This
is a pre-existing characteristic of the circuit, not a regression from the Chebyshev refit
(`02-eval-exp-chebyshev-refit.md`).

## Where the magnitude is likely being lost (not yet localized further)

The self-attention output is computed via:

1. `matmulScores` (`FHEController.cpp:1056`) — raw `Q·Kᵀ` dot products, scaled by 1/64 via `mask_heads`
2. `eval_exp` (`:1289`) — softmax numerator
3. `eval_inverse_naive`/`_2` (`:1345`/`:1351`) — softmax denominator
4. `matmulRE(unwrapped_scores, V_wrapped, 128, 128)` (the ciphertext-weight overload, `:950`) — the
   actual `scores · V` weighted-sum aggregation, using `mult` + `rotsum` (`:875`, a rotate-and-add
   reduction over 128-slot-aligned blocks)

Any of these four steps could be where the ~100-1000x attenuation originates — this was not narrowed
down further before this report was written. The most direct next diagnostic would be decrypting the
`scores` ciphertext immediately after softmax normalization (before step 4) to check whether the
softmax weights themselves are already near-zero/miscalibrated, or whether the loss happens specifically
in the final `scores · V` aggregation.

## Practical implications

- The ~3-4x logit compression, the positive-sentiment misclassification (7/7 in this investigation), and
  likely at least some of the input-dependent decode failures are downstream symptoms of this one root
  cause, not independent, diffusely-distributed CKKS noise as earlier assumed.
- Fixing softmax/attention-value aggregation specifically is a far more targeted, higher-leverage next
  step than further tuning polynomial domains, LayerNorm statistics, or circuit depth headroom — all of
  which were tested and found not to be the bottleneck (see `01-precision-investigation.md`).
