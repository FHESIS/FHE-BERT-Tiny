# Plaintext Implementation — Optimization Techniques

## Scope

"Plaintext implementation" here means `src/python/PlainCircuit.py` (the
reference implementation the FHE circuit in `src/ServerCircuit.cpp` is
checked against) and its unused variant `src/python/PlainCircuitSimplified.py`.
Profiled with `cProfile` and manual timers on this machine (CPU, no HF token
configured), invoking `PlainCircuit.py` the same way `run_benchmark.py` /
`run_sentiment_comparison.py` do.

## Headline finding: 99% of measured "plaintext inference time" is process startup, not inference

`run_plaintext()` in both benchmark scripts shells out to a **fresh Python
subprocess per sentence** (`subprocess.run(["python3", "PlainCircuit.py", sentence])`).
The real 50-sentence run in `sentiment_comparison_results.json` recorded:

| | value |
|---|---|
| total plaintext time (50 sentences) | 528.3s |
| mean per sentence | 10.57s |

Breaking down one invocation with manual timers:

| Stage | Time | Fixed cost per process? |
|---|---|---|
| `import transformers`, `import torch` | ~7.1s | yes |
| `AutoTokenizer.from_pretrained("bert-base-uncased")` | ~0.8s | yes (network round-trip, see below) |
| `BertForSequenceClassification.from_pretrained(...)` + `load_state_dict` | ~0.8s | yes |
| **actual forward pass (the thing being measured)** | **~0.14s** | no — this is the only part that scales with input |

So of the ~10.5s charged to "plaintext inference" per sentence, **~10.4s is
identical, repeated process-startup overhead** and only ~0.14s is the model
actually running. Across the 50-sentence benchmark that's ~520s spent
re-importing `transformers`/`torch` and reloading identical weights 50 times,
versus ~7s if paid once.

**Fix:** move the loop into the process instead of the process into the loop
— load the tokenizer/model once, then classify each sentence in a plain
Python `for` loop (or expose a small `--jsonl`/stdin-per-line mode in
`PlainCircuit.py` that the benchmark scripts feed sentences into over one
subprocess). Expected effect: the 50-sentence plaintext benchmark drops from
~528s to roughly `7s (import) + 50 × 0.14s (forward) ≈ 14s` — a **~38x**
reduction in wall-clock time for that benchmark, with zero change to the
logits it reports (this only removes redundant setup, not compute).

This is the dominant win by nearly two orders of magnitude over everything
below; the rest of this report only matters once it's applied.

## Secondary findings (in the script itself)

1. **Unused tokenizer call.** `PlainCircuit.py:21` — `tokenized = tokenizer(text)`
   — builds a full tokenizer encoding (`input_ids`/`attention_mask`/etc.) that
   is never read; the code re-tokenizes on the next line via
   `tokenizer.tokenize(text)` instead. Dead work on every invocation; delete
   line 21.

2. **Network round-trip for a tokenizer that never changes.**
   `AutoTokenizer.from_pretrained("bert-base-uncased")` hits the HF Hub for
   an unauthenticated etag check every run (visible in the profile as 1.3s in
   `_ssl.SSLSocket.read`, and as the "sending unauthenticated requests to the
   HF Hub" warning). Setting `HF_HUB_OFFLINE=1` (once the tokenizer is cached
   locally, which it already is) cuts `AutoTokenizer.from_pretrained` from
   ~0.78s to ~0.10s per process. Minor once finding #1 is applied (paid once
   instead of 50 times), but free to fix and combines with it.

3. **Per-token Python loops instead of vectorized ops.** `PlainCircuit.py`
   applies each of the four (precomputed-mean/var) LayerNorms with a Python
   `for i in range(seq_len)` loop that re-fetches `...LayerNorm.weight` /
   `.bias` from the model on every iteration (loop-invariant — same tensor
   every time), does a single-row op, and rebuilds the sequence with
   `torch.cat`. This pattern is repeated 4 times (once per LayerNorm across
   the 2 encoder layers), ~55-110 lines of near-identical code.
   `PlainCircuitSimplified.py` already contains the fix as a `PrecomputedLayerNorm`
   `nn.Module` (lines 39-65): index the per-position mean/var arrays with
   `torch.arange(seq_len).clamp(...)` and apply them to the whole `[1, seq_len, 128]`
   tensor in one broadcasted expression — no Python-level loop, no
   `torch.cat`, no repeated attribute lookups. Given the sequence lengths
   here (tens of tokens), the wall-clock effect is small, but it removes ~130
   lines of duplicated, harder-to-audit code and is a strictly better
   drop-in replacement.

4. **`float64` throughout.** Both scripts run entirely in double precision
   (`.double()` on every tensor) to match the FHE circuit's numerics for
   precision comparisons. That's a deliberate, correct choice for
   `Precision.py`-style validation runs, but it's needless for a plain
   "what's the sentiment" call — `float32` roughly doubles matmul throughput
   at this model size, though the effect is negligible (<10ms) relative to
   the fixed costs in finding #1, so only worth doing if a `float32` fast
   path is added deliberately for non-validation callers.

## Not changed / out of scope

- `ServerCircuit.cpp` (the encrypted circuit) mirrors this same math but
  under CKKS; none of the above generalizes to it — the encrypted side's
  bottlenecks (bootstrap count, GPU kernel scheduling, per-request plaintext
  re-encoding) were already addressed separately in `PERFORMANCE_REPORT.md`.
- Model/tokenizer loading logic itself (`bert-base-uncased` tokenizer paired
  with a `prajjwal1/bert-tiny` model) is an existing, intentional modeling
  choice per `FIDESLIB_GPU_MIGRATION.md` — not touched here.

## Recommended priority

| # | Change | Effort | Impact |
|---|---|---|---|
| 1 | Load model/tokenizer once; loop sentences in-process (or batch-mode CLI) | small | ~38x on any multi-sentence benchmark |
| 2 | `HF_HUB_OFFLINE=1` (or `local_files_only=True`) | trivial | ~0.7s/process, folds into #1 |
| 3 | Delete unused `tokenizer(text)` call | trivial | negligible but free |
| 4 | Adopt `PrecomputedLayerNorm`-style vectorization from `PlainCircuitSimplified.py` | small | code quality; negligible wall-clock at this seq length |
