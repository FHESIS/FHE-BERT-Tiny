# FHE-BERT-Tiny Performance Investigation & Fixes

## Summary

Investigated why encrypted (FHE) inference was taking ~166-172s per sentence and
applied a set of incremental, verified fixes that bring it down to **~41-50s per
sentence (~3.5-4x speedup)**, with no change in output correctness.

## Baseline (before changes)

| Sentence | Plaintext | Encrypted | Result |
|---|---|---|---|
| "This was a complete waste of my time and money" | 9.4s | 168.0s | correct |
| "The plot was boring and the acting felt wooden" | 9.4s | 165.9s | correct |
| "This is one of the best performances I have seen all year" | 9.4s | 171.7s | correct |
| "A brilliant, funny, and touching film that exceeded my expectations" | 9.4s | 224.0s | **crashed** (SIGABRT after 3 retries) |

## Root cause

`CMakeLists.txt` never set `CMAKE_BUILD_TYPE`. With no build type, CMake applies
no optimization flags at all (effectively `-O0`) to `main.cpp` and
`FHEController.cpp` — the host-side orchestration code that builds plaintext
masks, encodes/decodes CKKS values, and drives every GPU call. This code runs
on the CPU around each (fast, GPU-accelerated) bootstrap/rotate/multiply, so an
unoptimized build meaningfully slows down the whole pipeline.

Confirmed via `build/CMakeFiles/FHE-BERT-Tiny.dir/flags.make`:
```
CXX_FLAGS = -fopenmp -std=gnu++20    # before: no -O flag at all
```

## Changes made

1. **`CMakeLists.txt`** — default `CMAKE_BUILD_TYPE` to `Release` (`-O3
   -DNDEBUG`) when the caller doesn't specify one. This was the dominant fix.

2. **`src/FHEController.cpp`** — removed unconditional debug `cout << ... <<
   endl` statements that fired on *every* call to `add`, `mult`, `rotsum`,
   `matmulRE`, `matmulCR`, `wrapUpRepeated`, and `eval_exp` (regardless of the
   existing `verbose` flag). Each `endl` forces a stream flush; these ran
   hundreds of times per inference.

3. **`src/FHEController.h` / `.cpp`** — added a small cache
   (`mask_ptxt_cache`) for the plaintext masks built by `mask_block`,
   `mask_heads`, `mask_mod_n`, and `mask_first_n`. These functions were
   rebuilding and re-NTT-encoding an identical 16384-element plaintext on
   every call, including inside per-token loops (e.g. `unwrapExpanded` calls
   `mask_mod_n` with the exact same arguments once per token). The cache is
   keyed by `(kind, params..., ciphertext level, mask_value)`. This reuse
   pattern is safe because the codebase already relies on reusing the same
   `Ptxt` object across many `mult()` calls (e.g. weight/bias plaintexts
   reused across all rows in `matmulRE`), which only works if `mult()` never
   mutates its plaintext argument in place — confirmed true here as well.

4. **Regression found and fixed** — caching the mask plaintexts as long-lived
   members of the global `FHEController controller` object introduced an
   intermittent **segfault at process exit**. Root-caused with `cuda-gdb`:

   ```
   FHEController::~FHEController()
     -> ~map<..., Ptxt>()               // destroying mask_ptxt_cache
       -> ~shared_ptr<PlaintextImpl>()
         -> CryptoContextImpl::EvictDevicePlaintext()
           -> FIDESlib::GPUfree()
             -> FIDESlib::Stream::init()   // SIGSEGV
   ```

   The global `controller` object is constructed before `main()` runs, but
   CUDA's own context/stream pool is lazily initialized *inside* `main()` (on
   first CUDA call) and registers its teardown via `atexit`. Because
   `atexit`/global-destructor order is LIFO by registration time, CUDA's
   stream-pool teardown — registered *after* `controller` was constructed —
   runs *before* `controller`'s destructor at process exit. Once the mask
   cache started holding onto GPU-backed plaintexts until that point,
   destroying them tried to free GPU memory through an already-torn-down
   stream pool.

   Fix: added `FHEController::clear_mask_cache()` and call it explicitly at
   every `main()` return path, *before* the program reaches the risky
   teardown window (i.e. while CUDA is still known-alive). Verified with 5/5
   clean runs on an input that previously crashed 3/3 times, plus a full
   benchmark re-run with no segfaults.

## Results (after changes)

| Sentence | Plaintext | Encrypted | Result |
|---|---|---|---|
| "This was a complete waste of my time and money" | 9.3s | 41.4s | correct |
| "The plot was boring and the acting felt wooden" | 9.4s | 48.3s (1 retry) | correct |
| "This is one of the best performances I have seen all year" | 9.5s | 50.4s | correct |
| "A brilliant, funny, and touching film that exceeded my expectations" | 9.4s | 61.0s | crashed (SIGABRT after 3 retries — see below) |

Plaintext logits are byte-identical to baseline in every case (Python code
untouched). Encrypted logits match baseline to within normal CKKS noise, with
identical predicted sentiment on every completed run.

## Known pre-existing issue (not fixed, out of scope)

One sentence intermittently fails with `SIGABRT` after 3 retries, both before
and after these changes. This is a documented, pre-existing bug: an
intermittent GPU race condition in FIDESlib's CUDA bootstrap kernels
(commented extensively in `src/main.cpp` around the retry loop). It is
data-dependent and unrelated to the performance changes in this report — it
reproduced in the baseline run before any code was touched.

## Files changed

- `CMakeLists.txt`
- `src/main.cpp`
- `src/FHEController.h`
- `src/FHEController.cpp`

Changes are currently uncommitted in the working tree.
