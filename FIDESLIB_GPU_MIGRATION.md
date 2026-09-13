# Porting FHE-BERT-Tiny to GPU via FIDESlib

This report documents the measures taken to build and run FHE-BERT-Tiny on top of
[FIDESlib](https://github.com/) — a GPU-accelerated CKKS backend — in place of the original
CPU-only OpenFHE circuit. It summarizes the diff on branch `gpu-build` relative to the
pre-GPU baseline.

## 1. Build system

- **CMake requirement raised** from 3.5.1 to 3.25.2, and the C++ standard bumped from 17 to 20
  (`CMakeLists.txt`), both required by FIDESlib.
- **Dependency swap**: `find_package(OpenFHE)` was replaced with
  `find_package(fideslib REQUIRED CONFIG)`. The executable now links
  `fideslib::fideslib` and `CUDA::cudart` instead of raw OpenFHE shared/static libraries.
- **Dual header dependency**: `FHEController` still calls a few raw `lbcrypto::` symbols that
  FIDESlib's wrapper API doesn't expose (e.g. `FHECKKSRNS::GetBootstrapDepth`). Because of this,
  the system-wide OpenFHE headers that FIDESlib itself was built and linked against
  (`/usr/local/include/openfhe`, patched OpenFHE 1.5.1) are added to the include path alongside
  FIDESlib's own headers.
- **New `build.sh`** wraps `cmake -S/-B` + `cmake --build` (with `--clean` and `--jobs`
  options) and documents a required run-time workaround: `LD_LIBRARY_PATH` must include
  `/usr/lib/x86_64-linux-gnu` because `ldconfig` otherwise resolves `libcuda.so.1` to a broken
  shim on this machine.

## 2. API migration (OpenFHE → FIDESlib namespace)

`src/FHEController.h` and `src/FHEController.cpp` were migrated from OpenFHE's top-level
`lbcrypto` types to FIDESlib's own wrapper types:

- `Ptxt`/`Ctxt` aliases changed from `Plaintext`/`Ciphertext<DCRTPoly>` to
  `fideslib::Plaintext`/`fideslib::Ciphertext<fideslib::DCRTPoly>`.
- `CryptoContext`, `KeyPair`, `CCParams<CryptoContextCKKSRNS>`, `SerType`, `Serial`, and the
  scheme-enable flags (`PKE`, `KEYSWITCH`, `LEVELEDSHE`, `ADVANCEDSHE`, `FHE`) are all now
  referenced through the `fideslib::` namespace.
- `context->SetDevices({0})` is now called after context creation to pin the context to GPU 0.
- Key-switch technique explicitly forced to `HYBRID` via `SetKeySwitchTechnique`.
- Several `FHEController` methods now copy `const Ctxt&`/`const Ptxt&` parameters into a local
  mutable variable (`c_mut`, `p_mut`) before calling `Decrypt`/`EvalAdd`/`EvalMult`, since
  FIDESlib's equivalents take non-const references where OpenFHE accepted `const&`.
- `EvalChebyshevFunction` (used for ReLU, GELU, tanh, and the two "inverse" approximations) is
  not exposed by FIDESlib. Every call site was rewritten as the two-step
  `GetChebyshevCoefficients(...)` + `EvalChebyshevSeries(...)`.
- `EvalPoly` (Paterson–Stockmeyer polynomial evaluation) has no FIDESlib equivalent either.
  `eval_exp` (Taylor series for `e^x`) was rewritten as an explicit Horner chain of
  `EvalMult`/`EvalAdd` calls, followed by 3 `EvalSquare` calls to reach `x^8`.
- `OPENFHE_THROW(config_error, "...")` (two-argument form) was updated to the
  single-argument `OPENFHE_THROW("...")` signature (`Utils.h`).
- Several `for (int i = 0; i < vec.size(); i++)` loops were changed to `size_t i` (or an
  explicit `static_cast<int>(vec.size())` comparison) to fix signed/unsigned comparison
  warnings/errors surfaced by the stricter C++20 build.

## 3. Context / key lifecycle changes

FIDESlib's `CryptoContext` has different lifecycle requirements than OpenFHE's factory-keyed
context, which required restructuring `generate_context`, `generate_bootstrapping_and_rotation_keys`,
`load_context`, and `load_bootstrapping_and_rotation_keys`:

- **`LoadContext()` must be the last precomputation step.** It pushes all generated
  material to the GPU; any further `EvalRotateKeyGen`/`EvalBootstrapSetup` call after it throws
  "Context is already loaded". `LoadContext()` calls were therefore moved to the very end of key
  generation (`generate_bootstrapping_and_rotation_keys`) and key loading
  (`load_bootstrapping_and_rotation_keys`), instead of being called immediately after context
  creation.
- **Serialization order changed.** Under FIDESlib, a crypto-context serialized before every key
  (mult, bootstrap, rotation) exists on it silently "forgets" which rotation indices are
  registered once deserialized — `EvalRotate` then fails with "Rotation index N not found" even
  though the automorphism key file itself deserializes fine. Serialization of
  `crypto-context.txt`/`public-key.txt`/`secret-key.txt` was moved out of `generate_context()`
  and is now deferred (via a new `serialize_context_pending` flag) until the end of
  `generate_bootstrapping_and_rotation_keys()`, after rotation and bootstrap keys are also
  in place.
- **No factory-keyed context registry.** `clear_bootstrapping_and_rotation_keys` /
  `clear_rotation_keys` / `load_context`'s old `ClearEvalMultKeys` / `ClearEvalAutomorphismKeys`
  / `CryptoContextFactory::ReleaseAllContexts()` calls were removed or turned into no-ops:
  FIDESlib's `CryptoContext` is a plain per-object `shared_ptr`, not an entry in a global
  factory-keyed cache the way OpenFHE's is.
- **Ciphertext (de)serialization is unsupported.** FIDESlib's `Serial` wrapper only covers the
  `CryptoContext` and key pair, not individual ciphertexts. The previously-used
  checkpoint path (`save`/`load_vector`/`load_ciphertext` writing
  `../checkpoint/*.bin`) is not exercised by the current circuit; those functions were changed
  to fail loudly (`cerr` + `exit(1)`) rather than silently no-op, and all call sites that wrote
  checkpoints in `main.cpp` were removed.

## 4. Parameter / precision tuning

- `level_budget` (CKKS bootstrap CtoS/StoC budget) raised from `{3, 3}` to `{4, 4}`.
- Scaling technique switched from `FLEXIBLEAUTO` to `FLEXIBLEAUTOEXT`, which reserves an extra
  modulus to preserve precision through the final decode — added to fix an
  "approximation error is too high" failure at the last bootstrap+decrypt step.
- `levelsUsedBeforeBootstrap` raised from 12 to 15 (first `generate_context` overload) / 14
  (second overload), in three separate increments, each documented inline with the observed
  failure that motivated it:
  1. **+1** — FIDESlib's `eval_exp` uses a Horner chain (no `EvalPoly`/Paterson–Stockmeyer),
     which costs one more multiplicative level than OpenFHE's version, so a bootstrap call was
     landing exactly at `circuit_depth` with zero headroom (crash confirmed via debug print:
     `c->GetLevel() == circuit_depth == 26`).
  2. **+1** — the final Meta-BTS bootstrap in `main()` (after `classifier()`) was still landing
     exactly at `circuit_depth` with zero headroom, causing `Decode()` to throw (confirmed via
     debug print: `classified->GetLevel() == circuit_depth == 29` right before that bootstrap
     call, decrypting fine at that point).
  3. **+1** — a specific input ("This movie was fantastic") still hit the same error at the same
     final bootstrap+decode step even though other inputs decoded fine, indicating the error is
     data-dependent and headroom was still occasionally insufficient. It was confirmed that
     FIDESlib's GPU `EvalBootstrap` ignores `numIterations`/`precision` entirely
     (`api/CryptoContext.cpp`), so the only available fix was more `circuit_depth` headroom via a
     full context/key regeneration.
- `generate_rotation_keys`'s rotation index list in `main.cpp` was extended with
  `-128, -256, -512` (previously stopped at `-64`).

## 5. Reliability workaround: retry on intermittent bootstrap noise

`main.cpp` wraps the entire circuit evaluation (encoder1 → encoder2 → pooler → classifier →
final Meta-BTS bootstrap → decrypt) in a retry loop (`max_attempts = 3`), catching
`lbcrypto::OpenFHEException`. This was added after observing (via instrumenting OpenFHE's
`Decode()`) that the *identical* circuit on the *identical* input decoded fine with ~+23 bits of
precision on some runs and failed ("approximation error is too high") with ~-4 bits on others —
a ~27-bit swing pointing to an intermittent bug (most likely a race condition in FIDESlib's CUDA
bootstrap kernels) rather than a per-input or per-parameter precision shortfall. Retrying
re-rolls that kernel's execution and empirically succeeds within a couple of attempts, though it
is not guaranteed (see §6).

A final Meta-BTS bootstrap (`numIterations = 2`, via `controller.bootstrap(x, 0, verbose)`) was
also added at several points in the circuit (pooler, self-attention scores, self-output,
intermediate, and immediately before the final decode in `main`) for extra decode precision.

## 6. Observed results (`TIMING_COMPARISON.md`)

A manual comparison of `PlainCircuit.py` vs. the GPU `FHE-BERT-Tiny` binary across 5 sentences:

| Outcome | Count |
|---|---|
| Matched plaintext sentiment | 4/5 |
| Required 1 retry due to noise | 2/5 |
| Failed after exhausting all 3 retries | 1/5 |

- Encrypted inference took 37–78s per sentence (vs. sub-second true plaintext compute, once
  fixed model-load overhead is excluded).
- All successful runs agreed with plaintext on predicted sentiment class; encrypted logits are
  consistently compressed relative to plaintext, expected given the circuit's polynomial
  approximations of softmax/GELU.
- The one hard failure exhausted all 3 retries with `"The decryption failed because the
  approximation error is too high. Check the parameters."`, confirming the retry-loop
  workaround in §5 is a mitigation, not a fix — the underlying bootstrap noise issue can still
  fail user-facing requests and may warrant further parameter tuning or a FIDESlib-side fix.

## 7. Misc application-level changes

- `main.cpp`: `IDE_MODE` default flipped from `true` to `false` (CLI-argument mode is now the
  default); relative paths changed from `../...` to `./...` throughout (binary is now expected
  to be run from the repo root, not from `build/`); `system(...)` calls replaced with a
  `run_command()` helper that logs a warning on non-zero exit instead of ignoring it silently.
- `src/python/ExtractEmbeddings.py` / `PlainCircuit.py`: switched from
  `AutoModelForSequenceClassification` to `BertForSequenceClassification` with
  `strict=False` state-dict loading, and the tokenizer source changed to `bert-base-uncased`
  (tokenizer only — the fine-tuned bert-tiny weights are still loaded separately).
- New `src/python/PlainCircuitSimplified.py`: a variant plaintext reference implementation using
  hardcoded per-position precomputed LayerNorm mean/variance constants (`PrecomputedLayerNorm`),
  for a more direct equivalence check against the FHE circuit's own precomputed-LayerNorm
  approximation.
- Several unrelated exploratory scripts added under `src/python/` (`bert-tiny.py`,
  `meta-llama.py`, `numpy-linear-reg.py`, `pytorch.py`, `vanilla-bert.py`) — not part of the
  GPU migration itself.
- `.gitignore` updated to exclude the HF model cache, the Python virtualenv, `*.log`, `*.patch`.
- Two reference PDFs added under `paper/` (original BERT and a second paper, presumably the
  FIDESlib paper or a related reference).

## Open items

- Debug `cout` tracing added throughout `FHEController.cpp` (`"Calling EvalAdd..."`,
  `"MatMulRE ..."`, `"Rotate Sum ..."`, etc.) is unconditional, not gated behind `verbose` —
  worth cleaning up or gating before this is considered production-ready.
- The intermittent GPU-bootstrap noise issue (§5/§6) is only mitigated, not fixed; a ~1/5 hard
  failure rate after exhausting retries remains.
- Ciphertext checkpointing (`save`/`load_vector`/`load_ciphertext`) has no FIDESlib-backed
  implementation; if a checkpointing workflow is needed again, it will require either a custom
  serialization format or an upstream FIDESlib feature.
