# FHE-BERT-Tiny Security Assessment

**Client/server trust-boundary separation, and an empirical chosen-plaintext (CPA) and
chosen-ciphertext (CCA) analysis of the resulting system.**

| | |
|---|---|
| Scheme | CKKS (FIDESlib GPU backend over OpenFHE 1.5.1) |
| Target | BERT-tiny / SST-2 sentiment classifier, 2 encoder layers |
| Branch | `incremental-updates` |
| Hardware | NVIDIA RTX A5000 |
| Scope | `src/`, `security_tests/`, the client/server trust boundary, CPA and CCA threat models |

An interactive version of this report, with charts, is published at:
https://claude.ai/code/artifact/998438d9-dc7e-43d7-b296-cb6af4303d7d

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Architecture: from one role to two](#2-architecture-from-one-role-to-two)
3. [Findings from the separation](#3-findings-from-the-separation)
4. [Chosen-plaintext (CPA) analysis](#4-chosen-plaintext-cpa-analysis)
5. [Chosen-ciphertext / malleability (CCA) analysis](#5-chosen-ciphertext--malleability-cca-analysis)
6. [Risk summary](#6-risk-summary)
7. [Recommendations](#7-recommendations)
8. [Methodology, reproduction, and scope limitations](#8-methodology-reproduction-and-scope-limitations)

---

## 1. Executive summary

FHE-BERT-Tiny ran client and server logic — tokenization, encryption, the homomorphic BERT
circuit, and decryption — as one undifferentiated code path inside a single `main()`, sharing
one `FHEController` object that held both the public encryption key and the client's secret key
for its entire lifetime. That structure was split along an explicit trust boundary:

- a **server role** that never has access to the secret key or the client's plaintext, and
- a **client role** that owns both, and only ever calls the server with already-encrypted
  ciphertexts.

Building and validating that split surfaced **three real defects in the original code**
(§3.1–3.3) and **one new defect introduced by the split itself**, caught by the test harness
before it shipped (§3.4).

With the boundary in place, two empirical questions were tested against the real GPU circuit and
a real decryption path:

- **Does anything the server can observe about a request leak the plaintext content?**
  (chosen-plaintext analysis, §4) — timing, ciphertext counts, and retry behavior were measured
  across 40 real end-to-end FHE circuit evaluations.
- **What happens if a decryption capability is ever exposed to ciphertexts an attacker can
  influence?** (chosen-ciphertext analysis, §5) — tested directly against the real CKKS context,
  150 trials.

**Headline results:**

| Question | Result |
|---|---|
| Is the server-role controller able to decrypt anything? | **No** — verified structurally (no code path) and at runtime (guarded, throws) |
| Does ciphertext count leak sentence length? | **Yes, exactly** — 1 ciphertext per token, always |
| Is there a content-dependent timing side channel? | **No convincing evidence** found at this sample size (n=40) |
| If a decryption oracle is ever exposed, can an attacker recover plaintext? | **Yes, exactly** — 100% success, 150/150 trials, recovery error ~1.2×10⁻¹³ |

---

## 2. Architecture: from one role to two

The circuit itself (2 BERT-tiny encoder layers, pooler, classifier — all evaluated
homomorphically) is unchanged. What moved is **who can call what**.

| Concern | Before | After |
|---|---|---|
| **Key material** | One `FHEController`; `load_context()` deserialized `secret-key.txt` unconditionally, before any evaluation began. | `load_context_server()` never opens `secret-key.txt`. `load_client_secret_key()` is a separate call, made only after the server-role code has already returned a result ciphertext. |
| **Input handling** | `encoder1()` — the first stage of the "server" circuit — read the client's plaintext token-embedding files off disk and encrypted them itself. | `client_main.cpp` reads and encrypts the embeddings; the server-side `encoder1()` in `ServerCircuit.cpp` only ever receives an already-encrypted `vector<Ctxt>`. |
| **Decryption surface** | `decrypt()`/`decrypt_tovector()` plus four debug helpers (`print*`) all called `Decrypt()` directly, reachable from circuit code under `--verbose`. | All six paths refuse to run unless a secret key was explicitly loaded; the four debug helpers were removed from server-side circuit code entirely. |
| **Server observability** | Ad hoc `cout` lines gated on `--verbose`. | Structured JSONL to `logs/server_processing.jsonl` — one line per stage, with timing, ciphertext level, and retry/exception events. Never a decrypted value. |

**New source layout:**

```
src/
  client_main.cpp      CLIENT-SIDE driver: args, tokenization call, encryption,
                        secret-key loading, decryption, output.
  ServerCircuit.h/.cpp  SERVER-SIDE circuit: encoder1, encoder2, pooler, classifier.
                        Takes an FHEController that was loaded via load_context_server()
                        only. Never calls decrypt()/decrypt_tovector().
  ServerLog.h           Structured JSONL logger used only by ServerCircuit.cpp.
  FHEController.h/.cpp  Shared crypto/circuit-op wrapper (unchanged API surface except
                        for the key-loading and decrypt-guard changes below).
```

### Why still one OS process

Genuine network/process separation — the client serializing a ciphertext, sending it to a
separate server process, and getting a serialized ciphertext back — is the natural end state,
but **FIDESlib's GPU-resident `Ciphertext` type does not currently support serialization**:
`FHEController::save()`, `load_ciphertext()`, and `load_vector()` are stubs that call `exit(1)`
(`src/FHEController.cpp:1376-1400`).

So the boundary enforced here is a strict **in-process** one: the secret key is not merely
permission-gated, it is **not deserialized into memory at all** until after the server-role code
has finished running. This eliminates the entire window during which server-side circuit
evaluation coexists in memory with the client's secret key — which is exactly the window the
original code got wrong (§3.1).

Implementing ciphertext (de)serialization is the natural next step for genuine network
separation. FIDESlib's `CiphertextImpl` already keeps a lazily-synced OpenFHE-native CPU copy
internally (`EnsureLazyCPUCopy()`, `api/Ciphertext.cpp`) that looks like the natural hook — see
§7, recommendation 2.

---

## 3. Findings from the separation

Three of these were latent in the code before this work started — the split simply made them
visible and fixable. The fourth was introduced by the split itself, and was only caught because
the CPA test harness ran 40 real end-to-end requests rather than a handful of manual smoke tests.

### 3.1 — Critical (fixed): secret key reachable from server-role code

**The circuit-evaluating controller held the decryption key for its entire life.**

`load_context()` deserialized `secret-key.txt` into the same `key_pair` used to run the
homomorphic circuit — before the first homomorphic operation executed, and for the whole process
lifetime after. The local variable holding it was even named `serverSecretKey`. The circuit
itself never called `Decrypt()`, so nothing was decrypted in practice — but the *capability* sat
in server-role memory the entire time, one bug or one added debug line away from being used.

```diff
--- src/FHEController.cpp (before)
- fideslib::PrivateKey<fideslib::DCRTPoly> serverSecretKey;
- fideslib::Serial::DeserializeFromFile(".../secret-key.txt", serverSecretKey, ...);
- key_pair.secretKey = serverSecretKey;  // loaded before ANY evaluation runs
  ...

+++ src/FHEController.cpp (after)
+ void FHEController::load_context_server(bool verbose) {
+     // secret-key.txt is deliberately never opened here.
+ }
+ void FHEController::load_client_secret_key(bool verbose) {
+     // called only AFTER server-side circuit evaluation has returned
+     ...
+     key_pair.secretKey = clientSecretKey;
+     secret_key_loaded = true;
+ }
```

`decrypt()`/`decrypt_tovector()` now also throw `std::runtime_error` if called before
`secret_key_loaded` is `true` — a second, independent enforcement layer on top of the key simply
not existing in memory yet.

### 3.2 — High (fixed): server code read the client's plaintext directly off disk

**`encoder1()` encrypted the client's input itself.**

The first stage of what is nominally the server's circuit read
`input_folder + "input_N.txt"` — the client's raw token embeddings, written to disk in plaintext
by `ExtractEmbeddings.py` — and encrypted them in place. In this single-process program that
distinction didn't change any output, but it meant "server" code had a direct file-read path to
unencrypted client data, and any future refactor toward a real separate server process would have
silently carried that read along with it.

```diff
--- src/main.cpp (before)
- vector<Ctxt> encoder1() {                 // "server" circuit
-     for (int i = 0; i < inputs_count; i++)
-         inputs.push_back(controller.read_expanded_input(
-             input_folder + "input_" + to_string(i) + ".txt"));
  ...

+++ src/client_main.cpp (after) — CLIENT-SIDE only
+ encrypted_inputs.push_back(controller.read_expanded_input(
+     input_folder + "input_" + to_string(i) + ".txt"));

+++ src/ServerCircuit.cpp (after) — encoder1() now takes ciphertexts as a parameter
+ vector<Ctxt> encoder1(FHEController& controller, vector<Ctxt> inputs,
+                        bool verbose, ServerLog& log) { ... }
```

### 3.3 — High (fixed): four undocumented decryption call sites inside the circuit

**`print()`, `print_padded()`, `print_expanded()`, `print_min_max()` all decrypt directly.**

Beyond the primary `decrypt()`/`decrypt_tovector()` API, four debug helpers call
`context->Decrypt()` directly, and were invoked throughout the encoder/pooler/classifier code
under `--verbose` to print intermediate ciphertext values for precision debugging. This was only
*discovered* because fixing 3.1 made it a crash: with the secret key genuinely absent from the
server-role controller, the first such call segfaulted on a null key rather than silently
succeeding. That crash is what surfaced this finding — the debug path had been quietly decrypting
inside "server" code the entire time finding 3.1 existed.

```diff
--- src/FHEController.cpp
  void FHEController::print(const Ctxt &c, int slots, string prefix) {
+     if (!secret_key_loaded) throw std::runtime_error(
+         "print(): no secret key loaded -- server-role code must not decrypt.");
      ...
      context->Decrypt(key_pair.secretKey, c_mut, &result);

--- src/ServerCircuit.cpp
  if (verbose) cout << "The evaluation of Self-Output took: ..." << endl;
- if (verbose) controller.print_expanded(output[0], 0, 128, "Self-Output (Expanded)");
+ // Removed: server-role code must never decrypt, even for --verbose debugging.
  log.stage_end("encoder1.self_output", log_start, {{"level", ...}});
```

Server-side `--verbose` output now reports ciphertext *level* and stage timing only — the same
information that now also goes to the structured JSONL log — never a decrypted value.

### 3.4 — Medium/reliability (fixed): introduced by the split, caught by the test harness

**Decode failures are only observable by the client, so only the client can retry them.**

The original code's retry loop wrapped encryption, evaluation, *and* decryption in one
`try`/`catch`, because all three ran as one role. CKKS's approximate decode can fail
("approximation error is too high") in a way that is **only detectable at `Decode()` time** — a
ciphertext can pass every server-side step, including the final bootstrap, without throwing, and
still fail to decode. Moving decryption to the client (correctly, per 3.1) took it out of the
server's retry loop with nothing replacing it: an early version of the split let that exception
propagate uncaught, aborting the process. **Four of the first five runs of the real CPA test
harness crashed this way** before it was caught and fixed.

```diff
--- src/client_main.cpp (before)
- plain_result = controller.decrypt_tovector(eval_result.output, 2);  // uncaught → abort
  ...

+++ src/client_main.cpp (after)
+ try {
+     plain_result = controller.decrypt_tovector(eval_result.output, 2);
+     break;
+ } catch (const lbcrypto::OpenFHEException &e) {
+     // Only the client can see this failure. Ask the server to recompute
+     // the whole circuit from scratch (up to 3 client-level attempts).
+     continue;
+ }
```

After the fix, **all 40 CPA-harness requests completed successfully** — 25 on the first
client-level attempt, 14 needed a second, 1 needed a third (§4). This is flagged as a reliability
finding, not a confidentiality one — but a system that silently aborts on roughly 1 request in 8
(4/5 in the pre-fix sample) is itself worth knowing about before it ships.

---

## 4. Chosen-plaintext (CPA) analysis

**Question:** does anything a server-side observer can see about a request — without ever
decrypting anything — leak information about the plaintext?

**Method:** 40 real end-to-end requests were sent through the built client/server binary
(`./build/FHE-BERT-Tiny`) with a corpus of length-matched positive/negative sentence pairs across
three length buckets:

- **short** ≈ 2–4 words (e.g. `"Good movie"` / `"Bad movie"`)
- **medium** ≈ 7–9 words
- **long** ≈ 13–17 words

Pairing on length by swapping only the sentiment-bearing word(s) isolates *content* effects from
the trivial effect of length. For each request, everything a network observer could plausibly see
without decrypting anything was recorded — wall-clock time, per-stage timings and ciphertext
levels from the server's JSONL log, retry counts, and the number of ciphertexts submitted —
alongside the sentence itself (known only to the experimenters, to check for correlation).

Full corpus and harness: `security_tests/run_cpa_test.py`. Raw results:
`security_tests/cpa_results.csv` (40 rows). Statistical tests: `security_tests/analyze_security.py`.

### 4.1 Structural leak: request size exactly reveals token count

This needs no statistics: the architecture sends one ciphertext per input token, so the
ciphertext count in a request *is* the token count, exactly, every time.

| Sentence | Words | Ciphertexts observed |
|---|---:|---:|
| Good movie | 2 | 4 |
| I loved it | 3 | 5 |
| A brilliant heartwarming story I really enjoyed | 7 | 11 |
| I absolutely loved every single minute of this wonderfully crafted and deeply moving film | 14 | 17 |

Distinct ciphertext counts observed across the corpus: `4, 5, 9, 10, 11, 15, 16, 17, 18` — each
one pinpointing a distinct sentence length. A fixed-size or padded-batch wire format (padding
every request to, say, 32 token slots) would close this specific channel — see §7.

### 4.2 Retry behavior: real but modest, and length-correlated only through cost

| | 1 attempt | 2 attempts | 3 attempts |
|---|---:|---:|---:|
| **Server-side** attempts per successful request (n=40) | 36 (90%) | 4 (10%) | — |
| **Client-side** attempts per successful request (n=40) | 25 (62.5%) | 14 (35%) | 1 (2.5%) |

Server-side retries (4/40, 10%) come from the pre-existing intermittent GPU-bootstrap noise
blow-up documented in the code as a likely CUDA kernel race condition. Client-side retries (15/40
needed >1) are the decode-time failures from finding 3.4. Neither showed a relationship to
sentiment in this sample — retries did track length loosely, which follows from longer circuits
simply running more bootstrap operations, each an independent chance to hit the noise issue.

### 4.3 Content dependence: is timing different for positive vs. negative sentences?

Total wall time (all attempts summed) per length-matched pair, tested two ways: a paired t-test
(parametric) and a Wilcoxon signed-rank test (non-parametric, more robust to outliers with small
n).

| Bucket | Pairs | Positive mean | Negative mean | Paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| **Short** | 5 | 3.8 s | 5.2 s | **0.025** | 0.125 |
| Medium | 5 | 7.6 s | 7.0 s | 0.945 | 0.875 |
| Long | 4 | 8.0 s | 8.75 s | 0.749 | 1.0 |

Per-bucket raw values (seconds, one per pair):

```
short   positive: [3.0, 3.0, 3.0, 7.0, 3.0]      short   negative: [5.0, 5.0, 5.0, 8.0, 3.0]
medium  positive: [11.0, 11.0, 5.0, 4.0, 7.0]     medium  negative: [8.0, 5.0, 8.0, 7.0, 7.0]
long    positive: [7.0, 7.0, 11.0, 7.0]           long    negative: [7.0, 11.0, 6.0, 11.0]
```

**Read on this cautiously.** The `short`-bucket result is the kind of finding that looks alarming
in isolation and falls apart under a second, more robust test on the same data — exactly what
happened here. With only 5 pairs, whole-second timing resolution (the client's own top-level
timer truncates to integer seconds), and no plausible mechanism in the code that would make
bootstrap-retry probability content-dependent (it's described in the source as a GPU kernel race
condition, not a data-dependent branch), the most likely explanation is that one or two pairs
happened to need an extra retry by chance.

**This assessment did not find convincing evidence of a content-dependent timing side channel** —
but n=40 real GPU runs (each 3–12 s of actual FHE computation, 35–55 s wall time including Python
tokenization and context loading) is a small sample by side-channel-research standards, and a
larger, finer-grained run is the natural way to either confirm or rule out the short-bucket
signal with confidence (§7, recommendation 4).

### 4.4 Timing noise floor (for calibration)

Repeated runs (n≥3) of the same fixed sentence within a bucket, used to gauge measurement noise
against the content-dependence comparisons above:

| Bucket | n | Mean | Stdev | CV | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Short | 5 | 3.0 s | 0.0 | 0.0 | 3.0 s | 3.0 s |
| Medium | 5 | 7.4 s | 2.51 | 0.34 | 5.0 s | 11.0 s |
| Long | 5 | 7.0 s | 0.0 | 0.0 | 7.0 s | 7.0 s |

The medium bucket's variance is entirely attributable to occasional retries pushing the whole
request to a higher whole-second bucket, not fine-grained timing drift.

---

## 5. Chosen-ciphertext / malleability (CCA) analysis

**Question:** what happens if a decryption capability is ever exposed to ciphertexts an attacker
can influence?

This system's architecture (§2–3) means the client never decrypts anything except its own
request's final result — **no decryption oracle is exposed today.** The question tested here is
what happens *if* one ever is: by an added debug endpoint, a verbose error message, a status code
that varies with decrypted content, or any future feature that decrypts something an attacker had
a hand in shaping.

**Method:** a standalone tool (`security_tests/cca_probe_main.cpp`, built as its own CMake
target, `cca-probe`) ran 150 trials of two textbook experiments directly against the real CKKS
context (public + evaluation keys loaded the same way the server role loads them; the secret key
is loaded only to play the role of the oracle being probed — see the file's header comment).
Results: `security_tests/cca_probe_results.csv`.

### 5.1 Experiment 1 — plaintext recovery via additive malleability

CKKS ciphertexts are additively homomorphic and carry no integrity tag. For a ciphertext
`c = Enc(s)` an attacker never held the key for, and any attacker-chosen `Δ`:

```
c' = c + Enc(Δ)          — computable with the public key alone
Dec(c') = s + Δ          — if any oracle will decrypt c'
```

subtracting back out the known `Δ` recovers `s` exactly.

| Metric (150 trials, 16-slot random vectors) | Value |
|---|---:|
| Max L∞ recovery error | 1.17 × 10⁻¹³ |
| Mean L∞ recovery error | 5.87 × 10⁻¹⁴ |
| Mean L2 recovery error | 1.10 × 10⁻¹³ |

The recovered value matched the true (unknown-to-the-oracle-caller) secret to within CKKS's own
floating-point precision floor — i.e. the "attack" is limited only by arithmetic precision, not
by anything cryptographic.

### 5.2 Experiment 2 — IND-CCA1-style distinguishing game

A hidden bit `b` selects one of two fixed messages; the attacker receives `Enc(m_b)` and one
oracle query against a *derived* ciphertext (never the challenge itself, to also cover an oracle
that special-cases "don't decrypt exactly what you handed me"), and must guess `b`. Chance is 50%.

| Metric (150 trials) | Value |
|---|---:|
| Correct guesses | 150 / 150 |
| Accuracy | **100%** |
| Chance accuracy | 50% |
| Binomial p-value vs. chance | 7.0 × 10⁻⁴⁶ |

### 5.3 Interpretation

**This is not a bug in FHE-BERT-Tiny's code** — it's the expected, well-documented behavior of
unauthenticated CKKS (see e.g. Li & Micciancio, *On the Security of Homomorphic Encryption on
Approximate Numbers*, EUROCRYPT 2021, which extends this class of result to full secret-key
recovery given a passive decryption oracle). It is included here because the architectural work
in §2–3 is precisely what keeps this system on the safe side of it today: **zero** decryption
oracles are currently reachable by anything but the legitimate client decrypting its own result.
The severity rating in §6 reflects the blast radius *if that ever changes*, not a
currently-exploitable path.

---

## 6. Risk summary

| # | Finding | Class | Severity | Status |
|---|---|---|---|---|
| 3.1 | Secret key reachable from server-role controller | Confidentiality | **Critical** | Fixed |
| 5 | No ciphertext integrity — exact recovery if any decryption oracle is exposed | Confidentiality | **Critical (conditional)** | Inherent to CKKS |
| 3.2 | Server code read client plaintext embeddings directly | Confidentiality | High | Fixed |
| 3.3 | Four undocumented decrypting debug helpers in circuit code | Confidentiality | High | Fixed |
| 4.1 | Ciphertext count exactly reveals input token count | Metadata leak | Medium | Open |
| 3.4 | Client had no retry on decode-time failure (~4/5 of an early sample crashed) | Availability | Medium | Fixed |
| 2 | No real network/process separation (FIDESlib ciphertext serialization gap) | Architecture | Medium | Open |
| 4.3 | Short-sentence timing difference (positive vs. negative) | Possible side channel | Inconclusive | Needs larger N |

---

## 7. Recommendations

1. **Never let anything decrypt a ciphertext an attacker (or an untrusted party) could have
   influenced.** §5's result is total and immediate the moment that line is crossed — there is no
   partial version of this mitigation. Treat `decrypt()`/`decrypt_tovector()` as a single audited
   call site (it now is exactly one, in `client_main.cpp`) and review any future change that adds
   another.

2. **Implement ciphertext (de)serialization in FIDESlib** and split the client and server into
   genuinely separate processes communicating over a real transport. The lazily-synced
   OpenFHE-native CPU copy already inside `CiphertextImpl` (`EnsureLazyCPUCopy()`) is the natural
   starting point. This closes the remaining in-process trust assumption and is a prerequisite
   for any real network deployment.

3. **Pad requests to a fixed token-slot count** (or a small set of size buckets) to remove the
   exact-token-count leak in §4.1. The cost is bounded extra computation on short inputs, which
   this codebase already has GPU headroom for relative to its current per-request cost.

4. **Re-run the CPA timing study at larger scale** to resolve the short-bucket borderline result
   in §4.3 — more pairs, sub-second server-side timing (the JSONL log already has millisecond
   resolution per stage; only the client's own top-level `timing` variable truncates to whole
   seconds), and a pre-registered analysis to avoid multiple-comparisons inflation across three
   buckets.

5. **Keep the server-side JSONL log as the durable audit trail** it now is
   (`logs/server_processing.jsonl`) — it made every finding in §3–4 possible to state precisely,
   and costs nothing it isn't already paying for in verbose-mode timing prints.

---

## 8. Methodology, reproduction, and scope limitations

Every number in this report comes from the real binary and the real CKKS context on the hardware
listed above — nothing here is simulated or estimated.

```bash
# build (from the repo root)
cmake -S . -B build -G Ninja && cmake --build build

# CPA: 40 length-matched positive/negative runs through the real client/server binary
./src/python/.venv/bin/python3 security_tests/run_cpa_test.py --repeats 1

# CCA: 150-trial malleability + IND-CCA1 distinguishing-game probe
./build/cca-probe 150

# statistics (paired t-test, Wilcoxon, exact binomial test)
./src/python/.venv/bin/python3 security_tests/analyze_security.py
```

Example of the structured server log this produces (one JSONL line per stage):

```json
{"request_id":"…-c1","t_ms":7413,"type":"event","stage":"pooler.start"}
{"request_id":"…-c1","t_ms":7506,"type":"event","stage":"pooler.end","elapsed_ms":92,"level":27}
{"request_id":"…-c1","t_ms":7645,"type":"event","stage":"request.completed","attempts":1,"output_level":18}
```

**Scope.** This assessment covers the client/server code boundary and the two threat models named
in the brief (chosen-plaintext, chosen-ciphertext). It does **not**:

- attempt the full secret-key-recovery attack from a passive decryption oracle (only the
  plaintext-recovery and distinguishing variants, which are sufficient to establish the same
  severity — see §5.3 for the citation covering the stronger result);
- cover the Python tokenization path, the CUDA kernels themselves, or FIDESlib's internals beyond
  the serialization gap noted in §2;
- constitute a production-scale side-channel study — the CPA timing analysis (§4) is bounded by
  what 40 real multi-second GPU circuit evaluations could cover in this session.

---

*FHE-BERT-Tiny Security Assessment · CKKS / FIDESlib GPU backend · branch `incremental-updates`*

🤖 Generated with [Claude Code](https://claude.com/claude-code)
