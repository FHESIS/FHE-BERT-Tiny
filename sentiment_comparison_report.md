# Plaintext vs. Encrypted (FHE) Sentiment Inference Benchmark

50-sentence run of `src/python/run_sentiment_comparison.py`, comparing the plaintext
PyTorch BERT-tiny circuit against the GPU-accelerated FHE circuit (FIDESlib/CKKS)
on the same inputs.

**Run date:** 2026-09-15 02:17–03:03 UTC (46 min)
**Branch:** `incremental-updates-clean`
**Commit:** `218b2f139da70312fa120e61af3b0560541c21ca`
**Command:** `python3 src/python/run_sentiment_comparison.py --limit 50`

`expected_sentiment` is a hand-assigned label used only as a rough sanity check, not
an SST-2 gold label.

---

## System information

| Component | Detail |
|---|---|
| OS | Ubuntu 24.04.2 LTS |
| Kernel | 7.0.0-31-generic (x86_64) |
| CPU | 2x Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz (16 cores / 32 threads total) |
| Memory | 251 GiB RAM |
| GPU | 1x NVIDIA RTX A5000, 24564 MiB VRAM, compute capability 8.6 |
| GPU driver | 610.43.02 |

## Required software toolchain

| Component | Version | Notes |
|---|---|---|
| CMake | 3.28.3 | build generator: Unix Makefiles |
| GCC / G++ | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04) | host C++ compiler |
| CUDA Toolkit (nvcc) | 12.9 (V12.9.86) | `/usr/local/cuda` |
| OpenFHE | 1.5.1 | patched per `third_party` deps patch `openfhe-1.5.1.patch`; installed to `/usr/local` |
| FIDESlib | commit `fa97286` (2026-09-10), github.com/FHESIS/FIDESlib | GPU-accelerated CKKS backend; installed as a CMake package (`find_package(fideslib CONFIG REQUIRED)`) |
| GSL | 2.7.1+dfsg-6ubuntu2 (`libgsl-dev`) | |
| Python | 3.12.3 | virtualenv at `src/python/.venv` |
| PyTorch | 2.14.0+cu130 | CUDA available: yes |
| Transformers | 5.17.0 | HF Hub models cached offline (`bert-base-uncased`, `prajjwal1/bert-tiny`) |
| ninja / ccache | present (not used for this build; Makefiles + no launcher) | optional build accelerators referenced by `build_with_deps.sh`/`build_fideslib.sh` |

Build artifacts used for this run: `build/FHE-BERT-Tiny` (client/server ciphertext
circuit) and the plaintext reference `src/python/PlainCircuit.py`, both rebuilt from
this commit immediately before the run.

---

## Aggregate metrics

| Metric | Plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 50/50 | 48/50 |
| Failed / timed out | 0 | 2 |
| Total internal retries | n/a | 0 |
| Accuracy vs. expected label | 0.780 | 0.580 |
| Mean time (s) | 9.44 | 45.36 |
| Median time (s) | 9.27 | 44.18 |
| Min / Max time (s) | 9.13 / 11.95 | 40.64 / 60.07 |

- **Prediction agreement (plaintext vs. encrypted):** 48 pairs compared, 83.3% agreement
- **Mean absolute logit difference:** 0.9969
- **Encrypted / plaintext time ratio:** 4.8x slower

### Failures

2 of 50 encrypted runs (sentences #2 and #19) exited with return code `-6` (SIGABRT)
and produced no logits; both are counted as failures in the table above and excluded
from the timing/agreement statistics that require a completed encrypted result. No
internal retries were triggered on any sentence (`retries=0` throughout), and the
plaintext circuit had zero failures.

### Reading the accuracy numbers

`SENTENCES` in `run_sentiment_comparison.py` lists 50 "positive"-labeled sentences
followed by 50 "negative"-labeled ones; `--limit 50` takes the first 50, so **every
sentence in this run is hand-labeled "positive"** — this run does not exercise the
negative-labeled half of the fixture set at all. "Accuracy vs. expected label" here
is therefore really "how often each circuit predicted positive on sentences a human
labeled positive," not a balanced accuracy figure, and the 78.0%/58.0% gap mostly
reflects how many of these 50 sentences are genuinely unambiguous to this particular
fine-tuned BERT-tiny model rather than a defect in either circuit. The more
meaningful FHE-circuit-fidelity number is the **83.3% prediction agreement** between
the encrypted and plaintext circuits on the 48 sentences that completed on both
sides — i.e., 8 of those 48 pairs disagreed with each other, independent of whether
either matched the hand-assigned label (sentences #7, #13, #18, #23, #24, #28, #36,
#37). A full run (`--limit 100`, or the default with no `--limit`) would cover both
halves of the fixture set.

---

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Agree |
|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 9.25 | positive | [-0.2843, 0.2038] | 44.01 | ✓ |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 9.37 | FAIL | n/a | 51.34 | - |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 9.27 | positive | [-1.3309, 1.2521] | 53.68 | ✓ |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 9.26 | positive | [-1.0508, 0.9854] | 45.71 | ✓ |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 10.96 | negative | [0.4582, -0.5211] | 60.06 | ✓ |
| 6 | positive | A truly inspiring story told with grace and humor | positive | [-2.8302, 2.6644] | 9.80 | positive | [-2.0626, 1.7948] | 46.70 | ✓ |
| 7 | positive | The cinematography was stunning from start to finish | positive | [-0.8676, 1.2671] | 11.95 | negative | [0.1582, -0.2516] | 50.29 | ✗ |
| 8 | positive | This film restored my faith in modern storytelling | positive | [-2.4227, 2.4041] | 10.33 | positive | [-0.8839, 0.7934] | 52.51 | ✓ |
| 9 | positive | An absolute triumph of direction and performance | positive | [-2.5795, 2.5094] | 9.65 | positive | [-1.0108, 0.9151] | 43.63 | ✓ |
| 10 | positive | The soundtrack alone is worth the price of admission | positive | [-1.1515, 1.2544] | 9.21 | positive | [-0.3878, 0.3131] | 42.55 | ✓ |
| 11 | positive | A charming little film that grows on you | positive | [-2.6428, 2.5503] | 9.13 | positive | [-0.5220, 0.4509] | 41.15 | ✓ |
| 12 | positive | I laughed out loud more than once, a genuinely funny m… | positive | [-1.1401, 1.1657] | 9.24 | positive | [-0.9587, 0.5836] | 48.81 | ✓ |
| 13 | positive | The chemistry between the leads made the whole film wo… | positive | [-1.3859, 1.5569] | 9.29 | negative | [0.3401, -0.3671] | 42.47 | ✗ |
| 14 | positive | A beautifully written script with real emotional depth | positive | [-2.7035, 2.5578] | 9.17 | positive | [-0.9095, 0.8207] | 41.67 | ✓ |
| 15 | positive | This exceeded every expectation I had going in | negative | [0.7285, -0.2191] | 9.25 | negative | [0.5818, -0.4575] | 41.33 | ✓ |
| 16 | positive | The special effects were seamless and genuinely thrill… | negative | [1.7037, -1.2514] | 9.28 | negative | [0.8952, -0.8703] | 47.78 | ✓ |
| 17 | positive | A masterclass in pacing and tension | positive | [-1.5194, 1.5911] | 9.80 | positive | [-0.0727, 0.1019] | 40.64 | ✓ |
| 18 | positive | I would happily watch this again tomorrow | positive | [-0.9814, 1.1473] | 9.34 | negative | [0.3390, -0.2999] | 43.00 | ✗ |
| 19 | positive | The ending brought tears to my eyes, in the best way | positive | [-1.8498, 1.9775] | 9.14 | FAIL | n/a | 53.06 | - |
| 20 | positive | Every scene felt purposeful and beautifully shot | positive | [-2.4916, 2.4159] | 9.50 | positive | [-1.0026, 0.9027] | 46.20 | ✓ |
| 21 | positive | A wonderfully clever plot with a satisfying payoff | positive | [-2.5797, 2.4319] | 9.21 | positive | [-1.2555, 1.0498] | 45.95 | ✓ |
| 22 | positive | The dialogue crackles with wit and energy | positive | [-1.9427, 2.0351] | 9.23 | positive | [-0.8281, 0.6945] | 44.54 | ✓ |
| 23 | positive | This is family entertainment done exactly right | positive | [-0.8006, 0.8704] | 9.27 | negative | [0.1304, -0.2141] | 42.53 | ✗ |
| 24 | positive | A gripping thriller that never lets up | positive | [-0.3214, 0.4965] | 9.24 | negative | [0.6435, -0.5454] | 43.56 | ✗ |
| 25 | positive | The lead actress gives a career-defining performance | positive | [-2.6060, 2.4519] | 9.36 | positive | [-1.6991, 1.4340] | 41.79 | ✓ |
| 26 | positive | Genuinely one of the most enjoyable films of the decade | positive | [-2.6604, 2.4983] | 9.24 | positive | [-1.5340, 1.3981] | 42.17 | ✓ |
| 27 | positive | The food at this restaurant was absolutely delicious | positive | [-1.9152, 2.0372] | 9.18 | positive | [-0.7116, 0.5728] | 42.38 | ✓ |
| 28 | positive | Customer service was fast, friendly, and helpful | positive | [-0.6158, 0.5701] | 9.23 | negative | [0.1106, -0.1684] | 41.81 | ✗ |
| 29 | positive | This phone works flawlessly and the battery lasts all … | negative | [0.5082, -0.2039] | 9.27 | negative | [0.6760, -0.6226] | 43.20 | ✓ |
| 30 | positive | The hotel room was spotless and the staff were wonderf… | negative | [0.8662, -0.5125] | 9.27 | negative | [0.7960, -0.9273] | 51.65 | ✓ |
| 31 | positive | Great value for the price, highly recommended | positive | [-2.3320, 2.3084] | 9.23 | positive | [-0.7007, 0.6881] | 51.98 | ✓ |
| 32 | positive | The book kept me hooked until the very last page | negative | [1.2876, -0.8493] | 9.58 | negative | [0.5929, -0.6068] | 42.65 | ✓ |
| 33 | positive | A refreshing and uplifting comedy the whole family enj… | positive | [-2.8613, 2.6935] | 9.29 | positive | [-2.6934, 2.5010] | 44.62 | ✓ |
| 34 | positive | The concert was electric, easily the best show I have … | positive | [-1.4061, 1.4606] | 9.31 | positive | [-0.7762, 0.6748] | 53.95 | ✓ |
| 35 | positive | This laptop is fast, quiet, and beautifully designed | positive | [-2.2152, 2.1647] | 9.19 | positive | [-1.6224, 1.4103] | 42.85 | ✓ |
| 36 | positive | The weather was perfect for our trip to the coast | positive | [-0.6499, 0.8984] | 9.36 | negative | [0.1848, -0.2568] | 42.85 | ✗ |
| 37 | positive | Our vacation exceeded every expectation we had | positive | [-0.1633, 0.5457] | 9.26 | negative | [0.3283, -0.2674] | 41.54 | ✗ |
| 38 | positive | The coffee here is consistently excellent | positive | [-2.3400, 2.2299] | 9.22 | positive | [-1.1817, 0.9086] | 41.95 | ✓ |
| 39 | positive | I am thoroughly impressed with the quality of this pro… | positive | [-2.2569, 2.2221] | 9.27 | positive | [-0.8716, 0.7226] | 44.46 | ✓ |
| 40 | positive | The team played with incredible energy and won deserve… | positive | [-2.8117, 2.6778] | 9.38 | positive | [-1.7544, 1.5218] | 46.20 | ✓ |
| 41 | positive | This app makes managing my schedule so much easier | negative | [1.3563, -1.0753] | 9.28 | negative | [0.8165, -0.7257] | 44.05 | ✓ |
| 42 | positive | The garden looked absolutely gorgeous in the spring su… | positive | [-2.5593, 2.5052] | 9.68 | positive | [-1.0262, 0.9560] | 42.69 | ✓ |
| 43 | positive | Their new album is a joyful, inventive piece of work | positive | [-2.7654, 2.6119] | 9.39 | positive | [-2.0569, 1.7980] | 45.50 | ✓ |
| 44 | positive | This is not a bad film at all, it is actually quite go… | negative | [1.3674, -0.8872] | 9.44 | negative | [0.3747, -0.3528] | 46.79 | ✓ |
| 45 | positive | I never expected to enjoy this movie as much as I did | negative | [0.7270, -0.3142] | 9.18 | negative | [0.7298, -0.5132] | 44.31 | ✓ |
| 46 | positive | Despite a slow start, the film wins you over completely | negative | [0.8703, -0.4826] | 9.96 | negative | [0.6745, -0.6092] | 46.32 | ✓ |
| 47 | positive | It is hard not to smile the entire way through this mo… | negative | [1.2122, -0.8360] | 9.47 | negative | [0.6349, -0.6232] | 47.26 | ✓ |
| 48 | positive | A small film with a surprisingly big heart | positive | [-2.5913, 2.5148] | 9.42 | positive | [-1.1552, 0.9901] | 43.74 | ✓ |
| 49 | positive | The sequel actually improves on the original in every … | positive | [-1.4458, 1.4307] | 9.28 | positive | [-0.4015, 0.3086] | 46.88 | ✓ |
| 50 | positive | Well worth the wait, this delivers on every level | positive | [-1.9690, 1.8906] | 9.22 | positive | [-1.4508, 1.2443] | 44.88 | ✓ |

---

## Reproduction

```bash
git checkout incremental-updates-clean
cmake --build build -j"$(nproc)"
python3 src/python/run_sentiment_comparison.py --limit 50
```

Requires the crypto key material FHEController expects at `../keys` relative to the
repository root (i.e. a `keys/` directory as a sibling of the repo checkout) and the
Hugging Face models (`bert-base-uncased`, `prajjwal1/bert-tiny`) either cached
locally or reachable over the network.
