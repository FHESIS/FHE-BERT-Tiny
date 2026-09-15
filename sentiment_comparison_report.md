# Plaintext vs. Encrypted (FHE) Sentiment Inference Benchmark

Full 100-sentence run of `src/python/run_sentiment_comparison.py`, comparing the
plaintext PyTorch BERT-tiny circuit against the GPU-accelerated FHE circuit
(FIDESlib/CKKS) on the same inputs — the entire fixture set (50 hand-labeled
"positive" sentences followed by 50 hand-labeled "negative" ones).

**Run date:** batch 1 (sentences 1-50) 2026-09-15 02:17-03:03 UTC; batch 2
(sentences 51-100) resumed later, completing 2026-09-15 07:25 UTC (~46 min of
active runtime per batch; the gap between batches is idle time between requests
in this session, not benchmark runtime).
**Branch:** `incremental-updates-clean`
**Commit:** `218b2f139da70312fa120e61af3b0560541c21ca` (batch 1); results updated
on top of that commit for batch 2
**Command:** `python3 src/python/run_sentiment_comparison.py --limit 50`, then
`python3 src/python/run_sentiment_comparison.py --limit 100 --resume` to continue
from sentence 51

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
| Successful runs | 100/100 | 98/100 |
| Failed / timed out | 0 | 2 |
| Total internal retries | n/a | 0 |
| Accuracy vs. expected label | 0.870 | 0.770 |
| Mean time (s) | 9.41 | 45.71 |
| Median time (s) | 9.29 | 44.52 |
| Min / Max time (s) | 9.13 / 11.95 | 40.64 / 60.07 |

- **Prediction agreement (plaintext vs. encrypted):** 98 pairs compared, 91.8% agreement
- **Mean absolute logit difference:** 0.8104
- **Encrypted / plaintext time ratio:** 4.9x slower

### Failures

2 of 100 encrypted runs (sentences #2 and #19, both in the first 50) exited with
return code `-6` (SIGABRT) and produced no logits; both are counted as failures in
the table above and excluded from the timing/agreement statistics that require a
completed encrypted result. All 50 sentences in the second batch (#51-100)
completed without a single failure or retry. No internal retries were triggered on
any sentence (`retries=0` throughout), and the plaintext circuit had zero failures
across all 100 sentences.

### Reading the accuracy numbers

`SENTENCES` in `run_sentiment_comparison.py` lists 50 hand-labeled "positive"
sentences followed by 50 hand-labeled "negative" ones; this report covers both
halves. "Accuracy vs. expected label" is plaintext 87.0% vs. encrypted 77.0% overall
— mostly a reflection of which of these 100 sentences are genuinely unambiguous to
this particular fine-tuned BERT-tiny model, not a defect specific to either circuit.

The more meaningful FHE-circuit-fidelity number is **prediction agreement between
the two circuits themselves**, and it splits sharply by half:

| Half | Pairs compared | Agreement |
|---|---|---|
| Positive-labeled (#1-50) | 48/48 completed | 40/48 = 83.3% |
| Negative-labeled (#51-100) | 50/50 completed | 50/50 = 100% |
| **Overall** | **98/98 completed** | **90/98 = 91.8%** |

All 8 plaintext/encrypted disagreements (sentences #7, #13, #18, #23, #24, #28,
#36, #37) and both encrypted-side crashes (#2, #19) occurred in the positive-labeled
half; the negative-labeled half had perfect agreement between the two circuits on
every sentence that completed. This run is too small to say why (dataset/label
distribution during CKKS parameter tuning, an approximation asymmetry in the
polynomial sign function used for the classifier head, or simple chance at n=50 per
half are all plausible) — it's flagged here as a pattern worth a targeted follow-up
rather than a conclusion.

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
| 51 | negative | This was a complete waste of my time and money | negative | [2.6095, -2.2907] | 9.37 | negative | [1.4936, -1.2857] | 44.20 | ✓ |
| 52 | negative | The plot was boring and the acting felt wooden | negative | [2.6530, -2.4051] | 9.27 | negative | [1.9334, -1.7904] | 46.85 | ✓ |
| 53 | negative | I hated every minute of this dull and pointless movie | negative | [2.4941, -2.1406] | 10.94 | negative | [1.6717, -1.4260] | 49.77 | ✓ |
| 54 | negative | A disappointing mess with no redeeming qualities | negative | [1.9164, -1.5631] | 9.25 | negative | [1.0767, -0.8943] | 41.51 | ✓ |
| 55 | negative | The worst film I have seen in years, truly terrible | negative | [2.1156, -1.6756] | 9.30 | negative | [1.4512, -1.1885] | 43.13 | ✓ |
| 56 | negative | This movie is a tedious, incoherent slog | negative | [2.4688, -2.2177] | 9.30 | negative | [1.6963, -1.4954] | 48.16 | ✓ |
| 57 | negative | The dialogue was cringeworthy and the pacing glacial | negative | [1.9937, -1.6995] | 9.25 | negative | [1.3231, -1.2238] | 41.93 | ✓ |
| 58 | negative | A lazy, uninspired sequel that adds nothing new | negative | [1.5195, -1.1765] | 9.18 | negative | [0.6848, -0.7395] | 42.46 | ✓ |
| 59 | negative | I walked out halfway through, it was that bad | negative | [2.4447, -2.1008] | 9.29 | negative | [1.3259, -1.1386] | 42.16 | ✓ |
| 60 | negative | The special effects looked cheap and unconvincing | negative | [2.4505, -2.0795] | 9.20 | negative | [1.7484, -1.4595] | 42.09 | ✓ |
| 61 | negative | A forgettable film with a plot full of holes | negative | [2.2541, -1.9730] | 9.19 | negative | [1.5070, -1.3695] | 42.32 | ✓ |
| 62 | negative | The lead performance was flat and utterly unconvincing | negative | [1.7935, -1.4376] | 9.21 | negative | [1.0146, -0.8716] | 46.46 | ✓ |
| 63 | negative | This film insults the intelligence of its audience | negative | [2.0121, -1.8932] | 9.28 | negative | [1.5828, -1.3980] | 40.85 | ✓ |
| 64 | negative | A bloated, self-indulgent film that never earns its ru… | negative | [1.2682, -0.8831] | 9.25 | negative | [1.3392, -1.0944] | 54.71 | ✓ |
| 65 | negative | The jokes fell flat and the story made no sense | negative | [2.4912, -2.1281] | 9.51 | negative | [1.5161, -1.3139] | 45.96 | ✓ |
| 66 | negative | An empty, soulless cash grab of a sequel | negative | [2.4072, -2.0668] | 9.40 | negative | [1.3530, -1.2349] | 48.34 | ✓ |
| 67 | negative | This is easily the worst movie of the year | negative | [2.1857, -1.7700] | 9.22 | negative | [1.3266, -1.0661] | 49.16 | ✓ |
| 68 | negative | The editing was choppy and the story dragged endlessly | negative | [2.6185, -2.3664] | 9.33 | negative | [1.8609, -1.6346] | 43.19 | ✓ |
| 69 | negative | A painfully dull film with nothing to say | negative | [2.3163, -1.9617] | 9.34 | negative | [1.4387, -1.1708] | 41.58 | ✓ |
| 70 | negative | The characters were flat and impossible to care about | negative | [2.1442, -1.7778] | 9.25 | negative | [1.3750, -1.1592] | 47.59 | ✓ |
| 71 | negative | A confused mess that never finds its footing | negative | [1.8454, -1.5589] | 9.32 | negative | [1.2055, -0.9595] | 46.00 | ✓ |
| 72 | negative | The score was intrusive and the script painfully clich… | negative | [1.9996, -1.6772] | 9.32 | negative | [1.2282, -1.2139] | 45.46 | ✓ |
| 73 | negative | I regret spending an evening on this dreadful movie | negative | [1.5641, -1.1521] | 9.28 | negative | [1.1659, -0.9879] | 43.56 | ✓ |
| 74 | negative | A shallow, forgettable film with no depth whatsoever | negative | [2.3654, -2.0552] | 9.57 | negative | [1.4766, -1.2389] | 44.81 | ✓ |
| 75 | negative | The pacing kills any tension the story might have had | negative | [2.1681, -1.8853] | 9.46 | negative | [1.6357, -1.3629] | 42.43 | ✓ |
| 76 | negative | This sequel betrays everything that made the original … | positive | [-0.5473, 0.7857] | 9.27 | positive | [-0.1776, 0.1400] | 46.03 | ✓ |
| 77 | negative | The food arrived cold and tasted terrible | negative | [2.0856, -1.6197] | 9.42 | negative | [1.3917, -1.1184] | 42.63 | ✓ |
| 78 | negative | Customer service was rude and unhelpful throughout | negative | [0.2992, -0.0314] | 9.34 | negative | [0.4585, -0.4802] | 44.50 | ✓ |
| 79 | negative | This phone constantly freezes and the battery is awful | negative | [2.3785, -2.1521] | 9.77 | negative | [1.4385, -1.3102] | 44.42 | ✓ |
| 80 | negative | The hotel room was filthy and the staff were dismissive | negative | [2.5200, -2.3004] | 9.40 | negative | [2.0558, -1.8680] | 44.21 | ✓ |
| 81 | negative | Overpriced and poorly made, do not waste your money | negative | [2.5299, -2.3011] | 9.28 | negative | [1.8275, -1.6037] | 43.53 | ✓ |
| 82 | negative | The book was tedious and I never finished it | negative | [1.7480, -1.4630] | 9.21 | negative | [1.4169, -1.1573] | 46.26 | ✓ |
| 83 | negative | A joyless comedy that never once made me laugh | negative | [0.8010, -0.4724] | 9.39 | negative | [0.9943, -0.8093] | 44.41 | ✓ |
| 84 | negative | The concert was a disorganized, disappointing mess | negative | [2.5042, -2.1977] | 9.56 | negative | [1.8989, -1.6550] | 44.28 | ✓ |
| 85 | negative | This laptop overheats constantly and the fan is deafen… | negative | [0.9480, -0.5489] | 9.20 | negative | [0.6162, -0.6094] | 45.71 | ✓ |
| 86 | negative | The weather ruined what could have been a nice trip | negative | [2.1508, -1.8033] | 9.18 | negative | [0.7327, -0.7823] | 44.88 | ✓ |
| 87 | negative | Our vacation was plagued with delays and bad service | negative | [1.9408, -1.5617] | 9.25 | negative | [1.3345, -1.1115] | 51.84 | ✓ |
| 88 | negative | The coffee here tastes burnt and watered down | negative | [2.5050, -2.2400] | 9.27 | negative | [1.8426, -1.5523] | 41.45 | ✓ |
| 89 | negative | I am thoroughly disappointed with the quality of this … | positive | [-0.5694, 0.7459] | 9.28 | positive | [-0.5352, 0.4055] | 44.68 | ✓ |
| 90 | negative | The team played sloppily and lost embarrassingly | negative | [2.1992, -1.8678] | 9.30 | negative | [1.3240, -1.2162] | 47.10 | ✓ |
| 91 | negative | This app crashes constantly and loses my data | negative | [2.2555, -2.0304] | 9.23 | negative | [1.5721, -1.3461] | 46.72 | ✓ |
| 92 | negative | The garden was overgrown and poorly maintained | negative | [2.4761, -2.2948] | 9.18 | negative | [1.7312, -1.4769] | 41.69 | ✓ |
| 93 | negative | Their new album is a tired, uninspired retread | negative | [2.1906, -1.9049] | 9.33 | negative | [1.3295, -1.2223] | 56.36 | ✓ |
| 94 | negative | This is not a good film, it is a genuine chore to watch | negative | [0.4927, -0.1454] | 9.81 | negative | [0.2696, -0.3450] | 53.73 | ✓ |
| 95 | negative | I never expected to dislike this movie as much as I did | negative | [1.2759, -0.9130] | 9.89 | negative | [1.0255, -0.7741] | 55.81 | ✓ |
| 96 | negative | Despite a promising start, the film collapses complete… | negative | [2.2333, -1.8849] | 9.55 | negative | [1.4307, -1.1377] | 48.65 | ✓ |
| 97 | negative | It is hard to sit through this movie without checking … | negative | [1.6083, -1.2773] | 9.58 | negative | [1.2377, -1.0093] | 50.35 | ✓ |
| 98 | negative | A big-budget film with a surprisingly hollow core | negative | [1.6223, -1.3205] | 9.52 | negative | [0.4078, -0.6003] | 49.22 | ✓ |
| 99 | negative | The sequel manages to be worse than the original in ev… | negative | [1.9787, -1.6065] | 9.35 | negative | [0.9647, -0.9145] | 44.04 | ✓ |
| 100 | negative | Not worth the wait, this fails on almost every level | negative | [1.6743, -1.4364] | 9.37 | negative | [1.0294, -0.8576] | 55.23 | ✓ |

---

## Reproduction

```bash
git checkout incremental-updates-clean
cmake --build build -j"$(nproc)"
python3 src/python/run_sentiment_comparison.py
```

Runs all 100 sentences by default (no `--limit` needed); pass `--limit N` for a
smaller smoke test or `--resume` to continue an interrupted `--out-json` file.
Requires the crypto key material FHEController expects at `../keys` relative to the
repository root (i.e. a `keys/` directory as a sibling of the repo checkout) and the
Hugging Face models (`bert-base-uncased`, `prajjwal1/bert-tiny`) either cached
locally or reachable over the network.
