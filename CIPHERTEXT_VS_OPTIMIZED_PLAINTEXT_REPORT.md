# Ciphertext (FHE) vs Optimized Plaintext Circuit — Inference Comparison

Generated: 2026-09-14 13:23:03 UTC
Progress: 50/50 sentences completed

Encrypted circuit: `./build/FHE-BERT-Tiny <sentence> --verbose` (one fresh subprocess per sentence, CKKS evaluation over the trained BERT-tiny/SST-2 weights). Plaintext circuit: the **optimized** `src/python/PlainCircuit.py` (`PlainCircuit.load_model()` called once, `classify()` timed per sentence in the same process — see `PLAINTEXT_OPTIMIZATION_REPORT.md`). Both implement the same precomputed-LayerNorm approximation of BERT-tiny/SST-2; `expected_sentiment` is a hand-assigned sanity-check label, not an SST-2 gold label.

## Aggregate metrics

| Metric | Optimized plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 50/50 | 47/50 |
| Failed / timed out | 0 | 3 |
| Total internal retries | n/a | 0 |
| Accuracy vs expected label | 0.780 | 0.580 |
| Mean time (s) | 0.0259 | 44.325 |
| Median time (s) | 0.0169 | 43.685 |
| Min / Max time (s) | 0.0112 / 0.4597 | 38.286 / 58.319 |

**Prediction agreement (optimized plaintext vs encrypted):** 47 pairs compared, 0.830 agreement rate
**Mean absolute logit difference (avg of both logits):** 0.9801
**Mean absolute diff, logit₀ / logit₁:** 0.9389 / 1.0213
**Max absolute logit difference:** 2.1088
**Encrypted / optimized-plaintext time ratio (slowdown):** 1714.3x

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries | Agree | |Δlogit| avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 0.4597 | positive | [-0.2773, 0.1972] | 41.17 | 0 | ✓ | 1.6932 |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 0.0154 | FAIL | n/a | 52.28 | 0 | - | n/a |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 0.0153 | positive | [-1.3495, 1.2701] | 39.94 | 0 | ✓ | 1.2648 |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 0.0220 | positive | [-1.0526, 0.9871] | 43.31 | 0 | ✓ | 1.6931 |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 0.0178 | negative | [0.4590, -0.5215] | 48.14 | 0 | ✓ | 0.1894 |
| 6 | positive | A truly inspiring story told with grace and humor | positive | [-2.8302, 2.6644] | 0.0168 | positive | [-2.0785, 1.8136] | 42.37 | 0 | ✓ | 0.8013 |
| 7 | positive | The cinematography was stunning from start to finish | positive | [-0.8676, 1.2671] | 0.0192 | negative | [0.1604, -0.2537] | 42.73 | 0 | ✗ | 1.2744 |
| 8 | positive | This film restored my faith in modern storytelling | positive | [-2.4227, 2.4041] | 0.0169 | positive | [-0.9009, 0.8091] | 45.82 | 0 | ✓ | 1.5584 |
| 9 | positive | An absolute triumph of direction and performance | positive | [-2.5795, 2.5094] | 0.0172 | positive | [-1.0162, 0.9203] | 38.29 | 0 | ✓ | 1.5762 |
| 10 | positive | The soundtrack alone is worth the price of admission | positive | [-1.1515, 1.2544] | 0.0210 | positive | [-0.3913, 0.3176] | 42.70 | 0 | ✓ | 0.8485 |
| 11 | positive | A charming little film that grows on you | positive | [-2.6428, 2.5503] | 0.0206 | positive | [-0.5237, 0.4519] | 45.44 | 0 | ✓ | 2.1088 |
| 12 | positive | I laughed out loud more than once, a genuinely funny m… | positive | [-1.1401, 1.1657] | 0.0178 | positive | [-0.9635, 0.5870] | 41.19 | 0 | ✓ | 0.3777 |
| 13 | positive | The chemistry between the leads made the whole film wo… | positive | [-1.3859, 1.5569] | 0.0165 | negative | [0.3393, -0.3667] | 40.42 | 0 | ✗ | 1.8244 |
| 14 | positive | A beautifully written script with real emotional depth | positive | [-2.7035, 2.5578] | 0.0156 | positive | [-0.9100, 0.8213] | 44.29 | 0 | ✓ | 1.7650 |
| 15 | positive | This exceeded every expectation I had going in | negative | [0.7285, -0.2191] | 0.0168 | negative | [0.5802, -0.4564] | 42.97 | 0 | ✓ | 0.1928 |
| 16 | positive | The special effects were seamless and genuinely thrill… | negative | [1.7037, -1.2514] | 0.0167 | FAIL | n/a | 48.10 | 0 | - | n/a |
| 17 | positive | A masterclass in pacing and tension | positive | [-1.5194, 1.5911] | 0.0112 | positive | [-0.0699, 0.0987] | 44.75 | 0 | ✓ | 1.4710 |
| 18 | positive | I would happily watch this again tomorrow | positive | [-0.9814, 1.1473] | 0.0169 | negative | [0.3383, -0.2990] | 41.47 | 0 | ✗ | 1.3830 |
| 19 | positive | The ending brought tears to my eyes, in the best way | positive | [-1.8498, 1.9775] | 0.0168 | positive | [-1.4410, 1.3431] | 44.17 | 0 | ✓ | 0.5216 |
| 20 | positive | Every scene felt purposeful and beautifully shot | positive | [-2.4916, 2.4159] | 0.0117 | FAIL | n/a | 45.47 | 0 | - | n/a |
| 21 | positive | A wonderfully clever plot with a satisfying payoff | positive | [-2.5797, 2.4319] | 0.0169 | positive | [-1.2562, 1.0504] | 40.74 | 0 | ✓ | 1.3525 |
| 22 | positive | The dialogue crackles with wit and energy | positive | [-1.9427, 2.0351] | 0.0218 | positive | [-0.8510, 0.7154] | 44.84 | 0 | ✓ | 1.2057 |
| 23 | positive | This is family entertainment done exactly right | positive | [-0.8006, 0.8704] | 0.0178 | negative | [0.1266, -0.2107] | 38.76 | 0 | ✗ | 1.0042 |
| 24 | positive | A gripping thriller that never lets up | positive | [-0.3214, 0.4965] | 0.0171 | negative | [0.6439, -0.5451] | 43.60 | 0 | ✗ | 1.0034 |
| 25 | positive | The lead actress gives a career-defining performance | positive | [-2.6060, 2.4519] | 0.0168 | positive | [-1.7287, 1.4600] | 46.54 | 0 | ✓ | 0.9346 |
| 26 | positive | Genuinely one of the most enjoyable films of the decade | positive | [-2.6604, 2.4983] | 0.0112 | positive | [-1.5348, 1.3989] | 43.34 | 0 | ✓ | 1.1125 |
| 27 | positive | The food at this restaurant was absolutely delicious | positive | [-1.9152, 2.0372] | 0.0170 | positive | [-0.7262, 0.5852] | 41.81 | 0 | ✓ | 1.3205 |
| 28 | positive | Customer service was fast, friendly, and helpful | positive | [-0.6158, 0.5701] | 0.0197 | negative | [0.1113, -0.1687] | 39.58 | 0 | ✗ | 0.7329 |
| 29 | positive | This phone works flawlessly and the battery lasts all … | negative | [0.5082, -0.2039] | 0.0172 | negative | [0.6759, -0.6230] | 47.20 | 0 | ✓ | 0.2934 |
| 30 | positive | The hotel room was spotless and the staff were wonderf… | negative | [0.8662, -0.5125] | 0.0168 | negative | [0.7974, -0.9306] | 43.69 | 0 | ✓ | 0.2435 |
| 31 | positive | Great value for the price, highly recommended | positive | [-2.3320, 2.3084] | 0.0171 | positive | [-0.7489, 0.7397] | 50.67 | 0 | ✓ | 1.5759 |
| 32 | positive | The book kept me hooked until the very last page | negative | [1.2876, -0.8493] | 0.0169 | negative | [0.5927, -0.6066] | 48.95 | 0 | ✓ | 0.4688 |
| 33 | positive | A refreshing and uplifting comedy the whole family enj… | positive | [-2.8613, 2.6935] | 0.0171 | positive | [-2.7018, 2.5101] | 41.49 | 0 | ✓ | 0.1715 |
| 34 | positive | The concert was electric, easily the best show I have … | positive | [-1.4061, 1.4606] | 0.0168 | positive | [-0.7749, 0.6747] | 48.42 | 0 | ✓ | 0.7086 |
| 35 | positive | This laptop is fast, quiet, and beautifully designed | positive | [-2.2152, 2.1647] | 0.0133 | positive | [-1.6322, 1.4191] | 43.79 | 0 | ✓ | 0.6643 |
| 36 | positive | The weather was perfect for our trip to the coast | positive | [-0.6499, 0.8984] | 0.0170 | negative | [0.1837, -0.2558] | 50.32 | 0 | ✗ | 0.9939 |
| 37 | positive | Our vacation exceeded every expectation we had | positive | [-0.1633, 0.5457] | 0.0183 | negative | [0.3272, -0.2668] | 41.89 | 0 | ✗ | 0.6515 |
| 38 | positive | The coffee here is consistently excellent | positive | [-2.3400, 2.2299] | 0.0170 | positive | [-1.1901, 0.9151] | 39.51 | 0 | ✓ | 1.2323 |
| 39 | positive | I am thoroughly impressed with the quality of this pro… | positive | [-2.2569, 2.2221] | 0.0168 | positive | [-0.8687, 0.7208] | 40.91 | 0 | ✓ | 1.4447 |
| 40 | positive | The team played with incredible energy and won deserve… | positive | [-2.8117, 2.6778] | 0.0215 | positive | [-1.7537, 1.5206] | 45.31 | 0 | ✓ | 1.1076 |
| 41 | positive | This app makes managing my schedule so much easier | negative | [1.3563, -1.0753] | 0.0124 | negative | [0.8176, -0.7263] | 43.27 | 0 | ✓ | 0.4439 |
| 42 | positive | The garden looked absolutely gorgeous in the spring su… | positive | [-2.5593, 2.5052] | 0.0172 | positive | [-1.0327, 0.9627] | 48.74 | 0 | ✓ | 1.5346 |
| 43 | positive | Their new album is a joyful, inventive piece of work | positive | [-2.7654, 2.6119] | 0.0170 | positive | [-2.0618, 1.8022] | 45.48 | 0 | ✓ | 0.7566 |
| 44 | positive | This is not a bad film at all, it is actually quite go… | negative | [1.3674, -0.8872] | 0.0169 | negative | [0.3745, -0.3524] | 58.32 | 0 | ✓ | 0.7639 |
| 45 | positive | I never expected to enjoy this movie as much as I did | negative | [0.7270, -0.3142] | 0.0168 | negative | [0.7296, -0.5127] | 44.19 | 0 | ✓ | 0.1006 |
| 46 | positive | Despite a slow start, the film wins you over completely | negative | [0.8703, -0.4826] | 0.0169 | negative | [0.6717, -0.6076] | 44.85 | 0 | ✓ | 0.1618 |
| 47 | positive | It is hard not to smile the entire way through this mo… | negative | [1.2122, -0.8360] | 0.0182 | negative | [0.6352, -0.6230] | 53.18 | 0 | ✓ | 0.3950 |
| 48 | positive | A small film with a surprisingly big heart | positive | [-2.5913, 2.5148] | 0.0207 | positive | [-1.1624, 0.9972] | 46.52 | 0 | ✓ | 1.4732 |
| 49 | positive | The sequel actually improves on the original in every … | positive | [-1.4458, 1.4307] | 0.0140 | positive | [-0.3975, 0.3049] | 45.19 | 0 | ✓ | 1.0870 |
| 50 | positive | Well worth the wait, this delivers on every level | positive | [-1.9690, 1.8906] | 0.0167 | positive | [-1.4506, 1.2443] | 43.00 | 0 | ✓ | 0.5824 |
