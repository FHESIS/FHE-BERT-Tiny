# Ciphertext (FHE) vs Optimized Plaintext Circuit — Inference Comparison

Generated: 2026-09-15 01:15:22 UTC
Progress: 50/50 sentences completed

Encrypted circuit: `./build/FHE-BERT-Tiny <sentence> --verbose` (one fresh subprocess per sentence, CKKS evaluation over the trained BERT-tiny/SST-2 weights). Plaintext circuit: the **optimized** `src/python/PlainCircuit.py` (`PlainCircuit.load_model()` called once, `classify()` timed per sentence in the same process — see `PLAINTEXT_OPTIMIZATION_REPORT.md`). Both implement the same precomputed-LayerNorm approximation of BERT-tiny/SST-2; `expected_sentiment` is a hand-assigned sanity-check label, not an SST-2 gold label.

## Aggregate metrics

| Metric | Optimized plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 50/50 | 49/50 |
| Failed / timed out | 0 | 1 |
| Total internal retries | n/a | 0 |
| Accuracy vs expected label | 0.780 | 0.600 |
| Mean time (s) | 0.0267 | 42.957 |
| Median time (s) | 0.0175 | 42.488 |
| Min / Max time (s) | 0.0114 / 0.4675 | 38.151 / 49.217 |

**Prediction agreement (optimized plaintext vs encrypted):** 49 pairs compared, 0.837 agreement rate
**Mean absolute logit difference (avg of both logits):** 0.9868
**Mean absolute diff, logit₀ / logit₁:** 0.9512 / 1.0223
**Max absolute logit difference:** 2.1030
**Encrypted / optimized-plaintext time ratio (slowdown):** 1611.9x

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries | Agree | |Δlogit| avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 0.4675 | positive | [-0.2795, 0.1989] | 45.91 | 0 | ✓ | 1.6912 |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 0.0261 | FAIL | n/a | 53.48 | 0 | - | n/a |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 0.0250 | positive | [-1.3177, 1.2391] | 40.25 | 0 | ✓ | 1.2962 |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 0.0212 | positive | [-1.0682, 1.0021] | 46.32 | 0 | ✓ | 1.6778 |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 0.0208 | negative | [0.4592, -0.5210] | 45.57 | 0 | ✓ | 0.1890 |
| 6 | positive | A truly inspiring story told with grace and humor | positive | [-2.8302, 2.6644] | 0.0189 | positive | [-2.0406, 1.7777] | 42.35 | 0 | ✓ | 0.8382 |
| 7 | positive | The cinematography was stunning from start to finish | positive | [-0.8676, 1.2671] | 0.0180 | negative | [0.1567, -0.2507] | 42.07 | 0 | ✗ | 1.2711 |
| 8 | positive | This film restored my faith in modern storytelling | positive | [-2.4227, 2.4041] | 0.0209 | positive | [-0.8771, 0.7871] | 41.01 | 0 | ✓ | 1.5813 |
| 9 | positive | An absolute triumph of direction and performance | positive | [-2.5795, 2.5094] | 0.0176 | positive | [-1.0054, 0.9103] | 41.32 | 0 | ✓ | 1.5866 |
| 10 | positive | The soundtrack alone is worth the price of admission | positive | [-1.1515, 1.2544] | 0.0182 | positive | [-0.3793, 0.3048] | 42.52 | 0 | ✓ | 0.8609 |
| 11 | positive | A charming little film that grows on you | positive | [-2.6428, 2.5503] | 0.0174 | positive | [-0.5293, 0.4578] | 44.86 | 0 | ✓ | 2.1030 |
| 12 | positive | I laughed out loud more than once, a genuinely funny m… | positive | [-1.1401, 1.1657] | 0.0180 | positive | [-0.9580, 0.5827] | 40.88 | 0 | ✓ | 0.3826 |
| 13 | positive | The chemistry between the leads made the whole film wo… | positive | [-1.3859, 1.5569] | 0.0171 | negative | [0.3396, -0.3669] | 39.95 | 0 | ✗ | 1.8247 |
| 14 | positive | A beautifully written script with real emotional depth | positive | [-2.7035, 2.5578] | 0.0191 | positive | [-0.9066, 0.8172] | 42.09 | 0 | ✓ | 1.7687 |
| 15 | positive | This exceeded every expectation I had going in | negative | [0.7285, -0.2191] | 0.0172 | negative | [0.5817, -0.4574] | 43.81 | 0 | ✓ | 0.1925 |
| 16 | positive | The special effects were seamless and genuinely thrill… | negative | [1.7037, -1.2514] | 0.0115 | negative | [0.8914, -0.8684] | 42.49 | 0 | ✓ | 0.5977 |
| 17 | positive | A masterclass in pacing and tension | positive | [-1.5194, 1.5911] | 0.0173 | positive | [-0.0719, 0.1011] | 38.15 | 0 | ✓ | 1.4688 |
| 18 | positive | I would happily watch this again tomorrow | positive | [-0.9814, 1.1473] | 0.0174 | negative | [0.3391, -0.2999] | 44.53 | 0 | ✗ | 1.3839 |
| 19 | positive | The ending brought tears to my eyes, in the best way | positive | [-1.8498, 1.9775] | 0.0173 | positive | [-1.4366, 1.3389] | 41.45 | 0 | ✓ | 0.5259 |
| 20 | positive | Every scene felt purposeful and beautifully shot | positive | [-2.4916, 2.4159] | 0.0116 | positive | [-0.9565, 0.8595] | 39.12 | 0 | ✓ | 1.5457 |
| 21 | positive | A wonderfully clever plot with a satisfying payoff | positive | [-2.5797, 2.4319] | 0.0174 | positive | [-1.2562, 1.0510] | 47.98 | 0 | ✓ | 1.3522 |
| 22 | positive | The dialogue crackles with wit and energy | positive | [-1.9427, 2.0351] | 0.0174 | positive | [-0.8355, 0.7009] | 49.22 | 0 | ✓ | 1.2207 |
| 23 | positive | This is family entertainment done exactly right | positive | [-0.8006, 0.8704] | 0.0209 | negative | [0.1264, -0.2105] | 42.28 | 0 | ✗ | 1.0040 |
| 24 | positive | A gripping thriller that never lets up | positive | [-0.3214, 0.4965] | 0.0175 | negative | [0.6442, -0.5453] | 40.66 | 0 | ✗ | 1.0037 |
| 25 | positive | The lead actress gives a career-defining performance | positive | [-2.6060, 2.4519] | 0.0174 | positive | [-1.7297, 1.4606] | 39.62 | 0 | ✓ | 0.9338 |
| 26 | positive | Genuinely one of the most enjoyable films of the decade | positive | [-2.6604, 2.4983] | 0.0114 | positive | [-1.5536, 1.4173] | 48.43 | 0 | ✓ | 1.0939 |
| 27 | positive | The food at this restaurant was absolutely delicious | positive | [-1.9152, 2.0372] | 0.0203 | positive | [-0.7134, 0.5751] | 43.23 | 0 | ✓ | 1.3319 |
| 28 | positive | Customer service was fast, friendly, and helpful | positive | [-0.6158, 0.5701] | 0.0172 | negative | [0.1089, -0.1664] | 43.59 | 0 | ✗ | 0.7306 |
| 29 | positive | This phone works flawlessly and the battery lasts all … | negative | [0.5082, -0.2039] | 0.0210 | negative | [0.6765, -0.6232] | 43.78 | 0 | ✓ | 0.2938 |
| 30 | positive | The hotel room was spotless and the staff were wonderf… | negative | [0.8662, -0.5125] | 0.0126 | negative | [0.8074, -0.9404] | 41.24 | 0 | ✓ | 0.2434 |
| 31 | positive | Great value for the price, highly recommended | positive | [-2.3320, 2.3084] | 0.0173 | positive | [-0.7016, 0.6918] | 43.70 | 0 | ✓ | 1.6235 |
| 32 | positive | The book kept me hooked until the very last page | negative | [1.2876, -0.8493] | 0.0175 | negative | [0.5925, -0.6064] | 42.43 | 0 | ✓ | 0.4690 |
| 33 | positive | A refreshing and uplifting comedy the whole family enj… | positive | [-2.8613, 2.6935] | 0.0179 | positive | [-2.7009, 2.5106] | 40.82 | 0 | ✓ | 0.1717 |
| 34 | positive | The concert was electric, easily the best show I have … | positive | [-1.4061, 1.4606] | 0.0173 | positive | [-0.7708, 0.6692] | 49.06 | 0 | ✓ | 0.7134 |
| 35 | positive | This laptop is fast, quiet, and beautifully designed | positive | [-2.2152, 2.1647] | 0.0114 | positive | [-1.6256, 1.4129] | 43.03 | 0 | ✓ | 0.6707 |
| 36 | positive | The weather was perfect for our trip to the coast | positive | [-0.6499, 0.8984] | 0.0115 | negative | [0.1841, -0.2560] | 44.18 | 0 | ✗ | 0.9942 |
| 37 | positive | Our vacation exceeded every expectation we had | positive | [-0.1633, 0.5457] | 0.0215 | negative | [0.3297, -0.2692] | 43.82 | 0 | ✗ | 0.6539 |
| 38 | positive | The coffee here is consistently excellent | positive | [-2.3400, 2.2299] | 0.0181 | positive | [-1.1849, 0.9108] | 40.31 | 0 | ✓ | 1.2371 |
| 39 | positive | I am thoroughly impressed with the quality of this pro… | positive | [-2.2569, 2.2221] | 0.0189 | positive | [-0.9226, 0.7705] | 40.17 | 0 | ✓ | 1.3929 |
| 40 | positive | The team played with incredible energy and won deserve… | positive | [-2.8117, 2.6778] | 0.0173 | positive | [-1.7401, 1.5084] | 40.89 | 0 | ✓ | 1.1205 |
| 41 | positive | This app makes managing my schedule so much easier | negative | [1.3563, -1.0753] | 0.0173 | negative | [0.8168, -0.7262] | 43.54 | 0 | ✓ | 0.4443 |
| 42 | positive | The garden looked absolutely gorgeous in the spring su… | positive | [-2.5593, 2.5052] | 0.0176 | positive | [-1.0304, 0.9601] | 41.33 | 0 | ✓ | 1.5370 |
| 43 | positive | Their new album is a joyful, inventive piece of work | positive | [-2.7654, 2.6119] | 0.0118 | positive | [-2.0504, 1.7914] | 44.20 | 0 | ✓ | 0.7677 |
| 44 | positive | This is not a bad film at all, it is actually quite go… | negative | [1.3674, -0.8872] | 0.0136 | negative | [0.3769, -0.3542] | 45.87 | 0 | ✓ | 0.7618 |
| 45 | positive | I never expected to enjoy this movie as much as I did | negative | [0.7270, -0.3142] | 0.0211 | negative | [0.7293, -0.5132] | 42.27 | 0 | ✓ | 0.1007 |
| 46 | positive | Despite a slow start, the film wins you over completely | negative | [0.8703, -0.4826] | 0.0174 | negative | [0.6744, -0.6091] | 45.17 | 0 | ✓ | 0.1612 |
| 47 | positive | It is hard not to smile the entire way through this mo… | negative | [1.2122, -0.8360] | 0.0209 | negative | [0.6345, -0.6229] | 43.41 | 0 | ✓ | 0.3954 |
| 48 | positive | A small film with a surprisingly big heart | positive | [-2.5913, 2.5148] | 0.0177 | positive | [-1.1581, 0.9935] | 41.88 | 0 | ✓ | 1.4772 |
| 49 | positive | The sequel actually improves on the original in every … | positive | [-1.4458, 1.4307] | 0.0179 | positive | [-0.3973, 0.3049] | 40.29 | 0 | ✓ | 1.0871 |
| 50 | positive | Well worth the wait, this delivers on every level | positive | [-1.9690, 1.8906] | 0.0173 | positive | [-1.4552, 1.2483] | 43.84 | 0 | ✓ | 0.5781 |
