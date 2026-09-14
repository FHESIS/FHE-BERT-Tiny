# Plaintext vs Encrypted Sentiment Inference Comparison

Generated: 2026-09-14 12:03:18 UTC
Progress: 50/50 sentences completed

`expected_sentiment` is a hand-assigned sentiment label used only as a rough sanity check (not an SST-2 gold label).

## Aggregate metrics

| Metric | Plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 50/50 | 47/50 |
| Failed / timed out | 0 | 3 |
| Total internal retries | n/a | 0 |
| Accuracy vs expected label | 0.780 | 0.580 |
| Mean time (s) | 10.566 | 45.688 |
| Median time (s) | 10.485 | 44.643 |
| Min / Max time (s) | 10.328 / 11.053 | 41.212 / 53.192 |

**Prediction agreement (plaintext vs encrypted):** 47 pairs compared, 0.851 agreement rate
**Mean absolute logit difference:** 0.9719
**Encrypted / plaintext time ratio (slowdown):** 4.3x

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries | Agree |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 10.44 | positive | [-0.2717, 0.1919] | 43.30 | 0 | ✓ |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 10.86 | FAIL | n/a | 51.93 | 0 | - |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 10.84 | FAIL | n/a | 50.23 | 0 | - |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 10.48 | positive | [-1.0494, 0.9837] | 49.21 | 0 | ✓ |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 10.49 | negative | [0.4584, -0.5212] | 45.71 | 0 | ✓ |
| 6 | positive | A truly inspiring story told with grace and humor | positive | [-2.8302, 2.6644] | 10.51 | positive | [-2.0781, 1.8138] | 41.96 | 0 | ✓ |
| 7 | positive | The cinematography was stunning from start to finish | positive | [-0.8676, 1.2671] | 10.71 | FAIL | n/a | 49.67 | 0 | - |
| 8 | positive | This film restored my faith in modern storytelling | positive | [-2.4227, 2.4041] | 10.46 | positive | [-0.8933, 0.8022] | 41.83 | 0 | ✓ |
| 9 | positive | An absolute triumph of direction and performance | positive | [-2.5795, 2.5094] | 10.75 | positive | [-1.0314, 0.9342] | 44.64 | 0 | ✓ |
| 10 | positive | The soundtrack alone is worth the price of admission | positive | [-1.1515, 1.2544] | 10.51 | positive | [-0.3823, 0.3072] | 47.41 | 0 | ✓ |
| 11 | positive | A charming little film that grows on you | positive | [-2.6428, 2.5503] | 10.42 | positive | [-0.5244, 0.4536] | 44.20 | 0 | ✓ |
| 12 | positive | I laughed out loud more than once, a genuinely funny m… | positive | [-1.1401, 1.1657] | 10.42 | positive | [-0.9554, 0.5805] | 43.40 | 0 | ✓ |
| 13 | positive | The chemistry between the leads made the whole film wo… | positive | [-1.3859, 1.5569] | 10.46 | negative | [0.3394, -0.3674] | 44.08 | 0 | ✗ |
| 14 | positive | A beautifully written script with real emotional depth | positive | [-2.7035, 2.5578] | 10.70 | positive | [-0.9043, 0.8156] | 44.38 | 0 | ✓ |
| 15 | positive | This exceeded every expectation I had going in | negative | [0.7285, -0.2191] | 10.65 | negative | [0.5805, -0.4568] | 42.34 | 0 | ✓ |
| 16 | positive | The special effects were seamless and genuinely thrill… | negative | [1.7037, -1.2514] | 10.87 | negative | [0.8938, -0.8692] | 42.29 | 0 | ✓ |
| 17 | positive | A masterclass in pacing and tension | positive | [-1.5194, 1.5911] | 10.36 | positive | [-0.0678, 0.0963] | 43.12 | 0 | ✓ |
| 18 | positive | I would happily watch this again tomorrow | positive | [-0.9814, 1.1473] | 10.71 | negative | [0.3385, -0.2992] | 45.82 | 0 | ✗ |
| 19 | positive | The ending brought tears to my eyes, in the best way | positive | [-1.8498, 1.9775] | 10.48 | positive | [-1.4258, 1.3288] | 48.88 | 0 | ✓ |
| 20 | positive | Every scene felt purposeful and beautifully shot | positive | [-2.4916, 2.4159] | 10.35 | positive | [-0.9960, 0.8950] | 41.21 | 0 | ✓ |
| 21 | positive | A wonderfully clever plot with a satisfying payoff | positive | [-2.5797, 2.4319] | 10.79 | positive | [-1.2480, 1.0427] | 47.05 | 0 | ✓ |
| 22 | positive | The dialogue crackles with wit and energy | positive | [-1.9427, 2.0351] | 10.78 | positive | [-0.8558, 0.7189] | 49.13 | 0 | ✓ |
| 23 | positive | This is family entertainment done exactly right | positive | [-0.8006, 0.8704] | 10.48 | negative | [0.1274, -0.2112] | 43.21 | 0 | ✗ |
| 24 | positive | A gripping thriller that never lets up | positive | [-0.3214, 0.4965] | 10.88 | negative | [0.6442, -0.5452] | 43.24 | 0 | ✗ |
| 25 | positive | The lead actress gives a career-defining performance | positive | [-2.6060, 2.4519] | 10.70 | positive | [-1.7261, 1.4580] | 45.51 | 0 | ✓ |
| 26 | positive | Genuinely one of the most enjoyable films of the decade | positive | [-2.6604, 2.4983] | 11.05 | positive | [-1.5316, 1.3951] | 53.19 | 0 | ✓ |
| 27 | positive | The food at this restaurant was absolutely delicious | positive | [-1.9152, 2.0372] | 10.45 | positive | [-0.7501, 0.6044] | 41.26 | 0 | ✓ |
| 28 | positive | Customer service was fast, friendly, and helpful | positive | [-0.6158, 0.5701] | 10.48 | negative | [0.1095, -0.1674] | 46.68 | 0 | ✗ |
| 29 | positive | This phone works flawlessly and the battery lasts all … | negative | [0.5082, -0.2039] | 10.57 | negative | [0.6756, -0.6227] | 51.10 | 0 | ✓ |
| 30 | positive | The hotel room was spotless and the staff were wonderf… | negative | [0.8662, -0.5125] | 10.44 | negative | [0.7951, -0.9244] | 48.01 | 0 | ✓ |
| 31 | positive | Great value for the price, highly recommended | positive | [-2.3320, 2.3084] | 10.38 | positive | [-0.7261, 0.7172] | 42.90 | 0 | ✓ |
| 32 | positive | The book kept me hooked until the very last page | negative | [1.2876, -0.8493] | 10.45 | negative | [0.5932, -0.6071] | 46.03 | 0 | ✓ |
| 33 | positive | A refreshing and uplifting comedy the whole family enj… | positive | [-2.8613, 2.6935] | 10.48 | positive | [-2.7008, 2.5084] | 49.49 | 0 | ✓ |
| 34 | positive | The concert was electric, easily the best show I have … | positive | [-1.4061, 1.4606] | 10.69 | positive | [-0.7853, 0.6831] | 47.40 | 0 | ✓ |
| 35 | positive | This laptop is fast, quiet, and beautifully designed | positive | [-2.2152, 2.1647] | 10.75 | positive | [-1.6251, 1.4129] | 46.70 | 0 | ✓ |
| 36 | positive | The weather was perfect for our trip to the coast | positive | [-0.6499, 0.8984] | 10.58 | negative | [0.1855, -0.2571] | 51.77 | 0 | ✗ |
| 37 | positive | Our vacation exceeded every expectation we had | positive | [-0.1633, 0.5457] | 10.33 | negative | [0.3288, -0.2676] | 46.70 | 0 | ✗ |
| 38 | positive | The coffee here is consistently excellent | positive | [-2.3400, 2.2299] | 10.45 | positive | [-1.1866, 0.9123] | 44.42 | 0 | ✓ |
| 39 | positive | I am thoroughly impressed with the quality of this pro… | positive | [-2.2569, 2.2221] | 10.40 | positive | [-0.8711, 0.7248] | 48.49 | 0 | ✓ |
| 40 | positive | The team played with incredible energy and won deserve… | positive | [-2.8117, 2.6778] | 10.44 | positive | [-1.7498, 1.5171] | 42.86 | 0 | ✓ |
| 41 | positive | This app makes managing my schedule so much easier | negative | [1.3563, -1.0753] | 10.67 | negative | [0.8174, -0.7263] | 42.56 | 0 | ✓ |
| 42 | positive | The garden looked absolutely gorgeous in the spring su… | positive | [-2.5593, 2.5052] | 10.44 | positive | [-1.0238, 0.9529] | 51.30 | 0 | ✓ |
| 43 | positive | Their new album is a joyful, inventive piece of work | positive | [-2.7654, 2.6119] | 10.43 | positive | [-2.0517, 1.7926] | 52.06 | 0 | ✓ |
| 44 | positive | This is not a bad film at all, it is actually quite go… | negative | [1.3674, -0.8872] | 10.48 | negative | [0.3809, -0.3567] | 45.86 | 0 | ✓ |
| 45 | positive | I never expected to enjoy this movie as much as I did | negative | [0.7270, -0.3142] | 10.47 | negative | [0.7296, -0.5130] | 43.78 | 0 | ✓ |
| 46 | positive | Despite a slow start, the film wins you over completely | negative | [0.8703, -0.4826] | 10.64 | negative | [0.6717, -0.6076] | 42.67 | 0 | ✓ |
| 47 | positive | It is hard not to smile the entire way through this mo… | negative | [1.2122, -0.8360] | 10.37 | negative | [0.6363, -0.6242] | 50.73 | 0 | ✓ |
| 48 | positive | A small film with a surprisingly big heart | positive | [-2.5913, 2.5148] | 10.59 | positive | [-1.1768, 1.0100] | 41.95 | 0 | ✓ |
| 49 | positive | The sequel actually improves on the original in every … | positive | [-1.4458, 1.4307] | 10.46 | positive | [-0.3908, 0.2986] | 44.42 | 0 | ✓ |
| 50 | positive | Well worth the wait, this delivers on every level | positive | [-1.9690, 1.8906] | 10.72 | positive | [-1.4605, 1.2529] | 43.05 | 0 | ✓ |
