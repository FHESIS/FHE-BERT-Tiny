# Plaintext vs Encrypted Sentiment Inference Comparison

Generated: 2026-09-14 01:43:13 UTC
Progress: 100/100 sentences completed

`expected_sentiment` is a hand-assigned sentiment label used only as a rough sanity check (not an SST-2 gold label).

## Aggregate metrics

| Metric | Plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 100/100 | 93/100 |
| Failed / timed out | 0 | 7 |
| Total internal retries | n/a | 67 |
| Accuracy vs expected label | 0.870 | 0.740 |
| Mean time (s) | 9.806 | 48.135 |
| Median time (s) | 9.752 | 47.699 |
| Min / Max time (s) | 9.592 / 12.321 | 40.151 / 65.538 |

**Prediction agreement (plaintext vs encrypted):** 93 pairs compared, 0.935 agreement rate
**Mean absolute logit difference:** 0.7805
**Encrypted / plaintext time ratio (slowdown):** 4.9x

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries | Agree |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 9.76 | positive | [-0.2741, 0.1939] | 58.09 | 1 | ✓ |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 9.71 | FAIL | n/a | 54.48 | 3 | - |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 10.06 | positive | [-1.3214, 1.2424] | 48.29 | 0 | ✓ |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 9.73 | positive | [-1.0519, 0.9869] | 40.59 | 0 | ✓ |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 9.83 | negative | [0.4587, -0.5217] | 50.16 | 0 | ✓ |
| 6 | positive | A truly inspiring story told with grace and humor | positive | [-2.8302, 2.6644] | 9.70 | positive | [-2.0719, 1.8090] | 48.04 | 0 | ✓ |
| 7 | positive | The cinematography was stunning from start to finish | positive | [-0.8676, 1.2671] | 9.75 | negative | [0.1591, -0.2523] | 40.15 | 0 | ✗ |
| 8 | positive | This film restored my faith in modern storytelling | positive | [-2.4227, 2.4041] | 9.77 | positive | [-0.8872, 0.7965] | 54.17 | 2 | ✓ |
| 9 | positive | An absolute triumph of direction and performance | positive | [-2.5795, 2.5094] | 9.73 | positive | [-1.0215, 0.9255] | 53.81 | 1 | ✓ |
| 10 | positive | The soundtrack alone is worth the price of admission | positive | [-1.1515, 1.2544] | 9.78 | positive | [-0.3821, 0.3078] | 47.02 | 0 | ✓ |
| 11 | positive | A charming little film that grows on you | positive | [-2.6428, 2.5503] | 9.72 | positive | [-0.5211, 0.4500] | 47.70 | 0 | ✓ |
| 12 | positive | I laughed out loud more than once, a genuinely funny m… | positive | [-1.1401, 1.1657] | 9.72 | positive | [-0.9168, 0.5505] | 50.00 | 0 | ✓ |
| 13 | positive | The chemistry between the leads made the whole film wo… | positive | [-1.3859, 1.5569] | 9.75 | FAIL | n/a | 59.40 | 3 | - |
| 14 | positive | A beautifully written script with real emotional depth | positive | [-2.7035, 2.5578] | 9.69 | FAIL | n/a | 54.63 | 3 | - |
| 15 | positive | This exceeded every expectation I had going in | negative | [0.7285, -0.2191] | 9.82 | negative | [0.5819, -0.4573] | 40.66 | 0 | ✓ |
| 16 | positive | The special effects were seamless and genuinely thrill… | negative | [1.7037, -1.2514] | 9.84 | negative | [0.8985, -0.8727] | 56.57 | 2 | ✓ |
| 17 | positive | A masterclass in pacing and tension | positive | [-1.5194, 1.5911] | 9.70 | positive | [-0.0726, 0.1014] | 59.94 | 2 | ✓ |
| 18 | positive | I would happily watch this again tomorrow | positive | [-0.9814, 1.1473] | 10.06 | negative | [0.3391, -0.2993] | 47.51 | 0 | ✗ |
| 19 | positive | The ending brought tears to my eyes, in the best way | positive | [-1.8498, 1.9775] | 9.78 | positive | [-1.4208, 1.3240] | 49.65 | 0 | ✓ |
| 20 | positive | Every scene felt purposeful and beautifully shot | positive | [-2.4916, 2.4159] | 9.69 | positive | [-0.9825, 0.8849] | 54.61 | 2 | ✓ |
| 21 | positive | A wonderfully clever plot with a satisfying payoff | positive | [-2.5797, 2.4319] | 9.72 | positive | [-1.2613, 1.0548] | 55.24 | 1 | ✓ |
| 22 | positive | The dialogue crackles with wit and energy | positive | [-1.9427, 2.0351] | 9.80 | positive | [-0.8373, 0.7026] | 55.31 | 2 | ✓ |
| 23 | positive | This is family entertainment done exactly right | positive | [-0.8006, 0.8704] | 9.74 | FAIL | n/a | 60.40 | 3 | - |
| 24 | positive | A gripping thriller that never lets up | positive | [-0.3214, 0.4965] | 9.77 | negative | [0.6444, -0.5457] | 46.54 | 0 | ✗ |
| 25 | positive | The lead actress gives a career-defining performance | positive | [-2.6060, 2.4519] | 9.76 | positive | [-1.7441, 1.4732] | 41.04 | 0 | ✓ |
| 26 | positive | Genuinely one of the most enjoyable films of the decade | positive | [-2.6604, 2.4983] | 9.71 | positive | [-1.5513, 1.4140] | 42.25 | 0 | ✓ |
| 27 | positive | The food at this restaurant was absolutely delicious | positive | [-1.9152, 2.0372] | 9.90 | positive | [-0.7125, 0.5723] | 40.59 | 0 | ✓ |
| 28 | positive | Customer service was fast, friendly, and helpful | positive | [-0.6158, 0.5701] | 9.85 | negative | [0.1092, -0.1671] | 40.77 | 0 | ✗ |
| 29 | positive | This phone works flawlessly and the battery lasts all … | negative | [0.5082, -0.2039] | 9.87 | negative | [0.6767, -0.6241] | 41.81 | 0 | ✓ |
| 30 | positive | The hotel room was spotless and the staff were wonderf… | negative | [0.8662, -0.5125] | 9.80 | negative | [0.8120, -0.9381] | 42.06 | 0 | ✓ |
| 31 | positive | Great value for the price, highly recommended | positive | [-2.3320, 2.3084] | 9.75 | positive | [-0.7501, 0.7425] | 47.93 | 1 | ✓ |
| 32 | positive | The book kept me hooked until the very last page | negative | [1.2876, -0.8493] | 9.68 | negative | [0.5933, -0.6070] | 42.38 | 0 | ✓ |
| 33 | positive | A refreshing and uplifting comedy the whole family enj… | positive | [-2.8613, 2.6935] | 9.75 | positive | [-2.7071, 2.5148] | 41.36 | 0 | ✓ |
| 34 | positive | The concert was electric, easily the best show I have … | positive | [-1.4061, 1.4606] | 9.78 | positive | [-0.7808, 0.6784] | 50.47 | 1 | ✓ |
| 35 | positive | This laptop is fast, quiet, and beautifully designed | positive | [-2.2152, 2.1647] | 9.79 | positive | [-1.6361, 1.4231] | 41.48 | 0 | ✓ |
| 36 | positive | The weather was perfect for our trip to the coast | positive | [-0.6499, 0.8984] | 9.65 | negative | [0.1870, -0.2581] | 56.80 | 1 | ✗ |
| 37 | positive | Our vacation exceeded every expectation we had | positive | [-0.1633, 0.5457] | 9.73 | negative | [0.3281, -0.2675] | 47.30 | 1 | ✗ |
| 38 | positive | The coffee here is consistently excellent | positive | [-2.3400, 2.2299] | 9.67 | positive | [-1.1904, 0.9154] | 42.23 | 0 | ✓ |
| 39 | positive | I am thoroughly impressed with the quality of this pro… | positive | [-2.2569, 2.2221] | 9.67 | positive | [-0.8857, 0.7372] | 48.83 | 0 | ✓ |
| 40 | positive | The team played with incredible energy and won deserve… | positive | [-2.8117, 2.6778] | 9.72 | positive | [-1.7423, 1.5102] | 46.76 | 0 | ✓ |
| 41 | positive | This app makes managing my schedule so much easier | negative | [1.3563, -1.0753] | 9.74 | negative | [0.8157, -0.7254] | 40.78 | 0 | ✓ |
| 42 | positive | The garden looked absolutely gorgeous in the spring su… | positive | [-2.5593, 2.5052] | 10.09 | positive | [-1.0184, 0.9484] | 40.48 | 0 | ✓ |
| 43 | positive | Their new album is a joyful, inventive piece of work | positive | [-2.7654, 2.6119] | 9.67 | positive | [-2.0574, 1.7983] | 43.56 | 0 | ✓ |
| 44 | positive | This is not a bad film at all, it is actually quite go… | negative | [1.3674, -0.8872] | 9.75 | negative | [0.3787, -0.3548] | 50.64 | 0 | ✓ |
| 45 | positive | I never expected to enjoy this movie as much as I did | negative | [0.7270, -0.3142] | 9.85 | negative | [0.7289, -0.5121] | 58.85 | 1 | ✓ |
| 46 | positive | Despite a slow start, the film wins you over completely | negative | [0.8703, -0.4826] | 9.79 | negative | [0.6664, -0.6045] | 50.95 | 1 | ✓ |
| 47 | positive | It is hard not to smile the entire way through this mo… | negative | [1.2122, -0.8360] | 9.74 | negative | [0.6354, -0.6236] | 42.95 | 0 | ✓ |
| 48 | positive | A small film with a surprisingly big heart | positive | [-2.5913, 2.5148] | 9.72 | positive | [-1.1684, 1.0022] | 47.14 | 1 | ✓ |
| 49 | positive | The sequel actually improves on the original in every … | positive | [-1.4458, 1.4307] | 9.70 | positive | [-0.3939, 0.3018] | 41.55 | 0 | ✓ |
| 50 | positive | Well worth the wait, this delivers on every level | positive | [-1.9690, 1.8906] | 9.69 | positive | [-1.4448, 1.2386] | 44.23 | 0 | ✓ |
| 51 | negative | This was a complete waste of my time and money | negative | [2.6095, -2.2907] | 9.67 | negative | [1.4933, -1.2852] | 56.66 | 2 | ✓ |
| 52 | negative | The plot was boring and the acting felt wooden | negative | [2.6530, -2.4051] | 9.80 | negative | [1.9362, -1.7939] | 47.74 | 0 | ✓ |
| 53 | negative | I hated every minute of this dull and pointless movie | negative | [2.4941, -2.1406] | 9.65 | negative | [1.6724, -1.4268] | 43.91 | 0 | ✓ |
| 54 | negative | A disappointing mess with no redeeming qualities | negative | [1.9164, -1.5631] | 9.95 | negative | [1.0781, -0.8947] | 48.38 | 1 | ✓ |
| 55 | negative | The worst film I have seen in years, truly terrible | negative | [2.1156, -1.6756] | 9.75 | negative | [1.4551, -1.1915] | 41.85 | 0 | ✓ |
| 56 | negative | This movie is a tedious, incoherent slog | negative | [2.4688, -2.2177] | 9.79 | negative | [1.6971, -1.4960] | 53.85 | 1 | ✓ |
| 57 | negative | The dialogue was cringeworthy and the pacing glacial | negative | [1.9937, -1.6995] | 9.80 | negative | [1.3268, -1.2281] | 48.88 | 1 | ✓ |
| 58 | negative | A lazy, uninspired sequel that adds nothing new | negative | [1.5195, -1.1765] | 9.74 | negative | [0.6773, -0.7320] | 48.23 | 0 | ✓ |
| 59 | negative | I walked out halfway through, it was that bad | negative | [2.4447, -2.1008] | 9.69 | negative | [1.3257, -1.1388] | 49.95 | 1 | ✓ |
| 60 | negative | The special effects looked cheap and unconvincing | negative | [2.4505, -2.0795] | 9.66 | negative | [1.7482, -1.4589] | 56.88 | 1 | ✓ |
| 61 | negative | A forgettable film with a plot full of holes | negative | [2.2541, -1.9730] | 9.79 | negative | [1.4987, -1.3618] | 43.35 | 0 | ✓ |
| 62 | negative | The lead performance was flat and utterly unconvincing | negative | [1.7935, -1.4376] | 9.82 | negative | [1.0100, -0.8684] | 45.67 | 0 | ✓ |
| 63 | negative | This film insults the intelligence of its audience | negative | [2.0121, -1.8932] | 9.59 | negative | [1.5893, -1.4042] | 40.66 | 0 | ✓ |
| 64 | negative | A bloated, self-indulgent film that never earns its ru… | negative | [1.2682, -0.8831] | 9.71 | negative | [1.3365, -1.0922] | 47.38 | 0 | ✓ |
| 65 | negative | The jokes fell flat and the story made no sense | negative | [2.4912, -2.1281] | 9.69 | negative | [1.5167, -1.3142] | 42.29 | 0 | ✓ |
| 66 | negative | An empty, soulless cash grab of a sequel | negative | [2.4072, -2.0668] | 9.74 | FAIL | n/a | 64.37 | 3 | - |
| 67 | negative | This is easily the worst movie of the year | negative | [2.1857, -1.7700] | 9.76 | negative | [1.3277, -1.0667] | 48.34 | 1 | ✓ |
| 68 | negative | The editing was choppy and the story dragged endlessly | negative | [2.6185, -2.3664] | 9.84 | FAIL | n/a | 65.63 | 3 | - |
| 69 | negative | A painfully dull film with nothing to say | negative | [2.3163, -1.9617] | 9.75 | negative | [1.4394, -1.1716] | 41.19 | 0 | ✓ |
| 70 | negative | The characters were flat and impossible to care about | negative | [2.1442, -1.7778] | 9.81 | negative | [1.3742, -1.1588] | 41.22 | 0 | ✓ |
| 71 | negative | A confused mess that never finds its footing | negative | [1.8454, -1.5589] | 9.78 | negative | [1.2069, -0.9599] | 52.21 | 1 | ✓ |
| 72 | negative | The score was intrusive and the script painfully clich… | negative | [1.9996, -1.6772] | 9.83 | negative | [1.2260, -1.2134] | 43.48 | 0 | ✓ |
| 73 | negative | I regret spending an evening on this dreadful movie | negative | [1.5641, -1.1521] | 9.67 | negative | [1.1661, -0.9886] | 48.00 | 0 | ✓ |
| 74 | negative | A shallow, forgettable film with no depth whatsoever | negative | [2.3654, -2.0552] | 9.78 | negative | [1.4758, -1.2380] | 65.54 | 2 | ✓ |
| 75 | negative | The pacing kills any tension the story might have had | negative | [2.1681, -1.8853] | 9.68 | negative | [1.6356, -1.3627] | 48.57 | 0 | ✓ |
| 76 | negative | This sequel betrays everything that made the original … | positive | [-0.5473, 0.7857] | 9.78 | positive | [-0.1751, 0.1371] | 41.69 | 0 | ✓ |
| 77 | negative | The food arrived cold and tasted terrible | negative | [2.0856, -1.6197] | 9.63 | negative | [1.3928, -1.1193] | 40.16 | 0 | ✓ |
| 78 | negative | Customer service was rude and unhelpful throughout | negative | [0.2992, -0.0314] | 9.77 | negative | [0.4576, -0.4795] | 41.48 | 0 | ✓ |
| 79 | negative | This phone constantly freezes and the battery is awful | negative | [2.3785, -2.1521] | 9.73 | negative | [1.4373, -1.3099] | 49.72 | 1 | ✓ |
| 80 | negative | The hotel room was filthy and the staff were dismissive | negative | [2.5200, -2.3004] | 9.75 | negative | [2.0545, -1.8666] | 52.53 | 1 | ✓ |
| 81 | negative | Overpriced and poorly made, do not waste your money | negative | [2.5299, -2.3011] | 9.76 | negative | [1.8287, -1.6050] | 43.33 | 0 | ✓ |
| 82 | negative | The book was tedious and I never finished it | negative | [1.7480, -1.4630] | 9.72 | negative | [1.4157, -1.1566] | 63.98 | 2 | ✓ |
| 83 | negative | A joyless comedy that never once made me laugh | negative | [0.8010, -0.4724] | 9.75 | negative | [0.9946, -0.8094] | 49.26 | 1 | ✓ |
| 84 | negative | The concert was a disorganized, disappointing mess | negative | [2.5042, -2.1977] | 9.72 | negative | [1.8990, -1.6552] | 42.37 | 0 | ✓ |
| 85 | negative | This laptop overheats constantly and the fan is deafen… | negative | [0.9480, -0.5489] | 9.80 | negative | [0.6181, -0.6109] | 57.46 | 1 | ✓ |
| 86 | negative | The weather ruined what could have been a nice trip | negative | [2.1508, -1.8033] | 9.96 | negative | [0.7285, -0.7812] | 54.47 | 1 | ✓ |
| 87 | negative | Our vacation was plagued with delays and bad service | negative | [1.9408, -1.5617] | 9.65 | negative | [1.3347, -1.1120] | 59.37 | 2 | ✓ |
| 88 | negative | The coffee here tastes burnt and watered down | negative | [2.5050, -2.2400] | 9.83 | negative | [1.8418, -1.5515] | 43.66 | 0 | ✓ |
| 89 | negative | I am thoroughly disappointed with the quality of this … | positive | [-0.5694, 0.7459] | 10.16 | positive | [-0.5513, 0.4205] | 55.93 | 1 | ✓ |
| 90 | negative | The team played sloppily and lost embarrassingly | negative | [2.1992, -1.8678] | 9.78 | negative | [1.3309, -1.2231] | 45.15 | 0 | ✓ |
| 91 | negative | This app crashes constantly and loses my data | negative | [2.2555, -2.0304] | 9.79 | negative | [1.5696, -1.3444] | 50.96 | 1 | ✓ |
| 92 | negative | The garden was overgrown and poorly maintained | negative | [2.4761, -2.2948] | 9.76 | negative | [1.7319, -1.4776] | 62.85 | 2 | ✓ |
| 93 | negative | Their new album is a tired, uninspired retread | negative | [2.1906, -1.9049] | 12.32 | negative | [1.3299, -1.2214] | 48.59 | 0 | ✓ |
| 94 | negative | This is not a good film, it is a genuine chore to watch | negative | [0.4927, -0.1454] | 9.63 | negative | [0.2659, -0.3421] | 45.22 | 0 | ✓ |
| 95 | negative | I never expected to dislike this movie as much as I did | negative | [1.2759, -0.9130] | 9.76 | negative | [1.0269, -0.7752] | 43.70 | 0 | ✓ |
| 96 | negative | Despite a promising start, the film collapses complete… | negative | [2.2333, -1.8849] | 9.71 | negative | [1.4320, -1.1387] | 44.79 | 0 | ✓ |
| 97 | negative | It is hard to sit through this movie without checking … | negative | [1.6083, -1.2773] | 9.71 | negative | [1.2387, -1.0098] | 49.78 | 0 | ✓ |
| 98 | negative | A big-budget film with a surprisingly hollow core | negative | [1.6223, -1.3205] | 11.26 | FAIL | n/a | 63.92 | 3 | - |
| 99 | negative | The sequel manages to be worse than the original in ev… | negative | [1.9787, -1.6065] | 10.02 | negative | [0.9638, -0.9139] | 61.87 | 2 | ✓ |
| 100 | negative | Not worth the wait, this fails on almost every level | negative | [1.6743, -1.4364] | 9.70 | negative | [1.0287, -0.8579] | 44.75 | 0 | ✓ |
