# Plaintext vs Encrypted Sentiment Inference Comparison

Generated: 2026-09-13 23:50:54 UTC
Progress: 5/5 sentences completed

`expected_sentiment` is a hand-assigned sentiment label used only as a rough sanity check (not an SST-2 gold label).

## Aggregate metrics

| Metric | Plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 5/5 | 4/5 |
| Failed / timed out | 0 | 1 |
| Total internal retries | n/a | 5 |
| Accuracy vs expected label | 0.800 | 0.600 |
| Mean time (s) | 9.746 | 45.505 |
| Median time (s) | 9.733 | 42.612 |
| Min / Max time (s) | 9.724 / 9.785 | 41.072 / 55.723 |

**Prediction agreement (plaintext vs encrypted):** 4 pairs compared, 1.000 agreement rate
**Mean absolute logit difference:** 1.2047
**Encrypted / plaintext time ratio (slowdown):** 4.7x

## Per-sentence results

| # | Expected | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries | Agree |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | This is one of the best performances I have seen all y… | positive | [-1.9175, 1.9433] | 9.76 | positive | [-0.2815, 0.2008] | 42.41 | 0 | ✓ |
| 2 | positive | A brilliant, funny, and touching film that exceeded my… | positive | [-2.7019, 2.6011] | 9.79 | FAIL | n/a | 55.33 | 3 | - |
| 3 | positive | I absolutely loved this film, it was a masterpiece | positive | [-2.6164, 2.5329] | 9.73 | positive | [-1.3575, 1.2775] | 41.07 | 0 | ✓ |
| 4 | positive | What a delightful and heartwarming movie experience | positive | [-2.7744, 2.6514] | 9.72 | positive | [-1.0628, 0.9964] | 55.72 | 2 | ✓ |
| 5 | positive | The acting was superb and the story kept me engaged th… | negative | [0.6219, -0.3057] | 9.73 | negative | [0.4577, -0.5197] | 42.82 | 0 | ✓ |
