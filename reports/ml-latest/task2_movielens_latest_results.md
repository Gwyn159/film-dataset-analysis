# MovieLens latest task 2 results

Dataset: `data/ml-latest`.

Variant: item-based collaborative filtering.

The train/test split is reproducible with NumPy `default_rng(42)` and an 0.8 threshold.

## Split and baseline

| metric | value |
| --- | --- |
| train_init rows | 27065403 |
| test rows | 6766759 |
| mean rating in train_init | 3.542706 |
| baseline RMSE on test | 1.063795 |

## Collaborative filtering result

| model | rmse | details |
| --- | --- | --- |
| Train mean baseline | 1.063795 | prediction = 3.542706 |
| Adjusted item-based collaborative filtering | 0.987658 | adjusted cosine similarity, min_common_users=5, top_similar_items=30, max_similarity_user_ratings=30, coverage=70.69% |

## Implementation notes

- Similarity is calculated only on `train_init`.
- Similarity metric: adjusted cosine similarity between item vectors centered by each user's train mean.
- Prediction formula: user train mean plus weighted centered ratings of similar items.
- The local run caps similarity-building users at 30 train ratings to avoid a multi-billion-row item-item join.
- RMSE is calculated on the full `test` subset, not on a sampled test subset.
- If a test row has no matching similar neighbor rated by the user, the user's train mean is used as fallback; the global train mean is used only when the user has no train ratings.
- Top-similarity table rows: 103478.
- Items with learned neighbors: 5403.
- Fallback predictions: 1983221 of 6766759 test rows.
- Runtime in this environment: 27.12 seconds.
