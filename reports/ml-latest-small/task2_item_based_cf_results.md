# MovieLens task 2 results

Variant: item-based collaborative filtering.

Split is reproducible: pandas `sample(frac=0.8, random_state=42)`.

## Split and baseline

| metric | value |
| --- | --- |
| train_init rows | 80669 |
| test rows | 20167 |
| mean rating in train_init | 3.503229 |
| baseline RMSE on test | 1.044674 |

## Collaborative filtering result

| model | rmse | details |
| --- | --- | --- |
| Train mean baseline | 1.044674 | prediction = 3.503229 |
| Item-based collaborative filtering | 0.977583 | cosine similarity, min_common_users=5, top_k=30, coverage=84.01% |

## Implementation notes

- Similarity is calculated only on `train_init`.
- Similarity metric: cosine similarity between item rating vectors.
- Prediction for a test row uses ratings that the same user has in `train_init`.
- Up to 30 most similar rated items are used for each prediction.
- If no suitable neighbors are found, the train mean is used as fallback.
- Fallback predictions: 3225 of 20167 test rows.
- Runtime in this environment: 10.51 seconds.
