from pathlib import Path
import time

import numpy as np
import pandas as pd


DATA_PATH = Path("/Users/mihail/Downloads/ml-latest-small/ratings.csv")
OUTPUT_PATH = Path(
    "/Users/mihail/Documents/Codex/2026-06-05/files-mentioned-by-the-user-links/"
    "task2_item_based_cf_results.md"
)

SEED = 42
TRAIN_FRAC = 0.8
MIN_COMMON_USERS = 5
TOP_K = 30


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = list(rows[0].keys())
    result = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def main() -> None:
    started_at = time.time()
    ratings = pd.read_csv(DATA_PATH)

    train_init = ratings.sample(frac=TRAIN_FRAC, random_state=SEED)
    test = ratings.drop(train_init.index).copy()

    train_mean = train_init["rating"].mean()
    baseline_prediction = np.full(len(test), train_mean)
    baseline_rmse = rmse(test["rating"].to_numpy(), baseline_prediction)

    left = train_init[["userId", "movieId", "rating"]].rename(
        columns={"movieId": "movie_i", "rating": "rating_i"}
    )
    right = train_init[["userId", "movieId", "rating"]].rename(
        columns={"movieId": "movie_j", "rating": "rating_j"}
    )

    item_pairs = left.merge(right, on="userId", how="inner")
    item_pairs = item_pairs[item_pairs["movie_i"] < item_pairs["movie_j"]]
    item_pairs["dot"] = item_pairs["rating_i"] * item_pairs["rating_j"]
    item_pairs["sq_i"] = item_pairs["rating_i"] * item_pairs["rating_i"]
    item_pairs["sq_j"] = item_pairs["rating_j"] * item_pairs["rating_j"]

    similarities = (
        item_pairs.groupby(["movie_i", "movie_j"], as_index=False)
        .agg(
            common_users=("userId", "size"),
            dot=("dot", "sum"),
            sq_i=("sq_i", "sum"),
            sq_j=("sq_j", "sum"),
        )
    )
    del item_pairs

    similarities["similarity"] = similarities["dot"] / np.sqrt(
        similarities["sq_i"] * similarities["sq_j"]
    )
    similarities = similarities[
        (similarities["common_users"] >= MIN_COMMON_USERS)
        & (similarities["similarity"] > 0)
    ][["movie_i", "movie_j", "similarity", "common_users"]]

    reverse_similarities = similarities.rename(
        columns={"movie_i": "movie_j", "movie_j": "movie_i"}
    )[["movie_i", "movie_j", "similarity", "common_users"]]
    similarities = pd.concat([similarities, reverse_similarities], ignore_index=True)
    del reverse_similarities

    test_with_id = (
        test.reset_index(drop=True)
        .reset_index()
        .rename(
            columns={
                "index": "prediction_id",
                "movieId": "target_movie",
                "rating": "true_rating",
            }
        )
    )
    user_train_ratings = train_init[["userId", "movieId", "rating"]].rename(
        columns={"movieId": "rated_movie", "rating": "neighbor_rating"}
    )

    candidates = test_with_id[
        ["prediction_id", "userId", "target_movie", "true_rating"]
    ].merge(user_train_ratings, on="userId", how="inner")
    candidates = candidates[candidates["target_movie"] != candidates["rated_movie"]]
    candidates = candidates.merge(
        similarities,
        left_on=["target_movie", "rated_movie"],
        right_on=["movie_i", "movie_j"],
        how="inner",
    )
    del similarities

    candidates["abs_similarity"] = candidates["similarity"].abs()
    candidates = candidates.sort_values(
        ["prediction_id", "abs_similarity", "common_users", "rated_movie"],
        ascending=[True, False, False, True],
    )
    candidates = candidates.groupby("prediction_id", as_index=False).head(TOP_K)
    candidates["weighted_rating"] = (
        candidates["similarity"] * candidates["neighbor_rating"]
    )

    predictions = (
        candidates.groupby("prediction_id", as_index=False)
        .agg(
            weighted_sum=("weighted_rating", "sum"),
            denom=("abs_similarity", "sum"),
            neighbor_count=("neighbor_rating", "size"),
        )
    )
    predictions["prediction"] = predictions["weighted_sum"] / predictions["denom"]

    scored = test_with_id[["prediction_id", "true_rating"]].merge(
        predictions[["prediction_id", "prediction", "neighbor_count"]],
        on="prediction_id",
        how="left",
    )
    scored["prediction"] = scored["prediction"].fillna(train_mean)
    scored["neighbor_count"] = scored["neighbor_count"].fillna(0).astype(int)

    item_cf_rmse = rmse(
        scored["true_rating"].to_numpy(),
        scored["prediction"].to_numpy(),
    )

    fallback_count = int((scored["neighbor_count"] == 0).sum())
    coverage = 1 - fallback_count / len(scored)

    rows = [
        {
            "model": "Train mean baseline",
            "rmse": f"{baseline_rmse:.6f}",
            "details": f"prediction = {train_mean:.6f}",
        },
        {
            "model": "Item-based collaborative filtering",
            "rmse": f"{item_cf_rmse:.6f}",
            "details": (
                f"cosine similarity, min_common_users={MIN_COMMON_USERS}, "
                f"top_k={TOP_K}, coverage={coverage:.2%}"
            ),
        },
    ]

    report = "\n".join(
        [
            "# MovieLens task 2 results",
            "",
            "Variant: item-based collaborative filtering.",
            "",
            "Split is reproducible: pandas `sample(frac=0.8, random_state=42)`.",
            "",
            "## Split and baseline",
            "",
            markdown_table(
                [
                    {
                        "metric": "train_init rows",
                        "value": len(train_init),
                    },
                    {
                        "metric": "test rows",
                        "value": len(test),
                    },
                    {
                        "metric": "mean rating in train_init",
                        "value": f"{train_mean:.6f}",
                    },
                    {
                        "metric": "baseline RMSE on test",
                        "value": f"{baseline_rmse:.6f}",
                    },
                ]
            ),
            "",
            "## Collaborative filtering result",
            "",
            markdown_table(rows),
            "",
            "## Implementation notes",
            "",
            "- Similarity is calculated only on `train_init`.",
            "- Similarity metric: cosine similarity between item rating vectors.",
            "- Prediction for a test row uses ratings that the same user has in `train_init`.",
            f"- Up to {TOP_K} most similar rated items are used for each prediction.",
            f"- If no suitable neighbors are found, the train mean is used as fallback.",
            f"- Fallback predictions: {fallback_count} of {len(scored)} test rows.",
            f"- Runtime in this environment: {time.time() - started_at:.2f} seconds.",
            "",
        ]
    )

    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"train_mean={train_mean:.6f}")
    print(f"baseline_rmse={baseline_rmse:.6f}")
    print(f"item_cf_rmse={item_cf_rmse:.6f}")
    print(f"coverage={coverage:.2%}")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
