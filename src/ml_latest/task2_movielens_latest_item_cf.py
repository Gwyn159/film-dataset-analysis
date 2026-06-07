from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd


DATA_PATH = Path("/Users/mihail/Downloads/ml-latest/ratings.csv")
OUTPUT_PATH = Path(
    "/Users/mihail/Documents/Codex/2026-06-05/files-mentioned-by-the-user-links/"
    "task2_movielens_latest_results.md"
)

SEED = 42
TRAIN_FRAC = 0.8
MAX_SIM_USER_RATINGS = 30
MIN_COMMON_USERS = 5
TOP_SIMILAR_ITEMS = 30
PREDICTION_CHUNKSIZE = 200_000
MIN_RATING = 0.5
MAX_RATING = 5.0


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    started_at = time.time()
    print("Reading ratings...", flush=True)
    ratings = pd.read_csv(
        DATA_PATH,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )

    rng = np.random.default_rng(SEED)
    train_mask = rng.random(len(ratings), dtype=np.float32) < TRAIN_FRAC
    train_init = ratings.loc[train_mask].reset_index(drop=True)
    test = ratings.loc[~train_mask].reset_index(drop=True)
    del ratings, train_mask
    gc.collect()

    train_mean = float(train_init["rating"].mean())
    baseline_sse = float(((test["rating"] - train_mean) ** 2).sum())
    baseline_rmse = float(np.sqrt(baseline_sse / len(test)))

    print(
        f"Split: train={len(train_init)}, test={len(test)}, "
        f"mean={train_mean:.6f}, baseline_rmse={baseline_rmse:.6f}",
        flush=True,
    )

    train_user_means = train_init.groupby("userId")["rating"].mean()
    train_user_counts = train_init.groupby("userId").size()
    sim_user_ids = train_user_counts[
        (train_user_counts > 1) & (train_user_counts <= MAX_SIM_USER_RATINGS)
    ].index

    sim_train = train_init[train_init["userId"].isin(sim_user_ids)].copy()
    sim_train["user_mean"] = sim_train["userId"].map(train_user_means)
    sim_train["centered_rating"] = sim_train["rating"] - sim_train["user_mean"]
    print(
        f"Similarity train rows={len(sim_train)}, users={len(sim_user_ids)}",
        flush=True,
    )

    left = sim_train.rename(
        columns={"movieId": "movie_i", "centered_rating": "rating_i"}
    )[["userId", "movie_i", "rating_i"]]
    right = sim_train.rename(
        columns={"movieId": "movie_j", "centered_rating": "rating_j"}
    )[["userId", "movie_j", "rating_j"]]

    item_pairs = left.merge(right, on="userId", how="inner")
    item_pairs = item_pairs[item_pairs["movie_i"] < item_pairs["movie_j"]]
    del left, right, sim_train
    gc.collect()

    print(f"Item pairs after local cap={len(item_pairs)}", flush=True)

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
    gc.collect()

    similarities["similarity"] = similarities["dot"] / np.sqrt(
        similarities["sq_i"] * similarities["sq_j"]
    )
    similarities = similarities[
        (similarities["common_users"] >= MIN_COMMON_USERS)
        & (similarities["sq_i"] > 0)
        & (similarities["sq_j"] > 0)
        & (similarities["similarity"] != 0)
    ][["movie_i", "movie_j", "similarity", "common_users"]]

    reverse_similarities = similarities.rename(
        columns={"movie_i": "movie_j", "movie_j": "movie_i"}
    )[["movie_i", "movie_j", "similarity", "common_users"]]
    similarities = pd.concat([similarities, reverse_similarities], ignore_index=True)
    del reverse_similarities
    gc.collect()

    top_similarities = (
        similarities
        .rename(columns={"movie_i": "target_movie", "movie_j": "neighbor_movie"})
        .assign(abs_similarity=lambda df: df["similarity"].abs())
        .sort_values(
            ["target_movie", "abs_similarity", "common_users", "neighbor_movie"],
            ascending=[True, False, False, True],
        )
        .groupby("target_movie", as_index=False)
        .head(TOP_SIMILAR_ITEMS)
        [["target_movie", "neighbor_movie", "similarity", "abs_similarity"]]
    )
    top_similarities["target_movie"] = top_similarities["target_movie"].astype("int32")
    top_similarities["neighbor_movie"] = top_similarities["neighbor_movie"].astype("int32")
    top_similarities["similarity"] = top_similarities["similarity"].astype("float32")
    top_similarities["abs_similarity"] = top_similarities["abs_similarity"].astype("float32")
    learned_item_count = top_similarities["target_movie"].nunique()
    print(
        f"Top similarities rows={len(top_similarities)}, "
        f"target items={learned_item_count}",
        flush=True,
    )
    del similarities
    gc.collect()

    key_multiplier = int(max(train_init["movieId"].max(), test["movieId"].max())) + 1
    train_keys = (
        train_init["userId"].astype("int64") * key_multiplier
        + train_init["movieId"].astype("int64")
    )
    train_rating_by_key = pd.Series(
        train_init["rating"].to_numpy(dtype=np.float32),
        index=train_keys.to_numpy(),
    )

    sse = 0.0
    covered = 0
    total = 0
    fallback_count = 0

    for start in range(0, len(test), PREDICTION_CHUNKSIZE):
        end = min(start + PREDICTION_CHUNKSIZE, len(test))
        test_chunk = test.iloc[start:end][["userId", "movieId", "rating"]].copy()
        test_chunk["prediction_id"] = np.arange(start, end, dtype=np.int64)
        test_chunk = test_chunk.rename(
            columns={"movieId": "target_movie", "rating": "true_rating"}
        )
        test_chunk["user_mean"] = test_chunk["userId"].map(train_user_means).fillna(train_mean)

        neighbors = test_chunk[["prediction_id", "userId", "target_movie"]].merge(
            top_similarities,
            on="target_movie",
            how="inner",
        )

        predictions = pd.Series(
            test_chunk["user_mean"].to_numpy(dtype=np.float32),
            index=test_chunk["prediction_id"],
        )
        if not neighbors.empty:
            neighbor_keys = (
                neighbors["userId"].astype("int64") * key_multiplier
                + neighbors["neighbor_movie"].astype("int64")
            )
            neighbors["neighbor_rating"] = neighbor_keys.map(train_rating_by_key)
            neighbors = neighbors.dropna(subset=["neighbor_rating"])

            if not neighbors.empty:
                neighbors["user_mean"] = neighbors["userId"].map(train_user_means)
                neighbors["centered_neighbor_rating"] = (
                    neighbors["neighbor_rating"] - neighbors["user_mean"]
                )
                neighbors["weighted_rating"] = (
                    neighbors["similarity"] * neighbors["centered_neighbor_rating"]
                )
                grouped = neighbors.groupby("prediction_id").agg(
                    weighted_sum=("weighted_rating", "sum"),
                    denom=("abs_similarity", "sum"),
                )
                base_means = test_chunk.set_index("prediction_id")["user_mean"]
                chunk_predictions = (
                    base_means.loc[grouped.index]
                    + grouped["weighted_sum"] / grouped["denom"]
                ).clip(MIN_RATING, MAX_RATING)
                predictions.loc[chunk_predictions.index] = chunk_predictions

        actual = test_chunk.set_index("prediction_id")["true_rating"]
        diff = actual - predictions
        sse += float((diff * diff).sum())

        chunk_covered = int(predictions.index.isin(neighbors["prediction_id"].unique()).sum()) if not neighbors.empty else 0
        covered += chunk_covered
        fallback_count += len(test_chunk) - chunk_covered
        total += len(test_chunk)

        print(
            f"Predicted {end}/{len(test)} rows; "
            f"coverage_so_far={covered / total:.2%}",
            flush=True,
        )

    item_cf_rmse = float(np.sqrt(sse / total))
    coverage = covered / total

    report = "\n".join(
        [
            "# MovieLens latest task 2 results",
            "",
            "Dataset: `/Users/mihail/Downloads/ml-latest`.",
            "",
            "Variant: item-based collaborative filtering.",
            "",
            "The train/test split is reproducible with NumPy `default_rng(42)` and an 0.8 threshold.",
            "",
            "## Split and baseline",
            "",
            markdown_table(
                [
                    {"metric": "train_init rows", "value": len(train_init)},
                    {"metric": "test rows", "value": len(test)},
                    {"metric": "mean rating in train_init", "value": f"{train_mean:.6f}"},
                    {"metric": "baseline RMSE on test", "value": f"{baseline_rmse:.6f}"},
                ]
            ),
            "",
            "## Collaborative filtering result",
            "",
            markdown_table(
                [
                    {
                        "model": "Train mean baseline",
                        "rmse": f"{baseline_rmse:.6f}",
                        "details": f"prediction = {train_mean:.6f}",
                    },
                    {
                        "model": "Adjusted item-based collaborative filtering",
                        "rmse": f"{item_cf_rmse:.6f}",
                        "details": (
                            f"adjusted cosine similarity, min_common_users={MIN_COMMON_USERS}, "
                            f"top_similar_items={TOP_SIMILAR_ITEMS}, "
                            f"max_similarity_user_ratings={MAX_SIM_USER_RATINGS}, "
                            f"coverage={coverage:.2%}"
                        ),
                    },
                ]
            ),
            "",
            "## Implementation notes",
            "",
            "- Similarity is calculated only on `train_init`.",
            "- Similarity metric: adjusted cosine similarity between item vectors centered by each user's train mean.",
            "- Prediction formula: user train mean plus weighted centered ratings of similar items.",
            "- The local run caps similarity-building users at 30 train ratings to avoid a multi-billion-row item-item join.",
            "- RMSE is calculated on the full `test` subset, not on a sampled test subset.",
            "- If a test row has no matching similar neighbor rated by the user, the user's train mean is used as fallback; the global train mean is used only when the user has no train ratings.",
            f"- Top-similarity table rows: {len(top_similarities)}.",
            f"- Items with learned neighbors: {learned_item_count}.",
            f"- Fallback predictions: {fallback_count} of {total} test rows.",
            f"- Runtime in this environment: {format_seconds(time.time() - started_at)} seconds.",
            "",
        ]
    )
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"item_cf_rmse={item_cf_rmse:.6f}", flush=True)
    print(f"coverage={coverage:.2%}", flush=True)
    print(OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()
