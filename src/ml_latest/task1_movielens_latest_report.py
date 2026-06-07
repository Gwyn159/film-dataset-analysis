from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ml-latest"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "ml-latest" / "task1_movielens_latest_results.md"
GENRES = ["Animation", "Romance", "Documentary"]
CHUNKSIZE = 2_000_000


def markdown_table(df: pd.DataFrame) -> str:
    rows = df.copy()
    for column in rows.columns:
        if pd.api.types.is_float_dtype(rows[column]):
            rows[column] = rows[column].map(lambda value: f"{value:.3f}")
        else:
            rows[column] = rows[column].astype(str)
        rows[column] = rows[column].str.replace("|", "\\|", regex=False)

    header = "| " + " | ".join(rows.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = [
        "| " + " | ".join(row) + " |"
        for row in rows.to_numpy(dtype=str)
    ]
    return "\n".join([header, separator, *body])


def section(lines: list[str], title: str, df: pd.DataFrame) -> None:
    lines.append(f"\n## {title}\n")
    lines.append(markdown_table(df))


def aggregate_ratings() -> pd.DataFrame:
    counts = pd.Series(dtype="int64")
    sums = pd.Series(dtype="float64")

    for chunk in pd.read_csv(
        DATA_PATH / "ratings.csv",
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
        chunksize=CHUNKSIZE,
    ):
        grouped = chunk.groupby("movieId")["rating"].agg(["count", "sum"])
        counts = counts.add(grouped["count"], fill_value=0)
        sums = sums.add(grouped["sum"], fill_value=0)

    stats = pd.DataFrame(
        {
            "movieId": counts.index.astype("int32"),
            "rating_count": counts.astype("int64").to_numpy(),
            "rating_sum": sums.to_numpy(),
        }
    )
    stats["avg_rating"] = stats["rating_sum"] / stats["rating_count"]
    return stats[["movieId", "rating_count", "avg_rating"]]


def main() -> None:
    movies = pd.read_csv(DATA_PATH / "movies.csv")
    movie_genres = (
        movies
        .assign(genre=movies["genres"].str.split("|"))
        .explode("genre")
        [["movieId", "title", "genre"]]
    )

    genre_counts = (
        movie_genres
        .groupby("genre", as_index=False)
        .agg(movie_count=("movieId", "nunique"))
        .sort_values(["movie_count", "genre"], ascending=[False, True])
    )

    rating_stats = aggregate_ratings()

    stats = (
        movie_genres[movie_genres["genre"].isin(GENRES)]
        .merge(rating_stats, on="movieId", how="inner")
    )

    lines = [
        "# MovieLens latest task 1 results",
        "",
        "Dataset: `/Users/mihail/Downloads/ml-latest`.",
        "",
        "Variant: Animation, Romance, Documentary.",
        "",
        "Genres are split with `explode`, so one movie is counted in every genre it belongs to.",
    ]

    section(lines, "Genres and number of movies", genre_counts)

    for genre in GENRES:
        genre_stats = stats[stats["genre"] == genre]
        section(
            lines,
            f"{genre}: top 10 by number of ratings",
            (
                genre_stats
                .sort_values(["rating_count", "title"], ascending=[False, True])
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )
        section(
            lines,
            f"{genre}: top 10 by smallest number of ratings, rating_count > 10",
            (
                genre_stats[genre_stats["rating_count"] > 10]
                .sort_values(["rating_count", "title"], ascending=[True, True])
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )
        section(
            lines,
            f"{genre}: top 10 by highest average rating, rating_count > 10",
            (
                genre_stats[genre_stats["rating_count"] > 10]
                .sort_values(
                    ["avg_rating", "rating_count", "title"],
                    ascending=[False, False, True],
                )
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )
        section(
            lines,
            f"{genre}: top 10 by lowest average rating, rating_count > 10",
            (
                genre_stats[genre_stats["rating_count"] > 10]
                .sort_values(
                    ["avg_rating", "rating_count", "title"],
                    ascending=[True, False, True],
                )
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
