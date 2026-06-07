from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ml-latest-small"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "ml-latest-small" / "task1_movielens_results.md"
GENRES = ["Animation", "Romance", "Documentary"]


def markdown_table(df: pd.DataFrame) -> str:
    rows = df.copy()
    for column in rows.columns:
        if rows[column].dtype == "float64":
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


def main() -> None:
    movies = pd.read_csv(DATA_PATH / "movies.csv")
    ratings = pd.read_csv(DATA_PATH / "ratings.csv")

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

    stats = (
        movie_genres[movie_genres["genre"].isin(GENRES)]
        .merge(ratings[["movieId", "rating"]], on="movieId", how="inner")
        .groupby(["genre", "movieId", "title"], as_index=False)
        .agg(
            rating_count=("rating", "size"),
            avg_rating=("rating", "mean"),
        )
    )

    lines = [
        "# MovieLens task 1 results",
        "",
        "Variant: Animation, Romance, Documentary.",
        "",
        "Genres are split with `explode`, so a movie is counted once in each of its genres.",
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
                .sort_values(["avg_rating", "rating_count", "title"], ascending=[False, False, True])
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )
        section(
            lines,
            f"{genre}: top 10 by lowest average rating, rating_count > 10",
            (
                genre_stats[genre_stats["rating_count"] > 10]
                .sort_values(["avg_rating", "rating_count", "title"], ascending=[True, False, True])
                .head(10)
                [["title", "rating_count", "avg_rating"]]
            ),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
