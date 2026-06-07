from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    asc,
    avg,
    col,
    count,
    countDistinct,
    desc,
    explode,
    round,
    row_number,
    split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ml-latest"
GENRES = ["Animation", "Romance", "Documentary"]


spark = (
    SparkSession.builder
    .appName("MovieLens latest task 1")
    .getOrCreate()
)

movies = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH / "movies.csv"))
)

ratings = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH / "ratings.csv"))
)

movie_genres = (
    movies
    .withColumn("genre", explode(split(col("genres"), "\\|")))
    .select("movieId", "title", "genre")
)

print("\nGenres and number of movies")
(
    movie_genres
    .groupBy("genre")
    .agg(countDistinct("movieId").alias("movie_count"))
    .orderBy(desc("movie_count"), asc("genre"))
    .show(50, truncate=False)
)

movie_rating_stats = (
    movie_genres
    .filter(col("genre").isin(GENRES))
    .join(ratings.select("movieId", "rating"), on="movieId", how="inner")
    .groupBy("genre", "movieId", "title")
    .agg(
        count("rating").alias("rating_count"),
        avg("rating").alias("avg_rating"),
    )
)


def show_top_10(title, df, order_columns):
    window = Window.partitionBy("genre").orderBy(*order_columns)
    ranked = (
        df
        .withColumn("rank", row_number().over(window))
        .filter(col("rank") <= 10)
        .select(
            "genre",
            "rank",
            "title",
            "rating_count",
            round(col("avg_rating"), 3).alias("avg_rating"),
        )
    )

    print(f"\n{title}")
    for genre in GENRES:
        print(f"\n{genre}")
        (
            ranked
            .filter(col("genre") == genre)
            .orderBy("rank")
            .show(10, truncate=False)
        )


show_top_10(
    "Top 10 movies by number of ratings",
    movie_rating_stats,
    [desc("rating_count"), asc("title")],
)

more_than_10 = movie_rating_stats.filter(col("rating_count") > 10)

show_top_10(
    "Top 10 movies with the smallest number of ratings, rating_count > 10",
    more_than_10,
    [asc("rating_count"), asc("title")],
)

show_top_10(
    "Top 10 movies with the highest average rating, rating_count > 10",
    more_than_10,
    [desc("avg_rating"), desc("rating_count"), asc("title")],
)

show_top_10(
    "Top 10 movies with the lowest average rating, rating_count > 10",
    more_than_10,
    [asc("avg_rating"), desc("rating_count"), asc("title")],
)

spark.stop()
