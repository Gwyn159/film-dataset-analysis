from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    abs as spark_abs,
    asc,
    avg,
    broadcast,
    coalesce,
    col,
    count,
    desc,
    lit,
    monotonically_increasing_id,
    rand,
    row_number,
    sqrt,
    sum as spark_sum,
    when,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ml-latest" / "ratings.csv"
SEED = 42
TRAIN_FRAC = 0.8
MAX_SIM_USER_RATINGS = 30
MIN_COMMON_USERS = 5
TOP_SIMILAR_ITEMS = 30
MIN_RATING = 0.5
MAX_RATING = 5.0


spark = (
    SparkSession.builder
    .appName("MovieLens latest task 2 adjusted item-based CF")
    .getOrCreate()
)

ratings = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH))
    .select("userId", "movieId", "rating")
    .withColumn("split_rand", rand(SEED))
)

train_init = ratings.filter(col("split_rand") < TRAIN_FRAC).drop("split_rand").cache()
test = ratings.filter(col("split_rand") >= TRAIN_FRAC).drop("split_rand").cache()

train_mean = train_init.agg(avg("rating").alias("mean_rating")).first()["mean_rating"]

baseline_rmse = (
    test
    .select(
        sqrt(
            avg((col("rating") - lit(train_mean)) * (col("rating") - lit(train_mean)))
        ).alias("rmse")
    )
    .first()["rmse"]
)

print(f"Mean rating in train_init: {train_mean:.6f}")
print(f"Baseline RMSE on test: {baseline_rmse:.6f}")

user_stats = (
    train_init
    .groupBy("userId")
    .agg(
        count("*").alias("train_rating_count"),
        avg("rating").alias("user_mean"),
    )
    .cache()
)

sim_train = (
    train_init
    .join(user_stats, on="userId", how="inner")
    .filter(
        (col("train_rating_count") > 1)
        & (col("train_rating_count") <= MAX_SIM_USER_RATINGS)
    )
    .withColumn("centered_rating", col("rating") - col("user_mean"))
    .select("userId", "movieId", "centered_rating")
)

left = sim_train.select(
    "userId",
    col("movieId").alias("movie_i"),
    col("centered_rating").alias("rating_i"),
)
right = sim_train.select(
    "userId",
    col("movieId").alias("movie_j"),
    col("centered_rating").alias("rating_j"),
)

item_pairs = (
    left
    .join(right, on="userId", how="inner")
    .filter(col("movie_i") < col("movie_j"))
)

similarities = (
    item_pairs
    .groupBy("movie_i", "movie_j")
    .agg(
        count("*").alias("common_users"),
        spark_sum(col("rating_i") * col("rating_j")).alias("dot"),
        spark_sum(col("rating_i") * col("rating_i")).alias("sq_i"),
        spark_sum(col("rating_j") * col("rating_j")).alias("sq_j"),
    )
    .withColumn("similarity", col("dot") / sqrt(col("sq_i") * col("sq_j")))
    .filter(
        (col("common_users") >= MIN_COMMON_USERS)
        & (col("sq_i") > 0)
        & (col("sq_j") > 0)
        & (col("similarity") != 0)
    )
    .select("movie_i", "movie_j", "similarity", "common_users")
)

similarities_reverse = similarities.select(
    col("movie_j").alias("movie_i"),
    col("movie_i").alias("movie_j"),
    "similarity",
    "common_users",
)

similarities_all = (
    similarities
    .unionByName(similarities_reverse)
    .withColumn("abs_similarity", spark_abs(col("similarity")))
)

similarity_window = Window.partitionBy("movie_i").orderBy(
    desc("abs_similarity"),
    desc("common_users"),
    asc("movie_j"),
)

top_similarities = (
    similarities_all
    .withColumn("similarity_rank", row_number().over(similarity_window))
    .filter(col("similarity_rank") <= TOP_SIMILAR_ITEMS)
    .select(
        col("movie_i").alias("target_movie"),
        col("movie_j").alias("neighbor_movie"),
        "similarity",
        "abs_similarity",
    )
    .cache()
)

test_with_id = test.select(
    "userId",
    col("movieId").alias("target_movie"),
    col("rating").alias("true_rating"),
).withColumn("prediction_id", monotonically_increasing_id())

train_neighbors = train_init.select(
    "userId",
    col("movieId").alias("neighbor_movie"),
    col("rating").alias("neighbor_rating"),
)

neighbors = (
    test_with_id
    .join(broadcast(top_similarities), on="target_movie", how="inner")
    .join(train_neighbors, on=["userId", "neighbor_movie"], how="inner")
    .join(user_stats.select("userId", "user_mean"), on="userId", how="left")
    .withColumn("centered_neighbor_rating", col("neighbor_rating") - col("user_mean"))
    .withColumn("weighted_rating", col("similarity") * col("centered_neighbor_rating"))
)

predictions = (
    neighbors
    .groupBy("prediction_id")
    .agg(
        spark_sum("weighted_rating").alias("weighted_sum"),
        spark_sum("abs_similarity").alias("denom"),
        count("*").alias("neighbor_count"),
    )
)

scored = (
    test_with_id
    .join(user_stats.select("userId", "user_mean"), on="userId", how="left")
    .withColumn("base_prediction", coalesce(col("user_mean"), lit(train_mean)))
    .join(predictions, on="prediction_id", how="left")
    .withColumn(
        "raw_prediction",
        when(
            col("denom").isNotNull(),
            col("base_prediction") + col("weighted_sum") / col("denom"),
        ).otherwise(col("base_prediction")),
    )
    .withColumn(
        "prediction",
        when(col("raw_prediction") < MIN_RATING, lit(MIN_RATING))
        .when(col("raw_prediction") > MAX_RATING, lit(MAX_RATING))
        .otherwise(col("raw_prediction")),
    )
)

item_cf_rmse = (
    scored
    .select(
        sqrt(
            avg(
                (col("true_rating") - col("prediction"))
                * (col("true_rating") - col("prediction"))
            )
        ).alias("rmse")
    )
    .first()["rmse"]
)

fallback_count = scored.filter(col("denom").isNull()).count()
test_count = scored.count()

print(f"Adjusted item-based CF RMSE on test: {item_cf_rmse:.6f}")
print(f"Fallback predictions: {fallback_count} of {test_count}")

spark.stop()
