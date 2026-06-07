from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    abs as spark_abs,
    asc,
    avg,
    coalesce,
    col,
    count,
    desc,
    lit,
    monotonically_increasing_id,
    row_number,
    sqrt,
    sum as spark_sum,
)


DATA_PATH = "/Users/mihail/Downloads/ml-latest-small/ratings.csv"
SEED = 42
MIN_COMMON_USERS = 5
TOP_K = 30


spark = (
    SparkSession.builder
    .appName("MovieLens task 2 item-based CF")
    .getOrCreate()
)

ratings = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

train_init, test = ratings.randomSplit([0.8, 0.2], seed=SEED)
train_init = train_init.cache()
test = test.cache()

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

left = train_init.select(
    "userId",
    col("movieId").alias("movie_i"),
    col("rating").alias("rating_i"),
)
right = train_init.select(
    "userId",
    col("movieId").alias("movie_j"),
    col("rating").alias("rating_j"),
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
    .filter((col("common_users") >= MIN_COMMON_USERS) & (col("similarity") > 0))
    .select("movie_i", "movie_j", "similarity", "common_users")
)

similarities_reverse = similarities.select(
    col("movie_j").alias("movie_i"),
    col("movie_i").alias("movie_j"),
    "similarity",
    "common_users",
)

similarities_all = similarities.unionByName(similarities_reverse).cache()

test_with_id = test.select(
    monotonically_increasing_id().alias("prediction_id"),
    "userId",
    col("movieId").alias("target_movie"),
    col("rating").alias("true_rating"),
)

user_train_ratings = train_init.select(
    "userId",
    col("movieId").alias("rated_movie"),
    col("rating").alias("neighbor_rating"),
)

neighbors = (
    test_with_id
    .join(user_train_ratings, on="userId", how="inner")
    .filter(col("target_movie") != col("rated_movie"))
    .join(
        similarities_all,
        (col("target_movie") == col("movie_i"))
        & (col("rated_movie") == col("movie_j")),
        how="inner",
    )
    .withColumn("abs_similarity", spark_abs(col("similarity")))
)

neighbor_window = Window.partitionBy("prediction_id").orderBy(
    desc("abs_similarity"),
    desc("common_users"),
    asc("rated_movie"),
)

top_neighbors = (
    neighbors
    .withColumn("rank", row_number().over(neighbor_window))
    .filter(col("rank") <= TOP_K)
)

predictions = (
    top_neighbors
    .groupBy("prediction_id")
    .agg(
        spark_sum(col("similarity") * col("neighbor_rating")).alias("weighted_sum"),
        spark_sum("abs_similarity").alias("denom"),
        count("*").alias("neighbor_count"),
    )
    .withColumn("prediction", col("weighted_sum") / col("denom"))
)

scored = (
    test_with_id
    .join(
        predictions.select("prediction_id", "prediction", "neighbor_count"),
        on="prediction_id",
        how="left",
    )
    .withColumn("prediction", coalesce(col("prediction"), lit(train_mean)))
    .withColumn("neighbor_count", coalesce(col("neighbor_count"), lit(0)))
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

fallback_count = scored.filter(col("neighbor_count") == 0).count()
test_count = scored.count()

print(f"Item-based CF RMSE on test: {item_cf_rmse:.6f}")
print(f"Fallback predictions: {fallback_count} of {test_count}")

spark.stop()
