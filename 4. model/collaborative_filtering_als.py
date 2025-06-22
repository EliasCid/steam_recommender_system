#%%
import os
import glob
import mlflow
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import collect_list, expr

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=348893414371187222)

# Start Spark session (local mode)
spark = SparkSession.builder \
    .appName("LoadParquetSpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

# Reduce log output to only errors
spark.sparkContext.setLogLevel("ERROR")

# Load JAVA
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Load dataset
main_folder = os.path.dirname(os.getcwd())

df_folder = os.path.join(main_folder, "3. data_analysis", "db")
df_files = glob.glob(os.path.join(df_folder, "*.parquet"))

if df_files:
    latest_file = max(df_files, key=os.path.getctime)
    df = spark.read.parquet(latest_file)
else:
    print("No parquet files found in the filtered folder.")

# Collaborative Filtering (ALS)
indexer_user = StringIndexer(inputCol="user_steamid", outputCol="userIndex").fit(df)
indexer_game = StringIndexer(inputCol="game_appid", outputCol="gameIndex").fit(df)

df_indexed = indexer_user.transform(df)
df_indexed = indexer_game.transform(df_indexed)

train, test = df_indexed.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run():

    alpha = 10
    regParam = 0.1
    rank = 10
    maxIter = 10

    als = ALS(
        userCol="userIndex",
        itemCol="gameIndex",
        ratingCol="rating",
        implicitPrefs=True,
        alpha=alpha,
        regParam=regParam,
        rank=rank,
        maxIter=maxIter,
        coldStartStrategy="drop",  # Drops users/items unseen in training
        nonnegative=True
    )

    model = als.fit(train)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("rank", rank)
    mlflow.log_param("maxIter", maxIter)
    mlflow.log_param("regParam", regParam)
    mlflow.spark.log_model(model, "als_model")

    predictions = model.transform(test)

    # True games played (ground truth)
    true_labels = test.groupBy("userIndex").agg(collect_list("gameIndex").alias("true_items"))

    # Top-K recommendations
    top_k = 10
    top_k_recs = model.recommendForAllUsers(top_k).withColumn(
        "pred_items",
        expr("transform(recommendations, x -> x.gameIndex)")
    ).select("userIndex", "pred_items")

    # Join them
    joined = top_k_recs.join(true_labels, on="userIndex").dropna()

    results_pd = joined.select("pred_items", "true_items").toPandas()
    def precision_at_k(pred, true, k):
        pred_k = pred[:k]
        return len(set(pred_k) & set(true)) / k

    def recall_at_k(pred, true, k):
        pred_k = pred[:k]
        return len(set(pred_k) & set(true)) / len(true) if true else 0

    def average_precision(pred, true, k):
        score = 0.0
        hits = 0
        for i, p in enumerate(pred[:k]):
            if p in true:
                hits += 1
                score += hits / (i + 1)
        return score / min(len(true), k) if true else 0

    # Apply metrics
    precisions = []
    recalls = []
    maps = []

    for _, row in results_pd.iterrows():
        pred = row["pred_items"]
        true = row["true_items"]
        
        precisions.append(precision_at_k(pred, true, top_k))
        recalls.append(recall_at_k(pred, true, top_k))
        maps.append(average_precision(pred, true, top_k))

    # Final scores
    mlflow.log_metric(f"Precision", float(np.mean(precisions)))
    mlflow.log_metric(f"Recall", float(np.mean(recalls)))
    mlflow.log_metric(f"MAP", float(np.mean(maps)))
# %%
