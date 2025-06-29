#%%
import mlflow
import pandas as pd
import os
import glob
from pyspark.sql import SparkSession

# Start Spark
spark = (
    SparkSession.builder
    .appName("UnratedUserGamePairs")
    .getOrCreate()
)

# Define user
user_id = '76561198147702705'

# Load unrated dataset
main_folder = os.path.dirname(os.getcwd())

unrated_df_folder = os.path.join(main_folder, "4. model", "db", "unrated")
unrated_df_files = glob.glob(os.path.join(unrated_df_folder, "*.parquet"))

if unrated_df_files:
    latest_file = max(unrated_df_files, key=os.path.getctime)
    df = spark.read.parquet(latest_file)
else:
    print("No parquet files found in the unrated folder.")

unrated_df = df.filter(df["user_steamid"] == user_id)
unrated_df = unrated_df.toPandas()

# Load rated dataset
rated_df_folder = os.path.join(main_folder, "4. model", "db", "rated")
rated_df_files = glob.glob(os.path.join(rated_df_folder, "*.parquet"))

if rated_df_files:
    latest_file = max(rated_df_files, key=os.path.getctime)
    df = spark.read.parquet(latest_file)
else:
    print("No parquet files found in the rated folder.")

rated_df = df.filter(df["user_steamid"] == user_id)
rated_df = rated_df.toPandas()

# Load games dataset
games_df_folder = os.path.join(main_folder, "4. model", "db", "games")
games_df_files = glob.glob(os.path.join(games_df_folder, "*.parquet"))

if games_df_files:
    latest_file = max(games_df_files, key=os.path.getctime)
    df = spark.read.parquet(latest_file)
else:
    print("No parquet files found in the games folder.")

games_df = df.toPandas()

# Load model
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
client = mlflow.client.MlflowClient()
version = max((int(v.version) for v in client.get_latest_versions("collaborative_filtering_cornac")))
model = mlflow.pyfunc.load_model(f'models:/collaborative_filtering_cornac/{version}')

# Call predict on Unrated dataset
predictions_unrated = model.predict(unrated_df)
predictions_unrated = predictions_unrated.reset_index(drop=True)
predictions_unrated = (
    predictions_unrated
    .merge(games_df, on="game_appid", how="left")
    .sort_values("prediction", ascending=False)
)

# Call predict on Rated dataset
predictions_rated = model.predict(rated_df)
predictions_rated = predictions_rated.reset_index(drop=True)
predictions_rated = (
    predictions_rated
    .merge(games_df, on="game_appid", how="left")
    .sort_values("prediction", ascending=False)
)

# Top results and merge with games_df
predictions_unrated = pd.merge(predictions_unrated, games_df, on="game_appid", how="left")
predictions_unrated = predictions_unrated.sort_values(by="prediction", ascending=False)

predictions_rated = pd.merge(predictions_rated, games_df, on="game_appid", how="left")
predictions_rated = predictions_rated.sort_values(by="prediction", ascending=False)

rated_df_filtered = rated_df[['game_appid', 'game_name', 'game_playtime_forever', 'rating']].sort_values(by="game_playtime_forever", ascending=False)

print(predictions_unrated.head(10))
print(predictions_rated.head(10))
print(rated_df_filtered.head(10))
# %%
