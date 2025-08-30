# This Script implements functions to load datasets from parquet files.
#%%
import os
import glob
from pyspark.sql import SparkSession

# Start Spark
spark = (
    SparkSession.builder
    .appName('RecommenderSystem')
      .config('spark.driver.memory', '12g')            
      .config('spark.driver.maxResultSize', '4g')      
    .getOrCreate()
)

main_folder = os.path.dirname(os.getcwd())

# Load unrated dataset
def load_unrated(user_ids):
    unrated_df_folder = os.path.join(main_folder, '4. model', 'db', 'unrated')
    unrated_df_files = glob.glob(os.path.join(unrated_df_folder, '*.parquet'))

    if unrated_df_files:
        latest_file = max(unrated_df_files, key=os.path.getctime)
        df = spark.read.parquet(latest_file)
    else:
        print('No parquet files found in the unrated folder.')

    unrated_df = df.filter(df['user_steamid'].isin(user_ids))
    unrated_df = unrated_df.toPandas()

    return unrated_df

# Load rated dataset
def load_rated(user_ids):
    rated_df_folder = os.path.join(main_folder, '4. model', 'db', 'rated')
    rated_df_files = glob.glob(os.path.join(rated_df_folder, '*.parquet'))

    if rated_df_files:
        latest_file = max(rated_df_files, key=os.path.getctime)
        df = spark.read.parquet(latest_file)
    else:
        print('No parquet files found in the rated folder.')

    rated_df = df.filter(df['user_steamid'].isin(user_ids))
    rated_df = rated_df.toPandas()

    return rated_df

# Load games dataset
def load_games():
    games_df_folder = os.path.join(main_folder, '4. model', 'db', 'games')
    games_df_files = glob.glob(os.path.join(games_df_folder, '*.parquet'))

    if games_df_files:
        latest_file = max(games_df_files, key=os.path.getctime)
        df = spark.read.parquet(latest_file)
    else:
        print('No parquet files found in the games folder.')

    games_df = df.toPandas()

    return games_df

# Load users dataset
def load_users():
    users_df_folder = os.path.join(main_folder, '4. model', 'db', 'users')
    users_df_files = glob.glob(os.path.join(users_df_folder, '*.parquet'))

    if users_df_files:
        latest_file = max(users_df_files, key=os.path.getctime)
        df = spark.read.parquet(latest_file)
    else:
        print('No parquet files found in the users folder.')

    users_df = df.toPandas()

    return users_df
# %%
