# This script creates a database of users and their games for predictions.
#%%
import os, glob
from pyspark.sql import SparkSession
import datetime

# 1. Start Spark
spark = (
    SparkSession.builder
    .appName('db_for_predictions')
    .config('spark.driver.memory', '16g')
    .config('spark.executor.memory', '16g')
    .getOrCreate()
)

# 2. Load latest parquet
main_folder = os.path.dirname(os.getcwd())
df_folder   = os.path.join(main_folder, '3. data_analysis', 'db')
df_files    = glob.glob(os.path.join(df_folder, '*.parquet'))
if not df_files:
    raise FileNotFoundError(f'No parquet files found in {df_folder}')
latest_file = max(df_files, key=os.path.getctime)
df = spark.read.parquet(latest_file)

# 3. Select only needed cols
df_filtered = df.select('user_steamid', 'user_personaname', 'game_appid', 'game_name', 'game_img_url','game_playtime_forever', 'rating')

# 4. Distinct users & games
users = df_filtered.select('user_steamid').distinct()
games = df_filtered.select('game_appid').distinct()
games_with_name = df_filtered.select('game_appid', 'game_name', 'game_img_url').distinct()
users_with_name = df_filtered.select('user_steamid', 'user_personaname').distinct()

# 5. All combos → anti‐join rated
all_pairs   = users.crossJoin(games)
rated_pairs = df_filtered.select('user_steamid','game_appid').distinct()

unrated = all_pairs.join(
    rated_pairs,
    on=['user_steamid','game_appid'],
    how='left_anti'
)

# 6. Compute count
num_unrated = unrated.count()

print(f'Total number of unrated (user, game) pairs: {num_unrated:,}')

# Save result
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
unrated.coalesce(1).write.parquet(f'db/unrated/{now}.parquet')
df_filtered.coalesce(1).write.parquet(f'db/rated/{now}.parquet')
games_with_name.coalesce(1).write.parquet(f'db/games/{now}.parquet')
users_with_name.coalesce(1).write.parquet(f'db/users/{now}.parquet')
# %%