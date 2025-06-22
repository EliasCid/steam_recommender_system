#%% 
import os
import glob
import json
import datetime
import pandas as pd
import re

def save_parquet_full(data):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    df = pd.DataFrame(data)
    df.to_parquet(f"db/full/{now}.parquet", index=False)

def save_parquet_filtered(data):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    df = pd.DataFrame(data)
    df.to_parquet(f"db/filtered/{now}.parquet", index=False)

main_folder = os.path.dirname(os.getcwd())


# Load users_detail
users_detail_folder = os.path.join(main_folder, "1. data_collection", "db", "users_detail")
users_detail_files = glob.glob(os.path.join(users_detail_folder, "*.json"))

if users_detail_files:
    latest_users_detail_file = max(users_detail_files, key=os.path.getctime)
    with open(latest_users_detail_file, 'r') as file:
        users_detail_json = json.load(file)
    users_detail = pd.DataFrame.from_dict(users_detail_json, orient='index')
else:
    print("No JSON files found in the users detail folder.")

# Load games
games_folder = os.path.join(main_folder, "1. data_collection", "db", "games")
games_files = glob.glob(os.path.join(games_folder, "*.json"))

if games_files:
    latest_games_file = max(games_files, key=os.path.getctime)
    with open(latest_games_file, 'r') as file:
        games_json = json.load(file)
    rows = []
    for steam_id, games in games_json.items():
        if games:
            for game in games:
                row = {"steam_id": steam_id, **game}
                rows.append(row)
        else:
            rows.append({"steam_id": steam_id})
    games = pd.DataFrame(rows)
else:
    print("No JSON files found in the games folder.")

# Load games_detail from files with the same date
games_detail_folder = os.path.join(main_folder, "1. data_collection", "db", "games_detail")
games_detail_files = glob.glob(os.path.join(games_detail_folder, "*.json"))

if games_detail_files:
    # Extract the latest file's date
    latest_file = max(games_detail_files, key=os.path.getctime)
    latest_file_date = re.search(r"\d{4}-\d{2}-\d{2}", latest_file).group()  # Extract date from filename

    # Filter files with the same date
    same_date_files = [f for f in games_detail_files if latest_file_date in f]

    # Load and combine only files from the same date
    games_detail_list = []
    for file_path in same_date_files:
        with open(file_path, 'r') as file:
            games_detail_json = json.load(file)
            games_detail_list.append(pd.DataFrame(games_detail_json))
    games_detail = pd.concat(games_detail_list, ignore_index=True)  # Combine selected files
else:
    print("No JSON files found in the games detail folder.")

games_detail.drop_duplicates(subset='gameid', inplace=True)

# Add prefixes to DataFrames
users_detail_prefixed = users_detail.add_prefix("user_")
games_prefixed = games.add_prefix("game_")
games_detail_prefixed = games_detail.add_prefix("game_detail_")

# Merge DataFrames
merged_users_games = pd.merge(
    users_detail_prefixed,
    games_prefixed, 
    how="left", 
    left_on="user_steamid", 
    right_on="game_steam_id"
)
merged_users_games['game_appid'] = (
    merged_users_games['game_appid']
    .fillna(0)  # Replace NaNs with 0
    .astype('int64')  # Convert to int64
)

merged_users_games_details = pd.merge(
    merged_users_games,
    games_detail_prefixed,
    how="left",
    left_on="game_appid",
    right_on="game_detail_gameid"
)

# Save full DataFrame
save_parquet_full(merged_users_games_details)

# Filter specific columns and save
filtered_df = merged_users_games_details[[
    "user_steamid",
    "user_personaname",
    "user_loccountrycode",
    "game_appid",
    "game_name",
    "game_playtime_forever",
    "game_detail_release_date",
    "game_detail_developer",
    "game_detail_publisher",
    "game_detail_genres",
    "game_detail_tags",
    "game_detail_rating"
]]

save_parquet_filtered(filtered_df)
# %%
