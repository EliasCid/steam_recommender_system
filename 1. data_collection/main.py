#%%
import os
import glob
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from steam_scraper import *
import random

# Function to load JSON file if it exists for today's date
def load_json_if_exists(directory, prefix):
    date_str = datetime.now().strftime('%Y-%m-%d')
    files = glob.glob(f'{directory}/{prefix}_{date_str}*.json')
    if files:
        with open(files[0], 'r') as file:
            return json.load(file)
    return None

# Function to save JSON data to a file with the current timestamp
def save_json(data, directory, prefix):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'{directory}/{prefix}_{timestamp}.json', 'w') as file:
        json.dump(data, file, indent=4)

# Main function
def main():
    # Load environment variables like API_KEY
    load_dotenv()

    # Set parameters
    STEAM_ID = '76561198147702705' # Initial steamid
    max_unique_ids = 260_000 # Maximum number of unique steamids to fetch
    max_workers = 2 # Maximum number of threads to use
    batch_size = 100 # Number of games to fetch in each batch

    # Start total time tracking
    total_start_time = time.time()

    # Step 1: Fetch users data
    print('Step 1: Fetching users data...')
    step_start_time = time.time()
    friends_data = load_json_if_exists('db/users', 'users')
    unique_ids_data = load_json_if_exists('db/unique_ids', 'unique_ids')

    if friends_data is None or unique_ids_data is None:
        print('No existing users data found. Fetching from API...')
        friends_data, unique_steamids = users.fetch_friend_data(
            api_key=os.getenv('API_KEY'),
            initial_steamid=STEAM_ID,
            max_unique_ids=max_unique_ids
        )
        save_json(friends_data, 'db/users', 'users')
        save_json(unique_steamids, 'db/unique_ids', 'unique_ids')
    else:
        unique_steamids = unique_ids_data

    step_time = (time.time() - step_start_time) / 60
    print(f'Step 1 completed in {step_time:.2f} minutes.')

    # Step 2: Fetch games data
    print('Step 2: Fetching games data...')
    step_start_time = time.time()
    games_data = load_json_if_exists('db/games', 'games')

    if games_data is None:
        print('No existing games data found. Fetching from API...')
        games_data = {}

        def fetch_data(steamid, max_retries=5):
            for attempt in range(max_retries):
                try:
                    response = games.get_games_data(
                        key=os.getenv('API_KEY'),
                        steamid=steamid,
                        include_appinfo=True,
                        include_played_free_games=True
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return steamid, data.get('response', {}).get('games', [])
                    elif response.status_code == 429:
                        wait = 2 ** attempt + random.uniform(0, 1)  # exponential backoff
                        print(f'Rate limited (429) for SteamID {steamid}. Retrying in {wait:.2f} seconds...')
                        time.sleep(wait)
                    else:
                        print(f'Non-200 response for SteamID {steamid}: {response.status_code} - {response.text}')
                        break
                except Exception as e:
                    print(f'Error fetching data for SteamID {steamid}: {e}')
                    break
            return steamid, []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fetch_data, steamid): steamid for steamid in unique_steamids}

            for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching user games'):
                steamid, user_games = future.result()
                games_data[steamid] = user_games

        save_json(games_data, 'db/games', 'games')

    unique_appids = {game['appid'] for steamid, games_list in games_data.items() for game in games_list}
    step_time = (time.time() - step_start_time) / 60
    print(f'Step 2 completed in {step_time:.2f} minutes.')

    # Step 3: Fetch user details
    print('Step 3: Fetching user details...')
    step_start_time = time.time()
    user_details = load_json_if_exists('db/users_detail', 'users_detail')

    if user_details is None:
        print('No existing user details found. Fetching from API...')
        user_details = {}

        def fetch_user_detail(steamid, max_retries=5):
            for attempt in range(max_retries):
                try:
                    response = users_detail.get_users_detail(
                        key=os.getenv('API_KEY'),
                        steamids=steamid
                    )
                    if response.status_code == 200:
                        data = response.json()
                        players = data.get('response', {}).get('players', [])
                        if players:
                            return steamid, players[0]
                        else:
                            print(f'No player data found for SteamID {steamid}.')
                            return steamid, None
                    elif response.status_code == 429:
                        wait = 2 ** attempt + random.uniform(0, 1)  # exponential backoff
                        print(f'Rate limited (429) for SteamID {steamid}. Retrying in {wait:.2f} seconds...')
                        time.sleep(wait)
                    else:
                        print(f'Non-200 response for SteamID {steamid}: {response.status_code} - {response.text}')
                        break
                except Exception as e:
                    print(f'Error fetching user detail for SteamID {steamid}: {e}')
                    break
            return steamid, None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fetch_user_detail, steamid): steamid for steamid in unique_steamids}

            for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching user details'):
                steamid, player_data = future.result()
                if player_data:
                    user_details[steamid] = player_data

        save_json(user_details, 'db/users_detail', 'users_detail')

    step_time = (time.time() - step_start_time) / 60
    print(f'Step 3 completed in {step_time:.2f} minutes.')

    # Step 4: Scrape game details
    print('Step 4: Scraping game details...')
    step_start_time = time.time()
    game_details = load_json_if_exists('db/games_detail', 'games_detail')

    if game_details is None:
        print('No existing game details found. Scraping data in batches...')
        game_details, failed_games = games_detail.scrape_games_in_batches(
            list(unique_appids), 
            batch_size=batch_size, 
            max_workers=max_workers
        )
        save_json(game_details, 'db/games_detail', 'games_detail')
        save_json(failed_games, 'db/failed_games', 'failed_games')

    step_time = (time.time() - step_start_time) / 60
    print(f'Step 4 completed in {step_time:.2f} minutes.')

    # Total time
    total_time = (time.time() - total_start_time) / 60
    print(f'All steps completed successfully in {total_time:.2f} minutes!')

if __name__ == '__main__':
    main()
# %%