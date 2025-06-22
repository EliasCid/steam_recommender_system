import requests

def get_games_data(**kwargs):
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    response = requests.get(url,params=kwargs)
    return response