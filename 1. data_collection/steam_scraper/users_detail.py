import requests

def get_users_detail(**kwargs):
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002"
    response = requests.get(url,params=kwargs)
    return response