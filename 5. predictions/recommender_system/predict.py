# This Script implements functions to predict ratings for unrated and rated datasets.
#%%
import pandas as pd

# Call predict on Unrated dataset
def predict_unrated(model, unrated_df, games_df):
    predictions_unrated = model.predict(unrated_df)
    predictions_unrated = predictions_unrated.reset_index(drop=True)
    predictions_unrated = pd.merge(predictions_unrated, games_df, on='game_appid', how='left')
    predictions_unrated = predictions_unrated.sort_values(by='prediction', ascending=False)
    
    return predictions_unrated

# Call predict on Rated dataset
def predict_rated(model, rated_df, games_df):
    predictions_rated = model.predict(rated_df)
    predictions_rated = predictions_rated.reset_index(drop=True)
    predictions_rated = pd.merge(predictions_rated, games_df, on='game_appid', how='left')
    predictions_rated = predictions_rated.sort_values(by='prediction', ascending=False)
    
    return predictions_rated

# Top results and merge with games_df
def real_top_results(rated_df):
    real_top_results = rated_df[['game_appid', 'game_name', 'game_playtime_forever', 'rating']].sort_values(by='game_playtime_forever', ascending=False)

    return real_top_results
# %%