#%%
import pandas as pd
from tqdm import tqdm
from recommender_system import *

def apply_weight(df):
    df = df.merge(cosine_similaraty_weight, on='game_appid', how='left')
    df['weighted_predition'] = df['prediction'] * df['similarity_score']
    return df.sort_values(by=['user_steamid', 'weighted_predition'], ascending=[True, False])

# Load users
users_df = load_dataset.load_users()

# Define users
user_ids = users_df["user_steamid"].tolist()
#user_ids = users_df["user_steamid"].sample(100).tolist()
#user_ids = ['76561198147702705', '76561197960463103', '76561197960860132']

# Load models
collaborative_filtering_model = load_models.load_model('collaborative_filtering_cornac')
content_based_model = load_models.load_model('content_based_cornac_cvae')
similar_items_model = load_models.load_model('content_based_sklearn_tfidf')

final_result_all_users = pd.DataFrame()
precision_at_10_all_users = pd.DataFrame()

for user_id in tqdm(user_ids, desc="Processing users"):

    # Load data
    unrated_df = load_dataset.load_unrated(user_id)
    rated_df = load_dataset.load_rated(user_id)
    games_df = load_dataset.load_games()

    # Predictions
    collaborative_filtering_predictions_unrated = predict.predict_unrated(collaborative_filtering_model, unrated_df, games_df)
    collaborative_filtering_predictions_rated = predict.predict_rated(collaborative_filtering_model, rated_df, games_df)
    content_based_predictions_unrated = predict.predict_unrated(content_based_model, unrated_df, games_df)
    content_based_predictions_rated = predict.predict_rated(content_based_model, rated_df, games_df)
    real_top_results = predict.real_top_results(rated_df)

    # Results with rating 5
    rating_5 = real_top_results[real_top_results['rating'] == 5]

    # Similar items
    similar_items_predictions = similar_items_model.predict(rating_5)
    cosine_similaraty_weight = similar_items_predictions[['game_appid', 'similarity_score']].groupby('game_appid').max().reset_index()

    # Weighted ranking collaborative filtering
    collaborative_filtering_weighted = apply_weight(collaborative_filtering_predictions_unrated)
    collaborative_filtering_weighted = collaborative_filtering_weighted.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])

    # Weighted ranking content based
    content_based_weighted = apply_weight(content_based_predictions_unrated)
    content_based_weighted = content_based_weighted.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])

    # Weighted ranking collaborative filtering rated
    collaborative_filtering_weighted_rated = apply_weight(collaborative_filtering_predictions_rated)
    collaborative_filtering_weighted_rated = collaborative_filtering_weighted_rated.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])

    # Weighted ranking content based rated
    content_based_weighted_rated = apply_weight(content_based_predictions_rated)
    content_based_weighted_rated = content_based_weighted_rated.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])

    # Final result
    final_prediction = pd.concat([collaborative_filtering_weighted, content_based_weighted], ignore_index=True)
    final_prediction = final_prediction.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])
    final_prediction = final_prediction.drop_duplicates(subset='game_appid', keep='first')

    # Final result rated
    final_prediction_rated = pd.concat([collaborative_filtering_weighted_rated, content_based_weighted_rated], ignore_index=True)
    final_prediction_rated = final_prediction_rated.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])
    final_prediction_rated = final_prediction_rated.drop_duplicates(subset='game_appid', keep='first')

    # Final result all
    final_prediction_all = pd.concat([final_prediction, final_prediction_rated], ignore_index=True)
    final_prediction_all = final_prediction_all.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])
    final_prediction_all = final_prediction_all.drop_duplicates(subset='game_appid', keep='first')
    final_prediction_all = final_prediction_all.head(10)

    # Precision @10
    predicted_top_10 = final_prediction_all['game_appid'].head(10).tolist()
    real_top_10 = real_top_results['game_appid'].head(10).tolist()
    matches = set(predicted_top_10) & set(real_top_10)
    precision_at_10 = len(matches) / 10
    precision_at_10_all_users = pd.concat([precision_at_10_all_users, pd.DataFrame({'user_steamid': [user_id], 'precision_at_10': [precision_at_10]})], ignore_index=True)
    
    final_prediction_all['user_steamid'] = user_id
    final_result_all_users = pd.concat([final_result_all_users, final_prediction_all], ignore_index=True)

final_result_all_users.to_csv('precision/final_result_all_users.csv', index=False)
precision_at_10_all_users.to_csv('precision/precision_at_10_all_users.csv', index=False)

print(precision_at_10_all_users)
print(precision_at_10_all_users['precision_at_10'].mean().round(2))
# %%