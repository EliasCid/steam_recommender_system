#%%
import pandas as pd
from recommender_system import *

# Define user
user_id = '76561198147702705'
#user_id = '76561197960463103'
#user_id = '76561197960860132'

# Load data
unrated_df = load_dataset.load_unrated(user_id)
rated_df = load_dataset.load_rated(user_id)
games_df = load_dataset.load_games()

# Load models
collaborative_filtering_model = load_models.load_model('collaborative_filtering_cornac')
content_based_model = load_models.load_model('content_based_cornac_cvae')
similar_items_model = load_models.load_model('content_based_sklearn_tfidf')

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
collaborative_filtering_weighted = collaborative_filtering_predictions_unrated.merge(cosine_similaraty_weight, on='game_appid', how='left')
collaborative_filtering_weighted['weighted_predition'] = collaborative_filtering_weighted['prediction'] * collaborative_filtering_weighted['similarity_score']
collaborative_filtering_weighted = collaborative_filtering_weighted.sort_values(by='weighted_predition', ascending=False)

# Weighted ranking content based
content_based_weighted = content_based_predictions_unrated.merge(cosine_similaraty_weight, on='game_appid', how='left')
content_based_weighted['weighted_predition'] = content_based_weighted['prediction'] * content_based_weighted['similarity_score']
content_based_weighted = content_based_weighted.sort_values(by='weighted_predition', ascending=False)

# Weighted ranking collaborative filtering rated
collaborative_filtering_weighted_rated = collaborative_filtering_predictions_rated.merge(cosine_similaraty_weight, on='game_appid', how='left')
collaborative_filtering_weighted_rated['weighted_predition'] = collaborative_filtering_weighted_rated['prediction'] * collaborative_filtering_weighted_rated['similarity_score']
collaborative_filtering_weighted_rated = collaborative_filtering_weighted_rated.sort_values(by='weighted_predition', ascending=False)

# Weighted ranking content based rated
content_based_weighted_rated = content_based_predictions_rated.merge(cosine_similaraty_weight, on='game_appid', how='left')
content_based_weighted_rated['weighted_predition'] = content_based_weighted_rated['prediction'] * content_based_weighted_rated['similarity_score']
content_based_weighted_rated = content_based_weighted_rated.sort_values(by='weighted_predition', ascending=False)

# Final result
final_prediction = pd.concat([collaborative_filtering_weighted, content_based_weighted], ignore_index=True)
final_prediction = final_prediction.sort_values(by='weighted_predition', ascending=False)
final_prediction = final_prediction.drop_duplicates(subset='game_appid', keep='first')

#Final result rated
final_prediction_rated = pd.concat([collaborative_filtering_weighted_rated, content_based_weighted_rated], ignore_index=True)
final_prediction_rated = final_prediction_rated.sort_values(by='weighted_predition', ascending=False)
final_prediction_rated = final_prediction_rated.drop_duplicates(subset='game_appid', keep='first')

# Show result
print(collaborative_filtering_predictions_unrated[['game_appid', 'prediction', 'game_name']].head(10))
print(content_based_predictions_unrated[['game_appid', 'prediction', 'game_name']].head(10))
print(final_prediction[['game_appid', 'weighted_predition', 'game_name']].head(10))

print(collaborative_filtering_predictions_rated[['game_appid', 'prediction', 'game_name']].head(10))
print(content_based_predictions_rated[['game_appid', 'prediction', 'game_name']].head(10))
print(final_prediction_rated[['game_appid', 'weighted_predition', 'game_name']].head(10))
# %%
