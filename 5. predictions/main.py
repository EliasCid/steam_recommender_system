#%%
from recommender_system import *

# Define user
user_id = '76561198147702705'

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

# Top 10 real results
top_10 = real_top_results[['game_appid']].head(10)

# Similar items
similar_items_predictions = similar_items_model.predict(top_10)

# Print results
print(collaborative_filtering_predictions_unrated.head(10))
print(collaborative_filtering_predictions_rated.head(10))
print(content_based_predictions_unrated.head(10))
print(content_based_predictions_rated.head(10))
print(real_top_results.head(10))
print(similar_items_predictions.head(10))


# %%
