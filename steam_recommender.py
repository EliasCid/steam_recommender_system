import streamlit as st
import pandas as pd
from recommender_system import *

def apply_weight(df):
    df = df.merge(cosine_similaraty_weight, on='game_appid', how='left')
    df['weighted_predition'] = df['prediction'] * df['similarity_score']
    return df.sort_values(by=['user_steamid', 'weighted_predition'], ascending=[True, False])

# Site configuration

st.set_page_config(
    page_title="Steam Recommender System",
    page_icon=":video_game:",
    layout="wide"
)

# Load users
users_df = load_dataset.load_users()

user_id_input = st.sidebar.selectbox(
    'User ID',
    users_df['user_steamid'].tolist(),
    index = None,
    placeholder = "Select user id..."
)

user_name_input = st.sidebar.selectbox(
    'User Name',
    users_df['user_personaname'].tolist(),
    index = None,
    placeholder = "Select user name..."
)

# Load data
if user_name_input is not None:
    user_id_input = users_df[users_df['user_personaname'] == user_name_input]['user_steamid'].values[0]
    unrated_df = load_dataset.load_unrated(user_id_input)
    rated_df = load_dataset.load_rated(user_id_input)

if user_id_input is not None:
    user_name_input = users_df[users_df['user_steamid'] == user_id_input]['user_personaname'].values[0]
    unrated_df = load_dataset.load_unrated(user_id_input)
    rated_df = load_dataset.load_rated(user_id_input)
    
games_df = load_dataset.load_games()

# Site

st.title("Steam Recommender System")
st.divider()
st.markdown("Placeholder for introcuction")

if user_id_input or user_name_input is not None:

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

    # Final result all
    final_prediction_rated = pd.concat([collaborative_filtering_weighted_rated, content_based_weighted_rated], ignore_index=True)
    final_prediction_rated = final_prediction_rated.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])
    final_prediction_rated = final_prediction_rated.drop_duplicates(subset='game_appid', keep='first')

    final_prediction_all = pd.concat([final_prediction, final_prediction_rated], ignore_index=True)
    final_prediction_all = final_prediction_all.sort_values(by=['user_steamid','weighted_predition'], ascending=[True, False])
    final_prediction_all = final_prediction_all.drop_duplicates(subset='game_appid', keep='first')

    # Recommendations
    st.markdown(
    f"## 🎮 Top 10 Game Recommendations for "
    f"[{user_name_input} - {user_id_input}](https://steamcommunity.com/profiles/{user_id_input})"
    )
    st.divider()

    for img_url, name, score, appid in zip(
        final_prediction['game_img_url'].head(10),
        final_prediction['game_name'].head(10),
        final_prediction['weighted_predition'].head(10),
        final_prediction['game_appid'].head(10)
    ):
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        with col1:
            st.image(img_url)
        with col2:
            st.markdown(f"{name}")
        with col3:
            st.markdown(f"🎯 **Score:** {round(score, 2)}")
        with col4:
            steam_url = f"https://store.steampowered.com/app/{appid}"
            st.markdown(f"[🔗 View on Steam]({steam_url})", unsafe_allow_html=True)

    # Analysis
    st.markdown("## Analysis")
    st.divider()

    st.markdown("### First recommendations")

    col1, col2 = st.columns(2)

    col1.write("Colaborative Filtering")
    col1.dataframe(collaborative_filtering_predictions_unrated[['game_appid','game_name', 'prediction']].head(10).reset_index(drop=True))

    col2.write("Content Based")
    col2.dataframe(content_based_predictions_unrated[['game_appid','game_name', 'prediction']].head(10).reset_index(drop=True))

    st.markdown("### Adjusted recommendations")

    col1, col2 = st.columns(2)

    col1.write("Colaborative Filtering")
    col1.dataframe(collaborative_filtering_weighted[['game_appid','game_name', 'weighted_predition']].head(10).reset_index(drop=True))

    col2.write("Content Based")
    col2.dataframe(content_based_weighted[['game_appid','game_name', 'weighted_predition']].head(10).reset_index(drop=True))

    st.markdown("### Final recommendations considering all dataset")

    # Precision @10
    predicted_top_10 = final_prediction_all['game_appid'].head(10).tolist()
    real_top_10 = real_top_results['game_appid'].head(10).tolist()
    matches = set(predicted_top_10) & set(real_top_10)
    precision_at_10 = len(matches) / 10
    st.markdown(f"🎯 Precision@10: {precision_at_10 * 100:.0f}%")
    
    col1, col2 = st.columns(2)

    col1.write("Real Top 10")
    col1.dataframe(real_top_results[['game_appid','game_name', 'game_playtime_forever']].head(10).reset_index(drop=True))

    col2.write("Predicted Top 10")
    col2.dataframe(final_prediction_all[['game_appid','game_name', 'weighted_predition']].head(10).reset_index(drop=True))

else:
    st.write("Please select a user")