#%%
import pandas as pd
import numpy as np
import glob
import os
import mlflow
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Model wrapper
class ContentBasedRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.vectorizer = pickle.load(open(context.artifacts["tfidf_model"], "rb"))
        self.tfidf_matrix = pickle.load(open(context.artifacts["tfidf_matrix"], "rb"))
        self.df_unique    = pickle.load(open(context.artifacts["df_unique"],  "rb"))

    def predict(self, context, model_input):
        all_recs = []
        for _, row in model_input.iterrows():
            game_id = row["game_appid"]
            # find its index in df_unique
            match = self.df_unique[self.df_unique["game_appid"] == game_id]
            if match.empty:
                # if you want, append an error row; here we just skip
                continue
            i = match.index[0]

            # compute similarities
            sims = cosine_similarity(self.tfidf_matrix[i], self.tfidf_matrix).flatten()
            sims[i] = -1

            # top N
            top_n = sims.argsort()[-5000:][::-1]
            recs = self.df_unique.iloc[top_n].copy()
            recs["similarity_score"] = sims[top_n]
            # tag which input game this block belongs to
            recs["query_game_appid"]  = game_id

            # result columns
            all_recs.append(
                recs[[
                   "query_game_appid",
                   "game_appid",
                   "similarity_score"
                ]]
            )

        if not all_recs:
            return pd.DataFrame()
        return pd.concat(all_recs, ignore_index=True)

# Validation metrics
def avg_cosine_similarity(tfidf_matrix, index, top_indices):
    similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix[top_indices]).flatten()
    return np.mean(similarities)

def avg_feature_overlap(df, query_idx, top_indices):
    query_features = set(df.iloc[query_idx]['features_str'].split(','))
    scores = []
    for i in top_indices:
        rec_features = set(df.iloc[i]['features_str'].split(','))
        scores.append(len(query_features & rec_features) / len(query_features | rec_features))
    return np.mean(scores)

# Load dataset
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

df_folder = os.path.join(main_folder, "3. data_analysis", "db")
df_files = glob.glob(os.path.join(df_folder, "*.parquet"))

if df_files:
    latest_file = max(df_files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)
else:
    print("No parquet files found in the filtered folder.")

df['features_str'] = df['feature'].apply(lambda x: ','.join(x) if isinstance(x, (list, np.ndarray)) else str(x))
df['features_str'] = (
    df['game_detail_rating'] + ',' +
    df['features_str']
)  
df_unique = df[['game_appid', 'features_str']].drop_duplicates().reset_index(drop=True)

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=311634255185090727)

with mlflow.start_run():

    mlflow.sklearn.autolog()
    
    # TF-IDF matrix
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_unique['features_str'])

    # Top-N similar items to retrieve   
    TOP_N = 5000
    similar_items = []

    # Calculate similarities row by row
    for i in tqdm(range(tfidf_matrix.shape[0]), desc="Finding top-N similar games"):
        # Get cosine similarity for item i with all others
        cosine_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()

        # Remove self-comparison by setting own index to -1
        cosine_sim[i] = -1

        # Get indices of top-N most similar items
        top_indices = cosine_sim.argsort()[-TOP_N:][::-1]
        top_similar_ids = df_unique.iloc[top_indices]['game_appid'].tolist()

        # Store result as tuple: (game_appid, [top-N similar game_appids])
        similar_items.append((df_unique.iloc[i]['game_appid'], top_similar_ids))

    # Save trained model
    with open('tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('df_unique.pkl', 'wb') as f:
        pickle.dump(df_unique, f)

    artifacts = {
        "tfidf_model": "tfidf_model.pkl",
        "tfidf_matrix": "tfidf_matrix.pkl",
        "df_unique": "df_unique.pkl"
    }

    avg_cos_sims = []
    avg_feature_overlaps = []

    # Collect validation metrics
    avg_cos_sims.append(avg_cosine_similarity(tfidf_matrix, i, top_indices))
    avg_feature_overlaps.append(avg_feature_overlap(df_unique, i, top_indices))

    metrics = {
        "avg_cosine_similarity": np.mean(avg_cos_sims),
        "avg_feature_overlap": np.mean(avg_feature_overlaps)
    }

    mlflow.log_metrics(metrics)

    # Log as an MLflow PyFunc model
    mlflow.pyfunc.log_model(
        name="model",
        python_model=ContentBasedRecommender(),
        artifacts=artifacts
    )
# %%
