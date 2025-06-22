#%%
import mlflow
import pandas as pd

# Load data
input_df = pd.DataFrame({'game_appid': [275850, 3416070]}) 

# Load model
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
client = mlflow.client.MlflowClient()
version = max((int(v.version) for v in client.get_latest_versions("content_based_sklearn_tfidf")))
model = mlflow.pyfunc.load_model(f'models:/content_based_sklearn_tfidf/{version}')

# Call predict
predictions = model.predict(input_df)
predictions
# %%
