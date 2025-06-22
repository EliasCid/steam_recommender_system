#%%
import mlflow
import pandas as pd
from surprise import Reader, Dataset

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
client = mlflow.client.MlflowClient()
version = max((int(v.version) for v in client.get_latest_versions("collaborative_filtering_surprise")))

# Prepare input dataframe
input_df = pd.DataFrame([{"user_id": '76561197976968076', "item_id": '440'},{"user_id": '76561197976968076', "item_id": '531960'},{"user_id": '76561197960314370', "item_id": '570'}])
model = mlflow.pyfunc.load_model(f'models:/collaborative_filtering_surprise/{version}')

# Call predict
predictions = model.predict(input_df)

print(predictions)
# %%
