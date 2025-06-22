#%%
import os
import glob
import mlflow
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

#Model Wrapper
class SurpriseModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        import pandas as pd

        if not {'user_id', 'item_id'}.issubset(model_input.columns):
            raise ValueError("Input DataFrame must contain 'user_id' and 'item_id' columns.")
        
        predictions = []
        for _, row in model_input.iterrows():
            pred = self.model.predict(uid=row['user_id'], iid=row['item_id'])
            predictions.append({
                "user_id": row['user_id'],
                "item_id": row['item_id'],
                "estimated_rating": pred.est
            })

        return pd.DataFrame(predictions)

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=488482081013367275)

# Load dataset
main_folder = os.path.dirname(os.getcwd())

df_folder = os.path.join(main_folder, "3. data_analysis", "db")
df_files = glob.glob(os.path.join(df_folder, "*.parquet"))

if df_files:
    latest_file = max(df_files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)
else:
    print("No parquet files found in the filtered folder.")

df_filtered = df[['user_steamid', 'game_appid', 'rating']] 

# A reader is still needed but only the rating_scale param is required.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df_filtered[['user_steamid', 'game_appid', 'rating']], reader)

'''
# Using movielens-100K for tests
data = Dataset.load_builtin("ml-100k")
'''

with mlflow.start_run():

    param_grid = {"n_epochs": [10, 100], "lr_all": [0.001, 0.09], "reg_all": [0.1, 2]}
    mlflow.log_params(param_grid)

    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae", "fcp"], cv=3)
    gs.fit(data)

    mlflow.log_metric("rmse", gs.best_score["rmse"])
    mlflow.log_metric("mae", gs.best_score["mae"])
    mlflow.log_metric("fcp", gs.best_score["fcp"])
    
    best_params_rmse = gs.best_params["rmse"]
    best_params_mae = gs.best_params["mae"]
    best_params_fcp = gs.best_params["fcp"]

    mlflow.log_dict(best_params_rmse, "best_params_rmse.json")
    mlflow.log_dict(best_params_mae, "best_params_mae.json")
    mlflow.log_dict(best_params_fcp, "best_params_fcp.json")

    # Fit final model on full trainset
    trainset = data.build_full_trainset()
    best_model = gs.best_estimator['rmse']
    best_model.fit(trainset)

    # Save trained model
    with open('surprise_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    artifacts = {
        "model": "surprise_model.pkl"
    }

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SurpriseModelWrapper(),
        artifacts=artifacts
    )
    # %%