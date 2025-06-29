#%%
import mlflow

# Load model
def load_model(model_name):
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    client = mlflow.client.MlflowClient()
    version = max((int(v.version) for v in client.get_latest_versions(model_name)))
    model = mlflow.pyfunc.load_model(f'models:/{model_name}/{version}')
    
    return model
# %%
