# This Script implements a content-based filtering model using CVAE from Cornac and logs it to MLflow.
#%%
import os
import glob
import pandas as pd
import numpy as np
import dill as pickle
import cornac
import mlflow
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer
from cornac.models import CVAE
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

# MLflow configuration
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=433389737882269030)

# Model Wrapper
class CornacModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the Cornac model artifact
        with open(context.artifacts['model'], 'rb') as f:
            self.model = pickle.load(f)

        # Grab global mean if it exists
        self.global_mean = getattr(self.model, 'global_mean', None)

        # Explicitly check for user biases
        if hasattr(self.model, 'u_biases'):
            self.user_bias = self.model.u_biases
        elif hasattr(self.model, 'u_bias'):
            self.user_bias = self.model.u_bias
        else:
            self.user_bias = None

        # Explicitly check for item biases
        if hasattr(self.model, 'i_biases'):
            self.item_bias = self.model.i_biases
        elif hasattr(self.model, 'i_bias'):
            self.item_bias = self.model.i_bias
        else:
            self.item_bias = None

        # Load raw→internal index maps
        self.user_map = getattr(self.model, 'user_map', getattr(self.model, 'uid_map', None))
        self.item_map = getattr(self.model, 'item_map', getattr(self.model, 'iid_map', None))

    def predict(self, context, model_input):
        if not {'user_steamid', 'game_appid'}.issubset(model_input.columns):
            raise ValueError("Input must have 'user_steamid' and 'game_appid' columns.")

        out = []
        for _, row in model_input.iterrows():
            raw_u, raw_i = row['user_steamid'], row['game_appid']

            # Helper to map raw → internal index
            def to_index(raw, mapping):
                if mapping is None:
                    return int(raw)
                if raw in mapping:
                    return mapping[raw]
                try:
                    return mapping.get(int(raw), None)
                except:
                    return None

            uidx = to_index(raw_u, self.user_map)
            iidx = to_index(raw_i, self.item_map)

            # Try CF score
            score = None
            if uidx is not None and iidx is not None:
                try:
                    s = self.model.score(uidx, iidx)
                    score = float(s[0] if hasattr(s, '__iter__') else s)
                except ScoreException:
                    score = None

            # Fallback logic
            if score is None:
                # both new
                if uidx is None and iidx is None and self.global_mean is not None:
                    score = float(self.global_mean)
                # new item
                elif uidx is not None and iidx is None and self.user_bias is not None:
                    score = float(self.global_mean + self.user_bias[uidx])
                # new user
                elif iidx is not None and uidx is None and self.item_bias is not None:
                    score = float(self.global_mean + self.item_bias[iidx])
                # final fallback
                elif self.global_mean is not None:
                    score = float(self.global_mean)
                else:
                    score = float('nan')

            out.append(score)

        return (
            model_input[['user_steamid', 'game_appid']]
            .assign(prediction=out)
            .reset_index(drop=True)
        )

# Load dataset
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

df_folder = os.path.join(main_folder, '3. data_analysis', 'db')
df_files = glob.glob(os.path.join(df_folder, '*.parquet'))

if df_files:
    latest_file = max(df_files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)
else:
    print('No parquet files found in the filtered folder.')

df['feature'] = df['feature'].apply(lambda x: ','.join(x) if isinstance(x, (list, np.ndarray)) else str(x))

data = list(df[['user_steamid', 'game_appid', 'rating']].itertuples(index=False, name=None))

# Initiate a TextModality
item_text_modality = TextModality(
    corpus=df['feature'].tolist(),
    ids=df['game_appid'].tolist(),
    tokenizer=BaseTokenizer(sep=',', stop_words='english'),
    max_vocab=5000,
    max_doc_freq=0.5,
)

# Split the data into train and test sets
rs = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=42,
)

# Define metrics to evaluate the models
metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

# Evaluate model
with mlflow.start_run():

    parameters = {
        'z_dim': 50,
        'vae_layers': [200, 100],
        'act_fn': 'sigmoid',
        'input_dim': 5000,
        'lr': 0.001,
        'batch_size': 128,
        'n_epochs': 100,
        'lambda_u': 1e-4,
        'lambda_v': 0.001,
        'lambda_r': 10,
        'lambda_w': 1e-4,
        'seed': 42
    }

    # Initialize models
    cvae = CVAE(**parameters)

    model = cvae
    mlflow.log_param('model', model.__class__.__name__)
    mlflow.log_params(parameters)

    test_result, val_result = rs.evaluate(  
        model=model,  
        metrics=metrics,  
        user_based=True,  
        show_validation=True  
    )

    metrics = {
        'mae': round(test_result.metric_avg_results['MAE'],4),
        'rmse': round(test_result.metric_avg_results['RMSE'],4),
        'auc': round(test_result.metric_avg_results['AUC'],4),
        'map': round(test_result.metric_avg_results['MAP'],4),
        'ndcg_10': round(test_result.metric_avg_results['NDCG@10'],4),
        'precision_10': round(test_result.metric_avg_results['Precision@10'],4),
        'recall_10': round(test_result.metric_avg_results['Recall@10'],4),
    }

    mlflow.log_metrics(metrics)

    # Save trained model
    with open('cornac_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    artifacts = {
        'model': 'cornac_model.pkl'
    }

    # Log as an MLflow PyFunc model
    mlflow.pyfunc.log_model(
        name='model',
        python_model=CornacModelWrapper(),
        artifacts=artifacts
    )
# %%