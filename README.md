# Steam Game Recommender System

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/status-active-success?style=flat-square)
![License](https://img.shields.io/github/license/EliasCid/steam_recommender_system?style=flat-square)

A hybrid recommendation system for Steam games that combines collaborative filtering and content-based approaches to provide personalized game recommendations to users.

![Hybrid Recommender System](images/Hybrid_RS.svg)

## Features

- **Hybrid Recommendation System**: Combines collaborative filtering and content-based filtering for improved recommendation quality
- **Data Collection**: Automated scraping of Steam game and user data
- **Multiple Recommendation Models**:
  - Collaborative Filtering with BPR
  - Content-Based Filtering with CVAE
  - TF-IDF based Similar Items
- **Evaluation Metrics**: Implements precision@k for model evaluation
- **Easy-to-Use Interface**: Simple Python API for generating recommendations

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. It is recommended to use `uv` to install dependencies. If you don't have `uv` installed, you can install it using `pip` or follow instructions [here](https://docs.astral.sh/uv/getting-started/installation/):

```bash
pip install uv
```

1. Clone the repository:
   ```bash
   git clone https://github.com/EliasCid/steam_recommender_system
   cd steam_recommender_system
   ```

2. Create a virtual environment:
   ```bash
   uv venv
   ```

3. Activate it:

   - On Linux or macOS:
   ```bash
   source .venv/bin/activate
   ```
   - On Windows:
   ```bash
   .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   uv sync
   ```

5. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Steam API key:
     ```
     API_KEY = "your_steam_api_key_here"
     ```

## Project Structure

```
.
├── 1. data_collection/                                  # Scripts for collecting Steam data
│   ├── steam_scraper/                                   # Core scraping functionality
│   │   ├── __init__.py
│   │   ├── games.py                              
│   │   ├── games_detail.py                       
│   │   ├── users.py                              
│   │   └── users_detail.py                       
│   └── main.py                                   
│
├── 2. data_processing/                                  # Data cleaning and preprocessing
│   └── steam_db.py                               
│
├── 3. data_analysis/                                    # Data analysis notebook
│   └── steam.ipynb                               
│
├── 4. model/                                            # Model training and evaluation
│   ├── collaborative_filtering_cornac_deep_learning.py  
│   ├── collaborative_filtering_cornac.py                 
│   ├── collaborative_filtering_cornac_mlp.py             
│   ├── collaborative_filtering_cornac_pmf.py             
│   ├── content_based_cornac_cvae.py                      
│   ├── content_based_cornac_hft.py                       
│   ├── content_based_sklearn_tfidf.py                    
│   └── 0_db_for_predictions.py                           
│
├── 5. predictions/                                      # Model prediction and evaluation
│   ├── collaborative_filtering_cornac_deep_learning_prediction.py  
│   ├── collaborative_filtering_cornac_prediction.py     
│   ├── content_based_cornac_cvae_prediction.py          
│   ├── content_based_sklearn_tfidf_prediction.py                     
│   ├── precision_at_10.py                               
│   └── recommender.py                                   
│
├── recommender_system/                                  # Core recommendation system package
│   ├── __init__.py
│   ├── load_models.py                            
│   ├── load_dataset.py                           
│   └── predict.py                                
│
├── mlruns/                                              # MLflow experiment tracking
├── artifacts/                                           # Model artifacts and outputs
├── mlartifacts/                                         # MLflow model artifacts
├── steam_recommender.py                                 # Main application entry point
├── pyproject.toml                                       # Project metadata and dependencies
└── README.md                                            # This file
```

## Models

### 1. Collaborative Filtering
- Uses Bayesian Personalized Ranking (BPR)
- Captures user-item interactions

### 2. Content-Based Filtering
- Implements Collaborative Variational Autoencoder (CVAE)
- Considers game genres and tags

### 3. TF-IDF Similarity
- Text-based similarity using game genres and tags
- Useful for finding similar games

## Results

The system provides recommendations with an average Precision@10 of 21%

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Steam Web API](https://developer.valvesoftware.com/wiki/Steam_Web_API) for providing game data
- [CORNAC](https://cornac.readthedocs.io/en/latest/) for providing the models.
- [MLflow](https://mlflow.org/) for experiment tracking.
- [uv](https://docs.astral.sh/uv/) for managing dependencies.