
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CATEGORY = "Video_Games"
DATASET_SAMPLE_SIZE = None

DATASET_SOURCE = os.getenv("DATASET_SOURCE", "movielens")

RATINGS_CSV = DATA_DIR / "ratings_Electronics.csv"
REVIEWS_PARQUET = PROCESSED_DATA_DIR / "reviews.parquet"
RATINGS_PARQUET = PROCESSED_DATA_DIR / "ratings.parquet"
TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DATA_DIR / "test.parquet"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://10.4.89.233:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "recommendation_system")
MONGODB_CONNECT_TIMEOUT_MS = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "15000"))
MONGODB_SERVER_SELECTION_TIMEOUT_MS = int(
    os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "15000")
)

MONGODB_CLUSTER_NAME = os.getenv("MONGODB_CLUSTER_NAME", "recommendation_mongodb_cluster")
MONGODB_NODES = os.getenv(
    "MONGODB_NODES",
    "10.4.89.233:27017",
).split(",")

MONGODB_COLLECTIONS = {
    "reviews": "reviews",
    "ratings": "ratings",
    "products": "products",
    "users": "users",
    "recommendations": "recommendations",
    "product_profiles": "product_profiles",
    "user_profiles": "user_profiles",
}

MONGODB_SHARD_KEYS = {
    "reviews": {"user_id": "hashed"},
    "ratings": {"user_id": "hashed"},
    "recommendations": {"user_id": "hashed"},
    "product_profiles": {"product_id": "hashed"},
    "user_profiles": {"user_id": "hashed"},
}

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "8g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
SPARK_APP_NAME = "HybridRecommender"

ALS_CONFIG = {
    "rank": 128,
    "max_iter": 75,
    "reg_param": 0.001,
    "cold_start_strategy": "drop",
    "nonnegative": False,
    "user_col": "user_index",
    "item_col": "item_index",
    "rating_col": "rating",
    "implicit_prefs": True,
    "alpha": 40.0,
    "factors": 128,
    "iterations": 75,
    "regularization": 0.001,
}

ALS_TUNING_GRID = {
    "factors": [64, 100, 128, 150, 200],
    "regularizations": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "iterations": [50, 75, 100],
    "ranks": [64, 100, 128, 150, 200],
    "reg_params": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "max_iters": [50, 75, 100],
}

CBF_CONFIG = {
    "pretrained_model_name": "en_core_web_md",
    "embedding_dim": 300,
    "min_reviews_per_product": 3,
    "min_reviews_per_user": 3,
}

HYBRID_CONFIG = {
    "alpha": 0.7,
    "beta": 0.3,
    "top_n": 10,
    "weight_search_step": 0.05,
}

EVAL_CONFIG = {
    "k_values": [3, 5, 10, 20, 50, 100],
    "rating_threshold": 3.0,
    "test_size": 0.2,
    "random_seed": 42,
}

PLM_CBF_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "batch_size": 256,
    "max_seq_length": 256,
    "min_reviews_per_product": 3,
    "min_reviews_per_user": 3,
}

GRAM_CONFIG = {
    "plm_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "plm_embed_dim": 384,
    "latent_dim": 128,
    "n_negatives": 4,
    "epochs": 15,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "freeze_plm": True,
    "top_n": 50,
    "grad_clip": 1.0,
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "application.log"
