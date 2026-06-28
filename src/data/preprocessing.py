
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.utils.logger import get_logger
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_CATEGORY,
    EVAL_CONFIG, CBF_CONFIG, TRAIN_PARQUET, TEST_PARQUET, REVIEWS_PARQUET
)

logger = get_logger(__name__)

class DataPreprocessor:

    def __init__(self):
        self.raw_df = None
        self.clean_df = None
        self.train_df = None
        self.test_df = None
        self.stats = {}

    def load_raw_data(self, file_path: Path = None) -> pd.DataFrame:
        if file_path is None:
            parquet_files = list(RAW_DATA_DIR.glob("reviews_*.parquet"))
            if parquet_files:
                file_path = parquet_files[0]
                logger.info(f"Found raw data: {file_path}")
            else:
                raise FileNotFoundError(
                    f"No raw data found in {RAW_DATA_DIR}. "
                    "Run download_dataset.py first."
                )

        if str(file_path).endswith(".parquet"):
            self.raw_df = pd.read_parquet(str(file_path))
        elif str(file_path).endswith(".csv"):
            self.raw_df = pd.read_csv(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        logger.info(f" Loaded {len(self.raw_df)} rows from {file_path}")
        logger.info(f"   Columns: {list(self.raw_df.columns)}")

        return self.raw_df

    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        df = df if df is not None else self.raw_df.copy()
        initial_count = len(df)

        logger.info(" Cleaning data...")

        required_cols = ["user_id", "product_id", "rating"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data.")

        df = df.dropna(subset=["user_id", "product_id", "rating"])

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]

        df = df.drop_duplicates(subset=["user_id", "product_id"], keep="first")

        if "review_text" in df.columns:
            df["review_text"] = df["review_text"].fillna("")
            df["review_text"] = df["review_text"].astype(str)
        else:
            df["review_text"] = ""

        df["user_id"] = df["user_id"].astype(str)
        df["product_id"] = df["product_id"].astype(str)

        removed_count = initial_count - len(df)
        logger.info(f"   Removed {removed_count} rows ({removed_count/initial_count*100:.1f}%)")
        logger.info(f"   Remaining: {len(df)} rows")

        self.clean_df = df
        return df

    def filter_interactions(self, df: pd.DataFrame = None,
                            min_user_reviews: int = None,
                            min_product_reviews: int = None) -> pd.DataFrame:
        df = df if df is not None else self.clean_df.copy()
        min_user = min_user_reviews or CBF_CONFIG["min_reviews_per_user"]
        min_product = min_product_reviews or CBF_CONFIG["min_reviews_per_product"]

        logger.info(f" Filtering: min {min_user} reviews/user, min {min_product} reviews/product")
        initial_count = len(df)

        prev_count = 0
        iteration = 0
        max_iterations = 10
        while prev_count != len(df) and iteration < max_iterations:
            prev_count = len(df)
            iteration += 1

            product_counts = df["product_id"].value_counts()
            valid_products = product_counts[product_counts >= min_product].index
            df = df[df["product_id"].isin(valid_products)]

            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            df = df[df["user_id"].isin(valid_users)]

            if len(df) == 0:
                break

        if len(df) == 0:
            logger.warning(" All data filtered out! Relaxing thresholds to min=1...")
            df = (self.clean_df if self.clean_df is not None else self.raw_df).copy()
            logger.info(f"   Using unfiltered data: {len(df)} rows")

        logger.info(f"   Iterations: {iteration}")
        logger.info(f"   Removed {initial_count - len(df)} rows")
        logger.info(f"   Remaining: {len(df)} rows, "
                     f"{df['user_id'].nunique()} users, "
                     f"{df['product_id'].nunique()} products")

        self.clean_df = df
        return df

    def add_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        df = df if df is not None else self.clean_df.copy()

        logger.info(" Adding engineered features...")

        if "review_text" in df.columns:
            df["review_length"] = df["review_text"].str.len()
            df["has_review"] = df["review_text"].str.len() > 10
            df["word_count"] = df["review_text"].str.split().str.len().fillna(0).astype(int)

        df["rating_normalized"] = (df["rating"] - 1.0) / 4.0

        self.clean_df = df
        return df

    def train_test_split(self, df: pd.DataFrame = None,
                         test_size: float = None,
                         random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df if df is not None else self.clean_df.copy()
        test_size = test_size or EVAL_CONFIG["test_size"]
        random_seed = random_seed or EVAL_CONFIG["random_seed"]

        logger.info(f" Per-user splitting: {1-test_size:.0%} train, {test_size:.0%} test")

        np.random.seed(random_seed)
        
        train_indices = []
        test_indices = []
        
        for user_id, group in df.groupby("user_id"):
            indices = group.index.tolist()
            n_interactions = len(indices)
            
            if n_interactions <= 1:
                train_indices.extend(indices)
            else:
                np.random.shuffle(indices)
                n_test = max(1, int(n_interactions * test_size))
                test_indices.extend(indices[:n_test])
                train_indices.extend(indices[n_test:])
        
        self.train_df = df.loc[train_indices].reset_index(drop=True)
        self.test_df = df.loc[test_indices].reset_index(drop=True)
        
        train_users = set(self.train_df["user_id"].unique())
        test_users = set(self.test_df["user_id"].unique())
        overlap = train_users & test_users

        logger.info(f"   Train: {len(self.train_df)} rows, {len(train_users)} users")
        logger.info(f"   Test:  {len(self.test_df)} rows, {len(test_users)} users")
        logger.info(f"   User overlap: {len(overlap)} users ({len(overlap)/max(len(test_users),1)*100:.1f}% of test users)")

        return self.train_df, self.test_df

    def save_processed(self, train_path: Path = None, test_path: Path = None,
                       reviews_path: Path = None):
        train_path = train_path or TRAIN_PARQUET
        test_path = test_path or TEST_PARQUET
        reviews_path = reviews_path or REVIEWS_PARQUET

        if self.train_df is not None:
            self.train_df.to_parquet(Path(train_path).as_posix(), index=False)
            logger.info(f" Saved train data: {train_path}")

        if self.test_df is not None:
            self.test_df.to_parquet(Path(test_path).as_posix(), index=False)
            logger.info(f" Saved test data: {test_path}")

        if self.clean_df is not None:
            self.clean_df.to_parquet(Path(reviews_path).as_posix(), index=False)
            logger.info(f" Saved full reviews: {reviews_path}")

    def get_stats(self) -> dict:
        df = self.clean_df if self.clean_df is not None else self.raw_df
        if df is None:
            return {}

        n_users = df["user_id"].nunique()
        n_products = df["product_id"].nunique()
        total_possible = n_users * n_products if n_users > 0 and n_products > 0 else 1

        self.stats = {
            "total_reviews": len(df),
            "unique_users": n_users,
            "unique_products": n_products,
            "avg_rating": df["rating"].mean() if len(df) > 0 else 0,
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
            "sparsity": 1 - len(df) / total_possible,
            "avg_reviews_per_user": len(df) / max(n_users, 1),
            "avg_reviews_per_product": len(df) / max(n_products, 1),
        }

        if "review_text" in df.columns:
            self.stats["has_review_text"] = (df["review_text"].str.len() > 10).sum()
            self.stats["avg_review_length"] = df["review_text"].str.len().mean()

        return self.stats

    def run_full_pipeline(self, file_path: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info(" STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        self.load_raw_data(file_path)

        self.clean_data()

        self.filter_interactions()

        self.add_features()

        train_df, test_df = self.train_test_split()

        self.save_processed()

        stats = self.get_stats()
        logger.info("\n Final Dataset Statistics:")
        for key, value in stats.items():
            if key != "rating_distribution":
                logger.info(f"   {key}: {value}")

        logger.info("=" * 60)
        logger.info(" PREPROCESSING PIPELINE COMPLETE")
        logger.info("=" * 60)

        return train_df, test_df
