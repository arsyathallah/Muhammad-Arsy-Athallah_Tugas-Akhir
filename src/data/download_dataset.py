
import os
import sys
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DATASET_NAME, DATASET_CATEGORY, DATASET_SAMPLE_SIZE,
    DATASET_SOURCE
)

logger = get_logger(__name__)

def download_movielens_25m(output_dir: Path = None, sample_size: int = None) -> pd.DataFrame:
    import urllib.request
    import shutil

    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = output_dir / "ml-25m.zip"
    extract_dir = output_dir / "ml-25m"

    ratings_csv = extract_dir / "ratings.csv"
    if ratings_csv.exists():
        logger.info(f"MovieLens-25M already downloaded: {ratings_csv}")
    else:
        logger.info(f" Downloading MovieLens-25M (~250MB)...")
        logger.info(f"   URL: {url}")

        if not zip_path.exists():
            urllib.request.urlretrieve(url, str(zip_path))
            logger.info(f"   Downloaded to: {zip_path}")

        logger.info("   Extracting ZIP...")
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            zip_ref.extractall(str(output_dir))
        logger.info(f"   Extracted to: {extract_dir}")

    logger.info("Loading ratings.csv...")
    ratings_df = pd.read_csv(str(ratings_csv))
    logger.info(f"   Loaded {len(ratings_df)} ratings")

    ratings_df = ratings_df.rename(columns={
        "userId": "user_id",
        "movieId": "product_id",
    })

    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["product_id"] = ratings_df["product_id"].astype(str)

    if sample_size and len(ratings_df) > sample_size:
        ratings_df = ratings_df.sample(n=sample_size, random_state=42)
        logger.info(f"   Sampled to {len(ratings_df)} ratings")

    movies_csv = extract_dir / "movies.csv"
    tags_csv = extract_dir / "tags.csv"

    movies_df = pd.read_csv(str(movies_csv))
    movies_df["movieId"] = movies_df["movieId"].astype(str)
    movies_df = movies_df.rename(columns={"movieId": "product_id"})
    logger.info(f"   Loaded {len(movies_df)} movies")

    logger.info("   Generating review text from genres + tags...")

    if tags_csv.exists():
        tags_df = pd.read_csv(str(tags_csv))
        tags_df["movieId"] = tags_df["movieId"].astype(str)
        tags_df["userId"] = tags_df["userId"].astype(str)

        movie_tags = tags_df.groupby("movieId")["tag"].apply(
            lambda x: " ".join(x.dropna().astype(str))
        ).to_dict()
    else:
        movie_tags = {}

    movie_info = {}
    for _, row in movies_df.iterrows():
        pid = row["product_id"]
        title = row.get("title", "")
        genres = row.get("genres", "").replace("|", ", ")
        tags = movie_tags.get(pid, "")

        review_parts = []
        if title:
            review_parts.append(title)
        if genres and genres != "(no genres listed)":
            review_parts.append(f"Genres: {genres}")
        if tags:
            review_parts.append(f"Tags: {tags[:500]}")

        movie_info[pid] = " . ".join(review_parts)

    ratings_df["review_text"] = ratings_df["product_id"].map(
        lambda pid: movie_info.get(pid, "")
    )
    ratings_df["review_title"] = ratings_df["product_id"].map(
        lambda pid: movies_df[movies_df["product_id"] == pid]["title"].iloc[0]
        if pid in movies_df["product_id"].values else ""
    )

    rating_adjectives = {
        0.5: "Terrible.",
        1.0: "Very bad.",
        1.5: "Bad.",
        2.0: "Below average.",
        2.5: "Mediocre.",
        3.0: "Average.",
        3.5: "Decent.",
        4.0: "Good.",
        4.5: "Very good.",
        5.0: "Excellent!",
    }
    ratings_df["review_text"] = ratings_df.apply(
        lambda row: f"{rating_adjectives.get(row['rating'], 'OK.')} {row['review_text']}",
        axis=1
    )

    output_path = output_dir / "reviews_MovieLens25M.parquet"
    ratings_df.to_parquet(str(output_path), index=False)

    logger.info(f"\n MovieLens-25M Summary:")
    logger.info(f"   Total ratings: {len(ratings_df):,}")
    logger.info(f"   Unique users: {ratings_df['user_id'].nunique():,}")
    logger.info(f"   Unique movies: {ratings_df['product_id'].nunique():,}")
    logger.info(f"   Avg ratings/user: {len(ratings_df)/ratings_df['user_id'].nunique():.1f}")
    logger.info(f"   Avg ratings/movie: {len(ratings_df)/ratings_df['product_id'].nunique():.1f}")
    n_u = ratings_df['user_id'].nunique()
    n_i = ratings_df['product_id'].nunique()
    sparsity = 1 - len(ratings_df) / (n_u * n_i)
    logger.info(f"   Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    logger.info(f"   Rating scale: {ratings_df['rating'].min()}-{ratings_df['rating'].max()}")
    logger.info(f"   Rating distribution:")
    for rating, count in ratings_df['rating'].value_counts().sort_index().items():
        logger.info(f"      {rating}: {count:,}")
    logger.info(f"   Saved to: {output_path}")

    return ratings_df

def download_amazon_reviews(
    category: str = None,
    sample_size: int = None,
    output_dir: Path = None,
) -> pd.DataFrame:
    category = category or DATASET_CATEGORY
    sample_size = sample_size or DATASET_SAMPLE_SIZE
    output_dir = output_dir or RAW_DATA_DIR

    logger.info(f"Downloading Amazon Reviews 2023 - Category: {category}")
    logger.info(f"   Sample size: {sample_size}")

    try:
        from datasets import load_dataset

        logger.info("Loading from Hugging Face (this may take a few minutes)...")

        dataset = load_dataset(
            DATASET_NAME,
            f"raw_review_{category}",
            split="full",
            trust_remote_code=True,
        )

        logger.info("Converting to DataFrame...")
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))

        column_mapping = {
            "user_id": "user_id",
            "parent_asin": "product_id",
            "rating": "rating",
            "text": "review_text",
            "title": "review_title",
            "timestamp": "timestamp",
            "helpful_vote": "helpful_votes",
            "verified_purchase": "verified_purchase",
        }
        available_cols = dataset.column_names
        if "parent_asin" not in available_cols and "asin" in available_cols:
            column_mapping["asin"] = "product_id"

        logger.info(f"   Available columns in dataset: {available_cols}")

        select_cols = [c for c in column_mapping.keys() if c in available_cols]
        if select_cols:
            dataset = dataset.select_columns(select_cols)

        df = dataset.to_pandas()

        rename_map = {}
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                rename_map[old_col] = new_col
        df = df.rename(columns=rename_map)

        cols_to_drop = []
        for col_name in df.columns:
            try:
                if df[col_name].dtype == object:
                    sample_val = df[col_name].dropna().iloc[0] if not df[col_name].dropna().empty else None
                    if isinstance(sample_val, (list, dict)):
                        logger.info(f"   Dropping complex column: {col_name}")
                        cols_to_drop.append(col_name)
            except Exception:
                pass
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        required_cols = ["user_id", "product_id", "rating", "review_text"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Available: {list(df.columns)}")

        output_path = output_dir / f"reviews_{category}.parquet"
        df.to_parquet(str(output_path), index=False)
        logger.info(f"Saved {len(df)} reviews to {output_path}")

        logger.info(f"\nDataset Summary:")
        logger.info(f"   Total reviews: {len(df)}")
        logger.info(f"   Unique users: {df['user_id'].nunique()}")
        logger.info(f"   Unique products: {df['product_id'].nunique()}")
        logger.info(f"   Rating distribution:\n{df['rating'].value_counts().sort_index()}")
        logger.info(f"   Columns: {list(df.columns)}")

        return df

    except ImportError:
        logger.error(
            "'datasets' library not installed. Run: pip install datasets"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Attempting fallback: using existing ratings CSV...")
        return _fallback_load_existing(output_dir)

def _fallback_load_existing(output_dir: Path) -> pd.DataFrame:
    from src.config import RATINGS_CSV

    if not RATINGS_CSV.exists():
        raise FileNotFoundError(
            f"No dataset found. Please either:\n"
            f"1. Install 'datasets' library: pip install datasets\n"
            f"2. Place ratings CSV at: {RATINGS_CSV}"
        )

    logger.info(f"Loading existing CSV: {RATINGS_CSV}")
    df = pd.read_csv(str(RATINGS_CSV), header=None,
                     names=["user_id", "product_id", "rating", "timestamp"])

    if DATASET_SAMPLE_SIZE and len(df) > DATASET_SAMPLE_SIZE:
        df = df.sample(n=DATASET_SAMPLE_SIZE, random_state=42)

    review_templates = {
        1.0: "Very disappointing product. Poor quality and not worth the money.",
        2.0: "Below average. Expected much better based on the description.",
        3.0: "It's okay. Nothing special but does the job adequately.",
        4.0: "Good product overall. Happy with my purchase, would recommend.",
        5.0: "Excellent! Exceeded my expectations. Amazing quality and value.",
    }

    df["review_text"] = df["rating"].map(
        lambda r: review_templates.get(float(r), "Average product.")
    )
    df["review_title"] = df["rating"].map(
        lambda r: {1.0: "Bad", 2.0: "Below average", 3.0: "OK",
                   4.0: "Good", 5.0: "Excellent"}.get(float(r), "Review")
    )

    output_path = output_dir / "reviews_fallback.parquet"
    df.to_parquet(str(output_path), index=False)
    logger.info(f" Loaded {len(df)} ratings with generated review text -> {output_path}")
    logger.warning(
        " Using generated review text (not real reviews). "
        "For better results, install 'datasets' library and re-download."
    )

    return df

def download_metadata(category: str = None) -> pd.DataFrame:
    category = category or DATASET_CATEGORY

    logger.info(f" Downloading metadata for category: {category}")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            DATASET_NAME,
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )

        df = dataset.to_pandas()

        rename_map = {
            "parent_asin": "product_id",
            "title": "product_title",
            "main_category": "category",
            "price": "price",
            "average_rating": "avg_rating",
            "rating_number": "num_ratings",
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        output_path = RAW_DATA_DIR / f"metadata_{category}.parquet"
        df.to_parquet(str(output_path), index=False)
        logger.info(f" Saved {len(df)} product metadata to {output_path}")

        return df

    except Exception as e:
        logger.warning(f"Could not download metadata: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    if DATASET_SOURCE == "movielens":
        df = download_movielens_25m()
    else:
        df = download_amazon_reviews()
    print(f"\nDownloaded {len(df)} reviews")
    print(df.head())
