import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download_dataset import (
    download_amazon_reviews, download_metadata,
    download_movielens_25m
)
from src.config import DATASET_CATEGORY, DATASET_SAMPLE_SIZE, DATASET_SOURCE

def main():
    print("=" * 60)
    print(f"STEP 1: DOWNLOAD DATASET")
    print(f"   Source: {DATASET_SOURCE}")
    print("=" * 60)

    if DATASET_SOURCE == "movielens":
        print(f"   Dataset: MovieLens-25M (Gold Standard)")
        print(f"   Sample size: {DATASET_SAMPLE_SIZE}")
        reviews_df = download_movielens_25m(sample_size=DATASET_SAMPLE_SIZE)
    else:
        print(f"   Dataset: Amazon Reviews 2023")
        print(f"   Category: {DATASET_CATEGORY}")
        print(f"   Sample size: {DATASET_SAMPLE_SIZE}")
        reviews_df = download_amazon_reviews()

    print(f"\nDownloaded {len(reviews_df)} reviews")
    print(f"   Columns: {list(reviews_df.columns)}")
    print(f"\nSample:")
    print(reviews_df.head(3).to_string())

    if DATASET_SOURCE != "movielens":
        try:
            meta_df = download_metadata()
            if not meta_df.empty:
                print(f"\nDownloaded {len(meta_df)} product metadata entries")
        except Exception as e:
            print(f"\nMetadata download skipped: {e}")

if __name__ == "__main__":
    main()
