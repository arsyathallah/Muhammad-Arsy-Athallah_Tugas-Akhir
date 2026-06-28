import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.mongodb_loader import MongoDBLoader

def main():
    print("=" * 60)
    print("STEP 2: PREPROCESS & LOAD TO MONGODB")
    print("=" * 60)

    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.run_full_pipeline()

    stats = preprocessor.get_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if key != "rating_distribution":
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Loading to MongoDB...")
    print("=" * 60)

    try:
        loader = MongoDBLoader()
        loader.run_full_load(preprocessor.clean_df)
        loader.close()
    except Exception as e:
        print(f"\nMongoDB loading skipped: {e}")
        print("   Data is saved as parquet files for offline use.")

    print("\nPreprocessing & loading complete!")
    print(f"   Train: {len(train_df)} rows")
    print(f"   Test:  {len(test_df)} rows")

if __name__ == "__main__":
    main()
