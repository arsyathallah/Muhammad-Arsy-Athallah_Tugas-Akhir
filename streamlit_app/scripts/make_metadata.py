
import pickle
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
ITEM_IDS   = DATA_DIR / "item_id_list.pkl"
OUT_CSV    = DATA_DIR / "game_metadata.csv"

PARQUET_CANDIDATES = [
    RAW_DIR / "metadata_Video_Games.parquet",
    DATA_DIR / "metadata_Video_Games.parquet",
    ROOT.parent / "data" / "raw" / "metadata_Video_Games.parquet",
]

def get_image(imgs):
    if imgs is None:
        return ""
    if isinstance(imgs, dict):
        for key in ("large", "hi_res", "thumb"):
            arr = imgs.get(key)
            if arr is None:
                continue
            for url in arr:
                if url and str(url) != "None" and str(url).startswith("http"):
                    return str(url)
    if isinstance(imgs, list):
        for item in imgs:
            if isinstance(item, str) and item.startswith("http"):
                return item
            if isinstance(item, dict):
                for key in ("large", "hi_res", "thumb"):
                    url = item.get(key, "")
                    if url and str(url).startswith("http"):
                        return str(url)
    if isinstance(imgs, str) and imgs.startswith("http"):
        return imgs
    return ""

def main():
    parquet_path = None
    for p in PARQUET_CANDIDATES:
        if p.exists():
            parquet_path = p
            break
    if parquet_path is None:
        raise FileNotFoundError(
            f"metadata_Video_Games.parquet tidak ditemukan. "
            f"Dicari di: {[str(p) for p in PARQUET_CANDIDATES]}"
        )
    print(f"Membaca: {parquet_path}")

    if ITEM_IDS.exists():
        with open(ITEM_IDS, "rb") as f:
            keep = set(pickle.load(f))
        print(f"Filter ke {len(keep):,} item dari item_id_list.pkl")
    else:
        keep = None
        print("item_id_list.pkl tidak ditemukan — menggunakan semua item")

    df = pd.read_parquet(parquet_path)
    print(f"Total baris parquet: {len(df):,}")
    print(f"Kolom tersedia    : {list(df.columns)}")

    id_col = "parent_asin" if "parent_asin" in df.columns else "asin"
    df = df.rename(columns={id_col: "product_id"})

    title_candidates = ["title", "product_title", "name", "item_name"]
    title_col = next((c for c in title_candidates if c in df.columns), None)
    if title_col and title_col != "title":
        df = df.rename(columns={title_col: "title"})
    elif title_col is None:
        df["title"] = ""

    if keep is not None:
        df = df[df["product_id"].isin(keep)]
        print(f"Setelah filter: {len(df):,} item")

    if "images" in df.columns:
        df["image_url"] = df["images"].apply(get_image)
    else:
        df["image_url"] = ""

    rename_map = {}
    for src, dst in [("rating_number", "rating_count"), ("num_ratings", "rating_count"),
                     ("average_rating", "avg_rating")]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    for col in ["rating_count", "avg_rating"]:
        if col not in df.columns:
            df[col] = 0

    df["product_url"] = "https://www.amazon.com/dp/" + df["product_id"].astype(str)

    out = (
        df[["product_id", "title", "image_url", "product_url", "rating_count", "avg_rating"]]
        .drop_duplicates("product_id")
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}  ({len(out):,} baris)")

    has_title = out["title"].str.strip().astype(bool).sum()
    has_image = out["image_url"].str.strip().astype(bool).sum()
    has_url   = out["product_url"].str.strip().astype(bool).sum()
    print(f"  Dengan judul  : {has_title:,} ({has_title/len(out)*100:.1f}%)")
    print(f"  Dengan gambar : {has_image:,} ({has_image/len(out)*100:.1f}%)")
    print(f"  Dengan URL    : {has_url:,} ({has_url/len(out)*100:.1f}%)")

if __name__ == "__main__":
    main()
