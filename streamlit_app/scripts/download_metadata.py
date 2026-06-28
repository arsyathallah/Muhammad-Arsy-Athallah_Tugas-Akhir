
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
META_URL  = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz"
META_GZ   = DATA_DIR / "meta_Video_Games.jsonl.gz"
OUT_CSV   = DATA_DIR / "game_metadata.csv"
ITEM_IDS  = DATA_DIR / "item_id_list.pkl"

def download_file(url: str, dest: Path):
    print(f"Downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved → {dest}")

def load_item_ids() -> set:
    if not ITEM_IDS.exists():
        print(f"WARNING: {ITEM_IDS} not found — including all items in metadata")
        return None
    with open(ITEM_IDS, "rb") as f:
        ids = pickle.load(f)
    return set(ids)

def parse_metadata(gz_path: Path, keep_ids: set | None) -> list[dict]:
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing metadata"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = item.get("parent_asin") or item.get("asin")
            if not pid:
                continue
            if keep_ids is not None and pid not in keep_ids:
                continue

            image_url = ""
            images = item.get("images", [])
            if images:
                img = images[0]
                image_url = img.get("large") or img.get("hi_res") or img.get("thumb", "")

            records.append({
                "product_id"  : pid,
                "title"       : item.get("title", ""),
                "image_url"   : image_url,
                "rating_count": item.get("rating_number", 0),
                "avg_rating"  : item.get("average_rating", 0),
                "main_category": item.get("main_category", ""),
            })

    return records

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not META_GZ.exists():
        download_file(META_URL, META_GZ)
    else:
        print(f"Already downloaded: {META_GZ}")

    keep_ids = load_item_ids()
    print(f"Filtering to {len(keep_ids):,} items from item_id_list.pkl" if keep_ids else "Including all items")

    records = parse_metadata(META_GZ, keep_ids)
    print(f"Parsed {len(records):,} items")

    df = pd.DataFrame(records).drop_duplicates("product_id")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}  ({len(df):,} rows)")

    has_title = df["title"].str.strip().astype(bool).sum()
    has_image = df["image_url"].str.strip().astype(bool).sum()
    print(f"  Items with title : {has_title:,} ({has_title/len(df)*100:.1f}%)")
    print(f"  Items with image : {has_image:,} ({has_image/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
