
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from database import get_db, upsert_items, is_connected, MONGO_URI

def main():
    print(f"MongoDB URI : {MONGO_URI}")
    print("Mencoba koneksi...", end=" ")

    if not is_connected():
        print("GAGAL")
        print("\nPastikan MongoDB Replica Set sudah berjalan.")
        print("Lihat instruksi setup di README atau jalankan setup_replica.py")
        sys.exit(1)

    print("OK ")

    csv_path = ROOT / "data" / "game_metadata.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} tidak ditemukan.")
        print("Jalankan dulu: python streamlit_app/scripts/make_metadata.py")
        sys.exit(1)

    print(f"\nMembaca {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"product_id": str})
    print(f"  {len(df):,} baris ditemukan")

    print("Menyimpan ke MongoDB (collection: items)...")
    total = upsert_items(df)
    print(f"   {total:,} item tersimpan di MongoDB")

    db    = get_db()
    count = db["items"].count_documents({})
    sample = db["items"].find_one({}, {"_id": 0, "product_id": 1, "title": 1})
    print(f"\n  Verifikasi — total dokumen: {count:,}")
    print(f"  Contoh: {sample}")

    print("\n Migrasi selesai!")
    print(f"   Database : {db.name}")
    print(f"   Collection items     : {db['items'].count_documents({}):,} dokumen")
    print(f"   Collection uat_results: {db['uat_results'].count_documents({})} dokumen")

if __name__ == "__main__":
    main()
