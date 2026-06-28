"""
Migrasi data dari MongoDB lokal (mongod, port 27017) ke MongoDB Sharded Cluster
(mongos Docker, port 37017).

Jalankan setelah cluster Docker aktif:
  cd docker && docker compose up -d
  python scripts/migrate_to_sharded_cluster.py
"""

import sys
import time
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError

SRC_URI = "mongodb://localhost:27017"
DST_URI = "mongodb://localhost:37017"
DB_NAME  = "recommendation_system"

BATCH = 2000

COLLECTIONS = [
    "reviews",
    "ratings",
    "products",
    "users",
    "recommendations",
    "product_profiles",
    "user_profiles",
]


def connect(uri: str, label: str) -> MongoClient:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print(f"  [{label}] terhubung ke {uri}")
    return client


def migrate_collection(src_db, dst_db, name: str):
    src_col = src_db[name]
    dst_col = dst_db[name]

    total = src_col.count_documents({})
    if total == 0:
        print(f"  {name}: kosong, skip.")
        return 0

    existing = dst_col.count_documents({})
    if existing >= total:
        print(f"  {name}: sudah ada {existing} docs di tujuan, skip.")
        return existing

    print(f"  {name}: memindahkan {total:,} docs ...")

    # Drop data lama di tujuan agar tidak duplikat
    dst_col.drop()

    ops = []
    moved = 0
    t0 = time.time()
    for doc in src_col.find({}, no_cursor_timeout=False):
        ops.append(InsertOne(doc))
        if len(ops) >= BATCH:
            try:
                dst_col.bulk_write(ops, ordered=False)
            except BulkWriteError:
                pass
            moved += len(ops)
            ops = []
            elapsed = time.time() - t0
            pct = moved / total * 100
            print(f"    {moved:,}/{total:,} ({pct:.1f}%) — {elapsed:.1f}s", end="\r")

    if ops:
        try:
            dst_col.bulk_write(ops, ordered=False)
        except BulkWriteError:
            pass
        moved += len(ops)

    elapsed = time.time() - t0
    final = dst_col.count_documents({})
    print(f"    {name}: {final:,} docs dimuat dalam {elapsed:.1f}s          ")
    return final


def verify_sharding(dst_client: MongoClient):
    try:
        status = dst_client.admin.command("listShards")
        shards = status.get("shards", [])
        print(f"\n  Shards aktif: {len(shards)}")
        for s in shards:
            print(f"    - {s['_id']}  =>  {s['host']}")

        # Distribusi chunk per collection
        config = dst_client["config"]
        print("\n  Distribusi chunk per collection:")
        for coll in ["reviews", "ratings", "recommendations", "product_profiles", "user_profiles"]:
            ns = f"{DB_NAME}.{coll}"
            dist = list(config["chunks"].aggregate([
                {"$match": {"ns": ns}},
                {"$group": {"_id": "$shard", "chunks": {"$sum": 1}}},
            ]))
            if dist:
                info = ", ".join(f"{d['_id']}:{d['chunks']}" for d in dist)
                print(f"    {coll}: {info}")
    except Exception as e:
        print(f"  Tidak bisa verifikasi sharding: {e}")


def main():
    print("=" * 60)
    print(" MIGRASI: MongoDB lokal → MongoDB Sharded Cluster")
    print("=" * 60)

    print("\n[1] Koneksi ke sumber (mongod lokal) ...")
    try:
        src = connect(SRC_URI, "SRC")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Pastikan mongod lokal berjalan di port 27017.")
        sys.exit(1)

    print("\n[2] Koneksi ke tujuan (mongos Docker) ...")
    try:
        dst = connect(DST_URI, "DST")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Pastikan Docker cluster sudah jalan: cd docker && docker compose up -d")
        sys.exit(1)

    src_db = src[DB_NAME]
    dst_db = dst[DB_NAME]

    print(f"\n[3] Migrasi collection dari '{DB_NAME}' ...")
    totals = {}
    for coll in COLLECTIONS:
        totals[coll] = migrate_collection(src_db, dst_db, coll)

    print("\n[4] Verifikasi sharding ...")
    verify_sharding(dst)

    print("\n[5] Ringkasan hasil migrasi:")
    for coll, n in totals.items():
        print(f"    {coll}: {n:,} docs")

    src.close()
    dst.close()

    print("\nMigrasi selesai!")
    print("Koneksi aplikasi → MONGODB_URI=mongodb://localhost:37017")


if __name__ == "__main__":
    main()
