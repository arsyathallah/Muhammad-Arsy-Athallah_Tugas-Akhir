"""
Migrasi data dari MongoDB lokal (mongod, port 27017) ke MongoDB Sharded Cluster
(mongos di 10.4.89.233:27017).

Jalankan setelah cluster sharded aktif & dapat diakses:
  python scripts/migrate_to_sharded_cluster.py
"""

import os
import sys
import time
from dotenv import load_dotenv
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError

load_dotenv()

SRC_URI = os.getenv("MIGRATE_SRC_URI", "mongodb://localhost:27017")
# Tujuan = cluster sharded ber-auth; baca dari MONGODB_URI (.env) agar kredensial
# tidak ter-hardcode di file yang masuk git.
DST_URI = os.getenv("MONGODB_URI", "mongodb://10.4.89.233:27017")
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

    # Kosongkan isi lama agar tidak duplikat — JANGAN drop(), karena drop
    # menghapus konfigurasi sharding sehingga collection jadi tidak terdistribusi.
    dst_col.delete_many({})

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

        # Distribusi dokumen per shard (MongoDB 6.0+: $shardedDataDistribution).
        # config.chunks tidak lagi punya field 'ns' sejak MongoDB 5.0, jadi
        # query lama berbasis 'ns' tidak akan menghasilkan apa pun di 8.0.
        wanted = {f"{DB_NAME}.{c}" for c in (
            "reviews", "ratings", "recommendations", "product_profiles", "user_profiles"
        )}
        print("\n  Distribusi dokumen per shard:")
        found = False
        for d in dst_client.admin.aggregate([{"$shardedDataDistribution": {}}]):
            if d["ns"] not in wanted:
                continue
            found = True
            coll = d["ns"].split(".", 1)[1]
            parts = ", ".join(
                f"{s['shardName']}:{s.get('numOwnedDocuments', 0):,}"
                for s in d.get("shards", [])
            )
            print(f"    {coll}: {parts}")
        if not found:
            print("    (belum ada collection sharded yang berisi data — "
                  "balancer mungkin masih mendistribusikan)")
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

    print("\n[2] Koneksi ke tujuan (mongos sharded cluster) ...")
    try:
        dst = connect(DST_URI, "DST")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Pastikan mongos {DST_URI} aktif dan dapat diakses dari jaringan Anda.")
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
    print("Koneksi aplikasi → MONGODB_URI=mongodb://10.4.89.233:27017")


if __name__ == "__main__":
    main()
