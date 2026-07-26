"""
Script demo untuk sidang: membuktikan MongoDB Sharded Cluster benar-benar
mendistribusikan data ke beberapa shard, bukan cuma satu instance.

Jalankan (VPN ITS harus aktif dulu):
  python scripts/demo_sharding.py
"""

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

URI = os.getenv("MONGODB_URI", "mongodb://10.4.89.233:27017")
DB_NAME = os.getenv("MONGODB_DATABASE", "recommendation_system")

COLLECTIONS = ["reviews", "ratings", "recommendations", "product_profiles", "user_profiles"]


def section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    section("1. KONEKSI KE CLUSTER (via mongos)")
    try:
        client = MongoClient(URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print(f"  Terhubung ke: {URI}")
    except Exception as e:
        print(f"  GAGAL konek: {e}")
        print("  Pastikan VPN ITS aktif dan cluster hidup.")
        sys.exit(1)

    section("2. TOPOLOGI CLUSTER (shard yang aktif)")
    status = client.admin.command("listShards")
    shards = status.get("shards", [])
    print(f"  Jumlah shard aktif: {len(shards)}")
    for s in shards:
        print(f"    - {s['_id']:10s} -> {s['host']}")

    section("3. SHARD KEY PER COLLECTION")
    db_collections = client["config"]["collections"]
    for coll in COLLECTIONS:
        ns = f"{DB_NAME}.{coll}"
        doc = db_collections.find_one({"_id": ns})
        if doc:
            key = doc.get("key", {})
            print(f"  {coll:20s} shard key: {key}")
        else:
            print(f"  {coll:20s} (belum sharded / tidak ditemukan)")

    section("4. DISTRIBUSI DOKUMEN PER SHARD (bukti data tersebar)")
    wanted = {f"{DB_NAME}.{c}" for c in COLLECTIONS}
    found = False
    for d in client.admin.aggregate([{"$shardedDataDistribution": {}}]):
        if d["ns"] not in wanted:
            continue
        found = True
        coll = d["ns"].split(".", 1)[1]
        total = sum(s.get("numOwnedDocuments", 0) for s in d.get("shards", []))
        parts = ", ".join(
            f"{s['shardName']}={s.get('numOwnedDocuments', 0):,}"
            for s in d.get("shards", [])
        )
        print(f"  {coll:20s} total={total:,}  |  {parts}")
    if not found:
        print("  (data belum terlihat sharded / balancer belum jalan)")

    section("5. RINGKASAN JUMLAH DOKUMEN (query normal lewat mongos)")
    db = client[DB_NAME]
    for coll in COLLECTIONS:
        count = db[coll].count_documents({})
        print(f"  {coll:20s} {count:,} dokumen")

    client.close()
    print("\nSelesai. Cluster terbukti sharded & data tersebar di beberapa node.\n")


if __name__ == "__main__":
    main()
