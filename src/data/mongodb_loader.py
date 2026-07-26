
"""Loader untuk memuat data hasil preprocessing ke MongoDB Sharded Cluster.

Terhubung ke mongos router (MONGODB_URI) lalu menulis data ke collection
yang sudah di-shard oleh docker/init-mongo-sharding.js. Karena shard key
memakai hashed user_id / product_id, dokumen otomatis terdistribusi merata
ke seluruh shard.
"""

import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import (
    BulkWriteError,
    ConnectionFailure,
    ServerSelectionTimeoutError,
)

from src.config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTIONS,
    MONGODB_SHARD_KEYS,
    MONGODB_CONNECT_TIMEOUT_MS,
    MONGODB_SERVER_SELECTION_TIMEOUT_MS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 1000


class MongoDBLoader:
    """Memuat reviews/ratings/products/users ke MongoDB sharded cluster."""

    def __init__(self, uri: str = None, database: str = None):
        self.uri = uri or MONGODB_URI
        self.database_name = database or MONGODB_DATABASE
        self.client = MongoClient(
            self.uri,
            connectTimeoutMS=MONGODB_CONNECT_TIMEOUT_MS,
            serverSelectionTimeoutMS=MONGODB_SERVER_SELECTION_TIMEOUT_MS,
            maxPoolSize=10,
            minPoolSize=1,
        )
        # Memicu pemilihan server -> gagal cepat bila cluster tidak aktif
        self.client.admin.command("ping")
        self.db = self.client[self.database_name]
        logger.info(f" Terhubung ke MongoDB: {self.uri} (db={self.database_name})")

    def is_sharded_cluster(self) -> bool:
        """True bila terhubung ke mongos (cluster sharded), bukan mongod tunggal."""
        try:
            self.client.admin.command("listShards")
            return True
        except Exception:
            return False

    def _bulk_upsert(self, collection_name: str, ops: list) -> int:
        """Jalankan UpdateOne(upsert) secara batch, kembalikan jumlah operasi."""
        if not ops:
            return 0
        col = self.db[collection_name]
        total = 0
        for i in range(0, len(ops), BATCH_SIZE):
            batch = ops[i : i + BATCH_SIZE]
            try:
                col.bulk_write(batch, ordered=False)
                total += len(batch)
            except BulkWriteError as e:
                logger.warning(f"   Sebagian batch gagal pada '{collection_name}': {e.details}")
        return total

    def load_reviews(self, df: pd.DataFrame) -> int:
        """Muat dokumen review lengkap. Sharded by hashed user_id."""
        coll = MONGODB_COLLECTIONS["reviews"]
        ops = []
        for row in df.itertuples(index=False):
            doc = {
                "user_id": str(row.user_id),
                "product_id": str(row.product_id),
                "rating": float(row.rating),
                "review_text": getattr(row, "review_text", ""),
                "rating_normalized": float(getattr(row, "rating_normalized", 0.0) or 0.0),
                "word_count": int(getattr(row, "word_count", 0) or 0),
            }
            ops.append(
                UpdateOne(
                    {"user_id": doc["user_id"], "product_id": doc["product_id"]},
                    {"$set": doc},
                    upsert=True,
                )
            )
        n = self._bulk_upsert(coll, ops)
        logger.info(f"   reviews: {n} dokumen dimuat")
        return n

    def load_ratings(self, df: pd.DataFrame) -> int:
        """Muat matriks interaksi rating ringkas. Sharded by hashed user_id."""
        coll = MONGODB_COLLECTIONS["ratings"]
        ops = []
        for row in df.itertuples(index=False):
            doc = {
                "user_id": str(row.user_id),
                "product_id": str(row.product_id),
                "rating": float(row.rating),
                "rating_normalized": float(getattr(row, "rating_normalized", 0.0) or 0.0),
            }
            ops.append(
                UpdateOne(
                    {"user_id": doc["user_id"], "product_id": doc["product_id"]},
                    {"$set": doc},
                    upsert=True,
                )
            )
        n = self._bulk_upsert(coll, ops)
        logger.info(f"   ratings: {n} dokumen dimuat")
        return n

    def load_products(self, df: pd.DataFrame) -> int:
        """Agregasi metadata per produk. Collection 'products' (tidak di-shard)."""
        coll = MONGODB_COLLECTIONS["products"]
        agg = (
            df.groupby("product_id")
            .agg(review_count=("rating", "size"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        ops = [
            UpdateOne(
                {"product_id": str(r.product_id)},
                {"$set": {
                    "product_id": str(r.product_id),
                    "review_count": int(r.review_count),
                    "avg_rating": round(float(r.avg_rating), 3),
                }},
                upsert=True,
            )
            for r in agg.itertuples(index=False)
        ]
        n = self._bulk_upsert(coll, ops)
        logger.info(f"   products: {n} dokumen dimuat")
        return n

    def load_users(self, df: pd.DataFrame) -> int:
        """Agregasi profil per user. Collection 'users' (tidak di-shard)."""
        coll = MONGODB_COLLECTIONS["users"]
        agg = (
            df.groupby("user_id")
            .agg(review_count=("rating", "size"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        ops = [
            UpdateOne(
                {"user_id": str(r.user_id)},
                {"$set": {
                    "user_id": str(r.user_id),
                    "review_count": int(r.review_count),
                    "avg_rating": round(float(r.avg_rating), 3),
                }},
                upsert=True,
            )
            for r in agg.itertuples(index=False)
        ]
        n = self._bulk_upsert(coll, ops)
        logger.info(f"   users: {n} dokumen dimuat")
        return n

    def get_shard_distribution(self) -> dict:
        """Distribusi chunk per shard untuk tiap collection yang di-shard.

        Berguna sebagai bukti data benar-benar terdistribusi (untuk laporan TA).
        """
        if not self.is_sharded_cluster():
            return {"sharded": False, "note": "Terhubung ke mongod tunggal, bukan cluster."}

        result = {"sharded": True, "shards": [], "collections": {}}
        try:
            shards = self.client.admin.command("listShards").get("shards", [])
            result["shards"] = [{"id": s["_id"], "host": s["host"]} for s in shards]

            config_db = self.client["config"]
            for coll in MONGODB_SHARD_KEYS:
                ns = f"{self.database_name}.{coll}"
                dist = list(config_db["chunks"].aggregate([
                    {"$match": {"ns": ns}},
                    {"$group": {"_id": "$shard", "chunks": {"$sum": 1}}},
                ]))
                result["collections"][coll] = {
                    d["_id"]: d["chunks"] for d in dist
                }
        except Exception as e:
            result["error"] = str(e)
        return result

    def run_full_load(self, clean_df: pd.DataFrame) -> dict:
        """Muat seluruh data hasil preprocessing ke cluster."""
        if clean_df is None or len(clean_df) == 0:
            logger.warning(" clean_df kosong, tidak ada yang dimuat.")
            return {}

        if self.is_sharded_cluster():
            logger.info(" Terdeteksi sharded cluster — data akan terdistribusi via hashed shard key.")
        else:
            logger.warning(
                " Terhubung ke mongod tunggal (bukan sharded cluster). "
                "Data tetap dimuat, namun tidak terdistribusi. "
                "Jalankan 'docker-compose up -d' di folder docker/ untuk cluster penuh."
            )

        counts = {
            "reviews": self.load_reviews(clean_df),
            "ratings": self.load_ratings(clean_df),
            "products": self.load_products(clean_df),
            "users": self.load_users(clean_df),
        }

        dist = self.get_shard_distribution()
        if dist.get("sharded"):
            logger.info(" Distribusi chunk per shard:")
            for coll, shards in dist.get("collections", {}).items():
                logger.info(f"   {coll}: {shards}")

        logger.info(" Pemuatan ke MongoDB selesai.")
        return counts

    def close(self):
        if self.client is not None:
            self.client.close()
            logger.info(" Koneksi MongoDB ditutup.")
