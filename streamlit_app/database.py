
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv(Path(__file__).parent.parent / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME          = "gamerecsys"
ITEMS_COLLECTION = "items"
UAT_COLLECTION   = "uat_results"

_client: Optional[MongoClient] = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
            maxPoolSize=10,
            minPoolSize=1,
        )
    return _client

def get_db():
    return get_client()[DB_NAME]

def is_connected() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return False

def upsert_items(df: pd.DataFrame) -> int:
    col = get_db()[ITEMS_COLLECTION]
    col.create_index("product_id", unique=True)

    ops = []
    from pymongo import UpdateOne
    for row in df.itertuples(index=False):
        pid = row.product_id
        ops.append(UpdateOne(
            {"product_id": pid},
            {"$set": {
                "product_id"  : pid,
                "title"       : getattr(row, "title", ""),
                "image_url"   : getattr(row, "image_url", ""),
                "product_url" : getattr(row, "product_url",
                                    f"https://www.amazon.com/dp/{pid}"),
                "rating_count": int(getattr(row, "rating_count", 0) or 0),
                "avg_rating"  : float(getattr(row, "avg_rating", 0) or 0),
            }},
            upsert=True,
        ))
        if len(ops) >= 1000:
            col.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        col.bulk_write(ops, ordered=False)

    return col.count_documents({})

def get_items_df() -> pd.DataFrame:
    col  = get_db()[ITEMS_COLLECTION]
    docs = list(col.find({}, {"_id": 0}))
    if not docs:
        return pd.DataFrame(columns=["product_id", "title", "image_url", "rating_count", "avg_rating"])
    return pd.DataFrame(docs)

def get_popular_items(n: int = 60) -> list[dict]:
    col  = get_db()[ITEMS_COLLECTION]
    docs = list(col.find({}, {"_id": 0}).sort("rating_count", -1).limit(n))
    return docs

def search_items(query: str, n: int = 30) -> list[dict]:
    col = get_db()[ITEMS_COLLECTION]
    try:
        col.create_index([("title", "text")], background=True)
        docs = list(col.find(
            {"$text": {"$search": query}},
            {"_id": 0, "score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})]).limit(n))
        if docs:
            return docs
    except Exception:
        pass
    docs = list(col.find(
        {"title": {"$regex": query, "$options": "i"}},
        {"_id": 0},
    ).limit(n))
    return docs

def get_item(product_id: str) -> Optional[dict]:
    col = get_db()[ITEMS_COLLECTION]
    return col.find_one({"product_id": product_id}, {"_id": 0})

def get_items_by_ids(product_ids: list[str]) -> dict[str, dict]:
    col  = get_db()[ITEMS_COLLECTION]
    docs = col.find({"product_id": {"$in": product_ids}}, {"_id": 0})
    return {d["product_id"]: d for d in docs}

def save_uat_result(row: dict) -> str:
    col = get_db()[UAT_COLLECTION]
    col.create_index("session_id")
    if "sus_answers" in row and isinstance(row["sus_answers"], dict):
        row["sus_answers"] = {str(k): v for k, v in row["sus_answers"].items()}
    row["created_at"] = datetime.utcnow()
    result = col.insert_one(row)
    return str(result.inserted_id)

def get_all_uat_results() -> pd.DataFrame:
    col  = get_db()[UAT_COLLECTION]
    docs = list(col.find({}, {"_id": 0, "created_at": 0}))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs)

def get_uat_stats() -> dict:
    col   = get_db()[UAT_COLLECTION]
    total = col.count_documents({})
    if total == 0:
        return {"total_respondents": 0}

    pipeline = [
        {"$group": {
            "_id"      : None,
            "avg_sus"  : {"$avg": "$sus_score"},
            "avg_rel"  : {"$avg": "$relevant_count"},
            "avg_rated": {"$avg": "$rated_count"},
        }}
    ]
    agg = list(col.aggregate(pipeline))
    stats = agg[0] if agg else {}
    return {
        "total_respondents": total,
        "avg_sus_score"    : round(stats.get("avg_sus", 0), 1),
        "avg_relevant"     : round(stats.get("avg_rel", 0), 1),
        "avg_rated"        : round(stats.get("avg_rated", 0), 1),
    }

def auto_seed_if_empty(csv_path: str) -> int:
    """Load items from CSV into MongoDB if the collection is empty. Returns count of items."""
    from pathlib import Path
    col = get_db()[ITEMS_COLLECTION]
    if col.count_documents({}) > 0:
        return col.count_documents({})
    p = Path(csv_path)
    if not p.exists():
        return 0
    import pandas as pd
    df = pd.read_csv(p, dtype={"product_id": str})
    return upsert_items(df)

def get_shard_info() -> dict:
    try:
        client = get_client()
        status = client.admin.command("listShards")
        shards = [
            {"id": s["_id"], "host": s["host"]}
            for s in status.get("shards", [])
        ]
        db_admin = client["config"]
        chunks   = list(db_admin["chunks"].aggregate([
            {"$match": {"ns": f"{DB_NAME}.{ITEMS_COLLECTION}"}},
            {"$group": {"_id": "$shard", "chunks": {"$sum": 1}}},
        ]))
        return {"shards": shards, "chunk_distribution": chunks}
    except Exception as e:
        return {"shards": [], "chunk_distribution": [], "error": str(e)}
