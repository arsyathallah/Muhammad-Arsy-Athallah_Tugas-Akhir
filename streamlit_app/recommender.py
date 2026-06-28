import numpy as np
import pandas as pd
import pickle

import database as db

class CBFRecommender:
    def __init__(self, embeddings_path, item_ids_path, metadata_path=None):
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        with open(item_ids_path, "rb") as f:
            self.item_ids = pickle.load(f)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.item_ids)}

        if db.is_connected():
            meta_df = db.get_items_df()
        else:
            if metadata_path:
                meta_df = pd.read_csv(metadata_path, dtype={"product_id": str})
            else:
                meta_df = pd.DataFrame(columns=["product_id","title","image_url","rating_count","avg_rating"])

        self.metadata = meta_df.set_index("product_id") if not meta_df.empty else pd.DataFrame()
        self._use_mongo = db.is_connected()

    def get_user_embedding(self, rated_items: dict) -> np.ndarray | None:
        embs, weights = [], []
        for pid, rating in rated_items.items():
            if pid in self.pid_to_idx:
                embs.append(self.embeddings[self.pid_to_idx[pid]])
                weights.append(max((rating - 1) / 4.0, 0.01))
        if not embs:
            return None
        embs    = np.array(embs, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        user_emb = (embs * weights[:, None]).sum(0) / weights.sum()
        norm = np.linalg.norm(user_emb)
        return user_emb / norm if norm > 0 else None

    def recommend(self, rated_items: dict, top_n: int = 10) -> list[dict]:
        user_emb = self.get_user_embedding(rated_items)
        if user_emb is None:
            return []

        scores = self.embeddings @ user_emb
        for pid in rated_items:
            if pid in self.pid_to_idx:
                scores[self.pid_to_idx[pid]] = -2.0

        actual_n = min(top_n, len(self.item_ids))
        top_idx  = np.argpartition(scores, -actual_n)[-actual_n:]
        top_idx  = top_idx[np.argsort(scores[top_idx])[::-1]]

        pids = [self.item_ids[i] for i in top_idx]

        if self._use_mongo:
            meta_batch = db.get_items_by_ids(pids)
        else:
            meta_batch = {}

        results = []
        for idx, pid in zip(top_idx, pids):
            meta = meta_batch.get(pid, {})
            results.append({
                "product_id"  : pid,
                "score"       : float(scores[idx]),
                "title"       : meta.get("title") or self._get_field_csv(pid, "title", f"Game ({pid[:8]})"),
                "image_url"   : meta.get("image_url") or self._get_field_csv(pid, "image_url", None),
                "product_url" : meta.get("product_url") or self._get_field_csv(pid, "product_url",
                                    f"https://www.amazon.com/dp/{pid}"),
                "rating_count": meta.get("rating_count") or self._get_field_csv(pid, "rating_count", 0),
            })
        return results

    def get_popular_games(self, n: int = 60) -> list[dict]:
        if self._use_mongo:
            docs = db.get_popular_items(n)
            return [d for d in docs if d.get("product_id") in self.pid_to_idx]

        col = "rating_count" if "rating_count" in self.metadata.columns else None
        df  = self.metadata.nlargest(n, col) if col else self.metadata.head(n)
        df  = df[df.index.isin(self.pid_to_idx)]
        return df.reset_index().rename(columns={"index": "product_id"}).to_dict("records")

    def search_games(self, query: str, n: int = 30) -> list[dict]:
        if self._use_mongo:
            docs = db.search_items(query, n)
            return [d for d in docs if d.get("product_id") in self.pid_to_idx]

        mask = self.metadata["title"].str.contains(query, case=False, na=False)
        df   = self.metadata[mask & self.metadata.index.isin(self.pid_to_idx)].head(n)
        return df.reset_index().rename(columns={"index": "product_id"}).to_dict("records")

    def _get_field_csv(self, pid: str, field: str, default):
        if not self.metadata.empty and pid in self.metadata.index:
            val = self.metadata.at[pid, field]
            if pd.notna(val) and val != "":
                return val
        return default
