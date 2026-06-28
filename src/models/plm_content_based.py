
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import PLM_CBF_CONFIG, MODELS_DIR, TRAIN_PARQUET
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PLMContentBasedModel:

    def __init__(self, config: dict = None):
        self.config = config or PLM_CBF_CONFIG
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        model_name = self.config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading Sentence Transformer PLM: {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            self.encoder = SentenceTransformer(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = self.encoder.to(self.device)
            logger.info(f"PLM loaded on device: {self.device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self.product_profiles: Dict[str, dict] = {}
        self.user_profiles: Dict[str, dict] = {}
        self.product_embeddings_matrix = None
        self.product_ids_list: List[str] = []
        self.product_embeddings: Dict[str, np.ndarray] = {}
        self._is_fitted = False

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing sentiment of reviews...")
        sentiments = []
        total = len(df)

        for idx, text in enumerate(df["review_text"].values):
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"   Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            text = str(text) if pd.notna(text) else ""
            if len(text.strip()) < 3:
                sentiments.append({
                    "sentiment_compound": 0.0,
                    "sentiment_pos": 0.0,
                    "sentiment_neg": 0.0,
                    "sentiment_neu": 1.0,
                })
                continue
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiments.append({
                "sentiment_compound": scores["compound"],
                "sentiment_pos": scores["pos"],
                "sentiment_neg": scores["neg"],
                "sentiment_neu": scores["neu"],
            })

        sentiment_df = pd.DataFrame(sentiments)
        result = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        result["sentiment_label"] = result["sentiment_compound"].apply(
            lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
        )
        label_counts = result["sentiment_label"].value_counts()
        logger.info(f"Sentiment complete — pos:{label_counts.get('positive',0)} "
                    f"neu:{label_counts.get('neutral',0)} neg:{label_counts.get('negative',0)}")
        return result

    def build_product_profiles(self, df: pd.DataFrame) -> Dict[str, dict]:
        logger.info("Building product profiles with PLM (Sentence Transformer) embeddings...")
        batch_size = self.config.get("batch_size", 256)
        max_len = self.config.get("max_seq_length", 256)
        min_reviews = self.config.get("min_reviews_per_product", 3)

        product_groups = df.groupby("product_id")

        product_texts = {}
        for product_id, group in product_groups:
            all_text = " ".join(group["review_text"].dropna().astype(str).values)
            all_text = all_text[:max_len * 6]
            if len(all_text.strip()) > 10:
                product_texts[product_id] = all_text

        self.product_ids_list = list(product_texts.keys())
        texts_list = [product_texts[pid] for pid in self.product_ids_list]

        if texts_list:
            logger.info(f"Encoding {len(texts_list)} products with PLM (batch_size={batch_size})...")
            t0 = time.time()
            embeddings = self.encoder.encode(
                texts_list,
                batch_size=batch_size,
                show_progress_bar=True,
                device=self.device,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            elapsed = time.time() - t0
            self.product_embeddings_matrix = embeddings.astype(np.float32)
            logger.info(f"PLM encoding done in {elapsed:.1f}s — shape: {self.product_embeddings_matrix.shape}")

            for i, pid in enumerate(self.product_ids_list):
                self.product_embeddings[pid] = embeddings[i]

        for product_id, group in product_groups:
            if len(group) < min_reviews:
                continue
            avg_sentiment = group["sentiment_compound"].mean()
            avg_rating = group["rating"].mean()
            review_count = len(group)
            label_counts = group["sentiment_label"].value_counts(normalize=True)
            confidence = 1 - 1 / (1 + np.log1p(review_count))
            review_score = ((avg_sentiment + 1) / 2) * confidence
            self.product_profiles[product_id] = {
                "product_id": product_id,
                "avg_sentiment": float(avg_sentiment),
                "sentiment_distribution": {
                    "positive": label_counts.get("positive", 0),
                    "neutral": label_counts.get("neutral", 0),
                    "negative": label_counts.get("negative", 0),
                },
                "review_count": int(review_count),
                "avg_rating": float(avg_rating),
                "confidence": float(confidence),
                "review_score": float(review_score),
            }

        logger.info(f"Built {len(self.product_profiles)} product profiles")
        return self.product_profiles

    def build_user_profiles(self, df: pd.DataFrame) -> Dict[str, dict]:
        logger.info("Building user profiles from PLM product embeddings...")
        t0 = time.time()
        embed_dim = (
            self.product_embeddings_matrix.shape[1]
            if self.product_embeddings_matrix is not None
            else self.config.get("embedding_dim", 384)
        )

        user_sums = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float32))
        user_weights = defaultdict(float)
        user_products = defaultdict(set)
        user_ratings = defaultdict(list)
        user_sentiments = defaultdict(list)

        has_sentiment = "sentiment_compound" in df.columns

        for row in df.itertuples():
            uid = row.user_id
            pid = row.product_id
            rating = row.rating
            user_products[uid].add(pid)
            user_ratings[uid].append(rating)
            if has_sentiment:
                user_sentiments[uid].append(row.sentiment_compound)
            if pid in self.product_embeddings:
                weight = (rating - 1) / 4.0
                user_sums[uid] += self.product_embeddings[pid] * weight
                user_weights[uid] += weight

        for uid in user_products.keys():
            w = user_weights[uid]
            user_emb = user_sums[uid] / w if w > 0 else np.zeros(embed_dim, dtype=np.float32)
            avg_rating = sum(user_ratings[uid]) / len(user_ratings[uid])
            avg_sentiment = (
                sum(user_sentiments[uid]) / len(user_sentiments[uid])
                if has_sentiment else 0.0
            )
            self.user_profiles[uid] = {
                "user_id": uid,
                "embedding_vector": user_emb,
                "reviewed_products": user_products[uid],
                "avg_rating_given": float(avg_rating),
                "avg_sentiment": float(avg_sentiment),
                "num_reviews": len(user_ratings[uid]),
            }

        elapsed = time.time() - t0
        logger.info(f"Built {len(self.user_profiles)} user profiles in {elapsed:.1f}s")
        return self.user_profiles

    def batch_generate_scores(self, target_user_ids, candidate_product_ids,
                               top_n=200, exclude_reviewed=True, batch_size=2000):
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        valid_pids = [
            pid for pid in candidate_product_ids
            if pid in self.product_profiles and pid in self.product_embeddings
        ]
        if not valid_pids:
            return pd.DataFrame(columns=["user_id", "product_id", "cbf_score"])

        n_products = len(valid_pids)
        logger.info(f"Batch PLM-CBF scoring: {len(target_user_ids)} users × {n_products} products")
        t0 = time.time()

        product_matrix_T = np.ascontiguousarray(
            np.vstack([self.product_embeddings[pid] for pid in valid_pids]).T.astype(np.float32)
        )

        review_scores = np.array(
            [self.product_profiles[pid]["review_score"] for pid in valid_pids], dtype=np.float32
        )
        rating_factors = np.clip(
            np.array(
                [(self.product_profiles[pid]["avg_rating"] - 1.0) / 4.0 for pid in valid_pids],
                dtype=np.float32
            ), 0.0, 1.0
        )
        static_scores = 0.1 * review_scores + 0.1 * rating_factors
        pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}

        all_uids, all_pids, all_scores = [], [], []

        for b_start in range(0, len(target_user_ids), batch_size):
            batch_uids = target_user_ids[b_start:b_start + batch_size]
            vectors, uids_valid = [], []
            for uid in batch_uids:
                prof = self.user_profiles.get(uid)
                if prof is not None:
                    vec = prof["embedding_vector"]
                    if np.any(vec != 0):
                        norm = np.linalg.norm(vec)
                        vectors.append(vec / norm if norm > 0 else vec)
                        uids_valid.append(uid)

            if not vectors:
                continue

            user_mat = np.vstack(vectors).astype(np.float32)
            sims = user_mat @ product_matrix_T
            np.clip(sims, 0, None, out=sims)

            scores_batch = 0.8 * sims + static_scores
            np.clip(scores_batch, 0.0, 1.0, out=scores_batch)

            actual_top = min(top_n, n_products)

            for i, uid in enumerate(uids_valid):
                row = scores_batch[i]
                if exclude_reviewed:
                    for rpid in self.user_profiles[uid].get("reviewed_products", set()):
                        j = pid_to_idx.get(rpid)
                        if j is not None:
                            row[j] = -1.0
                if actual_top < n_products:
                    top_idx = np.argpartition(row, -actual_top)[-actual_top:]
                else:
                    top_idx = np.arange(n_products)
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
                for j in top_idx:
                    if row[j] > 0:
                        all_uids.append(uid)
                        all_pids.append(valid_pids[j])
                        all_scores.append(row[j])

        elapsed = time.time() - t0
        logger.info(f"PLM-CBF batch scoring done: {len(all_uids)} scores in {elapsed:.1f}s")

        if all_uids:
            return pd.DataFrame({
                "user_id": all_uids,
                "product_id": all_pids,
                "cbf_score": np.array(all_scores, dtype=np.float32),
            })
        return pd.DataFrame(columns=["user_id", "product_id", "cbf_score"])

    def get_cbf_scores_for_evaluation(self, user_id: str,
                                       candidate_products: List[str]) -> Dict[str, float]:
        if user_id not in self.user_profiles:
            return {}
        prof = self.user_profiles[user_id]
        user_vec = prof["embedding_vector"]
        norm = np.linalg.norm(user_vec)
        if norm == 0:
            return {}
        user_vec = (user_vec / norm).reshape(1, -1)

        scores = {}
        for pid in candidate_products:
            if pid not in self.product_profiles:
                continue
            p = self.product_profiles[pid]
            review_score = p["review_score"]
            rating_factor = np.clip((p["avg_rating"] - 1.0) / 4.0, 0.0, 1.0)
            if pid in self.product_embeddings:
                pv = self.product_embeddings[pid].reshape(1, -1)
                sim = float(cosine_similarity(user_vec, pv)[0][0])
                sim = max(0.0, sim)
                score = 0.8 * sim + 0.1 * review_score + 0.1 * rating_factor
            else:
                score = 0.6 * review_score + 0.4 * rating_factor
            scores[pid] = float(min(1.0, max(0.0, score)))
        return scores

    def fit(self, df: pd.DataFrame = None) -> "PLMContentBasedModel":
        logger.info("=" * 60)
        logger.info("TRAINING PLM CONTENT-BASED FILTERING")
        logger.info("=" * 60)
        if df is None:
            df = pd.read_parquet(str(TRAIN_PARQUET))
            logger.info(f"Loaded training data: {len(df)} rows")

        df = self.analyze_sentiment(df)
        self.build_product_profiles(df)
        self.build_user_profiles(df)
        self._is_fitted = True

        logger.info("=" * 60)
        logger.info("PLM-CBF TRAINING COMPLETE")
        logger.info("=" * 60)
        return self

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "plm_cbf"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "product_profiles.pkl", "wb") as f:
            pickle.dump(self.product_profiles, f)

        user_profiles_simple = {}
        for uid, profile in self.user_profiles.items():
            user_profiles_simple[uid] = {
                k: v for k, v in profile.items()
                if k not in ("embedding_vector", "reviewed_products")
            }
            user_profiles_simple[uid]["reviewed_products"] = list(profile["reviewed_products"])

        with open(path / "user_profiles.pkl", "wb") as f:
            pickle.dump(user_profiles_simple, f)

        with open(path / "embeddings.pkl", "wb") as f:
            pickle.dump({
                "product_vectors": self.product_embeddings,
                "product_ids": self.product_ids_list,
                "user_vectors": {
                    uid: p["embedding_vector"]
                    for uid, p in self.user_profiles.items()
                },
            }, f)

        logger.info(f"PLM-CBF model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "plm_cbf"
        path = Path(path)

        with open(path / "product_profiles.pkl", "rb") as f:
            self.product_profiles = pickle.load(f)

        with open(path / "embeddings.pkl", "rb") as f:
            vectors = pickle.load(f)
            self.product_embeddings = vectors["product_vectors"]
            self.product_ids_list = vectors["product_ids"]
            user_vectors = vectors["user_vectors"]

        with open(path / "user_profiles.pkl", "rb") as f:
            user_profiles_simple = pickle.load(f)

        self.user_profiles = {}
        for uid, profile in user_profiles_simple.items():
            self.user_profiles[uid] = profile
            self.user_profiles[uid]["reviewed_products"] = set(profile["reviewed_products"])
            self.user_profiles[uid]["embedding_vector"] = user_vectors.get(
                uid, np.zeros(self.config.get("embedding_dim", 384), dtype=np.float32)
            )

        self._is_fitted = True
        logger.info(f"PLM-CBF model loaded from {path}")
        logger.info(f"   Products: {len(self.product_profiles)}, Users: {len(self.user_profiles)}")
