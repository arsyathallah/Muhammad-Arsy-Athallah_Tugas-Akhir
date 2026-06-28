
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import sys
import subprocess

from src.config import CBF_CONFIG, MODELS_DIR, TRAIN_PARQUET
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ContentBasedModel:

    def __init__(self, config: dict = None):
        self.config = config or CBF_CONFIG
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        model_name = self.config.get("pretrained_model_name", "en_core_web_md")
        logger.info(f"Loading SpaCy Pre-trained model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Attempting to download...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)

        self.product_profiles: Dict[str, dict] = {}
        self.user_profiles: Dict[str, dict] = {}
        self.product_embeddings_matrix = None
        self.product_ids_list: List[str] = []
        self.product_embeddings: Dict[str, np.ndarray] = {}
        self._is_fitted = False

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(" Analyzing sentiment of reviews...")

        sentiments = []
        total = len(df)

        for idx, text in enumerate(df["review_text"].values):
            if idx % 5000 == 0 and idx > 0:
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
        logger.info(f"Sentiment analysis complete:")
        logger.info(f"   Positive: {label_counts.get('positive', 0)}")
        logger.info(f"   Neutral:  {label_counts.get('neutral', 0)}")
        logger.info(f"   Negative: {label_counts.get('negative', 0)}")

        return result

    def build_product_profiles(self, df: pd.DataFrame) -> Dict[str, dict]:
        logger.info(" Building product profiles with Pre-Trained SpaCy Embeddings...")

        min_reviews = self.config.get("min_reviews_per_product", 3)
        product_groups = df.groupby("product_id")

        logger.info("   Extracting texts for embedding...")
        product_texts = {}
        for product_id, group in product_groups:
            all_text = " ".join(group["review_text"].dropna().astype(str).values)
            if len(all_text.strip()) > 10:
                product_texts[product_id] = all_text

        self.product_ids_list = list(product_texts.keys())
        texts_list = [product_texts[pid] for pid in self.product_ids_list]

        if texts_list:
            logger.info(f"   Encoding {len(texts_list)} products using SpaCy (this may take a few minutes)...")
            
            embeddings = []
            for doc in self.nlp.pipe(texts_list, batch_size=256, n_process=1):
                embeddings.append(doc.vector)
                
            embeddings = np.vstack(embeddings).astype(np.float32)
            self.product_embeddings_matrix = embeddings
            logger.info(f"   Embeddings matrix shape: {self.product_embeddings_matrix.shape}")

            for i, pid in enumerate(self.product_ids_list):
                self.product_embeddings[pid] = embeddings[i]

        for product_id, group in product_groups:
            if len(group) < min_reviews:
                continue

            review_count = len(group)
            avg_sentiment = group["sentiment_compound"].mean()
            avg_rating = group["rating"].mean()

            label_counts = group["sentiment_label"].value_counts(normalize=True)
            sentiment_dist = {
                "positive": label_counts.get("positive", 0),
                "neutral": label_counts.get("neutral", 0),
                "negative": label_counts.get("negative", 0),
            }

            confidence = 1 - 1 / (1 + np.log1p(review_count))
            review_score = ((avg_sentiment + 1) / 2) * confidence

            self.product_profiles[product_id] = {
                "product_id": product_id,
                "avg_sentiment": float(avg_sentiment),
                "sentiment_distribution": sentiment_dist,
                "review_count": int(review_count),
                "avg_rating": float(avg_rating),
                "confidence": float(confidence),
                "review_score": float(review_score),
            }

        logger.info(f"Built {len(self.product_profiles)} product profiles")
        return self.product_profiles

    def build_user_profiles(self, df: pd.DataFrame) -> Dict[str, dict]:
        logger.info(" Building user profiles...")
        t0 = time.time()

        actual_dim = self.product_embeddings_matrix.shape[1] if self.product_embeddings_matrix is not None else self.config.get("embedding_dim", 300)

        user_sums = defaultdict(lambda: np.zeros(actual_dim, dtype=np.float32))
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
            w_sum = user_weights[uid]
            if w_sum > 0:
                user_embedding = user_sums[uid] / w_sum
            else:
                user_embedding = np.zeros(actual_dim, dtype=np.float32)

            avg_rating = sum(user_ratings[uid]) / len(user_ratings[uid])
            avg_sentiment = sum(user_sentiments[uid]) / len(user_sentiments[uid]) if has_sentiment else 0.0

            self.user_profiles[uid] = {
                "user_id": uid,
                "embedding_vector": user_embedding,
                "reviewed_products": user_products[uid],
                "avg_rating_given": float(avg_rating),
                "avg_sentiment": float(avg_sentiment),
                "num_reviews": len(user_ratings[uid]),
            }

        elapsed = time.time() - t0
        logger.info(f"Built {len(self.user_profiles)} user profiles in {elapsed:.1f}s")
        return self.user_profiles

    def get_cbf_scores(self, user_id: str,
                       candidate_products: List[str] = None) -> Dict[str, float]:
        if user_id not in self.user_profiles:
            return {}

        user_profile = self.user_profiles[user_id]
        user_vector = user_profile["embedding_vector"].reshape(1, -1)
        user_has_vector = np.any(user_vector != 0)
        reviewed = user_profile["reviewed_products"]

        if candidate_products is None:
            candidate_products = [
                pid for pid in self.product_profiles.keys()
                if pid not in reviewed
            ]
        else:
            candidate_products = [pid for pid in candidate_products if pid not in reviewed]

        scores = {}
        for pid in candidate_products:
            if pid not in self.product_profiles:
                continue

            profile = self.product_profiles[pid]
            review_score = profile["review_score"]
            rating_factor = (profile["avg_rating"] - 1.0) / 4.0
            rating_factor = max(0.0, min(1.0, rating_factor))

            if pid in self.product_embeddings and user_has_vector:
                product_vector = self.product_embeddings[pid].reshape(1, -1)
                
                sim = cosine_similarity(user_vector, product_vector)[0][0]
                sim = max(0.0, sim)
                
                cbf_score = 0.8 * sim + 0.1 * review_score + 0.1 * rating_factor
            else:
                cbf_score = 0.6 * review_score + 0.4 * rating_factor

            scores[pid] = float(min(1.0, max(0.0, cbf_score)))

        return scores

    def get_cbf_scores_for_evaluation(self, user_id: str,
                                       candidate_products: List[str]) -> Dict[str, float]:
        if user_id not in self.user_profiles:
            return {}

        user_profile = self.user_profiles[user_id]
        user_vector = user_profile["embedding_vector"].reshape(1, -1)
        user_has_vector = np.any(user_vector != 0)

        scores = {}
        for pid in candidate_products:
            if pid not in self.product_profiles:
                continue

            profile = self.product_profiles[pid]
            review_score = profile["review_score"]
            rating_factor = (profile["avg_rating"] - 1.0) / 4.0
            rating_factor = max(0.0, min(1.0, rating_factor))

            if pid in self.product_embeddings and user_has_vector:
                product_vector = self.product_embeddings[pid].reshape(1, -1)
                sim = cosine_similarity(user_vector, product_vector)[0][0]
                sim = max(0.0, sim)
                cbf_score = 0.8 * sim + 0.1 * review_score + 0.1 * rating_factor
            else:
                cbf_score = 0.6 * review_score + 0.4 * rating_factor

            scores[pid] = float(min(1.0, max(0.0, cbf_score)))

        return scores

    def batch_generate_scores(self, target_user_ids, candidate_product_ids,
                               top_n=200, exclude_reviewed=True, batch_size=2000):
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        valid_pids = [pid for pid in candidate_product_ids
                      if pid in self.product_profiles and pid in self.product_embeddings]

        if not valid_pids:
            return pd.DataFrame(columns=["user_id", "product_id", "cbf_score"])

        n_products = len(valid_pids)
        logger.info(f"Batch CBF scoring: {len(target_user_ids)} users × {n_products} products")

        t0 = time.time()
        product_matrix = np.vstack(
            [self.product_embeddings[pid] for pid in valid_pids]
        ).astype(np.float32)

        p_norms = np.linalg.norm(product_matrix, axis=1, keepdims=True)
        p_norms[p_norms == 0] = 1.0
        product_matrix /= p_norms
        product_matrix_T = np.ascontiguousarray(product_matrix.T)

        del product_matrix

        review_scores = np.array(
            [self.product_profiles[pid]["review_score"] for pid in valid_pids],
            dtype=np.float32
        )
        rating_factors = np.clip(
            np.array(
                [(self.product_profiles[pid]["avg_rating"] - 1.0) / 4.0
                 for pid in valid_pids], dtype=np.float32
            ), 0.0, 1.0
        )
        static_scores = 0.1 * review_scores + 0.1 * rating_factors

        pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}

        all_uids = []
        all_pids = []
        all_cbf_scores = []
        users_scored = 0

        for b_start in range(0, len(target_user_ids), batch_size):
            b_end = min(b_start + batch_size, len(target_user_ids))
            batch_uids = target_user_ids[b_start:b_end]

            vectors = []
            uids_valid = []
            for uid in batch_uids:
                prof = self.user_profiles.get(uid)
                if prof is not None:
                    vec = prof["embedding_vector"]
                    if np.any(vec != 0):
                        vectors.append(vec)
                        uids_valid.append(uid)

            if not vectors:
                continue

            user_mat = np.vstack(vectors).astype(np.float32)
            u_norms = np.linalg.norm(user_mat, axis=1, keepdims=True)
            u_norms[u_norms == 0] = 1.0
            user_mat /= u_norms

            sims = user_mat @ product_matrix_T
            np.clip(sims, 0, None, out=sims)

            scores_batch = 0.8 * sims
            scores_batch += static_scores
            np.clip(scores_batch, 0.0, 1.0, out=scores_batch)

            actual_top = min(top_n, n_products)

            for i, uid in enumerate(uids_valid):
                row = scores_batch[i]

                if exclude_reviewed:
                    reviewed = self.user_profiles[uid].get("reviewed_products", set())
                    for rpid in reviewed:
                        j = pid_to_idx.get(rpid)
                        if j is not None:
                            row[j] = -1.0

                if actual_top < n_products:
                    top_idx = np.argpartition(row, -actual_top)[-actual_top:]
                else:
                    top_idx = np.arange(n_products)

                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

                for j in top_idx:
                    s = row[j]
                    if s > 0:
                        all_uids.append(uid)
                        all_pids.append(valid_pids[j])
                        all_cbf_scores.append(s)

                users_scored += 1

        if all_uids:
            result_df = pd.DataFrame({
                "user_id": all_uids,
                "product_id": all_pids,
                "cbf_score": np.array(all_cbf_scores, dtype=np.float32),
            })
        else:
            result_df = pd.DataFrame(columns=["user_id", "product_id", "cbf_score"])

        elapsed = time.time() - t0
        logger.info(f" Batch CBF complete: {len(result_df)} scores in {elapsed:.1f}s")
        return result_df

    def get_product_review_score(self, product_id: str) -> float:
        if product_id in self.product_profiles:
            return self.product_profiles[product_id]["review_score"]
        return 0.0

    def fit(self, df: pd.DataFrame = None) -> "ContentBasedModel":
        logger.info("=" * 60)
        logger.info("TRAINING CONTENT-BASED FILTERING (SPACY PRE-TRAINED)")
        logger.info("=" * 60)

        if df is None:
            df = pd.read_parquet(str(TRAIN_PARQUET))
            logger.info(f"Loaded training data: {len(df)} rows")

        df = self.analyze_sentiment(df)
        self.build_product_profiles(df)
        self.build_user_profiles(df)

        self._is_fitted = True
        logger.info("=" * 60)
        logger.info("CONTENT-BASED FILTERING TRAINING COMPLETE")
        logger.info("=" * 60)

        return self

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "cbf"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        profiles_path = path / "product_profiles.pkl"
        with open(profiles_path, "wb") as f:
            pickle.dump(self.product_profiles, f)

        user_profiles_simple = {}
        for uid, profile in self.user_profiles.items():
            user_profiles_simple[uid] = {
                k: v for k, v in profile.items()
                if k != "embedding_vector" and k != "reviewed_products"
            }
            user_profiles_simple[uid]["reviewed_products"] = list(profile["reviewed_products"])

        user_path = path / "user_profiles.pkl"
        with open(user_path, "wb") as f:
            pickle.dump(user_profiles_simple, f)

        vectors_path = path / "embeddings.pkl"
        with open(vectors_path, "wb") as f:
            pickle.dump({
                "product_vectors": self.product_embeddings,
                "product_ids": self.product_ids_list,
                "user_vectors": {
                    uid: profile["embedding_vector"]
                    for uid, profile in self.user_profiles.items()
                },
            }, f)

        logger.info(f"CBF model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "cbf"
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
                uid, np.zeros(self.config.get("embedding_dim", 300))
            )

        self._is_fitted = True
        logger.info(f" CBF model loaded from {path}")
        logger.info(f"   Products: {len(self.product_profiles)}, Users: {len(self.user_profiles)}")
