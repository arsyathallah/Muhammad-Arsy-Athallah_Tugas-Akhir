
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.gram_model import GRAMModel
from src.config import GRAM_CONFIG, MODELS_DIR, TRAIN_PARQUET, TEST_PARQUET
from src.utils.logger import get_logger

logger = get_logger(__name__)

GRAM_MODEL_DIR = MODELS_DIR / "gram"

class GRAMTrainer:

    def __init__(self, config: dict = None):
        self.config = config or GRAM_CONFIG

        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"GRAM Trainer device: {self.device}")

        self.model: Optional[GRAMModel] = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.idx_to_item: List[str] = []
        self.item_plm_embeddings: Optional[np.ndarray] = None
        self.training_history: List[dict] = []

    def compute_item_embeddings(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        save_path = save_path or (GRAM_MODEL_DIR / "item_plm_embeddings.npy")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = save_path.parent / "item_id_map.pkl"

        if save_path.exists() and meta_path.exists():
            logger.info(f"Loading pre-computed PLM item embeddings from {save_path}")
            self.item_plm_embeddings = np.load(str(save_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.item_id_map = meta["item_id_map"]
            self.idx_to_item = meta["idx_to_item"]
            logger.info(f"Loaded {len(self.idx_to_item)} item embeddings")
            return self.item_plm_embeddings

        logger.info("Computing PLM item embeddings (this runs once)...")
        from sentence_transformers import SentenceTransformer

        plm_name = self.config.get("plm_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        encoder = SentenceTransformer(plm_name).to(self.device)
        max_len = 256

        item_texts_raw: Dict[str, List[str]] = {}
        for row in df.itertuples():
            pid = row.product_id
            text = str(row.review_text) if pd.notna(row.review_text) else ""
            if text.strip():
                item_texts_raw.setdefault(pid, []).append(text)

        self.idx_to_item = sorted(item_texts_raw.keys())
        self.item_id_map = {pid: i for i, pid in enumerate(self.idx_to_item)}

        texts_list = []
        for pid in self.idx_to_item:
            combined = " ".join(item_texts_raw[pid])[:max_len * 6]
            texts_list.append(combined)

        batch_size = self.config.get("batch_size", 512)
        logger.info(f"Encoding {len(texts_list)} items with PLM...")
        t0 = time.time()
        embeddings = encoder.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=True,
            device=str(self.device),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        self.item_plm_embeddings = embeddings.astype(np.float32)
        logger.info(f"Item PLM encoding done in {elapsed:.1f}s — shape: {self.item_plm_embeddings.shape}")

        np.save(str(save_path), self.item_plm_embeddings)
        with open(meta_path, "wb") as f:
            pickle.dump({"item_id_map": self.item_id_map, "idx_to_item": self.idx_to_item}, f)
        logger.info(f"Saved PLM item embeddings to {save_path}")

        return self.item_plm_embeddings

    def _build_user_map(self, df: pd.DataFrame):
        users = sorted(df["user_id"].unique())
        self.user_id_map = {uid: i for i, uid in enumerate(users)}

    def _build_interaction_matrix(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        user_items: Dict[int, List[int]] = {}
        for row in df.itertuples():
            uid = self.user_id_map.get(row.user_id)
            iid = self.item_id_map.get(row.product_id)
            if uid is not None and iid is not None:
                user_items.setdefault(uid, []).append(iid)
        return user_items

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        logger.info("=" * 60)
        logger.info("TRAINING GRAM-CF MODEL")
        logger.info("=" * 60)

        all_df = pd.concat([train_df, val_df], ignore_index=True) if val_df is not None else train_df
        self._build_user_map(all_df)
        self.compute_item_embeddings(all_df)

        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        plm_dim = self.config.get("plm_embed_dim", 384)
        latent_dim = self.config.get("latent_dim", 128)
        logger.info(f"Users: {n_users}, Items: {n_items}, PLM dim: {plm_dim}, Latent dim: {latent_dim}")

        self.model = GRAMModel(n_users, plm_dim, latent_dim).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        user_items_train = self._build_interaction_matrix(train_df)
        all_item_indices = list(range(n_items))
        n_negatives = self.config.get("n_negatives", 4)
        batch_size = self.config.get("batch_size", 512)
        epochs = self.config.get("epochs", 15)
        grad_clip = self.config.get("grad_clip", 1.0)

        item_embeds_tensor = torch.tensor(self.item_plm_embeddings, dtype=torch.float32).to(self.device)

        logger.info("Building training pairs...")
        pairs = []
        for u_idx, i_list in user_items_train.items():
            for i_idx in i_list:
                pairs.append((u_idx, i_idx))
        pairs = np.array(pairs, dtype=np.int64)
        logger.info(f"Total positive pairs: {len(pairs)}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            np.random.shuffle(pairs)
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for b_start in range(0, len(pairs), batch_size):
                batch = pairs[b_start:b_start + batch_size]
                if len(batch) == 0:
                    continue

                u_idx = torch.tensor(batch[:, 0], dtype=torch.long).to(self.device)
                i_pos_idx = batch[:, 1]

                i_neg_idx = np.random.randint(0, n_items, size=len(batch))

                pos_embeds = item_embeds_tensor[i_pos_idx]
                neg_embeds = item_embeds_tensor[i_neg_idx]

                optimizer.zero_grad()
                loss = self.model(u_idx, pos_embeds, neg_embeds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            logger.info(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f} — {elapsed:.1f}s")
            self.training_history.append({"epoch": epoch, "train_loss": avg_loss})

        logger.info("GRAM-CF training complete.")

    def recommend_for_all_users(self, n: int = 50, batch_size: int = 1024) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.model.eval()

        item_embeds_tensor = torch.tensor(
            self.item_plm_embeddings, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            all_item_proj = self.model.get_item_embedding(item_embeds_tensor)

        n_users = len(self.user_id_map)
        idx_to_user = {v: k for k, v in self.user_id_map.items()}

        all_uids, all_pids, all_scores = [], [], []

        logger.info(f"Generating top-{n} recommendations for {n_users} users...")
        t0 = time.time()

        for b_start in range(0, n_users, batch_size):
            b_end = min(b_start + batch_size, n_users)
            u_idx = torch.arange(b_start, b_end, dtype=torch.long).to(self.device)

            with torch.no_grad():
                u_embeds = self.model.get_user_embedding(u_idx)
                scores = u_embeds @ all_item_proj.T
                scores_np = scores.cpu().numpy()

            actual_top = min(n, scores_np.shape[1])
            top_idx = np.argpartition(scores_np, -actual_top, axis=1)[:, -actual_top:]

            for i in range(len(u_idx)):
                uid_str = idx_to_user[b_start + i]
                row_scores = scores_np[i]
                top_i = top_idx[i]
                top_i_sorted = top_i[np.argsort(row_scores[top_i])[::-1]]
                for j in top_i_sorted:
                    all_uids.append(uid_str)
                    all_pids.append(self.idx_to_item[j])
                    all_scores.append(float(row_scores[j]))

        elapsed = time.time() - t0
        logger.info(f"Recommendations generated in {elapsed:.1f}s")

        return pd.DataFrame({
            "user_id": all_uids,
            "product_id": all_pids,
            "cf_score": np.array(all_scores, dtype=np.float32),
        })

    def predict_ratings(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.eval()

        item_embeds_tensor = torch.tensor(
            self.item_plm_embeddings, dtype=torch.float32
        ).to(self.device)

        rows = []
        batch_size = 2048

        records = [
            (row.user_id, row.product_id, row.rating)
            for row in test_df.itertuples()
            if row.user_id in self.user_id_map and row.product_id in self.item_id_map
        ]

        for b_start in range(0, len(records), batch_size):
            batch = records[b_start:b_start + batch_size]
            u_idx = torch.tensor(
                [self.user_id_map[r[0]] for r in batch], dtype=torch.long
            ).to(self.device)
            i_idx = [self.item_id_map[r[1]] for r in batch]
            i_embeds = item_embeds_tensor[i_idx]

            with torch.no_grad():
                u_emb = self.model.get_user_embedding(u_idx)
                i_emb = self.model.get_item_embedding(i_embeds)
                scores = (u_emb * i_emb).sum(dim=-1).cpu().numpy()

            for k, (uid, pid, rating) in enumerate(batch):
                pred = 1.0 + 4.0 * torch.sigmoid(torch.tensor(scores[k])).item()
                rows.append({"user_id": uid, "product_id": pid,
                             "rating": rating, "prediction": pred})

        return pd.DataFrame(rows)

    def save(self, path: Optional[Path] = None):
        path = path or GRAM_MODEL_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "gram_model.pt")
        with open(path / "mappings.pkl", "wb") as f:
            pickle.dump({
                "user_id_map": self.user_id_map,
                "item_id_map": self.item_id_map,
                "idx_to_item": self.idx_to_item,
                "config": self.config,
                "training_history": self.training_history,
            }, f)
        logger.info(f"GRAM model saved to {path}")

    def load(self, path: Optional[Path] = None):
        path = path or GRAM_MODEL_DIR
        path = Path(path)

        with open(path / "mappings.pkl", "rb") as f:
            meta = pickle.load(f)
        self.user_id_map = meta["user_id_map"]
        self.item_id_map = meta["item_id_map"]
        self.idx_to_item = meta["idx_to_item"]
        self.config = meta.get("config", self.config)
        self.training_history = meta.get("training_history", [])

        emb_path = path / "item_plm_embeddings.npy"
        if emb_path.exists():
            self.item_plm_embeddings = np.load(str(emb_path))

        n_users = len(self.user_id_map)
        plm_dim = self.config.get("plm_embed_dim", 384)
        latent_dim = self.config.get("latent_dim", 128)
        self.model = GRAMModel(n_users, plm_dim, latent_dim).to(self.device)
        self.model.load_state_dict(torch.load(path / "gram_model.pt", map_location=self.device))
        self.model.eval()
        logger.info(f"GRAM model loaded from {path}")
