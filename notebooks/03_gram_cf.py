

import os, sys, json, pickle, time, glob
import numpy as np
import pandas as pd
from collections import defaultdict

OUT_DIR = "/kaggle/working"

matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
if not matches:
    raise FileNotFoundError("train.parquet tidak ditemukan. Pastikan dataset sudah di-attach ke notebook ini.")
DATA_DIR = os.path.dirname(matches[0])

print(f"DATA_DIR : {DATA_DIR}")
print(f"OUT_DIR  : {OUT_DIR}")

os.system("pip install sentence-transformers -q")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU — aktifkan GPU di Settings!'}")
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {total_mem:.1f} GB")

train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

print(f"Train : {len(train_df):,} | users: {train_df['user_id'].nunique():,} | items: {train_df['product_id'].nunique():,}")
print(f"Test  : {len(test_df):,}")

MAX_CHARS = 512
item_texts_raw = defaultdict(list)
for row in train_df.itertuples():
    text = str(row.review_text) if pd.notna(row.review_text) else ""
    if text.strip():
        item_texts_raw[row.product_id].append(text)

item_id_list = sorted(item_texts_raw.keys())
item_id_map  = {pid: i for i, pid in enumerate(item_id_list)}
item_texts   = [" ".join(item_texts_raw[pid])[:MAX_CHARS] for pid in item_id_list]
N_ITEMS      = len(item_id_list)

all_users   = pd.concat([train_df['user_id'], test_df['user_id']]).unique()
user_id_map = {uid: i for i, uid in enumerate(sorted(all_users))}
N_USERS     = len(user_id_map)

print(f"\nItems with text : {N_ITEMS:,}")
print(f"Total users     : {N_USERS:,}")

pairs = np.array([
    (user_id_map[row.user_id], item_id_map[row.product_id])
    for row in train_df.itertuples()
    if row.user_id in user_id_map and row.product_id in item_id_map
], dtype=np.int64)
print(f"Training pairs  : {len(pairs):,}")

with open(f"{OUT_DIR}/item_id_list.pkl", "wb") as f:
    pickle.dump(item_id_list, f)

PLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PLM_DIM   = 384

print(f"Loading PLM: {PLM_MODEL}")
sentence_encoder = SentenceTransformer(PLM_MODEL)
sentence_encoder = sentence_encoder.to(DEVICE)

plm_encoder = sentence_encoder[0].auto_model
pooling     = sentence_encoder[1]
tokenizer   = sentence_encoder.tokenizer

print(f"PLM loaded: {sum(p.numel() for p in plm_encoder.parameters()):,} params")

class GRAMModel(nn.Module):
    def __init__(self, n_users: int, plm_dim: int, latent_dim: int):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, latent_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        self.item_proj = nn.Sequential(
            nn.Linear(plm_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        nn.init.xavier_uniform_(self.item_proj[0].weight)
        nn.init.zeros_(self.item_proj[0].bias)

    def forward(self, user_ids, emb_pos, emb_neg):
        u     = self.user_emb(user_ids)
        i_pos = self.item_proj(emb_pos)
        i_neg = self.item_proj(emb_neg)
        loss  = -F.logsigmoid((u * i_pos).sum(-1) - (u * i_neg).sum(-1)).mean()
        return loss

    def score(self, user_ids, all_item_proj):
        return self.user_emb(user_ids) @ all_item_proj.T

    def project_items(self, item_embeds):
        return self.item_proj(item_embeds)

def encode_texts_batch(texts, tokenizer, plm_encoder, pooling, device, max_length=128):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.set_grad_enabled(plm_encoder.training):
        output     = plm_encoder(**encoded)
        token_embs = output.last_hidden_state
        mask       = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(embeddings, p=2, dim=-1)

LATENT_DIM    = 128
BATCH_SIZE_P1 = 512
EPOCHS_P1     = 20
LR_P1         = 1e-3
GRAD_CLIP     = 1.0

print("Pre-computing item embeddings (PLM frozen)...")
t0 = time.time()
plm_encoder.eval()
for p in plm_encoder.parameters():
    p.requires_grad_(False)

all_item_embs = []
ENC_BATCH     = 256
with torch.no_grad():
    for b in range(0, len(item_texts), ENC_BATCH):
        batch_texts = item_texts[b:b + ENC_BATCH]
        embs = encode_texts_batch(batch_texts, tokenizer, plm_encoder, pooling, DEVICE)
        all_item_embs.append(embs.cpu())
        if b % 5000 == 0 and b > 0:
            print(f"  {b}/{len(item_texts)} encoded...")

item_tensor_p1 = torch.cat(all_item_embs, dim=0).to(DEVICE)
print(f"Pre-computed {item_tensor_p1.shape[0]:,} item embeddings in {time.time()-t0:.1f}s")
print(f"VRAM after encoding: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model = GRAMModel(N_USERS, PLM_DIM, LATENT_DIM).to(DEVICE)
optimizer_p1  = optim.Adam(model.parameters(), lr=LR_P1, weight_decay=1e-5)
scheduler_p1  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_p1, mode='min', factor=0.5, patience=3
)

print(f"\n{'='*55}")
print(f"PHASE 1: PLM FROZEN — Train GRAMModel ({EPOCHS_P1} epochs)")
print(f"{'='*55}")

history   = []
best_loss = float('inf')
patience_count = 0
EARLY_STOP_PATIENCE = 6

for epoch in range(1, EPOCHS_P1 + 1):
    model.train()
    np.random.shuffle(pairs)
    total_loss, n_batch = 0.0, 0
    t0 = time.time()

    for b_start in range(0, len(pairs), BATCH_SIZE_P1):
        batch  = pairs[b_start:b_start + BATCH_SIZE_P1]
        u_idx  = torch.tensor(batch[:, 0], dtype=torch.long).to(DEVICE)
        i_pos  = item_tensor_p1[batch[:, 1]]
        i_neg  = item_tensor_p1[np.random.randint(0, N_ITEMS, len(batch))]

        optimizer_p1.zero_grad()
        loss = model(u_idx, i_pos, i_neg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer_p1.step()

        total_loss += loss.item()
        n_batch    += 1

    avg = total_loss / max(n_batch, 1)
    history.append({"phase": 1, "epoch": epoch, "loss": round(avg, 4)})
    scheduler_p1.step(avg)
    print(f"  P1 Epoch {epoch:2d}/{EPOCHS_P1}  loss={avg:.4f}  lr={optimizer_p1.param_groups[0]['lr']:.2e}  ({time.time()-t0:.1f}s)")

    if avg < best_loss - 1e-4:
        best_loss = avg
        patience_count = 0
        torch.save(model.state_dict(), f"{OUT_DIR}/gram_best.pt")
    else:
        patience_count += 1
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"  Early stop di epoch {epoch} (loss tidak turun {EARLY_STOP_PATIENCE} epoch)")
            break

model.load_state_dict(torch.load(f"{OUT_DIR}/gram_best.pt"))
print("Phase 1 complete   (loaded best checkpoint)")

np.save(f"{OUT_DIR}/item_plm_embeddings.npy", item_tensor_p1.cpu().numpy())
print("Saved → item_plm_embeddings.npy")

TOP_N       = 50
BATCH_SCORE = 1024

model.eval()
with torch.no_grad():
    all_item_proj = model.project_items(item_tensor_p1)

idx_to_user = {v: k for k, v in user_id_map.items()}

print(f"\nGenerating top-{TOP_N} recommendations untuk {N_USERS:,} users...")
t0 = time.time()

all_uids, all_pids, all_scores_list = [], [], []
for b_start in range(0, N_USERS, BATCH_SCORE):
    b_end = min(b_start + BATCH_SCORE, N_USERS)
    u_idx = torch.arange(b_start, b_end, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        scores = model.score(u_idx, all_item_proj).cpu().numpy()

    actual_top = min(TOP_N, N_ITEMS)
    for i in range(len(u_idx)):
        uid_str = idx_to_user[b_start + i]
        row     = scores[i]
        top_idx = np.argpartition(row, -actual_top)[-actual_top:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        for j in top_idx:
            all_uids.append(uid_str)
            all_pids.append(item_id_list[j])
            all_scores_list.append(float(row[j]))

gram_scores = pd.DataFrame({
    "user_id"   : all_uids,
    "product_id": all_pids,
    "gram_score": np.array(all_scores_list, dtype=np.float32),
})
print(f"Done in {time.time()-t0:.1f}s — {len(gram_scores):,} rows")
gram_scores.to_parquet(f"{OUT_DIR}/gram_scores.parquet", index=False)
print(f"Saved → gram_scores.parquet")

valid_test = test_df[
    test_df['user_id'].isin(user_id_map) & test_df['product_id'].isin(item_id_map)
].copy()

records    = [(r.user_id, r.product_id, r.rating) for r in valid_test.itertuples()]
BATCH_PRED = 2048
preds_rows = []
model.eval()

for b_start in range(0, len(records), BATCH_PRED):
    batch  = records[b_start:b_start + BATCH_PRED]
    u_idx  = torch.tensor([user_id_map[r[0]] for r in batch], dtype=torch.long).to(DEVICE)
    i_embs = item_tensor_p1[[item_id_map[r[1]] for r in batch]]
    with torch.no_grad():
        u_emb  = model.user_emb(u_idx)
        i_proj = model.project_items(i_embs)
        scores = (u_emb * i_proj).sum(-1).cpu().numpy()
    pred_r = 1.0 + 4.0 * torch.sigmoid(torch.tensor(scores)).numpy()
    for k, (uid, pid, r) in enumerate(batch):
        preds_rows.append({"user_id": uid, "product_id": pid,
                           "rating": r, "prediction": float(pred_r[k])})

preds_df = pd.DataFrame(preds_rows)
rmse = float(np.sqrt(((preds_df['rating'] - preds_df['prediction'])**2).mean()))
mae  = float((preds_df['rating'] - preds_df['prediction']).abs().mean())
print(f"GRAM RMSE : {rmse:.4f}")
print(f"GRAM MAE  : {mae:.4f}")
preds_df.to_parquet(f"{OUT_DIR}/gram_predictions.parquet", index=False)

def compute_ranking_metrics(rec_df, test_df, score_col, k_list, threshold=3.0):
    user_relevant = {}
    for row in test_df.itertuples():
        if row.rating >= threshold:
            user_relevant.setdefault(row.user_id, set()).add(row.product_id)
    recs_by_user = (
        rec_df.sort_values(score_col, ascending=False)
        .groupby("user_id")["product_id"].apply(list).to_dict()
    )
    results = {}
    for k in k_list:
        ndcg_l, prec_l, rec_l, hit_l = [], [], [], []
        for uid, relevant in user_relevant.items():
            if uid not in recs_by_user:
                continue
            top  = recs_by_user[uid][:k]
            hits = sum(1 for p in top if p in relevant)
            dcg  = sum(1/np.log2(i+2) for i, p in enumerate(top) if p in relevant)
            idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
            ndcg_l.append(dcg/idcg if idcg > 0 else 0.0)
            prec_l.append(hits/k)
            rec_l.append(hits/len(relevant) if relevant else 0.0)
            hit_l.append(1 if hits > 0 else 0)
        results[k] = {
            f"NDCG@{k}"     : round(float(np.mean(ndcg_l)), 4) if ndcg_l else 0.0,
            f"Precision@{k}": round(float(np.mean(prec_l)), 4) if prec_l else 0.0,
            f"Recall@{k}"   : round(float(np.mean(rec_l)),  4) if rec_l  else 0.0,
            f"HitRate@{k}"  : round(float(np.mean(hit_l)),  4) if hit_l  else 0.0,
            "users_evaluated": len(ndcg_l),
        }
    return results

K_LIST  = [5, 10, 20]
metrics = compute_ranking_metrics(gram_scores, test_df, "gram_score", K_LIST)

print("\n" + "="*55)
print("GRAM-CF (Full PLM Fine-tuning) — Ranking Metrics")
print("="*55)
for k, m in metrics.items():
    print(f"\n  K={k}:")
    for mk, mv in m.items():
        print(f"    {mk:<22} {mv}")

metrics["RMSE"]             = round(rmse, 4)
metrics["MAE"]              = round(mae,  4)
metrics["model"]            = "GRAM-CF (PLM frozen, Phase 1)"
metrics["training_history"] = history

with open(f"{OUT_DIR}/gram_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved → {OUT_DIR}/gram_metrics.json")
print("\n Notebook 3 (GRAM-CF PLM Frozen) selesai!")
print("   Output: gram_scores.parquet, gram_predictions.parquet,")
print("           gram_metrics.json, item_plm_embeddings.npy, item_id_list.pkl")
print("   → Upload output sebagai Kaggle dataset untuk Notebook 4")
