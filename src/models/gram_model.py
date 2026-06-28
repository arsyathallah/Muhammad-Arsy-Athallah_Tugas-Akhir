
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class GRAMModel(nn.Module):

    def __init__(self, n_users: int, plm_embed_dim: int = 384, latent_dim: int = 128):
        super().__init__()
        self.n_users = n_users
        self.plm_embed_dim = plm_embed_dim
        self.latent_dim = latent_dim

        self.user_embedding = nn.Embedding(n_users, latent_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.item_proj = nn.Sequential(
            nn.Linear(plm_embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        nn.init.xavier_uniform_(self.item_proj[0].weight)
        nn.init.zeros_(self.item_proj[0].bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_embeds_pos: torch.Tensor,
        item_embeds_neg: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_embedding(user_ids)
        i_pos = self.item_proj(item_embeds_pos)
        i_neg = self.item_proj(item_embeds_neg)

        pos_score = (u * i_pos).sum(dim=-1)
        neg_score = (u * i_neg).sum(dim=-1)

        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    def score_users_items(
        self,
        user_ids: torch.Tensor,
        item_embeds: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_embedding(user_ids)
        items = self.item_proj(item_embeds)
        return u @ items.T

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_ids)

    def get_item_embedding(self, item_embeds: torch.Tensor) -> torch.Tensor:
        return self.item_proj(item_embeds)
