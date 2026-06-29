"""
Model architecture for the MCI-to-AD longitudinal risk research demo.

This file was extracted from the audited web-demo / model-binding scripts.

Important:
- This file defines the model architecture only.
- It does not contain trained weights.
- Trained weights are stored separately in a PyTorch checkpoint (.pt).
- The default demo checkpoint is:
  results/web_demo_assets/research_demo_v1/models/default_low_overlap_demo_mmse_faq.pt

Model task:
- Input: longitudinal clinical visit token with shape [batch, visits, input_dim]
- Default token shape used by the web demo: [1, 8, 36]
- Output: four logits for 1-, 2-, 3-, and 5-year MCI-to-AD dementia conversion risk

Research use only. Not clinically validated.
"""

from __future__ import annotations

import torch
from torch import nn


class WebCompatibleDynamicTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,
        d_model: int = 64,
        max_visits: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        head_hidden_1: int = 128,
        head_hidden_2: int = 96,
        out_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.max_visits = max_visits

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_visits + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attn_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(head_hidden_1),
            nn.Linear(head_hidden_1, head_hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T,36], got {tuple(x.shape)}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        if x.shape[1] > self.max_visits:
            raise ValueError(f"Expected at most {self.max_visits} visits, got {x.shape[1]}")

        batch_size, n_visits, _ = x.shape

        h = self.input_proj(self.input_norm(x))

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embedding[:, : n_visits + 1, :]

        h = self.encoder(h)

        cls_h = h[:, 0, :]
        visit_h = h[:, 1:, :]

        attn_logits = self.attn_pool(visit_h).squeeze(-1)

        visit_present = x.abs().sum(dim=-1) > 0
        if visit_present.any():
            attn_logits = attn_logits.masked_fill(~visit_present, -1e9)

        attn_w = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_w * visit_h, dim=1)

        z = torch.cat([cls_h, pooled], dim=-1)
        logits = self.head(z)

        return logits
