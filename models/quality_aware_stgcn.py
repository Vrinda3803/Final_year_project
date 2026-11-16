"""Quality-aware spatiotemporal graph encoder for MMASD kinematic sequences."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class QualityAwareSTGCN(nn.Module):
    """Spatial graph message passing, temporal GRU, and masked-node decoder.

    Inputs use shape [batch, time, body_group, feature]. `quality` is a binary
    observed-node mask supplied by controlled masking/corruption. The local
    MMASD cache has no detector confidence values, so this model must be
    described as missing-feature reliability aware, not pose-confidence aware.
    """

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64,
                 gru_dim: int = 96, n_classes: int = 11,
                 dropout: float = 0.25, use_quality: bool = True) -> None:
        super().__init__()
        self.use_quality = use_quality
        self.input = nn.Linear(feature_dim + int(use_quality), hidden_dim)
        self.spatial1 = nn.Linear(hidden_dim, hidden_dim)
        self.spatial2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.GRU(hidden_dim * 2 + 1, gru_dim, batch_first=True,
                               bidirectional=True)
        self.temporal_score = nn.Linear(gru_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim * 2, gru_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(gru_dim, n_classes),
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, feature_dim))

    def encode_nodes(self, x: torch.Tensor, quality: torch.Tensor,
                     adjacency: torch.Tensor) -> torch.Tensor:
        q = quality if self.use_quality else torch.ones_like(quality)
        inp = torch.cat([x, q.unsqueeze(-1)], dim=-1) if self.use_quality else x
        h = F.gelu(self.input(inp))
        msg = torch.einsum("ij,btjh->btih", adjacency, h)
        h = self.norm1(h + self.dropout(F.gelu(self.spatial1(msg))))
        msg = torch.einsum("ij,btjh->btih", adjacency, h)
        h = self.norm2(h + self.dropout(F.gelu(self.spatial2(msg))))
        return h

    def forward(self, x: torch.Tensor, quality: torch.Tensor,
                valid_time: torch.Tensor, adjacency: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = quality if self.use_quality else torch.ones_like(quality)
        h = self.encode_nodes(x, q, adjacency)
        qf = q.unsqueeze(-1)
        denom = qf.sum(dim=2).clamp_min(1.0)
        pooled_mean = (h * qf).sum(dim=2) / denom
        masked = h.masked_fill(qf == 0, -1e4)
        pooled_max = masked.max(dim=2).values
        pooled_max = torch.where(qf.sum(dim=2) > 0, pooled_max,
                                 torch.zeros_like(pooled_max))
        coverage = q.mean(dim=2, keepdim=True)
        frame = torch.cat([pooled_mean, pooled_max, coverage], dim=-1)
        lengths = valid_time.sum(dim=1).clamp_min(1).to(device="cpu")
        packed = pack_padded_sequence(frame, lengths, batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.temporal(packed)
        temporal, _ = pad_packed_sequence(packed_out, batch_first=True,
                                          total_length=frame.shape[1])
        scores = self.temporal_score(temporal).squeeze(-1)
        scores = scores.masked_fill(~valid_time, -1e4)
        attn = torch.softmax(scores, dim=1)
        embedding = torch.sum(temporal * attn.unsqueeze(-1), dim=1)
        logits = self.classifier(embedding)
        reconstruction = self.decoder(h)
        return logits, reconstruction, embedding
