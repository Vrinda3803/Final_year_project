"""
Edge-Aware Graph Attention Network (E-GAT) and Multimodal Fusion Network for Autism Behaviour Recognition.
Implemented in standard PyTorch (no external C++ graph compilation dependencies required).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    """
    Edge-Aware Graph Attention Layer.
    Computes message passing where attention coefficients alpha_ij are directly
    modulated by continuous edge attributes (e.g. temporal proximity w_ij = 1 / (1 + delta_t)).
    """
    def __init__(self, in_features, out_features, edge_dim=1, dropout=0.3, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout

        self.W_node = nn.Linear(in_features, out_features, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_features, bias=False)

        self.a_src = nn.Parameter(torch.empty(size=(1, out_features)))
        self.a_dst = nn.Parameter(torch.empty(size=(1, out_features)))
        self.a_edge = nn.Parameter(torch.empty(size=(1, out_features)))
        self.leaky_relu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_node.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)
        nn.init.xavier_uniform_(self.a_edge, gain=gain)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, in_features]
        edge_index: [2, E] (long tensor of source and destination indices)
        edge_attr: [E, edge_dim] (float tensor of edge attributes/weights)
        """
        N = x.size(0)
        E = edge_index.size(1)

        h = self.W_node(x) # [N, out_features]

        if E == 0:
            # Graph has no edges: fallback to node self-transformation
            return F.dropout(F.relu(h), p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        e_proj = self.W_edge(edge_attr) # [E, out_features]

        # Compute unnormalized attention logits
        # attn_raw = (h_src * a_src).sum(-1) + (h_dst * a_dst).sum(-1) + (e_proj * a_edge).sum(-1)
        score_src = (h[src] * self.a_src).sum(dim=-1) # [E]
        score_dst = (h[dst] * self.a_dst).sum(dim=-1) # [E]
        score_edge = (e_proj * self.a_edge).sum(dim=-1) # [E]

        attn_logits = self.leaky_relu(score_src + score_dst + score_edge) # [E]

        # Numerically stable destination-wise Softmax over incoming edges
        # Find max per destination
        max_score = torch.full((N,), -1e9, device=x.device, dtype=x.dtype)
        max_score.scatter_reduce_(0, dst, attn_logits, reduce="amax", include_self=False)
        max_score = torch.where(max_score == -1e9, torch.zeros_like(max_score), max_score)

        exp_logits = torch.exp(attn_logits - max_score[dst])
        denom = torch.zeros(N, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst, exp_logits)
        denom = denom.clamp(min=1e-8)

        alpha = exp_logits / denom[dst]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # [E]

        # Aggregate messages: out_i = sum_{j in N(i)} alpha_ji * h_j
        weighted_msgs = h[src] * alpha.unsqueeze(-1) # [E, out_features]
        out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_features), weighted_msgs)

        return F.relu(out)


class EdgeAwareGAT(nn.Module):
    """
    2-Layer Edge-Aware GAT with Global Pooling for Video Graph Classification.
    """
    def __init__(self, in_dim=8, hidden_dim=64, n_classes=4, edge_dim=1, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeAwareGATLayer(in_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def extract_embedding(self, x, edge_index, edge_attr):
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        # Global mean pool across nodes
        graph_embed = h2.mean(dim=0, keepdim=True) # [1, hidden_dim]
        return graph_embed

    def forward(self, x, edge_index, edge_attr):
        embed = self.extract_embedding(x, edge_index, edge_attr)
        logits = self.classifier(embed)
        return logits


class MultimodalMEATGNN(nn.Module):
    """
    Multimodal Edge-Aware Spatio-Temporal Graph Neural Network (MEAT-GNN).
    Fuses the visual pose graph representation with acoustic descriptors via a learnable reliability gate.
    """
    def __init__(self, in_dim=8, audio_dim=32, hidden_dim=64, n_classes=4, edge_dim=1, dropout=0.3):
        super().__init__()
        self.pose_encoder = EdgeAwareGAT(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=n_classes, edge_dim=edge_dim, dropout=dropout)
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Reliability Gate: learnable attention between visual evidence and acoustic evidence
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x, edge_index, edge_attr, audio_feat):
        z_vis = self.pose_encoder.extract_embedding(x, edge_index, edge_attr) # [1, hidden]
        
        # Audio feature shape [1, audio_dim]
        if audio_feat.dim() == 1:
            audio_feat = audio_feat.unsqueeze(0)
        z_aud = self.audio_encoder(audio_feat) # [1, hidden]

        # Compute modality reliability gate
        concat = torch.cat([z_vis, z_aud], dim=-1)
        g = self.gate(concat) # [1, hidden]

        # Gated fusion
        z_fused = g * z_vis + (1.0 - g) * z_aud

        logits = self.classifier(z_fused)
        return logits, g
