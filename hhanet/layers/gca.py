# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCrossAlignment(nn.Module):
    """Graph-based Cross-level Alignment (GCA) module.

    Formalizes skip-connection fusion as directed bipartite graph matching,
    explicitly aligning semantic and spatial features via top-k neighborhood
    aggregation.
    """

    def __init__(self, dim: int = 768, dropout: float = 0.0):
        super().__init__()
        self.head_proj = nn.Linear(dim, dim)
        self.tail_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cls_tokens: torch.Tensor, feats: torch.Tensor, tok_ratio: float = 0.5) -> torch.Tensor:
        """cls_tokens as heads (B,H,D), feats as tails (B,T,D)."""
        B, H, D = cls_tokens.shape
        T = feats.shape[1]
        k = max(1, min(T, int(tok_ratio * max(1, H))))

        x = torch.cat([cls_tokens, feats.detach()], dim=1)
        e_h = self.head_proj(cls_tokens)  # (B,H,D)
        e_t = self.tail_proj(x)  # (B,H+T,D)

        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # (B,H,H+T)
        weight_topk, index_topk = torch.topk(attn_logit, k=k, dim=-1)
        index_topk = index_topk.to(torch.long)

        batch_idx = torch.arange(B, device=index_topk.device).view(B, 1, 1)
        Nb_h = e_t[batch_idx, index_topk, :]  # (B,H,k,D)

        topk_prob = F.softmax(weight_topk, dim=2)  # (B,H,k)
        eh_r = torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2)) + (topk_prob.unsqueeze(-1) * Nb_h)

        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('bhkd,bhkd->bhk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(2)  # (B,H,1,k)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(2)  # (B,H,D)

        sum_embedding = self.activation(self.linear1((e_h + e_Nh) * 0.1 + cls_tokens))
        bi_embedding = self.activation(self.linear2(e_h * e_Nh * 0.1 + cls_tokens))
        embedding = sum_embedding + bi_embedding

        return self.norm(self.dropout(embedding))


class PrototypePool(nn.Module):
    """Learnable query pooling: tokens (B,N,D) -> prototypes (B,K,D)."""

    def __init__(self, d_model: int = 256, num_proto: int = 32):
        super().__init__()
        self.num_proto = num_proto
        self.q_latent = nn.Parameter(torch.randn(num_proto, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        Q = self.q_proj(self.q_latent).unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B,K,N)
        w = attn.softmax(dim=-1)
        proto = w @ V  # (B,K,D)
        return proto


class GCAAlign(nn.Module):
    """Graph-based Cross-level Alignment (GCA).

    Pipeline:
        deep C_map -> prototypes (K)
        optional GCA: refine prototypes using shallow S_map tokens
        cross-attn injection: shallow tokens query prototypes
        fuse + residual back to S channels
    """

    def __init__(
        self,
        dim_S: int,
        dim_C: int,
        d_model: int = 256,
        num_proto: int = 32,
        dropout: float = 0.0,
        use_gca: bool = False,
    ):
        super().__init__()
        self.use_gca = use_gca
        self.proj_S = nn.Conv2d(dim_S, d_model, 1, bias=False)
        self.proj_C = nn.Conv2d(dim_C, d_model, 1, bias=False)
        self.pool = PrototypePool(d_model=d_model, num_proto=num_proto)

        # S <- Proto cross-attn
        self.qS = nn.Linear(d_model, d_model)
        self.kP = nn.Linear(d_model, d_model)
        self.vP = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

        self.fuse = nn.Sequential(
            nn.Conv2d(d_model * 2, dim_S, 1, bias=False),
            nn.BatchNorm2d(dim_S),
            nn.ReLU(inplace=True),
        )
        self.out_norm = nn.BatchNorm2d(dim_S)
        self.out_drop = nn.Dropout2d(dropout)

        if use_gca:
            self.gca = GraphCrossAlignment(dim=d_model, dropout=dropout)

    def forward(self, S_map: torch.Tensor, C_map: torch.Tensor, gca_topk_ratio: float = 0.5):
        B, CS, HS, WS = S_map.shape

        # 1) deep -> prototypes
        C_tokens = self.proj_C(C_map).flatten(2).transpose(1, 2).contiguous()  # (B,Nc,D)
        prototypes = self.pool(C_tokens)  # (B,K,D)

        # 2) optional refine prototypes with S tokens
        if self.use_gca:
            S_tokens_for_gca = self.proj_S(S_map).flatten(2).transpose(1, 2).contiguous()
            prototypes = self.gca(prototypes, S_tokens_for_gca, tok_ratio=gca_topk_ratio)

        # 3) cross-attn inject
        S_tokens = self.proj_S(S_map).flatten(2).transpose(1, 2).contiguous()  # (B,Ns,D)
        attn = (self.qS(S_tokens) @ self.kP(prototypes).transpose(1, 2)) * self.scale  # (B,Ns,K)
        attn = attn.softmax(dim=-1)
        injected = attn @ self.vP(prototypes)  # (B,Ns,D)
        injected_map = injected.transpose(1, 2).reshape(B, -1, HS, WS).contiguous()

        S_proj = self.proj_S(S_map)
        fused = self.fuse(torch.cat([S_proj, injected_map], dim=1))
        out = self.out_norm(self.out_drop(fused)) + S_map
        return out, prototypes
