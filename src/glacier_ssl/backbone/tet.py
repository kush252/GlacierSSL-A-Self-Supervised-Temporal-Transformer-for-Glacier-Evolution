"""Phase 7 — Temporal Evolution Transformer (Temptrans_aarchi.md).

Factorized temporal-then-spatial attention + MLP over regional tokens, plus a global
sequence context token whose communication path is resolved explicitly (decision ledger
`tet.context_mode`). Produces the three head-facing representations:

    regional_out   (B, T, 64, 768)
    timestamp_out  (B, T, 768)     masked mean over valid regions per timestamp
    sequence_out   (B, 768)        final context-token hidden state (NOT pooled)

Padding (invalid timestamps) is inert: padded timesteps are masked as temporal keys and
excluded from pooling and from the context-token aggregation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .layers import Attention, Mlp
from .temporal_encoding import make_encoding

_SEASONS = ["DJF", "MAM", "JJA", "SON"]
_SEASON_TO_ID = {s: i for i, s in enumerate(_SEASONS)}


def seasons_to_ids(seasons: list[list[str]], T: int, device) -> torch.Tensor:
    """(B, T) long tensor of season ids; unknown/pad -> 0 (masked downstream)."""
    B = len(seasons)
    out = torch.zeros(B, T, dtype=torch.long, device=device)
    for b, row in enumerate(seasons):
        for t, s in enumerate(row[:T]):
            out[b, t] = _SEASON_TO_ID.get(s, 0)
    return out


class CrossAttention(nn.Module):
    """Single-query-set cross-attention for the context-token path (context_mode=cross)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, kv, key_padding_mask=None):
        # query: (B, Q, D), kv: (B, M, D), key_padding_mask: (B, M) True=keep
        B, Q, D = query.shape
        M = kv.shape[1]
        q = self.q(query).reshape(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        kv2 = self.kv(kv).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv2[0], kv2[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale        # (B,heads,Q,M)
        if key_padding_mask is not None:
            attn = attn.masked_fill(~key_padding_mask.view(B, 1, 1, M), float("-inf"))
        attn = torch.nan_to_num(attn.softmax(dim=-1), nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(B, Q, D)
        return self.proj(self.drop(out))


class TETBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config.tet
        dim = int(c.embed_dim)
        heads = int(c.num_heads)
        mlp_hidden = int(dim * float(c.mlp_ratio))
        attn_drop = float(c.attn_dropout)
        mlp_drop = float(c.mlp_dropout)
        self.context_mode = str(c.context_mode)
        self.num_regions = int(c.num_regions)

        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = Attention(dim, heads, attn_drop, mlp_drop)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, heads, attn_drop, mlp_drop)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden, mlp_drop)

        if self.context_mode == "cross_attention":
            self.ctx_norm_q = nn.LayerNorm(dim)
            self.ctx_norm_kv = nn.LayerNorm(dim)
            self.ctx_attn = CrossAttention(dim, heads, attn_drop)

    def forward(self, regional, context, tmask):
        """regional: (B,T,R,D); context: (B,D); tmask: (B,T) True=valid timestamp."""
        B, T, R, D = regional.shape

        # ---- 1. temporal attention (per region over time) + context communication ----
        if self.context_mode == "broadcast_mean":
            # arrange (B,R,T,D); broadcast context as an extra time-step token at pos 0.
            xr = regional.permute(0, 2, 1, 3).contiguous()            # (B,R,T,D)
            ctx_tok = context.view(B, 1, 1, D).expand(B, R, 1, D)     # (B,R,1,D)
            seq = torch.cat([ctx_tok, xr], dim=2)                     # (B,R,T+1,D)
            seq = seq.reshape(B * R, T + 1, D)
            # key padding: context always valid + per-timestamp mask (shared over regions)
            kpm = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=regional.device),
                             tmask], dim=1)                           # (B,T+1)
            kpm = kpm.unsqueeze(1).expand(B, R, T + 1).reshape(B * R, T + 1)
            attn_out = self.temporal_attn(self.temporal_norm(seq), key_padding_mask=kpm)
            seq = seq + attn_out                                      # residual
            seq = seq.view(B, R, T + 1, D)
            regional = seq[:, :, 1:, :].permute(0, 2, 1, 3).contiguous()   # (B,T,R,D)
            # update context: masked mean over the R region-copies of the context slot.
            ctx_updated = seq[:, :, 0, :].mean(dim=1)                 # (B,D)
            context = ctx_updated
        elif self.context_mode == "cross_attention":
            # plain temporal self-attention per region over time.
            xr = regional.permute(0, 2, 1, 3).reshape(B * R, T, D)
            kpm = tmask.unsqueeze(1).expand(B, R, T).reshape(B * R, T)
            xr = xr + self.temporal_attn(self.temporal_norm(xr), key_padding_mask=kpm)
            regional = xr.view(B, R, T, D).permute(0, 2, 1, 3).contiguous()
            # context attends over all valid regional tokens (flattened over T and R).
            kv = regional.reshape(B, T * R, D)
            kv_mask = tmask.unsqueeze(-1).expand(B, T, R).reshape(B, T * R)
            q = self.ctx_norm_q(context).unsqueeze(1)                # (B,1,D)
            ctx_attn = self.ctx_attn(q, self.ctx_norm_kv(kv), key_padding_mask=kv_mask)
            context = context + ctx_attn.squeeze(1)
        else:
            raise ValueError(f"unknown context_mode '{self.context_mode}'")

        # ---- 2. spatial attention (per timestamp over 64 regions), context excluded ----
        xs = regional.reshape(B * T, R, D)
        xs = xs + self.spatial_attn(self.spatial_norm(xs))           # all 64 regions valid
        regional = xs.view(B, T, R, D)

        # ---- 3. MLP on regional tokens and on the context token ----
        regional = regional + self.mlp(self.mlp_norm(regional))
        context = context + self.mlp(self.mlp_norm(context))
        return regional, context


class TemporalEvolutionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config.tet
        self.dim = int(c.embed_dim)
        self.num_regions = int(c.num_regions)
        self.depth = int(c.depth)

        self.season_embed = nn.Embedding(len(_SEASONS), self.dim)
        self.time_encoding = make_encoding(str(c.time_encoding), self.dim)
        self.span_encoding = make_encoding(str(c.span_encoding), self.dim)
        self.sequence_token = nn.Parameter(torch.zeros(1, self.dim))
        nn.init.trunc_normal_(self.sequence_token, std=0.02)

        self.blocks = nn.ModuleList([TETBlock(config) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.dim)
        self.context_norm = nn.LayerNorm(self.dim)
        nn.init.trunc_normal_(self.season_embed.weight, std=0.02)

    def add_temporal_embeddings(self, regional, season_ids, elapsed_years):
        """regional (B,T,R,D) + season + elapsed-time, broadcast over the R regions."""
        season = self.season_embed(season_ids).unsqueeze(2)         # (B,T,1,D)
        elapsed = self.time_encoding(elapsed_years).unsqueeze(2)    # (B,T,1,D)
        return regional + season + elapsed

    def forward(self, regional, season_ids, elapsed_years, span_years, tmask,
                region_mask=None):
        """
        regional:      (B, T, 64, 768) from HPE
        season_ids:    (B, T) long
        elapsed_years: (B, T) float (from dates, measured from first obs in window)
        span_years:    (B,) float
        tmask:         (B, T) bool, True = valid timestamp
        region_mask:   (B, T, 64) bool optional, True = valid region
        """
        B, T, R, D = regional.shape
        x = self.add_temporal_embeddings(regional, season_ids, elapsed_years)
        # zero temporal embeddings on padded timesteps (kept inert, extra safety).
        x = x * tmask.view(B, T, 1, 1).float()

        context = self.sequence_token.expand(B, D) + self.span_encoding(span_years)

        for blk in self.blocks:
            x, context = blk(x, context, tmask)

        regional_out = self.norm(x)                                  # (B,T,64,768)
        sequence_out = self.context_norm(context)                    # (B,768)

        # timestamp_out: masked mean over valid regions per timestamp.
        if region_mask is None:
            region_mask = torch.ones(B, T, R, dtype=torch.bool, device=regional.device)
        rm = region_mask.unsqueeze(-1).float()                       # (B,T,R,1)
        denom = rm.sum(dim=2).clamp(min=1.0)                         # (B,T,1)
        timestamp_out = (regional_out * rm).sum(dim=2) / denom       # (B,T,768)

        return {"regional_out": regional_out, "timestamp_out": timestamp_out,
                "sequence_out": sequence_out}
