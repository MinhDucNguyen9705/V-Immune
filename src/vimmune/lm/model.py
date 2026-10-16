"""SpectrumLM: decoder-only Transformer over burst tokens with six next-burst field heads.

Implements report §2.4. Each burst is the sum of six field embeddings plus a Fourier time encoding of
cumulative time (no index positions). Pre-LN blocks, GELU, causal attention with separate Q/K/V
projections so LoRA (§2.7) can wrap Q and V.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vimmune.config import LMCfg


class TimeEncoding(nn.Module):
    """Fourier features of cumulative time t at 32 log-spaced periods (10 µs … 10 s), projected to d."""

    def __init__(self, d: int, n: int, period_range: tuple[float, float]):
        super().__init__()
        periods = np.logspace(np.log10(period_range[0]), np.log10(period_range[1]), n)
        self.register_buffer("omega", torch.tensor(2 * np.pi / periods, dtype=torch.float64))
        self.proj = nn.Linear(2 * n, d)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ph = (t.double()[..., None] * self.omega) % (2 * math.pi)  # float64 phase: µs periods over seconds
        return self.proj(torch.cat([ph.sin(), ph.cos()], -1).float())


class LoRALinear(nn.Module):
    """y = W x + (B A x)·(alpha/r); W frozen (Hu et al. 2022)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class Attention(nn.Module):
    export_mode = False  # explicit-mask attention for ONNX export (SDPA's is_causal is not exported faithfully)

    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = (m(x).view(B, L, self.h, self.dk).transpose(1, 2) for m in (self.q, self.k, self.v))
        # sequences are right-padded, so the causal mask alone keeps valid queries off padded keys
        if Attention.export_mode:
            a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            a = a.masked_fill(torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            y = a.softmax(-1) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), key_pad))
        return x + self.drop(self.mlp(self.ln2(x)))


class SpectrumLM(nn.Module):
    """Six independent next-burst softmax heads (or field-autoregressive heads if enabled)."""

    def __init__(self, vocab_sizes: list[int], cfg: LMCfg, n_classes: int = 0):
        super().__init__()
        d = cfg.d_model
        self.vocab_sizes = list(vocab_sizes)
        self.cfg = cfg
        self.embed = nn.ModuleList(nn.Embedding(v, d) for v in vocab_sizes)
        self.time = TimeEncoding(d, cfg.n_time_freqs, cfg.time_period_range_s)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, v) for v in vocab_sizes)
        self.ar = cfg.autoregressive_fields
        if self.ar:  # head a also sees the true fields < a of the target burst (teacher forcing)
            self.ar_embed = nn.ModuleList(nn.Embedding(v, d) for v in vocab_sizes[:-1])
        self.cls = nn.Linear(d, n_classes) if n_classes else None

    def hidden(self, tok: torch.Tensor, t: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        x = sum(e(tok[..., a]) for a, e in enumerate(self.embed)) + self.time(t)
        for b in self.blocks:
            x = b(x, valid)
        return self.ln_f(x)

    def field_logits(self, h: torch.Tensor, target: torch.Tensor | None = None) -> list[torch.Tensor]:
        if not self.ar:
            return [hd(h) for hd in self.heads]
        assert target is not None, "autoregressive fields need the target burst"
        out, ctx = [], h
        for a, hd in enumerate(self.heads):
            out.append(hd(ctx))
            if a < len(self.heads) - 1:
                ctx = ctx + self.ar_embed[a](target[..., a])
        return out

    def forward(self, tok: torch.Tensor, t: torch.Tensor, valid: torch.Tensor | None = None,
                target: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = self.hidden(tok, t, valid)
        return self.field_logits(h, target), h

    @staticmethod
    def pool(h: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is None:
            return h.mean(1)
        w = valid.float()[..., None]
        return (h * w).sum(1) / w.sum(1).clamp(min=1)

    def classify(self, h: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        assert self.cls is not None
        return self.cls(self.pool(h, valid))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
