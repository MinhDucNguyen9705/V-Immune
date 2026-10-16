"""Conditional waveform DDPM: 1-D U-Net on 2×1024 I/Q, FiLM conditioning on (centroid z, BW, PW).

Implements report §2.7 (waveform level). Cosine noise schedule with T = 1000 (Ho et al. 2020; Nichol &
Dhariwal 2021), ε-prediction, LoRA (rank 4) on every convolution for few-shot fine-tuning (Hu et al.
2022), DDIM sampling (Song et al. 2021).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vimmune.config import DDPMCfg


def cosine_alphas_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-8, 1.0)
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


def t_embed(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class LoRAConv1d(nn.Module):
    """Conv1d with a frozen base weight and a rank-r update ΔW = (B A).view(out, in, k)."""

    def __init__(self, base: nn.Conv1d, r: int):
        super().__init__()
        self.base = base
        ci, k = base.in_channels, base.kernel_size[0]
        self.A = nn.Parameter(torch.randn(r, ci * k) / math.sqrt(ci * k))
        self.B = nn.Parameter(torch.zeros(base.out_channels, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base.weight + (self.B @ self.A).view_as(self.base.weight)
        return F.conv1d(x, w, self.base.bias, self.base.stride, self.base.padding)


class ResBlock(nn.Module):
    def __init__(self, ci: int, co: int, cond: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(ci, co, 3, padding=1), nn.Conv1d(co, co, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(min(8, ci), ci), nn.GroupNorm(min(8, co), co)
        self.film = nn.Linear(cond, 2 * co)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(c).chunk(2, 1)
        h = self.n2(h) * (1 + g[..., None]) + b[..., None]
        return self.skip(x) + self.c2(F.silu(h))


class UNet1D(nn.Module):
    """Small 1-D U-Net; every ResBlock is FiLM-modulated by [t-embedding, cond(z, log BW, log PW)]."""

    def __init__(self, cfg: DDPMCfg, z_dim: int):
        super().__init__()
        ch, cd = cfg.channels, cfg.cond_dim
        self.cond = nn.Sequential(nn.Linear(z_dim + 2, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.temb = nn.Sequential(nn.Linear(cd, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.cd = cd
        self.inp = nn.Conv1d(2, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = ch[0]
        for co in ch:
            self.down.append(ResBlock(c, co, cd))
            self.pool.append(nn.Conv1d(co, co, 4, 2, 1))
            c = co
        self.mid = ResBlock(c, c, cd)
        self.up = nn.ModuleList()
        self.ups = nn.ModuleList()
        for co in reversed(ch):
            self.ups.append(nn.ConvTranspose1d(c, c, 4, 2, 1))
            self.up.append(ResBlock(c + co, co, cd))
            c = co
        self.out = nn.Conv1d(c, 2, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.temb(t_embed(t, self.cd)) + self.cond(cond)
        h = self.inp(x)
        skips = []
        for blk, pool in zip(self.down, self.pool):
            h = blk(h, c)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, c)
        for blk, up in zip(self.up, self.ups):
            h = up(h)
            h = blk(torch.cat([h, skips.pop()], 1), c)
        return self.out(h)


def add_lora(model: nn.Module, r: int) -> list[nn.Parameter]:
    """Freeze ``model`` and wrap every Conv1d with :class:`LoRAConv1d`. Returns the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for mod in list(model.modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Conv1d) and not isinstance(child, LoRAConv1d):
                lc = LoRAConv1d(child, r)
                setattr(mod, cn, lc)
                params += [lc.A, lc.B]
    return params


def make_cond(z: np.ndarray, bw: np.ndarray, pw: np.ndarray) -> torch.Tensor:
    """Conditioning vector [z, log10 BW − 6, log10 PW + 3] (roughly zero-centred)."""
    return torch.from_numpy(np.concatenate([z, (np.log10(bw) - 6)[:, None], (np.log10(pw) + 3)[:, None]], 1).astype(np.float32))


def train_ddpm(model: UNet1D, X: torch.Tensor, cond: torch.Tensor, steps: int, batch: int, lr: float, T: int,
               seed: int, params: list[nn.Parameter] | None = None, log=None) -> list[float]:
    """ε-prediction training (all params, or only ``params`` for LoRA fine-tuning)."""
    g = torch.Generator().manual_seed(seed)
    ab = cosine_alphas_bar(T)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr)
    model.train()
    hist = []
    for step in range(steps):
        i = torch.randint(0, len(X), (batch,), generator=g)
        x0, c = X[i], cond[i]
        t = torch.randint(0, T, (batch,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        a = ab[t][:, None, None]
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, c), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if log and (step + 1) % max(1, steps // 5) == 0:
            log(f"ddpm step {step + 1}/{steps} loss {np.mean(hist[-20:]):.4f}")
    model.eval()
    return hist


@torch.no_grad()
def ddim_sample(model: UNet1D, cond: torch.Tensor, n_steps: int, T: int, L: int, seed: int, batch: int = 250) -> torch.Tensor:
    """Deterministic DDIM (η = 0) sampling of ``len(cond)`` waveforms (N, 2, L)."""
    model.eval()
    ab = cosine_alphas_bar(T)
    ts = torch.linspace(T - 1, 0, n_steps).long()
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(cond), batch):
        c = cond[i : i + batch]
        x = torch.randn((len(c), 2, L), generator=g)
        for j, t in enumerate(ts):
            a = ab[t]
            a_prev = ab[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0)
            eps = model(x, t.repeat(len(c)), c)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-6, 6)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        out.append(x)
    return torch.cat(out)


def save_ddpm(model: UNet1D, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_ddpm(path: str | Path, cfg: DDPMCfg, z_dim: int) -> UNet1D:
    m = UNet1D(cfg, z_dim)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()
