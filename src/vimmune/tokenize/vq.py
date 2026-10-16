"""Waveform code: burst preprocessing + 1-D CNN VQ-VAE (EMA codebook) trained unlabeled on all bursts.

Implements report §2.2 (waveform code). The continuous pre-quantization ``z`` is kept for association
(§2.3); the code index is the 6th token field.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from vimmune.config import TokenizerCfg
from vimmune.dsp import lowpass, resample, shift
from vimmune.io.iq import BurstStore


# ------------------------------------------------------------------------------------------- preprocessing
def prep_burst(iq: np.ndarray, iq_fs: float, iq_fc: float, fc_est: float, bw_est: float, t: TokenizerCfg) -> np.ndarray:
    """Downconvert by FC, low-pass 1.2·BW, resample to occupancy 80 %, crop/pad to 1024, unit power."""
    x = shift(iq, -(fc_est - iq_fc), iq_fs)
    x = lowpass(x, t.lpf_factor * bw_est / 2, iq_fs, numtaps=65)
    x = resample(x, iq_fs, bw_est / t.occupancy)
    out = np.zeros(t.wave_len, np.complex64)
    out[: min(len(x), t.wave_len)] = x[: t.wave_len]
    p = np.mean(np.abs(out[: max(1, min(len(x), t.wave_len))]) ** 2)
    return out / np.sqrt(p + 1e-20)


def _prep_chunk(args: tuple) -> np.ndarray:
    root, rows, t = args
    store = BurstStore(root)
    out = np.zeros((len(rows), t.wave_len), np.complex64)
    for i, (ref, fs, ifc, fc, bw) in enumerate(rows):
        out[i] = prep_burst(store.get(ref), fs, ifc, fc, bw, t)
    return out


def prep_bursts(df: pd.DataFrame, store_root: str | Path, t: TokenizerCfg, workers: int = 8) -> np.ndarray:
    """Preprocess every burst of a BDW frame (parallel) -> (N, wave_len) complex64."""
    rows = list(zip(df.burst_iq_ref, df.iq_fs, df.iq_fc_hz, df.fc_hz, df.bw_hz))
    if not rows:
        return np.zeros((0, t.wave_len), np.complex64)
    n = max(1, len(rows) // (workers * 4))
    chunks = [(store_root, rows[i : i + n], t) for i in range(0, len(rows), n)]
    if workers <= 1:
        return np.concatenate([_prep_chunk(c) for c in chunks])
    with ProcessPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_prep_chunk, chunks)))


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """(N, L) complex -> (N, 2, L) float32."""
    return torch.from_numpy(np.stack([x.real, x.imag], 1).astype(np.float32))


# ------------------------------------------------------------------------------------------- model
def _block(ci: int, co: int, transpose: bool = False) -> nn.Sequential:
    conv = nn.ConvTranspose1d(ci, co, 4, 2, 1) if transpose else nn.Conv1d(ci, co, 7, 2, 3)
    return nn.Sequential(conv, nn.BatchNorm1d(co), nn.GELU())


class EMAQuantizer(nn.Module):
    """VQ codebook with EMA updates (van den Oord et al. 2017) and dead-code restarts."""

    def __init__(self, n: int, d: int, decay: float = 0.99, beta: float = 0.25):
        super().__init__()
        self.n, self.d, self.decay, self.beta = n, d, decay, beta
        self.register_buffer("embed", torch.randn(n, d) * 0.1)
        self.register_buffer("size", torch.ones(n))
        self.register_buffer("sum", self.embed.clone())
        self.register_buffer("usage", torch.full((n,), 1.0 / n))

    def codes(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cdist(z, self.embed).argmin(1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.codes(z)
        q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(idx, self.n).type_as(z)
                self.size.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                self.sum.mul_(self.decay).add_(oh.t() @ z, alpha=1 - self.decay)
                n = self.size.sum()
                size = (self.size + 1e-5) / (n + self.n * 1e-5) * n
                self.embed.copy_(self.sum / size[:, None])
                self.usage.mul_(0.9).add_(oh.mean(0), alpha=0.1)  # fast tracker for dead-code detection
                dead = self.usage < 0.1 / self.n  # restart rarely used codes at (jittered) batch encodings
                if dead.any():
                    r = z[torch.randint(0, len(z), (int(dead.sum()),), device=z.device)]
                    r = r + 0.01 * r.std() * torch.randn_like(r)
                    self.embed[dead] = r
                    self.sum[dead] = r * self.size[dead, None]
                    self.usage[dead] = 1.0 / self.n
        commit = self.beta * F.mse_loss(z, q.detach())
        return z + (q - z).detach(), idx, commit


class VQVAE(nn.Module):
    """1-D CNN encoder (4 strided conv-BN-GELU blocks) -> 64-d z -> VQ -> mirrored decoder."""

    def __init__(self, t: TokenizerCfg):
        super().__init__()
        c = t.channels
        self.L = t.wave_len
        self.enc = nn.Sequential(_block(2, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3]))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.to_z = nn.Linear(c[3] * 8, t.z_dim)
        self.vq = EMAQuantizer(t.codebook, t.z_dim, t.ema_decay, t.commitment)
        self.from_z = nn.Linear(t.z_dim, c[3] * (self.L // 16))
        self.dec = nn.Sequential(_block(c[3], c[2], True), _block(c[2], c[1], True), _block(c[1], c[0], True),
                                 _block(c[0], c[0], True), nn.Conv1d(c[0], 2, 7, 1, 3))
        self.c3 = c[3]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_z(self.pool(self.enc(x)).flatten(1))

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        return self.dec(self.from_z(q).view(len(q), self.c3, self.L // 16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        q, idx, commit = self.vq(z)
        return self.decode(q), z, idx, commit


def stft_l1(x: torch.Tensor, y: torch.Tensor, ffts: tuple[int, ...] = (64, 128, 256)) -> torch.Tensor:
    """Multi-resolution STFT-magnitude L1 between (N, 2, L) I/Q tensors."""
    xc, yc = torch.complex(x[:, 0], x[:, 1]), torch.complex(y[:, 0], y[:, 1])
    loss = x.new_zeros(())
    for n in ffts:
        w = torch.hann_window(n, device=x.device)
        X = torch.stft(xc, n, n // 4, window=w, return_complex=True, onesided=False).abs()
        Y = torch.stft(yc, n, n // 4, window=w, return_complex=True, onesided=False).abs()
        loss = loss + F.l1_loss(X, Y)
    return loss / len(ffts)


def train_vq(waves: np.ndarray, t: TokenizerCfg, seed: int = 0, steps: int | None = None, log=None) -> tuple[VQVAE, list[float]]:
    """Unlabeled VQ-VAE training on preprocessed bursts. Returns model and loss history."""
    torch.manual_seed(seed)
    model = VQVAE(t)
    opt = torch.optim.Adam(model.parameters(), lr=t.lr)
    X = to_tensor(waves)
    g = torch.Generator().manual_seed(seed)
    hist = []
    n_steps = steps or t.steps
    # k-means++-ish init: codebook from encodings of random bursts
    with torch.no_grad():
        model.eval()
        z0 = model.encode(X[torch.randint(0, len(X), (min(len(X), 4 * t.codebook),), generator=g)])
        model.vq.embed.copy_(z0[torch.randint(0, len(z0), (t.codebook,), generator=g)])
        model.vq.sum.copy_(model.vq.embed)
    model.train()
    for step in range(n_steps):
        xb = X[torch.randint(0, len(X), (t.batch,), generator=g)]
        rec, z, _, commit = model(xb)
        loss = stft_l1(rec, xb) + F.mse_loss(rec, xb) + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if log and step % max(1, n_steps // 10) == 0:
            log(f"vq step {step}/{n_steps} loss {loss.item():.4f}")
    model.eval()
    return model, hist


@torch.no_grad()
def encode_waves(model: VQVAE, waves: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(z (N, 64) float32, code (N,) int64) for preprocessed bursts."""
    model.eval()
    zs, cs = [], []
    for i in range(0, len(waves), batch):
        z = model.encode(to_tensor(waves[i : i + batch]))
        zs.append(z.numpy())
        cs.append(model.vq.codes(z).numpy())
    if not zs:
        return np.zeros((0, model.to_z.out_features), np.float32), np.zeros(0, np.int64)
    return np.concatenate(zs).astype(np.float32), np.concatenate(cs).astype(np.int64)


def codebook_usage(codes: np.ndarray, n: int) -> float:
    return len(np.unique(codes)) / n


def save_vq(model: VQVAE, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_vq(path: str | Path, t: TokenizerCfg) -> VQVAE:
    m = VQVAE(t)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def encode_df(df: pd.DataFrame, store_root: str | Path, model: VQVAE, t: TokenizerCfg, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess + encode every burst of a BDW frame -> (z, code)."""
    return encode_waves(model, prep_bursts(df, store_root, t, workers))
