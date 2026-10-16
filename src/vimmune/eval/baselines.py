"""Blob-level baselines: a single-burst CNN classifier (mimicking the existing closed-set classifier) with
MSP, energy, OpenMax and nearest-centroid unknown scores.

Implements report §2.9 (baselines): Hendrycks & Gimpel 2017 (MSP), Liu et al. 2020 (energy), Bendale &
Boult 2016 (OpenMax, Weibull fit on activation-vector distances), nearest class-mean distance. Per-burst
scores are averaged over the same 32-burst windows SpectrumLM scores, for a like-for-like comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import weibull_min
from torch import nn


def burst_features(waves: np.ndarray, bw: np.ndarray, pw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-magnitude STFT image of the preprocessed burst (1×64×61) + [log10 BW, log10 PW] (the blob's shape)."""
    x = torch.from_numpy(waves)
    S = torch.stft(x, 64, 16, window=torch.hann_window(64), return_complex=True, onesided=False).abs()
    img = torch.log1p(torch.fft.fftshift(S, dim=1))[:, None].float()
    sc = torch.from_numpy(np.stack([np.log10(bw) - 6, np.log10(pw) + 3], 1).astype(np.float32))
    return img, sc


class BlobCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 2, 64), nn.ReLU())
        self.out = nn.Linear(64, n_classes)

    def features(self, img: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.conv(img).flatten(1), sc], 1))

    def forward(self, img: torch.Tensor, sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(img, sc)
        return self.out(f), f


def train_cnn(img: torch.Tensor, sc: torch.Tensor, y: np.ndarray, n_classes: int, steps: int, seed: int, batch: int = 128) -> BlobCNN:
    torch.manual_seed(seed)
    m = BlobCNN(n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    yt = torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(yt), (batch,), generator=g)
        loss = F.cross_entropy(m(img[i], sc[i])[0], yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


@torch.no_grad()
def cnn_outputs(m: BlobCNN, img: torch.Tensor, sc: torch.Tensor, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    L, Fe = [], []
    for i in range(0, len(img), batch):
        lg, f = m(img[i : i + batch], sc[i : i + batch])
        L.append(lg.numpy())
        Fe.append(f.numpy())
    return np.concatenate(L), np.concatenate(Fe)


class OpenMax:
    """Per-class Weibull models of the distance between correct training activation vectors and the class MAV."""

    def __init__(self, tail: int = 20, alpha: int = 2):
        self.tail, self.alpha = tail, alpha

    def fit(self, logits: np.ndarray, y: np.ndarray) -> OpenMax:
        K = logits.shape[1]
        self.mav = np.zeros((K, K))
        self.wb: list[tuple[str, object]] = []
        for c in range(K):
            av = logits[(y == c) & (logits.argmax(1) == c)]
            self.mav[c] = av.mean(0)
            d = np.sort(np.linalg.norm(av - self.mav[c], axis=1))[-self.tail :]
            try:
                self.wb.append(("weibull", weibull_min.fit(d, floc=0)))
            except Exception:  # degenerate tail (e.g. near-identical distances): empirical tail CDF instead
                self.wb.append(("empirical", d))
        return self

    def _cdf(self, c: int, x: float) -> float:
        kind, p = self.wb[c]
        if kind == "weibull":
            return float(weibull_min.cdf(x, *p))
        return float(np.searchsorted(p, x, side="right") / len(p))

    def unknown_prob(self, logits: np.ndarray) -> np.ndarray:
        K = logits.shape[1]
        a = min(self.alpha, K)
        out = np.empty(len(logits))
        for n, av in enumerate(logits):
            rank = np.argsort(-av)
            w = np.ones(K)
            for i, c in enumerate(rank[:a]):
                cdf = self._cdf(c, float(np.linalg.norm(av - self.mav[c])))
                w[c] = 1 - (a - i) / a * cdf
            rev = av * w
            unk = np.sum(av * (1 - w))
            z = np.r_[rev, unk]
            p = np.exp(z - z.max())
            out[n] = p[-1] / p.sum()
        return out


def baseline_scores(logits: np.ndarray, feats: np.ndarray, centroids: np.ndarray, om: OpenMax) -> dict[str, np.ndarray]:
    """Per-burst unknown scores (higher = more unknown)."""
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return {"MSP (blob CNN)": -p.max(1),
            "Energy (blob CNN)": -np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) - logits.max(1),
            "OpenMax (blob CNN)": om.unknown_prob(logits),
            "Nearest centroid (blob CNN)": (1 - fn @ centroids.T).min(1)}
