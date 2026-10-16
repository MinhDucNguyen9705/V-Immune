"""SpectrumLM pretraining (next-burst LM loss over all six fields), classifier fine-tune, class centroids.

Implements report §2.4. L = −Σ_i Σ_a log p(x_{i,a} | b_{<i}); AdamW, cosine schedule with warmup, grad
clipping, bf16 autocast on CUDA. The classifier head sits on the mean-pooled last hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vimmune.config import LMCfg
from vimmune.lm.data import Track, sample_batch, scoring_windows, to_batch, window_arrays
from vimmune.lm.model import SpectrumLM
from vimmune.tokenize.fields import PAD


def lm_loss(logits: list[torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
    """Sum over fields of mean token NLL (PAD targets ignored)."""
    loss = tgt.new_zeros((), dtype=torch.float32)
    for a, lg in enumerate(logits):
        loss = loss + F.cross_entropy(lg.flatten(0, 1).float(), tgt[..., a].flatten(), ignore_index=PAD)
    return loss


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def eval_lm(model: SpectrumLM, tracks: list[Track], L: int, device: str, max_windows: int = 256) -> float:
    model.eval()
    items = []
    for tr in tracks:
        for s in range(0, max(1, len(tr) - L + 1), L):
            items.append(window_arrays(tr, s, min(len(tr), s + L), L))
    items = items[:max_windows]
    tot, n = 0.0, 0
    for i in range(0, len(items), 64):
        b = {k: v.to(device) for k, v in to_batch(items[i : i + 64]).items()}
        logits, _ = model(b["inp"], b["t"], b["valid"], b["tgt"])
        tot += float(lm_loss(logits, b["tgt"])) * len(items[i : i + 64])
        n += len(items[i : i + 64])
    return tot / max(n, 1)


def pretrain(model: SpectrumLM, train: list[Track], val: list[Track], cfg: LMCfg, seed: int, device: str = "cpu",
             steps: int | None = None, log=None) -> dict[str, list]:
    """Unlabeled next-burst pretraining. Returns history {'step', 'loss', 'val_step', 'val_loss'}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = steps or cfg.steps
    hist: dict[str, list] = {"step": [], "loss": [], "val_step": [], "val_loss": []}
    for step in range(n):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, n, min(cfg.warmup, n // 5 + 1), cfg.lr)
        b = {k: v.to(device) for k, v in sample_batch(train, cfg.batch, cfg.ctx, rng, cfg.mask_prob).items()}
        with _autocast(device):
            logits, _ = model(b["inp"], b["t"], b["valid"], b["tgt"])
            loss = lm_loss(logits, b["tgt"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist["step"].append(step)
        hist["loss"].append(loss.item())
        if (step + 1) % max(1, n // 10) == 0 or step == n - 1:
            vl = eval_lm(model, val, cfg.ctx, device)
            model.train()
            hist["val_step"].append(step)
            hist["val_loss"].append(vl)
            if log:
                log(f"lm step {step + 1}/{n} loss {np.mean(hist['loss'][-50:]):.3f} val {vl:.3f}")
    model.eval()
    return hist


def finetune_classifier(model: SpectrumLM, train: list[Track], classes: list[str], cfg: LMCfg, seed: int,
                        device: str = "cpu", steps: int | None = None, log=None) -> list[float]:
    """Fine-tune with CE on known classes + the LM loss (kept so the surprise score stays calibrated)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.weight_decay)
    lab = [t for t in train if t.label in classes]
    hist = []
    n = steps or cfg.ft_steps
    for step in range(n):
        b = {k: v.to(device) for k, v in sample_batch(lab, cfg.batch, cfg.ctx, rng, cfg.mask_prob, classes).items()}
        with _autocast(device):
            logits, h = model(b["inp"], b["t"], b["valid"], b["tgt"])
            ce = F.cross_entropy(model.classify(h, b["valid"]).float(), b["label"])
            loss = ce + lm_loss(logits, b["tgt"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist.append(ce.item())
        if log and (step + 1) % max(1, n // 5) == 0:
            log(f"ft step {step + 1}/{n} ce {np.mean(hist[-20:]):.3f}")
    model.eval()
    return hist


@torch.no_grad()
def embed_windows(model: SpectrumLM, tracks: list[Track], ctx: int, score_len: int, device: str = "cpu",
                  stride: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """Pooled hidden states h and class logits for every scoring window."""
    model.eval()
    items, labels = [], []
    for tr in tracks:
        for s, e in scoring_windows(tr, ctx, score_len, stride):
            items.append(window_arrays(tr, s, e, ctx))
            labels.append(tr.label)
    H, C = [], []
    for i in range(0, len(items), 128):
        b = {k: v.to(device) for k, v in to_batch(items[i : i + 128]).items()}
        h = model.hidden(b["inp"], b["t"], b["valid"])
        H.append(model.pool(h, b["valid"]).cpu().numpy())
        if model.cls is not None:
            C.append(model.classify(h, b["valid"]).cpu().numpy())
    if not H:
        return np.zeros((0, model.cfg.d_model)), np.zeros((0, 0)), []
    return np.concatenate(H), (np.concatenate(C) if C else np.zeros((len(items), 0))), labels


def centroids(h: np.ndarray, labels: list[str | None], classes: list[str]) -> np.ndarray:
    """Per-class centroid μ_k = L2-normalized mean of L2-normalized h."""
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    lab = np.array(labels, dtype=object)
    mu = np.stack([hn[lab == c].mean(0) for c in classes])
    return mu / np.linalg.norm(mu, axis=1, keepdims=True)


@dataclass
class LMBundle:
    """Everything needed to rebuild a trained SpectrumLM (+ classifier and centroids)."""

    vocab_sizes: list[int]
    cfg: LMCfg
    classes: list[str]
    state: dict[str, Any]
    centroids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def build(self) -> SpectrumLM:
        m = SpectrumLM(self.vocab_sizes, self.cfg, len(self.classes))
        m.load_state_dict(self.state)
        return m.eval()

    def save(self, path: str | Path) -> None:
        torch.save({"vocab_sizes": self.vocab_sizes, "cfg": self.cfg.model_dump(), "classes": self.classes,
                    "state": self.state, "centroids": self.centroids, "meta": self.meta}, path)

    @staticmethod
    def load(path: str | Path) -> LMBundle:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return LMBundle(d["vocab_sizes"], LMCfg.model_validate(d["cfg"]), d["classes"], d["state"], d["centroids"], d["meta"])
