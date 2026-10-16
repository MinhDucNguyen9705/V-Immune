"""Per-burst and per-window surprise from SpectrumLM.

Implements report §2.5 (surprise). For burst i: s_i = Σ_{a∈A′} w_a·[−log p(x_{i,a} | b_<i)] with A′ = all
fields except SNR and w_a = 1/log|V_a| (|V_a| = number of bins). A window's S is the mean of s_i over its
last ``score_len`` bursts; h is the mean-pooled last hidden state over the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vimmune.lm.data import Track, scoring_windows, to_batch, window_arrays
from vimmune.lm.model import SpectrumLM
from vimmune.tokenize.fields import FIELDS, MASK, PAD

SNR = FIELDS.index("snr")
WCODE = FIELDS.index("wcode")


def field_weights(n_bins: list[int], exclude: tuple[int, ...] = (SNR,)) -> np.ndarray:
    w = np.array([1.0 / np.log(n) for n in n_bins])
    w[list(exclude)] = 0.0
    return w


@torch.no_grad()
def field_nll(model: SpectrumLM, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L, 6) NLL per target field (0 where the target is PAD) and hidden states."""
    logits, h = model(b["inp"], b["t"], b["valid"], b["tgt"])
    nll = torch.stack([F.cross_entropy(lg.transpose(1, 2).float(), b["tgt"][..., a], ignore_index=PAD, reduction="none")
                       for a, lg in enumerate(logits)], -1)
    return nll, h


@dataclass
class WindowScores:
    key: list[str]
    label: list[str | None]
    end: np.ndarray  # index (exclusive) of the last scored burst in its track
    S: np.ndarray  # window surprise
    snr: np.ndarray  # mean SNR of the scored bursts
    h: np.ndarray  # pooled hidden state
    logits: np.ndarray  # classifier logits (may be empty)
    t_end: np.ndarray  # TOA of the last scored burst


@torch.no_grad()
def score_windows(model: SpectrumLM, tracks: list[Track], ctx: int, score_len: int, weights: np.ndarray,
                  stride: int | None = None, mask_fields: tuple[int, ...] = (), no_context: bool = False,
                  batch: int = 128) -> WindowScores:
    """Score every scoring window of every track (see :func:`scoring_windows`)."""
    model.eval()
    jobs = [(tr, s, e) for tr in tracks for s, e in scoring_windows(tr, ctx, score_len, stride)]
    w = torch.tensor(weights, dtype=torch.float32)
    S, snr, H, C, ends, tend = [], [], [], [], [], []
    for i in range(0, len(jobs), batch):
        chunk = jobs[i : i + batch]
        b = to_batch([window_arrays(tr, s, e, ctx) for tr, s, e in chunk])
        if mask_fields:
            for a in mask_fields:
                b["inp"][:, 1:, a] = torch.where(b["valid"][:, 1:], torch.full_like(b["inp"][:, 1:, a], MASK), b["inp"][:, 1:, a])
        if no_context:
            s_burst = _single_burst_surprise(model, b, w)
            h = model.hidden(b["inp"], b["t"], b["valid"])
        else:
            nll, h = field_nll(model, b)
            s_burst = (nll * w).sum(-1)
        for j, (tr, s, e) in enumerate(chunk):
            n = e - s
            S.append(float(s_burst[j, n - score_len : n].mean()))
            snr.append(float(tr.snr[e - score_len : e].mean()))
            ends.append(e)
            tend.append(float(tr.toa[e - 1]))
        H.append(model.pool(h, b["valid"]).numpy())
        if model.cls is not None:
            C.append(model.classify(h, b["valid"]).numpy())
    d = model.cfg.d_model
    return WindowScores([tr.key for tr, _, _ in jobs], [tr.label for tr, _, _ in jobs], np.array(ends, int), np.array(S),
                        np.array(snr), np.concatenate(H) if H else np.zeros((0, d)),
                        np.concatenate(C) if C else np.zeros((len(jobs), 0)), np.array(tend))


@torch.no_grad()
def _single_burst_surprise(model: SpectrumLM, b: dict[str, torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    """Ablation N = 1: every burst predicted from [BOS] only (no sequence context)."""
    B, L, _ = b["tgt"].shape
    tgt = b["tgt"].reshape(B * L, 1, 6)
    one = {"inp": torch.full((B * L, 1, 6), 2, dtype=torch.long), "tgt": tgt,
           "t": torch.zeros(B * L, 1, dtype=torch.float64), "valid": torch.ones(B * L, 1, dtype=torch.bool)}
    nll, _ = field_nll(model, one)
    return (nll[:, 0] * w).sum(-1).view(B, L)


@torch.no_grad()
def burst_surprise(model: SpectrumLM, tr: Track, ctx: int, weights: np.ndarray) -> np.ndarray:
    """Causal per-burst surprise s_n for a whole track (each burst sees ≥ ctx/2 bursts of history when available)."""
    half = max(1, ctx // 2)
    out = np.zeros(len(tr))
    jobs, spans = [], []
    for e in range(min(ctx, len(tr)), len(tr) + half, half):
        e = min(e, len(tr))
        s = max(0, e - ctx)
        lo = 0 if s == 0 else e - half  # positions of this window that are written to the output
        jobs.append(window_arrays(tr, s, e, ctx))
        spans.append((s, e, lo))
        if e == len(tr):
            break
    w = torch.tensor(weights, dtype=torch.float32)
    for i in range(0, len(jobs), 128):
        nll, _ = field_nll(model, to_batch(jobs[i : i + 128]))
        sb = (nll * w).sum(-1).numpy()
        for j, (s, e, lo) in enumerate(spans[i : i + 128]):
            out[max(lo, s) : e] = sb[j, max(lo, s) - s : e - s]
    return out
