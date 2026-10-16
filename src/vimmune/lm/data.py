"""Track/sequence datasets for SpectrumLM: tokenized tracks, session splits, random and scoring windows.

Implements report §2.4 (data side). Splits are made by ``session_id`` only, never by burst.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from vimmune.tokenize.fields import BOS, MASK, PAD, Tokenizer


@dataclass
class Track:
    key: str
    tokens: np.ndarray  # (N, 6) int64
    toa: np.ndarray  # (N,) seconds
    snr: np.ndarray  # (N,) dB
    label: str | None
    session: str
    rows: np.ndarray  # positional row indices into the source BDW frame

    def __len__(self) -> int:
        return len(self.toa)

    def slice(self, s: int, e: int) -> Track:
        return Track(self.key, self.tokens[s:e], self.toa[s:e], self.snr[s:e], self.label, self.session, self.rows[s:e])


def build_tracks(df: pd.DataFrame, codes: np.ndarray, tok: Tokenizer, by: str = "source_id",
                 min_len: int = 2, label_col: str = "protocol") -> list[Track]:
    """One track per (session, ``by``) group, bursts sorted by TOA. ``codes`` is row-aligned with ``df``."""
    df = df.reset_index(drop=True)
    out = []
    for (sess, key), g in df.groupby(["session_id", by], sort=True):
        if (isinstance(key, (int, np.integer)) and key < 0) or len(g) < min_len:
            continue
        g = g.sort_values("toa_s", kind="stable")
        rows = g.index.to_numpy()
        lab = g[label_col].mode() if label_col in g and g[label_col].notna().any() else None
        out.append(Track(f"{sess}/{key}", tok.encode_track(g, codes[rows]), g.toa_s.to_numpy(), g.snr_db.to_numpy(),
                         None if lab is None or len(lab) == 0 else str(lab.iloc[0]), str(sess), rows))
    return out


def split_sessions(sessions: list[str], fracs: tuple[float, ...], seed: int, strata: list[str] | None = None) -> dict[str, str]:
    """Assign each session to train/val/cal/test (stratified by ``strata`` if given)."""
    names = ["train", "val", "cal", "test"][: len(fracs)]
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for i, s in enumerate(sessions):
        groups.setdefault(strata[i] if strata else "", []).append(s)
    for g in groups.values():
        g = sorted(set(g))
        rng.shuffle(g)
        cut = np.round(np.cumsum(fracs) / sum(fracs) * len(g)).astype(int)
        for i, s in enumerate(g):
            out[s] = names[int(np.searchsorted(cut, i, side="right"))]
    return out


def window_arrays(tr: Track, s: int, e: int, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Input/target/time/valid arrays (length L) for predicting bursts [s, e) of ``tr``.

    Input position 0 is [BOS] (time 0); position j>0 holds burst s+j-1. Times are seconds since burst s.
    """
    n = e - s
    inp = np.full((L, 6), PAD, np.int64)
    tgt = np.full((L, 6), PAD, np.int64)
    t = np.zeros(L, np.float64)
    valid = np.zeros(L, bool)
    inp[0] = BOS
    inp[1:n] = tr.tokens[s : e - 1]
    tgt[:n] = tr.tokens[s:e]
    t[1:n] = tr.toa[s : e - 1] - tr.toa[s]
    valid[:n] = True
    return inp, tgt, t, valid


def to_batch(items: list[tuple[np.ndarray, ...]]) -> dict[str, torch.Tensor]:
    inp, tgt, t, valid = (np.stack(x) for x in zip(*items))
    return {"inp": torch.from_numpy(inp), "tgt": torch.from_numpy(tgt), "t": torch.from_numpy(t), "valid": torch.from_numpy(valid)}


def sample_batch(tracks: list[Track], B: int, L: int, rng: np.random.Generator, mask_prob: float = 0.0,
                 classes: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Random training windows (tracks weighted by length) with [MASK] field augmentation."""
    w = np.array([len(tr) for tr in tracks], float)
    idx = rng.choice(len(tracks), B, p=w / w.sum())
    items, labels = [], []
    for i in idx:
        tr = tracks[i]
        s = int(rng.integers(0, max(1, len(tr) - L + 1)))
        items.append(window_arrays(tr, s, min(len(tr), s + L), L))
        if classes is not None:
            labels.append(classes.index(tr.label) if tr.label in classes else -100)
    b = to_batch(items)
    if mask_prob > 0:
        m = (torch.rand(b["inp"].shape, generator=torch.Generator().manual_seed(int(rng.integers(1 << 31)))) < mask_prob)
        m[:, 0] = False
        m &= b["valid"][..., None]
        b["inp"] = torch.where(m, torch.full_like(b["inp"], MASK), b["inp"])
    if classes is not None:
        b["label"] = torch.tensor(labels)
    return b


def scoring_windows(tr: Track, ctx: int, score_len: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Windows (s, e) whose last ``score_len`` bursts are scored with up to ``ctx`` bursts of context."""
    stride = stride or score_len
    out = []
    for e in range(score_len, len(tr) + 1, stride):
        out.append((max(0, e - ctx), e))
    return out
