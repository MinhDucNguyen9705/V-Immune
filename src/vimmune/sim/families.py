"""Burst schedules of the fictional protocol families P1–P8.

Implements report §2.1 (spec §4.2). :func:`schedule` returns the ground-truth bursts (TOA, PW, FC, BW,
channel, source) of one emitter over a time span. P3 (adaptive) needs the already-scheduled background
bursts so it can avoid channels that are occupied at transmit time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vimmune.config import Family, ProtocolsCfg

COLS = ["toa_s", "pw_s", "fc_off_hz", "bw_hz", "channel", "source_id", "protocol", "link"]


def channel_centers(f: Family) -> np.ndarray:
    """Channel center offsets (Hz from band center)."""
    if f.channel_centers_hz is not None:
        return np.asarray(f.channel_centers_hz, float)
    n = f.n_channels
    return (np.arange(n) - (n - 1) / 2) * f.channel_spacing_hz


def hop_sequence(f: Family) -> np.ndarray:
    """P1: fixed deterministic hop sequence (family-level, length ``hop_seq_len``)."""
    return np.random.default_rng(f.hop_seed).permutation(f.n_channels)[: f.hop_seq_len or f.n_channels]


def lfsr16(seed: int, n: int) -> np.ndarray:
    """16-bit Fibonacci LFSR (taps 16,14,13,11) states; P2 pseudo-random hopping."""
    s = (seed & 0xFFFF) or 0xACE1
    out = np.empty(n, np.int64)
    for i in range(n):
        for _ in range(16):  # advance 16 bits per hop so successive channels decorrelate
            bit = ((s >> 0) ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1
            s = (s >> 1) | (bit << 15)
        out[i] = s
    return out


def mean_interval(f: Family) -> float:
    """Mean time between bursts of one source (for sizing sessions)."""
    if f.timing == "periodic":
        return float(f.frame_s)
    if f.timing == "pair":
        return float(f.frame_s)
    if f.timing == "csma":
        lo, hi = f.pw_range_s or (f.pw_s, f.pw_s)
        return float((lo + hi) / 2 + f.csma_difs_s + f.csma_mean_gap_s)
    return float((f.adv_interval_s + f.adv_rand_s / 2) / f.n_channels)


def _busy(bg: pd.DataFrame | None, t0: float, t1: float, centers: np.ndarray, bw: float) -> np.ndarray:
    """Boolean per channel: does any background burst overlap [t0, t1] x channel band?"""
    if bg is None or len(bg) == 0:
        return np.zeros(len(centers), bool)
    toa = bg["toa_s"].to_numpy()
    i0 = np.searchsorted(toa, t0 - 0.005)
    i1 = np.searchsorted(toa, t1)
    sel = bg.iloc[i0:i1]
    sel = sel[(sel["toa_s"] + sel["pw_s"]) > t0]
    if len(sel) == 0:
        return np.zeros(len(centers), bool)
    lo = (sel["fc_off_hz"] - sel["bw_hz"] / 2).to_numpy()[:, None]
    hi = (sel["fc_off_hz"] + sel["bw_hz"] / 2).to_numpy()[:, None]
    c = centers[None, :]
    return ((c + bw / 2 > lo) & (c - bw / 2 < hi)).any(0)


def schedule(f: Family, protos: ProtocolsCfg, t_start: float, t_stop: float, rng: np.random.Generator,
             source_id: str, center_freq: float, channel: int | None = None,
             background: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ground-truth burst list of one emitter of family ``f`` in [t_start, t_stop)."""
    centers = channel_centers(f)
    nch = len(centers)
    cfo = rng.uniform(-1, 1) * protos.cfo_ppm * 1e-6 * center_freq
    jit = protos.jitter_frac
    rows: dict[str, list] = {k: [] for k in COLS}

    def add(toa: np.ndarray, pw: np.ndarray, ch: np.ndarray, sid: str, link: str | None = None) -> None:
        rows["toa_s"].extend(toa)
        rows["pw_s"].extend(pw)
        rows["channel"].extend(ch)
        rows["fc_off_hz"].extend(centers[ch] + cfo)
        rows["bw_hz"].extend(np.full(len(toa), f.bw_hz))
        rows["source_id"].extend([sid] * len(toa))
        rows["protocol"].extend([f.id] * len(toa))
        rows["link"].extend([link] * len(toa))

    if f.timing in ("periodic", "pair"):
        T = float(f.frame_s)
        t0 = t_start + rng.uniform(0, T)
        k = np.arange(int(np.ceil((t_stop - t0) / T)) + 1)
        toa = t0 + k * T + rng.uniform(-jit, jit, len(k)) * T
        keep = (toa >= t_start) & (toa + f.pw_s < t_stop)
        toa, k = toa[keep], k[keep]
        pw = np.full(len(toa), f.pw_s)
        ch0 = int(rng.integers(nch)) if channel is None else channel
        if f.hop == "fixed_seq":
            seq = hop_sequence(f)
            ch = seq[(int(rng.integers(len(seq))) + k) % len(seq)]
        elif f.hop == "prng":
            ch = lfsr16(int(rng.integers(1, 2**16)), int(k.max()) + 1 if len(k) else 0)[k] % nch
        elif f.hop == "adaptive":
            ch = np.empty(len(toa), int)
            for i, t in enumerate(toa):
                free = np.flatnonzero(~_busy(background, t, t + f.pw_s, centers, f.bw_hz))
                ch[i] = rng.choice(free) if len(free) else rng.integers(nch)
        elif f.hop == "retune":
            ch = np.empty(len(toa), int)
            cur, t_next = ch0, t_start + rng.uniform(0.5, 1.5) * f.retune_s
            for i, t in enumerate(toa):
                if t >= t_next:
                    cur = int(rng.choice([c for c in range(nch) if c != cur]))
                    t_next = t + rng.uniform(0.8, 1.2) * f.retune_s
                ch[i] = cur
        else:  # fixed
            ch = np.full(len(toa), ch0)
        if f.timing == "pair":
            ch_dn = (ch0 + 1 + int(rng.integers(nch - 1))) % nch
            add(toa, pw, ch, f"{source_id}.up", "up")
            d = f.pair_delay_s * (1 + rng.uniform(-jit, jit, len(toa)))
            keep = toa + d + f.pw_s < t_stop
            add(toa[keep] + d[keep], pw[keep], np.full(keep.sum(), ch_dn), f"{source_id}.dn", "dn")
        else:
            add(toa, pw, ch, source_id)
    elif f.timing == "csma":
        lo, hi = f.pw_range_s or (f.pw_s, f.pw_s)
        n = int((t_stop - t_start) / mean_interval(f) * 1.5) + 10
        pw = rng.uniform(lo, hi, n)
        gaps = f.csma_difs_s + rng.exponential(f.csma_mean_gap_s, n)
        toa = t_start + rng.exponential(f.csma_mean_gap_s) + np.concatenate([[0], np.cumsum(pw + gaps)[:-1]])
        keep = toa + pw < t_stop
        ch = np.full(keep.sum(), int(rng.integers(nch)) if channel is None else channel)
        add(toa[keep], pw[keep], ch, source_id)
    else:  # advertising: one PDU on each advertising channel per event
        n = int((t_stop - t_start) / f.adv_interval_s) + 2
        ev = t_start + rng.uniform(0, f.adv_interval_s) + np.cumsum(f.adv_interval_s + rng.uniform(0, f.adv_rand_s, n)) - f.adv_interval_s
        toa = (ev[:, None] + np.arange(nch)[None, :] * f.adv_gap_s).ravel()
        ch = np.tile(np.arange(nch), n)
        keep = toa + f.pw_s < t_stop
        add(toa[keep], np.full(keep.sum(), f.pw_s), ch[keep], source_id)

    df = pd.DataFrame(rows)
    df["channel"] = df["channel"].astype(int)
    return df.sort_values("toa_s", kind="stable").reset_index(drop=True)
