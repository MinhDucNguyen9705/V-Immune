"""Channel model: path loss -> SNR, 2–4 tap multipath, AWGN.

Implements report §2.1 (spec §4.3). CFO (±5 ppm of the RF center) is drawn per emitter in
:func:`vimmune.sim.families.schedule`. SNR is *in-band*: signal power over noise power inside the
signal bandwidth, with unit noise power over the full sample rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from vimmune.config import ChannelCfg


def snr_from_distance(d_m: float, cfg: ChannelCfg) -> float:
    """Log-distance path loss mapped to in-band SNR (dB)."""
    return cfg.snr_ref_db - 10 * cfg.pathloss_exp * np.log10(max(d_m, 1e-3) / cfg.d_ref_m)


def multipath_taps(rng: np.random.Generator, fs: float, cfg: ChannelCfg) -> np.ndarray:
    """Random 2–4 tap channel with exponentially decaying power, unit total energy."""
    n = int(rng.integers(cfg.taps_min, cfg.taps_max + 1))
    max_d = max(1, int(round(cfg.max_delay_s * fs)))
    delays = np.sort(np.r_[0, rng.choice(np.arange(1, max_d + 1), n - 1, replace=max_d < n - 1)])
    p = cfg.tap_decay ** np.arange(n)
    h = np.zeros(delays[-1] + 1, np.complex64)
    h[delays] += (np.sqrt(p / p.sum()) * np.exp(2j * np.pi * rng.uniform(size=n))).astype(np.complex64)
    return h


def apply_multipath(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return signal.lfilter(h, 1.0, x).astype(np.complex64) if len(h) > 1 else (x * h[0]).astype(np.complex64)


def awgn(n: int, rng: np.random.Generator, power: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)).astype(np.complex64)


def amplitude_for_snr(snr_db: float, bw_hz: float, fs: float) -> float:
    """Signal amplitude giving in-band SNR ``snr_db`` against unit-power full-band noise."""
    return float(np.sqrt(10 ** (snr_db / 10) * min(bw_hz / fs, 1.0)))


def band_gain_db(h: np.ndarray, fs: float, f_off: float, bw: float, n: int = 16) -> float:
    """Mean power gain (dB) of channel ``h`` over [f_off - bw/2, f_off + bw/2]."""
    f = f_off + np.linspace(-bw / 2, bw / 2, n)
    H = np.exp(-2j * np.pi * np.outer(f / fs, np.arange(len(h)))) @ h
    return float(10 * np.log10(np.mean(np.abs(H) ** 2) + 1e-12))
