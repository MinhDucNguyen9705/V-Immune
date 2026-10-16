"""Small shared DSP helpers (PSD bandwidth, frequency shift, rational resampling).

Used by report §2.1 (simulator / detector) and §2.7 (vaccine quality filter).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


def shift(iq: np.ndarray, f_hz: float, fs: float, t0: float = 0.0) -> np.ndarray:
    """Multiply by exp(j 2π f t) (positive f moves the spectrum up)."""
    n = np.arange(len(iq))
    return (iq * np.exp(2j * np.pi * f_hz * (n / fs + t0))).astype(np.complex64)


def resample(iq: np.ndarray, fs_in: float, fs_out: float, max_den: int = 64) -> np.ndarray:
    """Polyphase resample by a rational approximation of fs_out/fs_in."""
    if abs(fs_out / fs_in - 1) < 1e-6:
        return iq.astype(np.complex64)
    fr = Fraction(fs_out / fs_in).limit_denominator(max_den)
    return signal.resample_poly(iq, fr.numerator, fr.denominator).astype(np.complex64)


def lowpass(iq: np.ndarray, cutoff_hz: float, fs: float, numtaps: int = 129) -> np.ndarray:
    c = min(cutoff_hz / (fs / 2), 0.999)
    h = signal.firwin(numtaps, c)
    return signal.fftconvolve(iq, h, mode="same").astype(np.complex64)


def psd(iq: np.ndarray, fs: float, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD, frequency-sorted."""
    f, p = signal.welch(iq, fs, nperseg=min(nfft, len(iq)), return_onesided=False, detrend=False)
    o = np.argsort(f)
    return f[o], p[o]


def extent_at_db(p: np.ndarray, db: float = -10.0, smooth: int = 5, max_gap: int | None = None) -> tuple[int, int, int]:
    """Index range around the peak where ``p`` stays above ``peak + db`` dB.

    Dips below threshold shorter than ``max_gap`` bins (default 2 % of the axis, e.g. an OFDM DC null)
    do not end the extent.
    """
    ps = np.convolve(p, np.ones(smooth) / smooth, mode="same") if smooth > 1 else p
    k = int(np.argmax(ps))
    above = ps >= ps[k] * 10 ** (db / 10)
    gap = max(1, len(ps) // 50) if max_gap is None else max_gap
    lo = hi = k
    i, miss = k - 1, 0
    while i >= 0 and miss <= gap:
        miss = 0 if above[i] else miss + 1
        lo = i if above[i] else lo
        i -= 1
    i, miss = k + 1, 0
    while i < len(ps) and miss <= gap:
        miss = 0 if above[i] else miss + 1
        hi = i if above[i] else hi
        i += 1
    return lo, hi, k


def bandwidth(iq: np.ndarray, fs: float, db: float = -10.0, nfft: int = 1024) -> float:
    """-10 dB (by default) occupied bandwidth from the Welch PSD."""
    f, p = psd(iq, fs, nfft)
    lo, hi, _ = extent_at_db(p, db, smooth=5)
    return float(f[hi] - f[lo] + (f[1] - f[0]))
