"""Waveform primitives at any requested sample rate: GFSK, OFDM (QPSK), band-limited noise.

Implements report §2.1 (spec §4.1). All functions return unit-average-power complex64 baseband.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import erf


def gfsk(n: int, fs: float, bw_hz: float, rng: np.random.Generator, bt: float = 0.5, h: float = 0.5,
         bw_per_rate: float = 1.0) -> np.ndarray:
    """GFSK with Gaussian BT, modulation index h and random bits.

    Symbol rate R = bw_hz / bw_per_rate, so the -10 dB width of the PSD is ``bw_hz``. The instantaneous
    frequency is Δf·Σ a_k p(t-kT) with p = rect(T) * Gaussian(σ = sqrt(ln2)/(2πB)), Δf = hR/2.
    """
    R = bw_hz / bw_per_rate
    T = 1.0 / R
    t = np.arange(n) / fs
    nsym = int(np.ceil(t[-1] / T)) + 6 if n else 0
    a = rng.choice([-1.0, 1.0], nsym)
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt / T)
    k0 = np.floor(t / T).astype(int)
    freq = np.zeros(n)
    for d in range(-3, 4):
        k = k0 + d
        tc = t - (k * T + T / 2)
        p = 0.5 * (erf((tc + T / 2) / (np.sqrt(2) * sigma)) - erf((tc - T / 2) / (np.sqrt(2) * sigma)))
        freq += a[np.clip(k + 3, 0, nsym - 1)] * p
    phase = 2 * np.pi * np.cumsum(freq * h * R / 2) / fs + rng.uniform(0, 2 * np.pi)
    return np.exp(1j * phase).astype(np.complex64)


def ofdm(n: int, fs: float, bw_hz: float, rng: np.random.Generator, scs_hz: float = 30e3, cp: float = 0.125) -> np.ndarray:
    """CP-OFDM with QPSK on the K = bw/scs subcarriers around DC (DC itself unused).

    A transmit FIR (cutoff 0.55·BW) enforces a spectral mask, as real transmitters do; unfiltered
    rectangular-symbol sidelobes would leak energy several MHz outside the channel.
    """
    N = max(8, int(round(fs / scs_hz)))
    K = min(N - 2, max(2, int(round(bw_hz / (fs / N)))))
    ncp = int(round(cp * N))
    nsym = int(np.ceil(n / (N + ncp))) + 1
    idx = np.r_[-(K // 2) : 0, 1 : K - K // 2 + 1] % N
    X = np.zeros((nsym, N), np.complex64)
    X[:, idx] = (rng.choice([-1, 1], (nsym, len(idx))) + 1j * rng.choice([-1, 1], (nsym, len(idx)))) / np.sqrt(2)
    x = np.fft.ifft(X, axis=1) * (N / np.sqrt(len(idx)))
    x = np.concatenate([x[:, N - ncp :], x], axis=1).ravel()[: n + 64]
    if bw_hz < 0.9 * fs:  # transmit spectral mask
        x = signal.lfilter(signal.firwin(129, 1.1 * bw_hz / fs), 1.0, x)[64 : 64 + n]
        x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)
    else:
        x = x[:n]
    return x.astype(np.complex64)


def noise_burst(n: int, fs: float, bw_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Colored (band-limited) complex Gaussian noise of the target bandwidth."""
    w = (rng.standard_normal(n + 256) + 1j * rng.standard_normal(n + 256)) / np.sqrt(2)
    hfir = signal.firwin(129, min(bw_hz / fs, 0.999))
    x = signal.lfilter(hfir, 1.0, w)[256:]
    return (x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)).astype(np.complex64)


def synth(modulation: str, n: int, fs: float, bw_hz: float, rng: np.random.Generator, **kw: float) -> np.ndarray:
    """Dispatch on modulation name ('gfsk' | 'ofdm' | 'noise')."""
    if modulation == "gfsk":
        return gfsk(n, fs, bw_hz, rng, bt=kw.get("bt", 0.5), h=kw.get("h", 0.5), bw_per_rate=kw.get("bw_per_rate", 1.0))
    if modulation == "ofdm":
        return ofdm(n, fs, bw_hz, rng, scs_hz=kw.get("scs_hz", 30e3), cp=kw.get("cp", 0.125))
    return noise_burst(n, fs, bw_hz, rng)
