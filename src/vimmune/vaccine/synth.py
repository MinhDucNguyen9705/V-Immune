"""Vaccine data synthesis: sequence generator from the card, parametric waveforms, channel augmentation,
quality filter.

Implements report §2.7 (sequence level, CPU parametric fallback, channel augmentation, filter). All
waveforms live in the tokenizer's preprocessed domain: 1024 complex samples at fs' = BW/0.8, unit power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vimmune.config import Config
from vimmune.dsp import bandwidth
from vimmune.grammar.card import Card
from vimmune.sim import channel as chan
from vimmune.sim.fastbdw import apply_noise_model
from vimmune.sim.waveforms import gfsk, ofdm


# ------------------------------------------------------------------------------------------- sequences
def transition_matrix(fc: np.ndarray, toa: np.ndarray, centers: np.ndarray, period: float | None, smoothing: float) -> np.ndarray:
    ch = np.argmin(np.abs(fc[:, None] - centers[None]), 1)
    n = len(centers)
    C = np.full((n, n), smoothing)
    ok = np.ones(len(ch) - 1, bool) if not period else (np.diff(toa) / period < 1.5)
    np.add.at(C, (ch[:-1][ok], ch[1:][ok]), 1.0)
    return C / C.sum(1, keepdims=True)


def generate_sequences(card: Card, learn: pd.DataFrame, cfg: Config, n_seq: int, seq_len: int, seed: int) -> pd.DataFrame:
    """Synthetic BDW sequences of the new protocol.

    Channels follow the transition matrix P estimated on the learn bursts, inter-arrivals the frame period
    with ±1 % jitter (empirical ΔTOA if aperiodic), PW/BW are resampled from the learn bursts, SNR ~ U(−10, 20)
    dB per sequence, CFO ±5 ppm per sequence; then the detector measurement-noise model (misses, jitter,
    split/merge) is applied.
    """
    rng = np.random.default_rng(seed)
    v = cfg.vaccine
    centers = np.asarray(card.channels_hz) if card.channels_hz else np.array([learn.fc_hz.median()])
    P = transition_matrix(learn.fc_hz.to_numpy(), learn.toa_s.to_numpy(), centers, card.frame_period_s, cfg.grammar.trans_smoothing)
    w, vec = np.linalg.eig(P.T)
    pi = np.abs(np.real(vec[:, np.argmin(np.abs(w - 1))]))
    pi /= pi.sum()
    dtoa_emp = np.diff(np.sort(learn.toa_s.to_numpy()))
    rows = []
    for s in range(n_seq):
        n = int(seq_len * 1.3) + 4  # oversample: the noise model drops some bursts
        if card.frame_period_s:
            gaps = card.frame_period_s * (1 + rng.uniform(-cfg.protos.jitter_frac, cfg.protos.jitter_frac, n))
        else:
            gaps = rng.choice(dtoa_emp, n) if len(dtoa_emp) else np.full(n, 0.01)
        toa = np.cumsum(gaps)
        ch = np.empty(n, int)
        ch[0] = rng.choice(len(centers), p=pi)
        for i in range(1, n):
            ch[i] = rng.choice(len(centers), p=P[ch[i - 1]])
        k = rng.integers(0, len(learn), n)
        cfo = rng.uniform(-1, 1) * cfg.protos.cfo_ppm * 1e-6 * cfg.sim.center_freq
        snr = rng.uniform(*v.aug_snr_db)
        rows.append(pd.DataFrame({"toa_s": toa, "pw_s": learn.pw_s.to_numpy()[k], "bw_hz": learn.bw_hz.to_numpy()[k],
                                  "fc_off_hz": centers[ch] - cfg.sim.center_freq + cfo, "snr_db": snr + rng.normal(0, 0.5, n),
                                  "channel": ch, "source_id": f"syn{s}", "protocol": "NEW", "seq": s}))
    df = pd.concat(rows, ignore_index=True)
    df["merged"] = False
    meas = apply_noise_model(df, cfg.sim.noise_model, rng)
    meas = meas.groupby("seq", sort=True).head(seq_len).reset_index(drop=True)
    meas["fc_hz"] = cfg.sim.center_freq + meas.fc_off_hz
    meas["tod_s"] = meas.toa_s + meas.pw_s
    meas["fc_off_norm"] = meas.fc_off_hz / cfg.sim.fs
    meas["session_id"] = "synthetic"
    return meas


# ------------------------------------------------------------------------------------------- waveforms
def classify_waveform(waves: np.ndarray) -> str:
    """'ofdm-like' (flat-top spectrum) vs 'gfsk-like' (peaked spectrum) from the mean PSD of the bursts.

    In the preprocessed domain the signal occupies ±0.4·fs'; the ratio of PSD near the band edges
    (0.35–0.45·fs') to the centre (< 0.1·fs') is ≈ 0.75–1 for OFDM and ≈ 0.3 for Gaussian-filtered FSK.
    """
    X = np.abs(np.fft.fftshift(np.fft.fft(waves[:, :512], axis=1), axes=1)) ** 2
    p = X.mean(0)
    f = np.fft.fftshift(np.fft.fftfreq(512))
    edge = p[(np.abs(f) >= 0.35) & (np.abs(f) <= 0.45)].mean()
    centre = p[np.abs(f) < 0.1].mean()
    return "ofdm-like" if edge / centre > 0.5 else "gfsk-like"


def param_waveforms(card: Card, kind: str, n: int, cfg: Config, seed: int, scs_candidates=(15e3, 30e3, 312.5e3)) -> np.ndarray:
    """CPU fallback: parametric GFSK/OFDM fit from the card, in the preprocessed domain (fs' = BW/0.8)."""
    rng = np.random.default_rng(seed)
    t = cfg.tokenizer
    fs = card.bw_hz / t.occupancy
    L = t.wave_len
    n_on = min(L, int(round(card.pw_s * fs)))
    out = np.zeros((n, L), np.complex64)
    for i in range(n):
        if kind == "gfsk-like":
            x = gfsk(n_on, fs, card.bw_hz, rng, cfg.protos.gfsk_bt, cfg.protos.gfsk_h, cfg.protos.gfsk_bw_per_rate)
        else:
            x = ofdm(n_on, fs, card.bw_hz, rng, float(rng.choice(scs_candidates)))
        out[i, :n_on] = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)
    return out


def augment(waves: np.ndarray, snr_db: np.ndarray, cfg: Config, rng: np.random.Generator, bw_hz: float) -> np.ndarray:
    """Channel augmentation in the preprocessed domain: multipath (2–4 taps), CFO ±5 ppm, background
    interference (simulated WiFi-/BLE-like, SIR ~ U(0, 20) dB, overlap prob 0.3), AWGN at the burst SNR."""
    v = cfg.vaccine
    fs = bw_hz / cfg.tokenizer.occupancy
    out = np.empty_like(waves)
    L = waves.shape[1]
    n_t = np.arange(L)
    for i, x in enumerate(waves):
        on = np.abs(x) > 0
        y = chan.apply_multipath(x, chan.multipath_taps(rng, fs, cfg.sim.channel))
        cfo = rng.uniform(-1, 1) * cfg.protos.cfo_ppm * 1e-6 * cfg.sim.center_freq
        y = y * np.exp(2j * np.pi * cfo * n_t / fs).astype(np.complex64)
        if rng.uniform() < v.aug_overlap_prob:
            m = int(rng.integers(L // 8, L))
            s0 = int(rng.integers(0, L - m + 1))
            bg = (gfsk(m, fs, min(1e6, 0.5 * fs), rng) if rng.uniform() < 0.5 else ofdm(m, fs, min(fs, 20e6), rng, 312.5e3))
            bg = bg * np.exp(2j * np.pi * rng.uniform(-0.4, 0.4) * fs * np.arange(m) / fs)
            y[s0 : s0 + m] += bg * 10 ** (-rng.uniform(*v.aug_sir_db) / 20)
        p_sig = np.mean(np.abs(y[on]) ** 2) if on.any() else 1.0
        noise = chan.awgn(L, rng, p_sig / 10 ** (snr_db[i] / 10) / cfg.tokenizer.occupancy)
        y = y + noise
        out[i] = y / np.sqrt(np.mean(np.abs(y[: max(1, on.sum())]) ** 2) + 1e-20)
    return out


def reestimate(waves: np.ndarray, bw_card: float, occupancy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-waveform BW (−10 dB, from the PSD at fs' = BW/0.8), PW (envelope extent) and 'saturated' flag."""
    fs = bw_card / occupancy
    bws, pws, sat = [], [], []
    for x in waves:
        bws.append(bandwidth(x, fs, nfft=256))
        e = np.convolve(np.abs(x) ** 2, np.ones(16) / 16, mode="same")
        on = np.flatnonzero(e > 0.25 * np.median(e[e > np.percentile(e, 10)]))
        pws.append((on[-1] - on[0] + 1) / fs if len(on) else 0.0)
        sat.append(len(on) and on[-1] >= len(x) - 20)
    return np.array(bws), np.array(pws), np.array(sat, bool)


def mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    """Squared MMD with an RBF kernel (median-distance bandwidth)."""
    x = np.concatenate([a, b])
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    s = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    K = np.exp(-d2 / s)
    n = len(a)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def quality_filter(z_gen: np.ndarray, z_real: np.ndarray, waves: np.ndarray, card: Card, cfg: Config) -> tuple[np.ndarray, dict]:
    """Keep samples with cos(z_gen, z̄_real) ≥ 0.8 and re-estimated BW/PW within 10 % of the card."""
    v = cfg.vaccine
    zbar = z_real.mean(0)
    cos = (z_gen @ zbar) / (np.linalg.norm(z_gen, axis=1) * np.linalg.norm(zbar) + 1e-12)
    bw, pw, sat = reestimate(waves, card.bw_hz, cfg.tokenizer.occupancy)
    ok_bw = np.abs(bw / card.bw_hz - 1) <= v.filter_rel
    # PW is only measurable when the burst fits in the 1024-sample window
    fits = card.pw_s * card.bw_hz / cfg.tokenizer.occupancy < 0.95 * cfg.tokenizer.wave_len
    ok_pw = (np.abs(pw / card.pw_s - 1) <= v.filter_rel) if fits else np.ones(len(pw), bool)
    keep = (cos >= v.filter_cos) & ok_bw & ok_pw
    pd_gen = np.mean(np.linalg.norm(z_gen[:, None] - z_gen[None], axis=-1)) if len(z_gen) > 1 else 0.0
    pd_real = np.mean(np.linalg.norm(z_real[:, None] - z_real[None], axis=-1)) if len(z_real) > 1 else 0.0
    kept = z_gen[keep]
    stats = {"n_generated": int(len(z_gen)), "n_kept": int(keep.sum()), "pass_cos": float(np.mean(cos >= v.filter_cos)),
             "pass_bw": float(ok_bw.mean()), "pass_pw": float(ok_pw.mean()), "pw_checked": bool(fits),
             "cos_median": float(np.median(cos)), "mmd_all": mmd_rbf(z_gen[:300], z_real), "mmd_kept": mmd_rbf(kept[:300], z_real) if len(kept) else None,
             "spread_gen": float(pd_gen), "spread_real": float(pd_real),
             "mode_collapse_warning": bool(pd_real > 0 and pd_gen < 0.5 * pd_real)}
    return keep, stats
