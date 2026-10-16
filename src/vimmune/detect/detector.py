"""Reference blob detector: STFT -> noise floor -> 2-D CA-CFAR -> morphology -> blobs -> BDW + burst IQ.

Implements report §2.1 (spec §5, the stand-in for the in-house pipeline). Any object implementing
:class:`Detector` can replace it.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import fft as sp_fft
from scipy import ndimage, signal

from vimmune.config import DetectCfg
from vimmune.io.bdw import SessionMeta, make_bdw
from vimmune.io.iq import BurstStore, IQSource


class Detector(Protocol):
    """Interface of a burst detector: IQ recording in, BDW table out (snippets go to ``store``)."""

    def detect(self, src: IQSource, store: BurstStore, session: SessionMeta) -> pd.DataFrame: ...


def stft_power(x: np.ndarray, nfft: int, hop: int, batch: int = 2048) -> np.ndarray:
    """|STFT|² with a Hann window, frames x bins, frequency axis fft-shifted (ascending). float32, batched."""
    n = 1 + (len(x) - nfft) // hop
    if n <= 0:
        return np.zeros((0, nfft), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.complex64), nfft)[::hop][:n]
    w = signal.windows.hann(nfft, sym=False).astype(np.float32)
    out = np.empty((n, nfft), np.float32)
    for i in range(0, n, batch):
        X = sp_fft.fft(frames[i : i + batch] * w, axis=1)
        out[i : i + batch] = np.fft.fftshift(X.real**2 + X.imag**2, axes=1)
    return out / np.float32(np.sum(w**2))


def _box1(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Running mean of length k along ``axis`` (edge-replicated), via cumsum differences."""
    if k <= 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2 + 1, k - 1 - k // 2)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    hi = [slice(None)] * 2
    lo = [slice(None)] * 2
    hi[axis], lo[axis] = slice(k, None), slice(None, -k)
    return (c[tuple(hi)] - c[tuple(lo)]) / np.float32(k)


def _box(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Separable 2-D box mean (frames x bins)."""
    return _box1(_box1(a, size[0], 0), size[1], 1)


def cfar_stat(R: np.ndarray, cfg: DetectCfg) -> np.ndarray:
    """Smoothed noise-normalized power over the CA reference, which is capped at the noise floor.

    ``R`` is first pooled by ``cfg.pool`` (non-overlapping means); the remaining smoothing, guard and training
    sizes are in pooled cells. The returned map is at pooled resolution.

    The training ring (guard excluded) gives the local cell average. It is capped at 1 (the whitened noise
    floor), so wide bursts that fill their own training cells are still detected (see DECISIONS).
    """
    pt, pf = cfg.pool
    n, m = (R.shape[0] // pt) * pt, (R.shape[1] // pf) * pf
    Rp = R[:n, :m].reshape(n // pt, pt, m // pf, pf).mean((1, 3))
    Rs = _box(Rp, (max(1, cfg.smooth[0] // pt), max(1, cfg.smooth[1] // pf)))
    gt, gf = cfg.cfar_guard
    tt, tf = cfg.cfar_train
    outer = (2 * (gt + tt) + 1, 2 * (gf + tf) + 1)
    inner = (2 * gt + 1, 2 * gf + 1)
    so, si = _box(Rs, outer) * np.prod(outer), _box(Rs, inner) * np.prod(inner)
    ca = (so - si) / (np.prod(outer) - np.prod(inner))
    return Rs / np.minimum(ca, 1.0)


def calibrate_alpha(cfg: DetectCfg) -> float:
    """CFAR threshold factor for the configured Pfa, measured on simulated pure noise (cached)."""
    return _alpha(cfg.model_dump_json())


@lru_cache(maxsize=8)
def _alpha(cfg_json: str, n_frames: int = 12000, seed: int = 0) -> float:
    cfg = DetectCfg.model_validate_json(cfg_json)
    rng = np.random.default_rng(seed)
    nf = cfg.nfft
    x = ((rng.standard_normal(n_frames * cfg.hop + nf) + 1j * rng.standard_normal(n_frames * cfg.hop + nf)) / np.sqrt(2)).astype(np.complex64)
    S = stft_power(x, nf, cfg.hop)
    return float(np.quantile(cfar_stat(S / S.mean(0, keepdims=True), cfg), 1 - cfg.pfa))


def noise_floor(S: np.ndarray, q: float, cap_db: float = 3.0) -> np.ndarray:
    """Per-bin noise power from the q-quantile of exponential bin powers, Q_q / (-ln(1-q)).

    Bins that are occupied almost all the time bias the quantile upwards, so each bin is capped at
    ``cap_db`` above the 10th percentile of the per-bin estimates (receiver response assumed smooth).
    """
    nf = np.quantile(S, q, axis=0) / -np.log1p(-q)
    return np.minimum(nf, np.quantile(nf, 0.1) * 10 ** (cap_db / 10))


def _crossing(p: np.ndarray, thr: float, first: bool) -> float:
    """Fractional index of the first (or last) threshold crossing of ``p`` (linear interpolation)."""
    above = np.flatnonzero(p >= thr)
    if len(above) == 0:
        return float(len(p) // 2)
    i = above[0] if first else above[-1]
    j = i - 1 if first else i + 1
    if 0 <= j < len(p) and p[i] != p[j]:
        return i + (j - i) * (p[i] - thr) / (p[i] - p[j])
    return float(i)


def _process_block(args: tuple) -> list[dict]:
    src, b0, b1, cfg, alpha = args
    fs, nfft, hop = src.fs, cfg.nfft, cfg.hop
    # noise floor: frames at 4x hop over the sliding 200 ms window centered on this block
    half_nf = int(cfg.noise_window_s * fs / 2)
    mid = (b0 + b1) // 2
    n0 = max(0, mid - half_nf)
    nfl = noise_floor(stft_power(src.read(n0, min(src.n_samples, mid + half_nf) - n0), nfft, 4 * hop), cfg.noise_quantile, cfg.noise_cap_db)
    # detection on [b0 - margin, b1 + margin]
    w0 = max(0, b0 - int(cfg.margin_s * fs))
    w1 = min(src.n_samples, b1 + int(cfg.margin_s * fs))
    S = stft_power(src.read(w0, w1 - w0), nfft, hop)
    if len(S) < 8:
        return []
    Rb = S / nfl
    del S
    f0 = 0
    mask = cfar_stat(Rb, cfg) > alpha
    mask = ndimage.binary_closing(mask, structure=np.ones(cfg.closing, bool))
    lab, n = ndimage.label(mask)
    pt, pf = cfg.pool
    out = []
    excess = np.maximum(Rb - 1.0, 0.0)
    fbin = fs / nfft
    t_frame = lambda i: (w0 + (f0 + i) * hop + nfft / 2) / fs  # noqa: E731
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ts, fsl = slice(sl[0].start * pt, sl[0].stop * pt), slice(sl[1].start * pf, sl[1].stop * pf)
        if ts.stop - ts.start < cfg.min_frames or fsl.stop - fsl.start < cfg.min_bins:
            continue
        # TOA/TOD: half-level crossings of the in-band excess power profile
        m = 6
        a, b = max(0, ts.start - m), min(len(Rb), ts.stop + m)
        prof = excess[a:b, fsl].mean(1)
        lvl = np.median(excess[ts, fsl].mean(1))
        i0 = _crossing(prof, 0.5 * lvl, True)
        i1 = _crossing(prof, 0.5 * lvl, False)
        toa, tod = t_frame(a + i0), t_frame(a + i1)
        if not (b0 / fs <= toa < b1 / fs) or tod <= toa:
            continue
        # spectrum over the burst's frames; FC = power-weighted centroid, BW = -10 dB extent
        ta, tb = int(np.ceil(a + i0)), int(np.floor(a + i1)) + 1
        if tb <= ta:
            ta, tb = ts.start, ts.stop
        fa, fb = max(0, fsl.start - 8), min(nfft, fsl.stop + 8)
        spec = excess[ta:tb, fa:fb].mean(0)
        sm = np.convolve(spec, np.ones(3) / 3, mode="same")
        pk = sm.max()
        above = np.flatnonzero(sm >= pk * 10 ** (cfg.bw_db / 10))
        lo, hi = above[0], above[-1]
        w = spec[lo : hi + 1]
        fc_bin = fa + lo + np.sum(np.arange(len(w)) * w) / max(w.sum(), 1e-12)
        fc_off = (fc_bin - nfft / 2) * fbin
        bw = (hi - lo + 1) * fbin
        snr = 10 * np.log10(max(float(w.mean()), 1e-3))
        out.append({"toa_s": toa, "tod_s": tod, "pw_s": tod - toa, "fc_off_hz": fc_off, "bw_hz": bw,
                    "snr_db": snr, "iq": extract_iq(src, toa, tod, fc_off, bw, cfg)})
    return out


def extract_iq(src: IQSource, toa: float, tod: float, fc_off: float, bw: float, cfg: DetectCfg) -> tuple[np.ndarray, float]:
    """Time-gate, mix FC to DC, FIR low-pass to ±0.6·BW (a 1.2·BW band-pass at FC), decimate to ≈1.25·BW."""
    fs = src.fs
    D = max(1, int(fs // (1.25 * bw)))
    taps = 129
    n = min(int(round((tod - toa) * fs)), cfg.iq_max_len * D)
    s0 = int(round(toa * fs))
    x = src.read(s0 - taps // 2, n + taps)
    t = (s0 - taps // 2 + np.arange(len(x))) / fs
    x = x * np.exp(-2j * np.pi * fc_off * t).astype(np.complex64)
    h = signal.firwin(taps, min(cfg.iq_bw_factor * bw / fs, 0.999))
    y = signal.fftconvolve(x, h, mode="valid")[::D]
    return y[: cfg.iq_max_len].astype(np.complex64), fs / D


class ReferenceDetector:
    """STFT + per-bin noise floor + 2-D CA-CFAR reference detector (spec §5)."""

    def __init__(self, cfg: DetectCfg):
        self.cfg = cfg

    def detect(self, src: IQSource, store: BurstStore, session: SessionMeta | None = None) -> pd.DataFrame:
        cfg = self.cfg
        alpha = calibrate_alpha(cfg)
        bs = int(cfg.block_s * src.fs)
        jobs = [(src, b0, min(b0 + bs, src.n_samples), cfg, alpha) for b0 in range(0, src.n_samples, bs)]
        if cfg.workers > 1:
            with ProcessPoolExecutor(cfg.workers) as ex:
                blobs = [b for part in ex.map(_process_block, jobs) for b in part]
        else:
            blobs = [b for j in jobs for b in _process_block(j)]
        blobs.sort(key=lambda b: b["toa_s"])
        refs, fss = [], []
        for b in blobs:
            iq, fs = b.pop("iq")
            refs.append(store.put(iq))
            fss.append(fs)
        store.flush()
        sess = session or SessionMeta(fs=src.fs, center_freq=src.center_freq)
        sig = pd.DataFrame(blobs, columns=["toa_s", "tod_s", "pw_s", "fc_off_hz", "bw_hz", "snr_db"])
        sig["fc_hz"] = src.center_freq + sig.pop("fc_off_hz")
        sig["burst_iq_ref"], sig["iq_fs"], sig["iq_fc_hz"] = refs, fss, sig["fc_hz"]
        return make_bdw(sig, sess)


def label_from_truth(det: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Copy protocol/source_id of the truth burst with the largest t-f overlap onto each detection (eval only)."""
    det = det.copy()
    tt, td = truth.toa_s.to_numpy(), truth.tod_s.to_numpy()
    tl, th = (truth.fc_hz - truth.bw_hz / 2).to_numpy(), (truth.fc_hz + truth.bw_hz / 2).to_numpy()
    order = np.argsort(tt)
    proto, src, idx = [], [], []
    for r in det.itertuples():
        j0, j1 = np.searchsorted(tt[order], [r.toa_s - 0.003, r.tod_s])
        cand = order[j0:j1]
        ot = np.minimum(td[cand], r.tod_s) - np.maximum(tt[cand], r.toa_s)
        of = np.minimum(th[cand], r.fc_hz + r.bw_hz / 2) - np.maximum(tl[cand], r.fc_hz - r.bw_hz / 2)
        ov = np.where((ot > 0) & (of > 0), ot * of, 0.0)
        if len(cand) and ov.max() > 0:
            j = cand[int(np.argmax(ov))]
            proto.append(truth.protocol.iloc[j])
            src.append(truth.source_id.iloc[j])
            idx.append(int(j))
        else:
            proto.append(None)
            src.append(None)
            idx.append(-1)
    det["protocol"], det["source_id"], det["truth_idx"] = proto, src, idx
    return det
