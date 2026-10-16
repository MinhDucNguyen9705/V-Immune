"""Detector evaluation against simulator truth and comparison with the fast-mode noise model.

Implements report §2.1 (spec §5 / M2 validation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def isolated(truth: pd.DataFrame, guard_s: float = 200e-6, guard_hz: float = 1e6) -> np.ndarray:
    """Truth bursts with no other burst within the t-f guard (collisions and touching blobs excluded)."""
    t0, t1 = truth.toa_s.to_numpy() - guard_s / 2, truth.tod_s.to_numpy() + guard_s / 2
    lo = (truth.fc_hz - truth.bw_hz / 2).to_numpy() - guard_hz / 2
    hi = (truth.fc_hz + truth.bw_hz / 2).to_numpy() + guard_hz / 2
    o = np.argsort(t0)
    ok = np.ones(len(truth), bool)
    for a in range(len(o)):
        i = o[a]
        for b in range(a + 1, len(o)):
            j = o[b]
            if t0[j] >= t1[i] + 0.003:
                break
            if t0[j] < t1[i] and t1[j] > t0[i] and lo[j] < hi[i] and hi[j] > lo[i]:
                ok[i] = ok[j] = False
    return ok


def match(det: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """For every truth burst: matched detection (TOA within PW/2 and FC within BW/2) and errors."""
    dt, dfc = det.toa_s.to_numpy(), det.fc_hz.to_numpy()
    o = np.argsort(dt)
    rows = []
    for r in truth.itertuples():
        j0, j1 = np.searchsorted(dt[o], [r.toa_s - r.pw_s / 2, r.toa_s + r.pw_s / 2])
        c = o[j0:j1]
        c = c[np.abs(dfc[c] - r.fc_hz) < r.bw_hz / 2]
        if len(c):
            j = c[np.argmin(np.abs(dt[c] - r.toa_s))]
            d = det.iloc[j]
            rows.append((True, d.toa_s - r.toa_s, d.fc_hz - r.fc_hz, d.pw_s / r.pw_s - 1, d.bw_hz / r.bw_hz - 1, d.snr_db - r.snr_db))
        else:
            rows.append((False, np.nan, np.nan, np.nan, np.nan, np.nan))
    m = pd.DataFrame(rows, columns=["hit", "toa_err", "fc_err", "pw_rel", "bw_rel", "snr_err"], index=truth.index)
    return pd.concat([truth[["protocol", "source_id", "snr_db", "bw_hz", "pw_s"]], m], axis=1)


def error_stats(m: pd.DataFrame, snr_bins: tuple[float, ...] = (-5, 0, 5, 10, 20, 40)) -> pd.DataFrame:
    """Recall and robust error spreads per SNR bin (used to compare with the fast-mode noise model)."""
    m = m.copy()
    m["bin"] = pd.cut(m.snr_db, snr_bins)
    mad = lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))) if np.isfinite(x).any() else np.nan  # noqa: E731
    return m.groupby("bin", observed=True).agg(
        n=("hit", "size"), recall=("hit", "mean"),
        toa_sd_us=("toa_err", lambda x: 1e6 * mad(x)), toa_bias_us=("toa_err", lambda x: 1e6 * np.nanmedian(x)),
        pw_rel_sd=("pw_rel", mad), bw_rel_sd=("bw_rel", mad), bw_rel_bias=("bw_rel", np.nanmedian),
        fc_frac_sd=("fc_err", mad), snr_sd=("snr_err", mad))


def noise_model_stats(nm, snr_db: float) -> dict[str, float]:
    """Spreads the fast-mode noise model predicts at ``snr_db``."""
    g = 1 / np.sqrt(10 ** (snr_db / 10))
    return {"toa_sd_us": 1e6 * (nm.toa_a_s + nm.toa_b_s * g), "pw_rel_sd": nm.pw_rel_a + nm.pw_rel_b * g,
            "bw_rel_sd": nm.bw_rel_a + nm.bw_rel_b * g,
            "recall": 1 - 1 / (1 + np.exp((snr_db - nm.miss_snr50_db) / nm.miss_slope_db))}
