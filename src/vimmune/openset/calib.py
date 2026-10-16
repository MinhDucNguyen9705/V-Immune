"""Open-set decision: SNR-calibrated surprise and distance scores, fusion, conformal p-values, CUSUM.

Implements report §2.5:
  z_S = (S − μ(s))/σ(s) with μ, σ per 2 dB bin of mean SNR, smoothed across bins (counts-weighted LOWESS-style
  local-linear fit); d = min_k (1 − cos(h, μ_k)) calibrated the same way; U = α z_S + (1−α) z_d;
  p = (1 + #{U_j ≥ U})/(n + 1), unknown if p < ε (Angelopoulos & Bates 2021);
  CUSUM g_n = max(0, g_{n−1} + ℓ_n − κ), alarm at g_n > h (Page 1954).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------------------- SNR calibration
@dataclass
class SNRCalib:
    """μ(s), σ(s) on a 2 dB SNR grid, smoothed across bins; constant extrapolation beyond the data."""

    centers: list[float] = field(default_factory=list)
    mu: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    counts: list[float] = field(default_factory=list)

    @staticmethod
    def fit(values: np.ndarray, snr: np.ndarray, bin_db: float = 2.0, bw_bins: float = 2.0, lo: float = -16.0,
            hi: float = 42.0) -> SNRCalib:
        edges = np.arange(lo, hi + bin_db, bin_db)
        c = (edges[:-1] + edges[1:]) / 2
        k = np.clip(np.digitize(snr, edges) - 1, 0, len(c) - 1)
        n = np.bincount(k, minlength=len(c)).astype(float)
        m = np.bincount(k, values, len(c)) / np.maximum(n, 1)
        v = np.bincount(k, values**2, len(c)) / np.maximum(n, 1) - m**2
        ok = n > 0
        mu_s = _local_linear(c, c[ok], m[ok], n[ok], bw_bins * bin_db)
        sd_s = np.sqrt(np.maximum(_local_linear(c, c[ok], np.maximum(v[ok], 0), n[ok], bw_bins * bin_db), 1e-12))
        sd_s = np.maximum(sd_s, 0.5 * (np.std(values) + 1e-9))  # sparse bins must not produce tiny σ
        lo_c, hi_c = c[ok].min(), c[ok].max()  # no extrapolation of the trend beyond observed SNRs
        mu_s = np.interp(np.clip(c, lo_c, hi_c), c, mu_s)
        sd_s = np.interp(np.clip(c, lo_c, hi_c), c, sd_s)
        return SNRCalib(c.tolist(), mu_s.tolist(), sd_s.tolist(), n.tolist())

    def z(self, values: np.ndarray, snr: np.ndarray) -> np.ndarray:
        c = np.asarray(self.centers)
        return (np.asarray(values) - np.interp(snr, c, self.mu)) / np.interp(snr, c, self.sigma)


def _local_linear(x0: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, bw: float) -> np.ndarray:
    """Counts-weighted local-linear regression with a tricube kernel (LOWESS-style smoothing across bins)."""
    out = np.empty(len(x0))
    for i, q in enumerate(x0):
        d = np.abs(x - q) / (2.5 * bw)
        k = w * np.clip(1 - d**3, 0, None) ** 3
        if k.sum() <= 0:
            k = w * np.exp(-0.5 * ((x - q) / bw) ** 2)
        if (k > 0).sum() >= 2:
            X = np.stack([np.ones_like(x), x - q], 1)
            beta = np.linalg.lstsq(X * np.sqrt(k)[:, None], y * np.sqrt(k), rcond=None)[0]
            out[i] = beta[0]
        else:
            out[i] = np.sum(k * y) / max(k.sum(), 1e-12)
    return out


def distance(h: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d = min_k (1 − cos(h, μ_k)) and the argmin class index."""
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    D = 1 - hn @ mu.T
    return D.min(1), D.argmin(1)


def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    """AUROC with 'unknown' as the positive class (higher score = more unknown)."""
    if len(known) == 0 or len(unknown) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(known)), np.ones(len(unknown))]
    return float(roc_auc_score(y, np.r_[known, unknown]))


def tpr_at_fpr(known: np.ndarray, unknown: np.ndarray, fpr: float = 0.05) -> float:
    thr = np.quantile(known, 1 - fpr)
    return float(np.mean(unknown > thr)) if len(unknown) else float("nan")


def choose_alpha(zs_k: np.ndarray, zd_k: np.ndarray, zs_u: np.ndarray, zd_u: np.ndarray, grid: list[float]) -> tuple[float, dict[float, float]]:
    """α maximizing validation AUROC of U = α z_S + (1−α) z_d (ties -> larger α)."""
    res = {a: auroc(a * zs_k + (1 - a) * zd_k, a * zs_u + (1 - a) * zd_u) for a in grid}
    best = max(res, key=lambda a: (res[a], a))
    return best, res


def p_value(U: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """Conformal p-value p = (1 + #{U_j ≥ U}) / (n + 1)."""
    s = np.sort(cal)
    ge = len(s) - np.searchsorted(s, np.asarray(U), side="left")
    return (1 + ge) / (len(s) + 1)


# ------------------------------------------------------------------------------------------- CUSUM
def cusum(ell: np.ndarray, kappa: float, h: float, reset: bool = True) -> tuple[np.ndarray, list[int]]:
    """g_n trace and alarm indices (g reset to 0 after an alarm if ``reset``)."""
    g, out, alarms = 0.0, np.empty(len(ell)), []
    for i, x in enumerate(ell):
        g = max(0.0, g + x - kappa)
        if g > h:
            alarms.append(i)
            if reset:
                g = 0.0
        out[i] = g
    return out, alarms


def simulate_alarm_rate(series: list[np.ndarray], kappa: float, hs: np.ndarray, n_streams: int, n_steps: int,
                        block: int, rng: np.random.Generator) -> np.ndarray:
    """False alarms per burst for each threshold in ``hs``: CUSUM on block-bootstrapped null ℓ streams.

    Streams are built from random contiguous blocks of the calibration series (keeps short-range
    correlation); all thresholds and streams run in one vectorized pass.
    """
    pool = [s for s in series if len(s) >= block]
    if not pool:
        pool = series
    g = np.zeros((len(hs), n_streams))
    alarms = np.zeros(len(hs))
    buf = np.empty((n_streams, 0))
    pos = 0
    for _ in range(n_steps):
        if pos >= buf.shape[1]:
            blocks = []
            for _ in range(n_streams):
                s = pool[rng.integers(len(pool))]
                b = min(block, len(s))
                i0 = rng.integers(0, len(s) - b + 1)
                blocks.append(s[i0 : i0 + b])
            m = min(len(b) for b in blocks)
            buf, pos = np.stack([b[:m] for b in blocks]), 0
        g = np.maximum(0.0, g + buf[:, pos][None, :] - kappa)
        hit = g > hs[:, None]
        alarms += hit.sum(1)
        g[hit] = 0.0
        pos += 1
    return alarms / (n_streams * n_steps)


def choose_kappa_h(series_by_family: dict[str, list[np.ndarray]], dt_by_family: dict[str, float], kappas: list[float],
                   h_max: float, target_hours: float, n_streams: int, sim_hours: float, block: int,
                   seed: int, shift: float) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """(κ, h) minimizing the expected detection delay h/(δ − κ) for a reference shift δ, subject to the MTBFA
    target on every known family. δ is the typical mean ℓ of validation pseudo-unknowns."""
    best: tuple[float, float, float, dict[str, float]] = (np.inf, kappas[0], np.inf, {})
    table = {}
    for k in sorted(kappas):
        if k >= shift:
            continue
        h, mt = choose_h(series_by_family, dt_by_family, k, target_hours, n_streams, sim_hours, block, seed, h_max)
        delay = h / (shift - k)
        table[str(k)] = float(h)
        if delay < best[0]:
            best = (delay, k, h, mt)
    return best[1], best[2], best[3], table


def choose_h(series_by_family: dict[str, list[np.ndarray]], dt_by_family: dict[str, float], kappa: float,
             target_hours: float, n_streams: int, sim_hours: float, block: int, seed: int,
             h_max: float = 60.0) -> tuple[float, dict[str, float]]:
    """Smallest h (grid) whose simulated mean time between false alarms ≥ target for *every* family.

    Each family is simulated for ``sim_hours`` of its own burst rate on block-bootstrapped null streams. h is
    the smaller of (a) the first grid h whose Poisson 95 % upper rate bound meets the target and (b) the h
    where a log-linear fit of log(alarm rate) vs h (on grid points with ≥ 10 alarms) crosses the target.
    Returns h = inf if no h ≤ h_max qualifies.
    """
    rng = np.random.default_rng(seed)
    hs = np.linspace(0.5, h_max, min(160, int(h_max / 0.5)))
    h_need = 0.0
    mtbfa: dict[str, float] = {}
    for fam, ser in series_by_family.items():
        dt = dt_by_family[fam]
        steps = max(1000, int(min(sim_hours * 3600 / dt / n_streams, 200_000)))
        T_h = n_streams * steps * dt / 3600  # simulated hours
        n_al = simulate_alarm_rate(ser, kappa, hs, n_streams, steps, block, rng) * n_streams * steps
        rate = n_al / T_h  # alarms per hour
        target = 1.0 / target_hours
        # (a) direct: Poisson 95 % upper bound (k + 3) / T below target
        ok = np.flatnonzero((n_al + 3) / T_h <= target)
        h_direct = hs[ok[0]] if len(ok) else np.inf
        # (b) log-linear tail extrapolation (CUSUM false-alarm rate decays ~exponentially in h)
        fit = n_al >= 10
        h_ext = np.inf
        if fit.sum() >= 3:
            b_, a_ = np.polyfit(hs[fit], np.log(rate[fit]), 1)
            if b_ < 0:
                h_ext = (np.log(target) - a_) / b_
        h_f = min(h_direct, max(h_ext, hs[fit][-1] if fit.any() else 0.0))
        h_need = max(h_need, h_f)
        mtbfa[fam] = _mtbfa_at(h_f, hs, rate, n_al, T_h, fit)
    if not np.isfinite(h_need) or h_need > h_max:
        return float("inf"), mtbfa
    return float(h_need), mtbfa


def _mtbfa_at(h: float, hs: np.ndarray, rate: np.ndarray, n_al: np.ndarray, T_h: float, fit: np.ndarray) -> float:
    """MTBFA (hours) at h: observed if alarms were seen there, else the log-linear extrapolation."""
    if not np.isfinite(h):
        return float(1.0 / max(rate[-1], 1e-12))
    i = int(np.clip(np.searchsorted(hs, h), 0, len(hs) - 1))
    if n_al[i] >= 10 or fit.sum() < 3:
        return float(T_h / max(n_al[i], 1))
    b_, a_ = np.polyfit(hs[fit], np.log(rate[fit]), 1)
    return float(np.exp(-(a_ + b_ * h)))


def siegmund_arl(kappa: float, h: float, mu: float = 0.0) -> float:
    """Siegmund's ARL approximation for N(mu, 1) increments with reference value κ (used in tests)."""
    d = mu - kappa
    b = h + 1.166
    if abs(d) < 1e-9:
        return b**2
    return (np.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d**2)


@dataclass
class OpenSetCalib:
    """Everything the open-set decision needs (stored in releases as JSON)."""

    weights: list[float]
    s_cal: SNRCalib
    d_cal: SNRCalib
    b_cal: SNRCalib  # per-burst surprise calibration for CUSUM
    alpha: float
    epsilon: float
    cal_scores: list[float]
    kappa: float
    h: float
    score_len: int
    ctx: int
    ell_clip: float = 10.0
    mtbfa_h: dict[str, float] = field(default_factory=dict)
    alpha_auroc: dict[str, float] = field(default_factory=dict)
    b_cal_class: dict[str, SNRCalib] = field(default_factory=dict)  # per-burst calibration per known class
    s_cal_class: dict[str, SNRCalib] = field(default_factory=dict)  # window-surprise calibration per class

    def z_s(self, S: np.ndarray, snr: np.ndarray, nearest: list[str] | None = None) -> np.ndarray:
        """Surprise z-score; standardized with the nearest class's table when one exists (else global)."""
        z = self.s_cal.z(S, snr)
        if nearest is not None and self.s_cal_class:
            for c, cal in self.s_cal_class.items():
                m = np.array([n == c for n in nearest], bool)
                if m.any():
                    z[m] = cal.z(np.asarray(S)[m], np.asarray(snr)[m])
        return z

    def fuse(self, S: np.ndarray, snr: np.ndarray, d: np.ndarray, nearest: list[str] | None = None) -> np.ndarray:
        return self.alpha * self.z_s(S, snr, nearest) + (1 - self.alpha) * self.d_cal.z(d, snr)

    def p(self, U: np.ndarray) -> np.ndarray:
        return p_value(U, np.asarray(self.cal_scores))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> OpenSetCalib:
        d = dict(d)
        for k in ("s_cal", "d_cal", "b_cal"):
            d[k] = SNRCalib(**d[k])
        for k in ("b_cal_class", "s_cal_class"):
            d[k] = {c: SNRCalib(**v) for c, v in d.get(k, {}).items()}
        return OpenSetCalib(**d)
