"""SDIF pulse deinterleaving (Milojević & Popović 1992; CDIF: Mardia 1989).

Implements report §2.3 step 2. Difference histograms of order 1–3 with the adaptive threshold
T(τ) = x·(E − c)·exp(−τ/(k·N)); each salient period's pulse trains are extracted within ±tol and removed,
and the search restarts until no salient period remains.
"""

from __future__ import annotations

import numpy as np

from vimmune.config import AssocCfg


def sdif_candidates(toa: np.ndarray, c: int, cfg: AssocCfg, max_period: float) -> list[float]:
    """Periods (s) whose order-``c`` difference-histogram bin exceeds the SDIF threshold, ascending."""
    if len(toa) <= c:
        return []
    d = toa[c:] - toa[:-c]
    d = d[(d > 0) & (d < max_period)]
    if len(d) == 0:
        return []
    N = cfg.sdif_bins
    h, edges = np.histogram(d, bins=N, range=(0, max_period))
    tau_bins = np.arange(N) + 0.5
    E = len(toa)
    thr = cfg.sdif_x * (E - c) * np.exp(-tau_bins / (cfg.sdif_k * N))
    hs = h + np.r_[h[1:], 0] + np.r_[0, h[:-1]]  # tolerate peaks straddling two bins
    peaks = np.flatnonzero((hs > thr) & (hs >= np.r_[hs[1:], 0]) & (hs >= np.r_[0, hs[:-1]]) & (h > 0))
    out = []
    for p in peaks:
        lo, hi = edges[max(p - 1, 0)], edges[min(p + 2, N)]
        sel = d[(d >= lo) & (d < hi)]
        out.append(float(np.median(sel)))
    out = sorted(set(np.round(out, 7)))
    return [float(v) for v in out]


def extract_trains(toa: np.ndarray, period: float, cfg: AssocCfg) -> list[np.ndarray]:
    """Greedy pulse-train extraction at ``period`` ± tol (missed pulses allowed). Returns index arrays."""
    used = np.zeros(len(toa), bool)
    tol = cfg.sdif_tol * period
    trains = []
    for i in range(len(toa)):
        if used[i]:
            continue
        chain, cur, miss = [i], toa[i], 0
        while miss <= cfg.sdif_max_missing:
            exp = cur + period
            j0, j1 = np.searchsorted(toa, [exp - tol, exp + tol])
            cand = [j for j in range(j0, j1) if not used[j] and j not in chain]
            if cand:
                j = min(cand, key=lambda j: abs(toa[j] - exp))
                chain.append(j)
                cur, miss = toa[j], 0
            else:
                cur, miss = exp, miss + 1
            if exp > toa[-1] + tol:
                break
        span = (toa[chain[-1]] - toa[chain[0]]) / period + 1
        if len(chain) >= cfg.sdif_min_train and len(chain) / span >= cfg.sdif_min_fill:
            used[chain] = True
            trains.append(np.array(chain))
    return _merge_phase_consistent(trains, toa, period, cfg)


def _merge_phase_consistent(trains: list[np.ndarray], toa: np.ndarray, period: float, cfg: AssocCfg) -> list[np.ndarray]:
    """Re-join chains of one emitter broken by a long fade: disjoint in time and on the same phase grid."""
    trains = sorted(trains, key=lambda t: toa[t[0]])
    out: list[np.ndarray] = []
    for t in trains:
        for k, o in enumerate(out):
            gap = toa[t[0]] - toa[o[-1]]
            ph = gap / period
            if gap > 0 and abs(ph - round(ph)) * period <= cfg.sdif_tol * period * max(1.0, np.sqrt(round(ph))):
                out[k] = np.concatenate([o, t])
                break
        else:
            out.append(t)
    return out


def deinterleave(toa: np.ndarray, cfg: AssocCfg, max_period: float = 0.15) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Split a TOA list into periodic trains. Returns (trains as index arrays, residual indices, periods)."""
    order = np.argsort(toa)
    t = toa[order]
    remaining = np.arange(len(t))
    trains: list[np.ndarray] = []
    periods: list[float] = []
    while len(remaining) >= cfg.sdif_min_train:
        found = False
        for c in range(1, cfg.sdif_max_order + 1):
            for tau in sdif_candidates(t[remaining], c, cfg, max_period):
                if tau <= 0:
                    continue
                got = extract_trains(t[remaining], tau, cfg)
                if got:
                    for g in got:
                        trains.append(order[remaining[g]])
                        periods.append(tau)
                    keep = np.ones(len(remaining), bool)
                    keep[np.concatenate(got)] = False
                    remaining = remaining[keep]
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return trains, order[remaining], periods
