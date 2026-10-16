"""Source association: sliding-window graph grouping -> same-type SDIF split -> Hungarian tracking.

Implements report §2.3. Input is a BDW frame plus the continuous waveform embeddings ``z`` (row-aligned).
Output is a ``track_id`` per burst (-1 = unassigned); tracks are released to SpectrumLM at ≥ 32 bursts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vimmune.associate.sdif import deinterleave
from vimmune.config import AssocCfg


def _unit(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def graph_groups(z: np.ndarray, bw: np.ndarray, pw: np.ndarray, cfg: AssocCfg) -> list[np.ndarray]:
    """Connected components of the 'same waveform, same BW, same PW' graph (FC deliberately ignored)."""
    n = len(z)
    if n == 0:
        return []
    zu = _unit(z)
    lb, lp = np.log(bw), np.log(pw)
    rows, cols = [], []
    for i0 in range(0, n, 2048):
        blk = slice(i0, min(n, i0 + 2048))
        e = (zu[blk] @ zu.T >= cfg.cos_min) & (np.abs(lb[blk, None] - lb[None]) < cfg.dlogbw_max) & (np.abs(lp[blk, None] - lp[None]) < cfg.dlogpw_max)
        r, c = np.nonzero(e)
        rows.append(r + i0)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    return [np.flatnonzero(lab == k) for k in range(lab.max() + 1)]


def merge_similar_groups(groups: list[np.ndarray], z: np.ndarray, bw: np.ndarray, pw: np.ndarray,
                         cfg: AssocCfg) -> list[np.ndarray]:
    """Re-join components of one source broken apart by outlier bursts (collision-inflated BW/PW).

    Components merge when their centroid z, median log BW and median log PW satisfy the same pairwise
    tolerances as single bursts. Group medians are far more stable than single measurements, so different
    families stay apart; same-family emitters are separated afterwards by the SDIF split.
    """
    if len(groups) < 2:
        return groups
    zc = _unit(np.stack([z[g].mean(0) for g in groups]))
    lb = np.array([np.median(np.log(bw[g])) for g in groups])
    lp = np.array([np.median(np.log(pw[g])) for g in groups])
    e = (zc @ zc.T >= cfg.cos_min) & (np.abs(lb[:, None] - lb[None]) < cfg.dlogbw_max) & (np.abs(lp[:, None] - lp[None]) < cfg.dlogpw_max)
    _, lab = connected_components(coo_matrix(e.astype(float)), directed=False)
    return [np.concatenate([groups[i] for i in np.flatnonzero(lab == k)]) for k in range(lab.max() + 1)]


def has_time_overlap_at_different_fc(toa: np.ndarray, tod: np.ndarray, fc: np.ndarray, bw: np.ndarray) -> bool:
    """Spec trigger: two bursts of one group overlapping in time at different FC => > 1 emitter."""
    o = np.argsort(toa)
    t0, t1, f, b = toa[o], tod[o], fc[o], bw[o]
    for i in range(len(o) - 1):
        j = i + 1
        while j < len(o) and t0[j] < t1[i]:
            if abs(f[j] - f[i]) > 0.5 * max(b[i], b[j]):
                return True
            j += 1
    return False


def split_group(idx: np.ndarray, toa: np.ndarray, tod: np.ndarray, fc: np.ndarray, bw: np.ndarray,
                cfg: AssocCfg, use_sdif: bool = True) -> list[tuple[np.ndarray, float | None]]:
    """SDIF same-type split. Returns [(indices, period or None)].

    Splits when the spec trigger fires (time overlap at different FC) *or* SDIF finds ≥ 2 trains / a train
    plus a large residual: same-period emitters rarely overlap inside a 2 s window (see DECISIONS).
    """
    if not use_sdif or len(idx) < 2 * cfg.sdif_min_train:
        return [(idx, None)]
    trains, res, periods = deinterleave(toa[idx], cfg)
    overlap = has_time_overlap_at_different_fc(toa[idx], tod[idx], fc[idx], bw[idx])
    big_res = len(res) >= max(cfg.sdif_min_train, 0.3 * len(idx))
    if len(trains) >= 2 or (len(trains) == 1 and big_res) or (overlap and trains):
        out = [(idx[t], p) for t, p in zip(trains, periods)]
        if len(res) >= cfg.sdif_min_train:
            out.append((idx[res], None))
        return out
    return [(idx, periods[0] if trains else None)]


@dataclass
class Track:
    tid: int
    z: np.ndarray
    period: float | None
    bursts: set[int] = field(default_factory=set)
    last_win: int = 0


def _period(toa: np.ndarray, p: float | None) -> float:
    if p is not None:
        return p
    d = np.diff(np.sort(toa))
    return float(np.median(d)) if len(d) else 1.0


class Associator:
    """Stateful tracker over sliding windows (batch: :func:`associate`; streaming: the edge runtime).

    The Hungarian cost adds a burst-overlap term ``1 - |g ∩ track| / |g ∩ previous window|`` to the spec's
    cosine + period cost, because co-channel same-family emitters have identical z and period.
    """

    def __init__(self, cfg: AssocCfg, use_sdif: bool = True):
        self.cfg, self.use_sdif = cfg, use_sdif
        self.tracks: dict[int, Track] = {}
        self.next_id = 0
        self.w = 0
        self.prev: set[int] = set()

    def step(self, toa, tod, fc, bw, pw, z, assign: np.ndarray, t0: float) -> np.ndarray:
        """Group + split + match the bursts with TOA in [t0, t0 + window); updates ``assign`` in place.

        Returns the indices of bursts that received a track id in this step.
        """
        cfg = self.cfg
        widx = np.flatnonzero((toa >= t0) & (toa < t0 + cfg.window_s))
        groups: list[tuple[np.ndarray, float | None]] = []
        comps = [widx[g] for g in graph_groups(z[widx], bw[widx], pw[widx], cfg)]
        for g in merge_similar_groups(comps, z, bw, pw, cfg):
            groups += split_group(np.sort(g), toa, tod, fc, bw, cfg, self.use_sdif)
        active = [t for t in self.tracks.values() if self.w - t.last_win <= cfg.max_missed_windows]
        cost = np.full((len(groups), len(active)), 1e6)
        ovm = np.zeros((len(groups), len(active)))
        for i, (g, p) in enumerate(groups):
            zg = _unit(z[g].mean(0))
            pg = _period(toa[g], p)
            seen = np.array([x in self.prev for x in g.tolist()], bool)
            for j, tr in enumerate(active):
                pt = tr.period or pg
                ov = (len(set(g[seen].tolist()) & tr.bursts) / seen.sum()) if seen.sum() else 0.0
                ovm[i, j] = ov
                cost[i, j] = (1 - float(zg @ tr.z)) + abs(pg - pt) / max(pg, pt) + (1 - ov)
        matched: dict[int, Track] = {}
        if cost.size:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] <= cfg.match_max_cost:
                    matched[i] = active[j]
        for i in range(len(groups)):  # extra fragments of an already-matched emitter continue its track
            if i not in matched and len(active) and ovm[i].max() >= 0.5:
                j = int(np.argmax(ovm[i]))
                if cost[i, j] <= cfg.match_max_cost:
                    matched[i] = active[j]
        newly = []
        for i, (g, p) in enumerate(groups):
            tr = matched.get(i)
            if tr is None:
                if len(g) < cfg.sdif_min_train:
                    continue
                tr = Track(self.next_id, _unit(z[g].mean(0)), p)
                self.tracks[self.next_id] = tr
                self.next_id += 1
            new = g[assign[g] < 0]
            assign[new] = tr.tid
            newly.append(new)
            tr.bursts.update(g.tolist())
            tr.z = _unit(0.7 * tr.z + 0.3 * _unit(z[g].mean(0)))
            tr.period = _period(toa[g], p) if p is not None else tr.period
            tr.last_win = self.w
        self.prev = set(widx.tolist())
        self.w += 1
        return np.concatenate(newly) if newly else np.zeros(0, int)


def associate(df: pd.DataFrame, z: np.ndarray, cfg: AssocCfg, use_sdif: bool = True) -> pd.DataFrame:
    """Assign ``track_id`` to each burst (sliding 2 s windows, 0.5 s step, Hungarian across windows)."""
    df = df.reset_index(drop=True)
    assign = np.full(len(df), -1, np.int64)
    if len(df) == 0:
        return df.assign(track_id=assign)
    toa, tod = df.toa_s.to_numpy(), df.tod_s.to_numpy()
    fc, bw, pw = df.fc_hz.to_numpy(), df.bw_hz.to_numpy(), df.pw_s.to_numpy()
    a = Associator(cfg, use_sdif)
    t0, t_end = toa.min(), toa.max()
    while True:
        a.step(toa, tod, fc, bw, pw, z, assign, t0)
        if t0 + cfg.window_s >= t_end:
            break
        t0 += cfg.step_s
    return df.assign(track_id=assign)


def released_tracks(df: pd.DataFrame, min_bursts: int) -> list[int]:
    """Tracks with at least ``min_bursts`` bursts (the ones handed to SpectrumLM)."""
    c = df[df.track_id >= 0].track_id.value_counts()
    return sorted(c[c >= min_bursts].index.tolist())


def assoc_metrics(df: pd.DataFrame, min_bursts: int = 0) -> dict[str, float]:
    """Purity, completeness and ID switches against ground-truth ``source_id`` (eval only)."""
    d = df[(df.track_id >= 0) & df.source_id.notna()]
    if min_bursts:
        d = d[d.track_id.isin(released_tracks(df, min_bursts))]
    if len(d) == 0:
        return {"purity": float("nan"), "completeness": float("nan"), "id_switches": 0, "n_tracks": 0}
    ct = pd.crosstab(d.track_id, d.source_id)
    purity = ct.max(1).sum() / ct.values.sum()
    comp = ct.max(0).sum() / ct.values.sum()
    sw = 0
    for _, g in d.sort_values("toa_s").groupby("source_id"):
        ids = g.track_id.to_numpy()
        sw += int((ids[1:] != ids[:-1]).sum())
    return {"purity": float(purity), "completeness": float(comp), "id_switches": sw, "n_tracks": int(len(ct))}
