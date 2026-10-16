"""Apply the calibrated open-set decision to tracks; pseudo-unknowns for choosing α.

Implements report §2.5 (decision side). Validation needs unknowns to choose α, but the hidden families
must stay unseen, so pseudo-unknown tracks are built by recombining known validation tracks: the timing
and frequency fields of one family with the BW/PW/waveform fields of another (a grammar no known family
has). See DECISIONS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vimmune.lm.data import Track
from vimmune.lm.model import SpectrumLM
from vimmune.lm.score import burst_surprise, score_windows
from vimmune.openset.calib import OpenSetCalib, cusum, distance


def pseudo_unknowns(tracks: list[Track], n: int, seed: int) -> list[Track]:
    """Recombined tracks: (ΔTOA, FC, times) from family A + (BW, PW, SNR, waveform) from family B ≠ A."""
    rng = np.random.default_rng(seed)
    labs = sorted({t.label for t in tracks if t.label})
    by = {lab: [t for t in tracks if t.label == lab] for lab in labs}
    out = []
    for i in range(n):
        a, b = rng.choice(labs, 2, replace=False)
        ta, tb = by[a][rng.integers(len(by[a]))], by[b][rng.integers(len(by[b]))]
        m = min(len(ta), len(tb))
        tok = ta.tokens[:m].copy()
        tok[:, 2:] = tb.tokens[:m, 2:]
        out.append(Track(f"pseudo{i}/{a}+{b}", tok, ta.toa[:m], tb.snr[:m], "pseudo", f"pseudo{i}", np.arange(m)))
    return out


class OpenSetScorer:
    """SpectrumLM + centroids + calibration -> per-window decisions and per-track CUSUM."""

    def __init__(self, model: SpectrumLM, classes: list[str], mu: np.ndarray, calib: OpenSetCalib):
        self.model, self.classes, self.mu, self.cal = model, classes, mu, calib

    def score(self, tracks: list[Track], stride: int | None = None, mask_fields: tuple[int, ...] = (),
              no_context: bool = False, weights: np.ndarray | None = None) -> pd.DataFrame:
        c = self.cal
        w = np.asarray(c.weights) if weights is None else weights
        ws = score_windows(self.model, tracks, c.ctx, c.score_len, w, stride, mask_fields, no_context)
        d, k = distance(ws.h, self.mu) if len(ws.h) else (np.zeros(0), np.zeros(0, int))
        zS, zd = c.s_cal.z(ws.S, ws.snr), c.d_cal.z(d, ws.snr)
        U = c.alpha * zS + (1 - c.alpha) * zd
        p = c.p(U)
        if ws.logits.size:
            pr = np.exp(ws.logits - ws.logits.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            pred = [self.classes[i] for i in pr.argmax(1)]
            conf = pr.max(1)
        else:
            pred, conf = [self.classes[i] for i in k], np.ones(len(k))
        return pd.DataFrame({"key": ws.key, "label": ws.label, "end": ws.end, "t_end": ws.t_end, "S": ws.S,
                             "snr": ws.snr, "d": d, "zS": zS, "zd": zd, "U": U, "p": p, "unknown": p < c.epsilon,
                             "pred": pred, "conf": conf, "nearest": [self.classes[i] for i in k]})

    def burst_ell(self, tr: Track, windows: pd.DataFrame | None = None) -> np.ndarray:
        """Per-burst standardized surprise ℓ_n.

        Burst n is standardized with the per-burst SNR calibration of the track's nearest known class, taken
        from the latest window ending at or before n (causal); before the first window, the global one.
        """
        c = self.cal
        s = burst_surprise(self.model, tr, c.ctx, np.asarray(c.weights))
        ell = c.b_cal.z(s, tr.snr)
        if c.b_cal_class:
            w = self.score([tr]) if windows is None else windows
            for e0, e1, cls in zip(w.end, list(w.end[1:]) + [len(tr)], w.nearest):
                if cls in c.b_cal_class:
                    ell[e0:e1] = c.b_cal_class[cls].z(s[e0:e1], tr.snr[e0:e1])
        if len(ell) and tr.tokens[0, 0] == 0:  # first burst has no ΔTOA: not a full observation
            ell[0] = 0.0
        return np.clip(ell, -c.ell_clip, c.ell_clip)

    def cusum_track(self, tr: Track) -> dict:
        ell = self.burst_ell(tr)
        g, alarms = cusum(ell, self.cal.kappa, self.cal.h, reset=False)
        ttd = float(tr.toa[alarms[0]] - tr.toa[0]) if alarms else None
        return {"ell": ell, "g": g, "alarms": alarms, "first_alarm_idx": alarms[0] if alarms else None, "ttd_s": ttd}
