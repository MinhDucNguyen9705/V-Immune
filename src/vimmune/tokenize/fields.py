"""Field quantizers: each burst becomes 6 discrete tokens (ΔTOA, FC offset, BW, PW, SNR, waveform code).

Implements report §2.2. Every field vocabulary reserves [PAD]=0, [MASK]=1, [BOS]=2; bins start at 3.
Out-of-range values clamp into the edge bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vimmune.config import Config

PAD, MASK, BOS = 0, 1, 2
N_SPECIAL = 3
FIELDS = ["dtoa", "fc", "bw", "pw", "snr", "wcode"]


@dataclass(frozen=True)
class FieldQuantizer:
    """Uniform bins on a linear or log10 scale over [lo, hi]."""

    name: str
    n_bins: int
    lo: float
    hi: float
    log: bool = False

    @property
    def vocab(self) -> int:
        return self.n_bins + N_SPECIAL

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(x, 1e-300)) if self.log else np.asarray(x, float)

    def encode(self, x: np.ndarray) -> np.ndarray:
        a, b = self._f(np.array(self.lo)), self._f(np.array(self.hi))
        u = (self._f(np.asarray(x, float)) - a) / (b - a)
        return (np.clip(np.floor(u * self.n_bins), 0, self.n_bins - 1).astype(np.int64) + N_SPECIAL)

    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Bin centers (specials decode to NaN)."""
        a, b = self._f(np.array(self.lo)), self._f(np.array(self.hi))
        k = np.asarray(ids) - N_SPECIAL
        v = a + (k + 0.5) / self.n_bins * (b - a)
        v = 10**v if self.log else v
        return np.where(k >= 0, v, np.nan)

    def bin_width(self) -> float:
        """Bin width in the quantizer's scale (log10 units for log fields)."""
        return float((self._f(np.array(self.hi)) - self._f(np.array(self.lo))) / self.n_bins)


class Tokenizer:
    """The six field quantizers of a profile (the waveform code comes from the VQ-VAE)."""

    def __init__(self, cfg: Config):
        t, fs = cfg.tokenizer, cfg.sim.fs
        self.q = {
            "dtoa": FieldQuantizer("dtoa", t.dtoa_bins, *t.dtoa_range_s, log=True),
            "fc": FieldQuantizer("fc", t.fc_bins, -0.5, 0.5),
            "bw": FieldQuantizer("bw", t.bw_bins, t.bw_min_hz, fs, log=True),
            "pw": FieldQuantizer("pw", t.pw_bins, *t.pw_range_s, log=True),
            # 1 dB steps centered on integers: [-15, 40] -> 56 bins
            "snr": FieldQuantizer("snr", t.snr_bins, t.snr_range_db[0] - 0.5, t.snr_range_db[1] + 0.5),
        }
        self.n_codes = t.codebook

    @property
    def vocab_sizes(self) -> list[int]:
        return [self.q[f].vocab for f in FIELDS[:5]] + [self.n_codes + N_SPECIAL]

    @property
    def n_bins(self) -> list[int]:
        return [self.q[f].n_bins for f in FIELDS[:5]] + [self.n_codes]

    def encode_track(self, df: pd.DataFrame, codes: np.ndarray, first_dtoa: float | None = None) -> np.ndarray:
        """(N, 6) token ids for one source's bursts sorted by TOA. The first ΔTOA is PAD unless given."""
        toa = df.toa_s.to_numpy()
        dtoa = np.diff(toa, prepend=toa[0] - (first_dtoa if first_dtoa is not None else 0.0))
        tok = np.stack([
            self.q["dtoa"].encode(dtoa),
            self.q["fc"].encode(df.fc_off_norm.to_numpy()),
            self.q["bw"].encode(df.bw_hz.to_numpy()),
            self.q["pw"].encode(df.pw_s.to_numpy()),
            self.q["snr"].encode(df.snr_db.to_numpy()),
            np.asarray(codes, np.int64) + N_SPECIAL,
        ], axis=1)
        if first_dtoa is None and len(tok):
            tok[0, 0] = PAD
        return tok
