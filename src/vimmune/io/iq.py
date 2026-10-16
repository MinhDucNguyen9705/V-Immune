"""IQ file I/O: interleaved int16 / float32, SigMF, memory-mapped chunked streaming, burst snippet store.

Implements report §2.1 (spec §3.2).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DTYPES = {"ci16_le": np.int16, "cf32_le": np.float32}


def write_iq(path: str | Path, iq: np.ndarray, fmt: str = "ci16_le", scale: float = 1.0, append: bool = False) -> None:
    """Append/write complex samples as interleaved int16 (scaled, clipped) or float32."""
    inter = np.empty(2 * len(iq), dtype=np.float32)
    inter[0::2], inter[1::2] = iq.real, iq.imag
    if fmt == "ci16_le":
        out = np.clip(np.rint(inter * scale), -32768, 32767).astype("<i2")
    else:
        out = (inter * scale).astype("<f4")
    with open(path, "ab" if append else "wb") as f:
        out.tofile(f)


def open_iq(path: str | Path, fmt: str = "ci16_le") -> np.memmap:
    """Memory-map an interleaved IQ file (no data is read until sliced)."""
    return np.memmap(path, dtype=np.dtype(DTYPES[fmt]).newbyteorder("<"), mode="r")


def to_complex(raw: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64) / np.float32(scale)


@dataclass
class IQSource:
    """A memory-mapped IQ recording with its metadata."""

    data_path: Path
    fmt: str
    fs: float
    center_freq: float
    scale: float = 1.0
    meta: dict[str, Any] | None = None

    @property
    def raw(self) -> np.memmap:
        if getattr(self, "_raw", None) is None:
            self._raw = open_iq(self.data_path, self.fmt)
        return self._raw

    def __getstate__(self) -> dict[str, Any]:  # memmaps are reopened in worker processes
        return {k: v for k, v in self.__dict__.items() if k != "_raw"}

    @property
    def n_samples(self) -> int:
        return len(self.raw) // 2

    def read(self, start: int, n: int) -> np.ndarray:
        start = max(0, start)
        return to_complex(self.raw[2 * start : 2 * (start + n)], self.scale)

    def chunks(self, chunk_s: float = 0.1) -> Iterator[tuple[int, np.ndarray]]:
        """Stream (start_sample, complex chunk) pairs of ``chunk_s`` seconds (default 100 ms)."""
        n = int(round(chunk_s * self.fs))
        for s in range(0, self.n_samples, n):
            yield s, self.read(s, n)


def write_sigmf_meta(path_meta: str | Path, fmt: str, fs: float, center_freq: float, scale: float = 1.0,
                     extra: dict[str, Any] | None = None) -> None:
    meta = {
        "global": {"core:datatype": fmt, "core:sample_rate": fs, "core:version": "1.0.0",
                   "core:description": "V-IMMUNE simulated scene (fictional protocols)",
                   "vimmune:scale": scale, **(extra or {})},
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }
    Path(path_meta).write_text(json.dumps(meta, indent=1))


def read_sigmf(path_meta: str | Path) -> IQSource:
    """Open a SigMF recording (``.sigmf-meta`` + ``.sigmf-data``). Supports ci16_le and cf32_le."""
    pm = Path(path_meta)
    meta = json.loads(pm.read_text())
    g = meta["global"]
    fmt = g["core:datatype"]
    if fmt not in DTYPES:
        raise ValueError(f"unsupported SigMF datatype {fmt}")
    fc = meta.get("captures", [{}])[0].get("core:frequency", 0.0)
    return IQSource(pm.with_suffix(".sigmf-data"), fmt, float(g["core:sample_rate"]), float(fc),
                    float(g.get("vimmune:scale", 1.0)), meta)


def open_raw(path: str | Path, fmt: str, fs: float, center_freq: float, scale: float = 1.0) -> IQSource:
    """Open a headerless interleaved int16/float32 file."""
    return IQSource(Path(path), fmt, fs, center_freq, scale)


class BurstStore:
    """Chunked append-only store of complex burst snippets addressed by ``burst_iq_ref``.

    Layout: ``<root>/chunk_XXXX.npy`` holding interleaved float16 (I, Q) samples. A ref is the string
    ``"<prefix><chunk>:<offset>:<length>"`` (offset/length in complex samples). Chunks rotate at
    ``chunk_samples``; the prefix lets parallel writers share one directory.
    """

    def __init__(self, root: str | Path, chunk_samples: int = 4_000_000, prefix: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_samples = chunk_samples
        self.prefix = prefix
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._chunk = len(list(self.root.glob(f"chunk_{prefix}[0-9][0-9][0-9][0-9].npy")))
        self._cache: dict[str, np.ndarray] = {}

    def put(self, iq: np.ndarray) -> str:
        if self._buf_len + len(iq) > self.chunk_samples and self._buf:
            self.flush()
        ref = f"{self.prefix}{self._chunk:04d}:{self._buf_len}:{len(iq)}"
        self._buf.append(iq.astype(np.complex64))
        self._buf_len += len(iq)
        return ref

    def flush(self) -> None:
        if not self._buf:
            return
        x = np.concatenate(self._buf)
        inter = np.empty(2 * len(x), np.float16)
        inter[0::2], inter[1::2] = x.real, x.imag
        np.save(self.root / f"chunk_{self.prefix}{self._chunk:04d}.npy", inter)
        self._chunk += 1
        self._buf, self._buf_len = [], 0

    def get(self, ref: str) -> np.ndarray:
        c, o_, n_ = ref.rsplit(":", 2)
        o, n = int(o_), int(n_)
        if c not in self._cache:
            self._cache[c] = np.load(self.root / f"chunk_{c}.npy", mmap_mode="r")
        a = np.asarray(self._cache[c][2 * o : 2 * (o + n)], dtype=np.float32)
        return (a[0::2] + 1j * a[1::2]).astype(np.complex64)
