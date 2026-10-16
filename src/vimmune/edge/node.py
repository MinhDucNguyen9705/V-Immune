"""Edge node: hot-loads signed releases without restarting, scores tracks, sends compact records upstream.

Implements report §2.8 (hot-load, federation). A watcher thread polls the node's release folder; a
verified package is loaded into a *new* model state that replaces the current one in a single reference
assignment under a lock, so the scoring loop always uses a complete state and never stops. Upstream
messages are BDW + int8 waveform embeddings (~96 bytes/burst) and calibration statistics for FedAvg.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from vimmune.edge.release import ReleaseError, semver_key, verify_release
from vimmune.lm.data import Track
from vimmune.lm.model import SpectrumLM
from vimmune.openset.calib import OpenSetCalib
from vimmune.openset.scorer import OpenSetScorer


@dataclass
class NodeState:
    version: str
    scorer: OpenSetScorer
    loaded_at: float = field(default_factory=time.time)


def state_from_release(base: SpectrumLM, blobs: dict[str, bytes], manifest: dict) -> NodeState:
    from vimmune.vaccine.adapter import load_adapter

    obj = torch.load(io.BytesIO(blobs["adapter.pt"]), map_location="cpu", weights_only=False)
    model, _ = load_adapter(base, obj)
    calib = OpenSetCalib.from_dict(json.loads(blobs["calib.json"]))
    mu = np.load(io.BytesIO(blobs["mu_ext.npy"]))
    return NodeState(manifest["version"], OpenSetScorer(model, obj["meta"]["classes"], mu, calib))


class EdgeNode:
    """Holds the current model state; ``watch()`` hot-swaps it when a newer verified release appears."""

    def __init__(self, node_id: str, base: SpectrumLM, base_state: NodeState, release_dir: Path, public_key: Path,
                 log_path: Path | None = None, interval: float = 0.2):
        self.node_id, self.base, self.release_dir, self.pub = node_id, base, release_dir, public_key
        self._state = base_state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.interval = interval
        self.events: list[dict] = []
        self.log_path = log_path
        self._seen: set[str] = set()
        release_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state(self) -> NodeState:
        with self._lock:
            return self._state

    def log(self, ev: dict) -> None:
        ev = {"t": time.time(), "node": self.node_id, **ev}
        self.events.append(ev)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev) + "\n")

    def poll_once(self) -> bool:
        """Load the newest unseen valid package (if newer than the current version). Returns True on swap."""
        swapped = False
        for p in sorted(self.release_dir.glob("*.vpkg")):
            if p.name in self._seen:
                continue
            self._seen.add(p.name)
            try:
                manifest, blobs = verify_release(p, self.pub)
            except ReleaseError as e:
                self.log({"event": "rejected", "package": p.name, "reason": str(e)})
                continue
            cur = self.state.version
            if semver_key(manifest["version"]) <= semver_key(cur):
                self.log({"event": "ignored_older", "package": p.name, "version": manifest["version"]})
                continue
            t0 = time.time()
            new = state_from_release(self.base, blobs, manifest)
            with self._lock:  # atomic swap: the scoring loop sees either the old or the new state
                old, self._state = self._state, new
            self.log({"event": "hot_load", "package": p.name, "from": old.version, "to": new.version,
                      "load_s": time.time() - t0, "restart": False})
            swapped = True
        return swapped

    def watch(self) -> threading.Thread:
        def loop():
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(self.interval)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------------------------------- upstream messages
RECORD = struct.Struct("<dfdfeie")  # toa, pw, fc, bw, snr(f16), track_id, z-scale(f16)


def pack_burst(toa: float, pw: float, fc: float, bw: float, snr: float, track_id: int, z: np.ndarray) -> bytes:
    """One upstream burst record: BDW fields + int8-quantized waveform embedding (~96 bytes)."""
    s = float(np.abs(z).max() / 127 + 1e-12)
    q = np.clip(np.round(z / s), -127, 127).astype(np.int8)
    return RECORD.pack(toa, pw, fc, bw, snr, track_id, s) + q.tobytes()


def unpack_burst(b: bytes, z_dim: int = 64) -> tuple[tuple, np.ndarray]:
    head = RECORD.unpack(b[: RECORD.size])
    z = np.frombuffer(b[RECORD.size : RECORD.size + z_dim], np.int8).astype(np.float32) * head[-1]
    return head, z


def calib_stats(S: np.ndarray, snr: np.ndarray, bin_db: float = 2.0, lo: float = -16.0, hi: float = 42.0) -> dict:
    """Sufficient statistics (count, Σ, Σ²) of the window surprise per SNR bin (FedAvg payload)."""
    edges = np.arange(lo, hi + bin_db, bin_db)
    k = np.clip(np.digitize(snr, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    return {"edges": edges.tolist(), "n": np.bincount(k, minlength=n).tolist(),
            "s1": np.bincount(k, S, n).tolist(), "s2": np.bincount(k, S**2, n).tolist()}


def fedavg_stats(parts: list[dict]) -> dict:
    """FedAvg of per-node calibration statistics (count-weighted means/variances per SNR bin)."""
    n = np.sum([p["n"] for p in parts], 0)
    s1 = np.sum([p["s1"] for p in parts], 0)
    s2 = np.sum([p["s2"] for p in parts], 0)
    mu = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, s2 / np.maximum(n, 1) - mu**2, np.nan)
    return {"edges": parts[0]["edges"], "n": n.tolist(), "mu": mu.tolist(), "sigma": np.sqrt(np.maximum(var, 0)).tolist()}


def score_stream(node: EdgeNode, tracks: list[Track], rescore_every: int, pace_s: float = 0.0,
                 progress: Callable[[float], object] | None = None) -> list[dict]:
    """Replay tracks burst by burst in time order; every ``rescore_every`` new bursts of a track, score its
    latest window with the node's *current* state. Returns one decision record per scoring event."""
    events = []
    order = sorted(((tr.toa[i], ti, i) for ti, tr in enumerate(tracks) for i in range(len(tr))))
    count = [0] * len(tracks)
    for j, (_, ti, i) in enumerate(order):
        count[ti] += 1
        tr = tracks[ti]
        if count[ti] >= rescore_every and count[ti] % rescore_every == 0:
            st = node.state  # snapshot: a hot-swap during scoring cannot mix states
            sub = tr.slice(max(0, count[ti] - st.scorer.cal.ctx), count[ti])
            row = st.scorer.score([sub], stride=max(1, len(sub) - st.scorer.cal.score_len)).iloc[-1]
            events.append({"track": tr.key, "label": tr.label, "n": count[ti], "t": float(tr.toa[i]), "version": st.version,
                           "pred": row.pred, "unknown": bool(row.unknown), "U": float(row.U), "S": float(row.S), "snr": float(row.snr)})
        if pace_s:
            time.sleep(pace_s)
        if progress and j % 200 == 0:
            progress(j / len(order))
    if progress:
        progress(1.0)
    return events
