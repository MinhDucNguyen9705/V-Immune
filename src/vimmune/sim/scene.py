"""Scenes: a band + duration + emitters; ground-truth BDW and streaming full-band IQ synthesis.

Implements report §2.1 (spec §4.4). IQ is rendered chunk by chunk (default 100 ms) in worker
processes that write disjoint slices of a pre-allocated memory-mapped SigMF ``ci16_le`` file.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from vimmune.config import Config, Family, ProtocolsCfg
from vimmune.io.bdw import SessionMeta, make_bdw, write_bdw
from vimmune.io.iq import write_sigmf_meta
from vimmune.sim import channel as chan
from vimmune.sim.families import schedule
from vimmune.sim.waveforms import synth


class EmitterCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    distance_m: float
    start_s: float = 0.0
    stop_s: float | None = None
    seed: int = 0
    channel: int | None = None
    source_id: str | None = None


class SceneCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "scene"
    fs: float = 30.72e6
    center_freq: float = 2.44e9
    duration_s: float = 6.0
    seed: int = 0
    environment: str = "sim"
    session_id: str | None = None
    node_id: str = "node-0"
    emitters: list[EmitterCfg]


def load_scene(path: str | Path) -> SceneCfg:
    return SceneCfg.model_validate(yaml.safe_load(Path(path).read_text()))


def burst_rng(scene_seed: int, src_idx: int, k: int) -> np.random.Generator:
    return np.random.default_rng([scene_seed % 2**32, src_idx, k])


def wave_kwargs(f: Family, protos: ProtocolsCfg) -> dict[str, float]:
    return {"bt": protos.gfsk_bt, "h": protos.gfsk_h, "bw_per_rate": protos.gfsk_bw_per_rate,
            "scs_hz": f.ofdm_scs_hz, "cp": f.ofdm_cp}


def build_truth(scene: SceneCfg, cfg: Config) -> pd.DataFrame:
    """Ground-truth bursts of every emitter (adaptive emitters scheduled last, against the others)."""
    protos = cfg.protos
    order = sorted(range(len(scene.emitters)), key=lambda i: protos.get(scene.emitters[i].family).hop == "adaptive")
    parts: list[pd.DataFrame] = []
    for i in order:
        e = scene.emitters[i]
        f = protos.get(e.family)
        rng = np.random.default_rng([scene.seed % 2**32, i, e.seed])
        bg = pd.concat(parts).sort_values("toa_s") if parts and f.hop == "adaptive" else None
        sid = e.source_id or f"{scene.name}.e{i}.{e.family}"
        df = schedule(f, protos, e.start_s, e.stop_s or scene.duration_s, rng, sid, scene.center_freq, e.channel, bg)
        base_snr = chan.snr_from_distance(e.distance_m, cfg.sim.channel)
        df["snr_db"] = base_snr + rng.normal(0, 0.5, len(df))  # small per-burst fading
        df["distance_m"] = e.distance_m
        df["src_idx"] = i
        df["k"] = np.arange(len(df))
        parts.append(df)
    truth = pd.concat(parts).sort_values("toa_s", kind="stable").reset_index(drop=True)
    truth["tod_s"] = truth["toa_s"] + truth["pw_s"]
    truth["fc_hz"] = scene.center_freq + truth["fc_off_hz"]
    truth["signal_type"] = truth["protocol"].map(lambda p: protos.get(p).kind)
    return truth


def truth_bdw(scene: SceneCfg, truth: pd.DataFrame) -> pd.DataFrame:
    sess = SessionMeta(fs=scene.fs, center_freq=scene.center_freq, environment=scene.environment,
                       session_id=scene.session_id or scene.name, node_id=scene.node_id)
    sig = truth[["toa_s", "tod_s", "pw_s", "fc_hz", "bw_hz", "snr_db", "protocol", "source_id", "signal_type"]].copy()
    sig["fc_off_norm"] = 0.0
    out = make_bdw(sig, sess)
    out["distance_m"] = truth["distance_m"].to_numpy()
    out["channel"] = truth["channel"].to_numpy()
    return out


def render_burst(f: Family, protos: ProtocolsCfg, row: pd.Series, fs: float, scene_seed: int,
                 cfg: Config) -> np.ndarray:
    """Waveform of one burst at band rate ``fs`` (amplitude for its in-band SNR, multipath, 1 µs ramps)."""
    n = max(1, int(round(row.pw_s * fs)))
    rng = burst_rng(scene_seed, int(row.src_idx), int(row.k))
    x = synth(f.modulation, n, fs, f.bw_hz, rng, **wave_kwargs(f, protos))
    h = chan.multipath_taps(np.random.default_rng([scene_seed % 2**32, int(row.src_idx), 10**6]), fs, cfg.sim.channel)
    x = chan.apply_multipath(x, h)
    r = min(n // 2, max(1, int(1e-6 * fs)))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    x[:r] *= ramp
    x[n - r :] *= ramp[::-1]
    return x * chan.amplitude_for_snr(row.snr_db, f.bw_hz, fs)


def _render_chunk(args: tuple) -> None:
    path, c0, n, bursts, scene_seed, fs, cfg, scale = args
    protos = cfg.protos
    rng = np.random.default_rng([scene_seed % 2**32, 77, c0])
    y = chan.awgn(n, rng)
    for _, row in bursts.iterrows():
        f = protos.get(row.protocol)
        x = render_burst(f, protos, row, fs, scene_seed, cfg)
        s0 = int(round(row.toa_s * fs))
        t = (s0 + np.arange(len(x))) / fs
        x = x * np.exp(2j * np.pi * row.fc_off_hz * t).astype(np.complex64)
        a, b = max(s0, c0), min(s0 + len(x), c0 + n)
        if b > a:
            y[a - c0 : b - c0] += x[a - s0 : b - s0]
    mm = np.memmap(path, dtype="<i2", mode="r+")
    inter = np.empty(2 * n, np.float32)
    inter[0::2], inter[1::2] = y.real, y.imag
    mm[2 * c0 : 2 * (c0 + n)] = np.clip(np.rint(inter * scale), -32768, 32767).astype("<i2")
    mm.flush()


def render_iq(scene: SceneCfg, truth: pd.DataFrame, cfg: Config, out_dir: Path, workers: int = 8) -> Path:
    """Stream the full-band IQ to ``out_dir/iq.sigmf-data`` (ci16_le) and write the SigMF meta."""
    fs = scene.fs
    N = int(round(scene.duration_s * fs))
    data = out_dir / "iq.sigmf-data"
    np.memmap(data, dtype="<i2", mode="w+", shape=(2 * N,)).flush()
    csz = int(round(cfg.sim.chunk_s * fs))
    jobs = []
    for c0 in range(0, N, csz):
        n = min(csz, N - c0)
        t0, t1 = c0 / fs, (c0 + n) / fs
        sel = truth[(truth.toa_s < t1) & (truth.tod_s > t0)]
        jobs.append((data, c0, n, sel, scene.seed, fs, cfg, cfg.sim.iq_scale))
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_render_chunk, jobs))
    write_sigmf_meta(out_dir / "iq.sigmf-meta", "ci16_le", fs, scene.center_freq, cfg.sim.iq_scale,
                     {"vimmune:scene": scene.name})
    return out_dir / "iq.sigmf-meta"


def simulate_scene(scene: SceneCfg, cfg: Config, out_dir: str | Path, render: bool = True, workers: int = 8) -> pd.DataFrame:
    """Write ``truth.parquet`` (ground-truth BDW with ``source_id``) and, if ``render``, the IQ."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = build_truth(scene, cfg)
    write_bdw(truth_bdw(scene, truth), out / "truth.parquet")
    truth.to_parquet(out / "truth_full.parquet")
    (out / "scene.json").write_text(json.dumps(scene.model_dump(), indent=1))
    if render:
        render_iq(scene, truth, cfg, out, workers)
    return truth
