"""Asynchronous edge pipeline: detector -> tokenizer -> association -> student LM (ONNX INT8).

Implements report §2.8 (edge runtime). Each stage is an asyncio task connected by queues; CPU-heavy work
runs in a thread pool. A track is re-scored every ``rescore_every`` new bursts. Reports throughput
(bursts/s of wall clock), mean/p99 decision latency (detector emission of the triggering burst -> decision)
and peak RAM.
"""

from __future__ import annotations

import asyncio
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vimmune.associate.assoc import Associator
from vimmune.config import Config
from vimmune.detect.detector import _process_block, calibrate_alpha
from vimmune.edge.distill import OnnxLM
from vimmune.io.iq import IQSource
from vimmune.lm.data import Track, to_batch, window_arrays
from vimmune.openset.calib import OpenSetCalib, distance
from vimmune.tokenize.fields import FIELDS, Tokenizer
from vimmune.tokenize.vq import encode_waves, prep_burst


def onnx_decide(lm: OnnxLM, tr: Track, calib: OpenSetCalib, classes: list[str], mu: np.ndarray) -> dict:
    """Open-set decision for the latest window of ``tr`` with the ONNX student."""
    ctx, L = calib.ctx, calib.score_len
    e = len(tr)
    s = max(0, e - ctx)
    b = to_batch([window_arrays(tr, s, e, ctx)])
    out = lm.run({"inp": b["inp"].numpy(), "t": b["t"].numpy(), "valid": b["valid"].float().numpy()})
    tgt = b["tgt"].numpy()[0]
    n = e - s
    w = np.asarray(calib.weights)
    nll = np.zeros((n, len(FIELDS)))
    for a in range(len(FIELDS)):
        lp = out[a][0, np.arange(n), tgt[:n, a]]
        nll[:, a] = np.where(tgt[:n, a] == 0, 0.0, -lp)
    S = float((nll @ w)[n - min(L, n) :].mean())
    snr = float(tr.snr[e - min(L, n) :].mean())
    d, k = distance(out[7], mu)
    nearest = [classes[int(k[0])]]
    U = float(calib.fuse(np.array([S]), np.array([snr]), d, nearest)[0])
    p = float(calib.p(np.array([U]))[0])
    return {"S": S, "U": U, "p": p, "unknown": p < calib.epsilon, "pred": classes[int(np.argmax(out[6][0]))]}


async def run_pipeline(src: IQSource, cfg: Config, vq, lm: OnnxLM, calib: OpenSetCalib, classes: list[str], mu: np.ndarray,
                       duration_s: float | None = None) -> dict:
    loop = asyncio.get_running_loop()
    d, a_cfg = cfg.detect, cfg.assoc
    alpha = calibrate_alpha(d)
    tok = Tokenizer(cfg)
    n_end = src.n_samples if duration_s is None else min(src.n_samples, int(duration_s * src.fs))
    bs = int(d.block_s * src.fs)
    q_det: asyncio.Queue = asyncio.Queue(maxsize=8)
    q_tok: asyncio.Queue = asyncio.Queue(maxsize=8)
    lat: list[float] = []
    decisions: list[dict] = []
    t_wall0 = time.time()
    n_bursts = [0]

    async def detector():
        for b0 in range(0, n_end, bs):
            blobs = await loop.run_in_executor(None, _process_block, (src, b0, min(b0 + bs, n_end), d, alpha))
            await q_det.put((time.time(), blobs, (b0 + bs) / src.fs))
        await q_det.put(None)

    async def tokenizer():
        while (item := await q_det.get()) is not None:
            t_emit, blobs, t_sig = item
            if blobs:
                waves = np.stack([prep_burst(b["iq"][0], b["iq"][1], 0.0, 0.0, b["bw_hz"], cfg.tokenizer) for b in blobs])
                z, codes = await loop.run_in_executor(None, encode_waves, vq, waves)
            else:
                z, codes = np.zeros((0, cfg.tokenizer.z_dim), np.float32), np.zeros(0, np.int64)
            await q_tok.put((t_emit, blobs, z, codes, t_sig))
        await q_tok.put(None)

    async def associate_and_score():
        assoc = Associator(a_cfg)
        rows: list[dict] = []
        Z: list[np.ndarray] = []
        codes_all: list[int] = []
        emit: list[float] = []
        assign = np.zeros(0, np.int64)
        t_next = None
        scored: dict[int, int] = {}
        while (item := await q_tok.get()) is not None:
            t_emit, blobs, z, codes, t_sig = item
            for b, zz, c in zip(blobs, z, codes):
                rows.append({k: b[k] for k in ("toa_s", "tod_s", "pw_s", "fc_off_hz", "bw_hz", "snr_db")})
                Z.append(zz)
                codes_all.append(int(c))
                emit.append(t_emit)
            n_bursts[0] = len(rows)
            assign = np.r_[assign, np.full(len(blobs), -1, np.int64)]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            toa = df.toa_s.to_numpy()
            t_next = toa.min() if t_next is None else t_next
            while t_next + a_cfg.window_s <= t_sig:  # a window is complete once signal time passes its end
                fc = src.center_freq + df.fc_off_hz.to_numpy()
                await loop.run_in_executor(None, assoc.step, toa, df.tod_s.to_numpy(), fc, df.bw_hz.to_numpy(),
                                           df.pw_s.to_numpy(), np.stack(Z), assign, t_next)
                t_next += a_cfg.step_s
                for tid in np.unique(assign[assign >= 0]):
                    idx = np.flatnonzero(assign == tid)
                    n = len(idx)
                    if n >= a_cfg.min_track_bursts and n // cfg.edge.rescore_every > scored.get(tid, 0):
                        scored[tid] = n // cfg.edge.rescore_every
                        g = df.iloc[idx].assign(fc_off_norm=lambda x: x.fc_off_hz / src.fs).sort_values("toa_s")
                        tr = Track(f"t{tid}", tok.encode_track(g, np.asarray(codes_all)[g.index]), g.toa_s.to_numpy(),
                                   g.snr_db.to_numpy(), None, "edge", g.index.to_numpy())
                        dec = await loop.run_in_executor(None, onnx_decide, lm, tr, calib, classes, mu)
                        lat.append(time.time() - max(emit[i] for i in idx))
                        decisions.append({"track": int(tid), "n": n, **dec})

    await asyncio.gather(detector(), tokenizer(), associate_and_score())
    wall = time.time() - t_wall0
    return {"signal_s": n_end / src.fs, "wall_s": wall, "bursts": n_bursts[0], "throughput_bursts_per_s": n_bursts[0] / wall,
            "realtime_factor": (n_end / src.fs) / wall, "n_decisions": len(decisions),
            "latency_mean_s": float(np.mean(lat)) if lat else None, "latency_p99_s": float(np.percentile(lat, 99)) if lat else None,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "decisions": decisions}


def bench_lm(lm: OnnxLM, feeds: list[dict[str, np.ndarray]], repeat: int = 1) -> dict:
    """Per-window inference latency of an ONNX LM (batch 1)."""
    ts = []
    for _ in range(repeat):
        for f in feeds:
            t0 = time.perf_counter()
            lm.run(f)
            ts.append(time.perf_counter() - t0)
    return {"windows": len(ts), "latency_mean_ms": 1e3 * float(np.mean(ts)), "latency_p99_ms": 1e3 * float(np.percentile(ts, 99)),
            "windows_per_s": len(ts) / max(sum(ts), 1e-9)}


def run_edge_bench(src_meta: Path, cfg: Config, vq, lm: OnnxLM, calib: OpenSetCalib, classes: list[str], mu: np.ndarray,
                   duration_s: float | None) -> dict:
    from vimmune.io.iq import read_sigmf

    return asyncio.run(run_pipeline(read_sigmf(src_meta), cfg, vq, lm, calib, classes, mu, duration_s))
