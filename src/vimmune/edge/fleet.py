"""Fleet / federation simulation: 3 edge-node processes with different backgrounds and SNR profiles.

Implements report §2.8 (hot-load in the field, federation). Each node simulates its own scene (fast BDW
mode), associates and scores tracks with its current model state, streams compact upstream records
(BDW + int8 embedding) and calibration statistics. A release dropped into the node folders mid-run is
hot-loaded without restarting the process. The aggregator runs FedAvg over the calibration statistics.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vimmune.config import CONFIG_DIR, Config


def load_nodes(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or CONFIG_DIR / "nodes.yaml").read_text())["nodes"]


def node_main(spec: dict, cfg: Config, paths: dict, out: Path, new_family: str) -> None:
    """Body of one node process."""
    import torch

    torch.set_num_threads(2)
    from vimmune.associate.assoc import associate
    from vimmune.edge.node import EdgeNode, NodeState, calib_stats, pack_burst, score_stream
    from vimmune.io.iq import BurstStore
    from vimmune.lm.data import build_tracks
    from vimmune.pipeline import encode_bdw, load_lm, load_scorer
    from vimmune.sim.fastbdw import fast_session
    from vimmune.sim.scene import EmitterCfg, SceneCfg
    from vimmune.tokenize.fields import Tokenizer

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["seed"])
    lo, hi = spec["distance_m"]
    em = [EmitterCfg(family=f, distance_m=float(rng.uniform(lo, hi)), seed=int(rng.integers(1 << 30))) for f in spec["background"]]
    em += [EmitterCfg(family=new_family, distance_m=float(rng.uniform(lo, hi)), seed=int(rng.integers(1 << 30)))]
    em += [EmitterCfg(family="P1", distance_m=float(rng.uniform(lo, hi)), seed=int(rng.integers(1 << 30)))]
    sc = SceneCfg(name=spec["node_id"], duration_s=spec["duration_s"], seed=spec["seed"], environment=spec["environment"],
                  session_id=spec["node_id"], node_id=spec["node_id"], emitters=em)
    store = BurstStore(out / "bursts")
    df = fast_session(sc, cfg, store)
    store.flush()
    z, codes = encode_bdw(cfg, df, out / "bursts", Path(paths["tok"]), workers=1)
    df = associate(df, z, cfg.assoc)
    tracks = [t for t in build_tracks(df, codes, Tokenizer(cfg), by="track_id") if len(t) >= cfg.assoc.min_track_bursts]
    base, _ = load_lm(Path(paths["lm"]))
    sc0 = load_scorer(Path(paths["lm"]), Path(paths["calib"]))
    node = EdgeNode(spec["node_id"], base, NodeState("1.0.0", sc0), out / "releases", Path(paths["pub"]), out / "events.jsonl",
                    cfg.edge.watch_interval_s)
    node.log({"event": "start", "pid": os.getpid(), "version": "1.0.0"})
    node.watch()
    up = 0
    for i, r in enumerate(df.itertuples()):  # upstream: BDW + int8 embedding per burst
        up += len(pack_burst(r.toa_s, r.pw_s, r.fc_hz, r.bw_hz, r.snr_db, int(r.track_id), z[i]))
    prog = out / "progress.json"
    pace = spec.get("run_s", 6.0) / max(1, sum(len(t) for t in tracks))  # equal wall time per node: the drop lands mid-run everywhere
    ev = score_stream(node, tracks, cfg.edge.rescore_every, pace,
                      progress=lambda f: prog.write_text(json.dumps({"progress": f, "version": node.state.version})))
    time.sleep(2 * cfg.edge.watch_interval_s)
    node.stop()
    dec = pd.DataFrame(ev)
    dec.to_parquet(out / "decisions.parquet")
    known = dec[(~dec.unknown) & (dec.version == "1.0.0")] if len(dec) else dec
    stats = calib_stats(known.S.to_numpy(), known.snr.to_numpy()) if len(known) else None
    stats_bytes = len(json.dumps(stats)) if stats else 0
    summary = {"node_id": spec["node_id"], "pid": os.getpid(), "n_bursts": len(df), "n_tracks": len(tracks),
               "processed_bursts": int(sum(len(t) for t in tracks)), "scoring_events": len(dec),
               "upstream_bytes": up + stats_bytes, "bytes_per_burst": up / max(1, len(df)), "duration_s": spec["duration_s"],
               "upstream_bytes_per_s": (up + stats_bytes) / spec["duration_s"], "final_version": node.state.version,
               "hot_loads": [e for e in node.events if e["event"] == "hot_load"], "calib_stats": stats}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


def run_fleet(cfg: Config, paths: dict, release: Path, out: Path, new_family: str, nodes: list[dict] | None = None,
              drop_at: float = 0.4, timeout_s: float = 600) -> dict:
    """Start the node processes, drop ``release`` into every node's folder once all are ``drop_at`` through
    their streams, wait, then FedAvg the calibration statistics."""
    from vimmune.edge.node import fedavg_stats

    nodes = nodes or load_nodes()
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # fork after torch thread pools start can deadlock the children
    procs = []
    for spec in nodes:
        d = out / spec["node_id"]
        if d.exists():
            shutil.rmtree(d)
        p = ctx.Process(target=node_main, args=(spec, cfg, paths, d, new_family))
        p.start()
        procs.append((spec, p, d))
    t0 = time.time()
    dropped = False
    while any(p.is_alive() for _, p, _ in procs) and time.time() - t0 < timeout_s:
        prog = [json.loads((d / "progress.json").read_text())["progress"] if (d / "progress.json").exists() else 0.0 for _, _, d in procs]
        if not dropped and min(prog) >= drop_at:
            for _, _, d in procs:
                (d / "releases").mkdir(parents=True, exist_ok=True)
                shutil.copy(release, d / "releases" / release.name)
            dropped = True
            t_drop = time.time()
        time.sleep(0.1)
    for _, p, _ in procs:
        p.join(timeout=5)
    summaries = [json.loads((d / "summary.json").read_text()) for _, _, d in procs if (d / "summary.json").exists()]
    if len(summaries) != len(procs):
        raise RuntimeError(f"{len(procs) - len(summaries)} fleet node(s) failed; see {out}")
    fed = fedavg_stats([s["calib_stats"] for s in summaries if s["calib_stats"]])
    raw_bps = {f"{fs / 1e6:.2f} MHz int16": fs * 2 * 2 for fs in (cfg.sim.fs, 245.76e6)}
    rep = {"nodes": [{k: v for k, v in s.items() if k != "calib_stats"} for s in summaries], "release": release.name,
           "dropped_after_s": (t_drop - t0) if dropped else None, "fedavg": fed,
           "raw_iq_bytes_per_s": raw_bps,
           "upstream_vs_raw": {n: {k: s["upstream_bytes_per_s"] / v for k, v in raw_bps.items()} for n, s in
                               ((s["node_id"], s) for s in summaries)}}
    (out / "fleet_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep
