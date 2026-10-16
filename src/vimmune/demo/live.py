"""Live-scene processing for the demo: full-IQ scene -> detector -> tokens -> association -> SpectrumLM.

Implements report §2.10 (demo, live scenes). Simulator ground truth is attached to detections *only*
for display and evaluation (``truth_*`` columns); no decision uses it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vimmune.associate.assoc import assoc_metrics, associate, released_tracks
from vimmune.config import Config
from vimmune.detect.detector import ReferenceDetector, label_from_truth
from vimmune.detect.metrics import isolated, match
from vimmune.io.bdw import SessionMeta, read_bdw, write_bdw
from vimmune.io.iq import BurstStore, read_sigmf
from vimmune.lm.data import Track, build_tracks
from vimmune.openset.scorer import OpenSetScorer
from vimmune.pipeline import encode_bdw, write_json
from vimmune.sim.scene import SceneCfg, simulate_scene
from vimmune.tokenize.fields import Tokenizer


def simulate_and_detect(cfg: Config, scene: SceneCfg, out: Path, tok_dir: Path) -> dict:
    """Render the scene, run the reference detector, encode waveforms, associate. Writes bdw.parquet + z.npy."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    truth = simulate_scene(scene, cfg, out, workers=cfg.detect.workers)
    t_sim = time.time() - t0
    t0 = time.time()
    sess = SessionMeta(fs=scene.fs, center_freq=scene.center_freq, environment=scene.environment,
                       session_id=scene.session_id or scene.name, node_id=scene.node_id)
    det = ReferenceDetector(cfg.detect).detect(read_sigmf(out / "iq.sigmf-meta"), BurstStore(out / "bursts"), sess)
    t_det = time.time() - t0
    truth_t = truth.assign(fc_hz=scene.center_freq + truth.fc_off_hz)
    lab = label_from_truth(det, truth_t)
    det = det.assign(truth_protocol=lab.protocol, truth_source=lab.source_id)  # display/eval only; protocol stays None
    z, codes = encode_bdw(cfg, det, out / "bursts", tok_dir)
    det["wcode"] = codes
    t0 = time.time()
    det = associate(det, z, cfg.assoc)
    t_assoc = time.time() - t0
    np.save(out / "z.npy", z)
    write_bdw(det, out / "bdw.parquet")
    save_waterfall(out)
    m = match(det, truth_t[isolated(truth_t)])
    am = assoc_metrics(det.assign(source_id=det.truth_source), cfg.assoc.min_track_bursts)
    info = {"n_truth": len(truth), "n_detected": len(det), "recall_isolated_snr_ge_0": float(m[m.snr_db >= 0].hit.mean()),
            "association": am, "sim_s": t_sim, "detect_s": t_det, "assoc_s": t_assoc,
            "released_tracks": len(released_tracks(det, cfg.assoc.min_track_bursts))}
    write_json(out / "detect_info.json", info)
    return info


def save_waterfall(out: Path, seconds: float = 0.4, nfft: int = 512, max_frames: int = 1200) -> None:
    """Downsampled spectrogram (dB) of the first ``seconds`` of the scene for display."""
    from vimmune.detect.detector import stft_power

    src = read_sigmf(out / "iq.sigmf-meta")
    n = int(seconds * src.fs)
    hop = max(nfft, n // max_frames)
    S = stft_power(src.read(0, n), nfft, hop)
    S = S.reshape(len(S), 128, nfft // 128).mean(2)  # 128 frequency rows
    np.savez_compressed(out / "waterfall.npz", db=(10 * np.log10(S + 1e-12)).astype(np.float32),
                        t=(np.arange(len(S)) * hop + nfft / 2) / src.fs, f=(np.arange(128) - 64 + 0.5) * src.fs / 128,
                        center=np.array([src.center_freq]))


def scene_tracks(cfg: Config, out: Path) -> tuple[pd.DataFrame, list[Track]]:
    det = read_bdw(out / "bdw.parquet").reset_index(drop=True)
    keep = released_tracks(det, cfg.assoc.min_track_bursts)
    d = det[det.track_id.isin(keep)]
    tracks = build_tracks(d.reset_index(drop=True), det.wcode.to_numpy()[d.index], Tokenizer(cfg), by="track_id", label_col="truth_protocol")
    for t in tracks:  # rows index the filtered frame; map them back to rows of ``det``
        t.rows = d.index.to_numpy()[t.rows]
    return det, tracks


def score_scene(sc: OpenSetScorer, tracks: list[Track], det: pd.DataFrame, out: Path, tag: str) -> pd.DataFrame:
    """Per-track summary (+ per-window table and CUSUM traces) with one scorer."""
    rows, traces = [], {}
    wins = []
    for t in tracks:
        w = sc.score([t])
        cu = sc.cusum_track(t)
        wins.append(w.assign(track=t.key))
        g = det.loc[t.rows]
        first_known = w[(~w.unknown)]
        last = w.iloc[-1] if len(w) else None
        rows.append({
            "track": t.key, "track_id": int(t.key.split("/")[-1]), "n_bursts": len(t),
            "truth_family": g.truth_protocol.mode().iloc[0] if g.truth_protocol.notna().any() else None,
            "truth_purity": float((g.truth_protocol == g.truth_protocol.mode().iloc[0]).mean()) if g.truth_protocol.notna().any() else None,
            "bw_mhz": float(g.bw_hz.median() / 1e6), "pw_ms": float(g.pw_s.median() * 1e3), "snr_db": float(g.snr_db.median()),
            "decision": ("UNKNOWN" if last is not None and bool(w.unknown.mean() > 0.5) else (last.pred if last is not None else None)),
            "frac_windows_unknown": float(w.unknown.mean()) if len(w) else None, "pred_last": last.pred if last is not None else None,
            "zS_median": float(w.zS.median()) if len(w) else None, "U_median": float(w.U.median()) if len(w) else None,
            "p_min": float(w.p.min()) if len(w) else None, "p_median": float(w.p.median()) if len(w) else None,
            "cusum_alarm": cu["first_alarm_idx"] is not None, "cusum_ttd_s": cu["ttd_s"],
            "first_known_decision_s": float(first_known.t_end.iloc[0] - t.toa[0]) if len(first_known) else None,
            "t_start": float(t.toa[0]),
        })
        traces[f"{t.key}|t"] = t.toa - t.toa[0]
        traces[f"{t.key}|g"] = cu["g"]
        traces[f"{t.key}|ell"] = cu["ell"]
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"tracks_{tag}.parquet")
    (pd.concat(wins) if wins else pd.DataFrame()).to_parquet(out / f"windows_{tag}.parquet")
    np.savez_compressed(out / f"cusum_{tag}.npz", h=np.array([sc.cal.h]), kappa=np.array([sc.cal.kappa]), **traces)
    (out / f"tracks_{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    return df
