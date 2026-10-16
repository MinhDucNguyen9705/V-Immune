"""Fast BDW mode: ground-truth bursts -> detector-like measured BDW + short burst IQ snippets.

Implements report §2.1 (spec §4.5). Skips full-band IQ synthesis. The measurement model mimics the
reference detector: t-f collisions merge into the stronger burst, missed bursts at low SNR, TOA jitter,
PW/BW/FC errors growing as 1/sqrt(SNR), occasional splits and merges. Snippets are generated directly at
the detector's output rate (1.25·BW, centered on the *measured* FC) so the tokenizer path is identical.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vimmune.config import Config, NoiseModelCfg
from vimmune.io.bdw import SessionMeta, make_bdw
from vimmune.io.iq import BurstStore
from vimmune.sim import channel as chan
from vimmune.sim.scene import EmitterCfg, SceneCfg, build_truth, burst_rng, wave_kwargs
from vimmune.sim.waveforms import noise_burst, synth


def resolve_collisions(truth: pd.DataFrame) -> pd.DataFrame:
    """Bursts overlapping in time *and* frequency become one blob, as connected components would.

    The weaker burst disappears; if it is within 10 dB of the stronger one the survivor's extent becomes
    the union bounding box and the label is that of the larger-area burst (as ``label_from_truth`` does for
    the real detector). Adds column ``merged``.
    """
    df = truth.sort_values("toa_s").reset_index(drop=True).copy()
    df["merged"] = False
    alive = np.ones(len(df), bool)
    toa, tod = np.array(df.toa_s), np.array(df.toa_s + df.pw_s)
    lo, hi = np.array(df.fc_off_hz - df.bw_hz / 2), np.array(df.fc_off_hz + df.bw_hz / 2)
    snr = np.array(df.snr_db)
    j0 = 0
    for i in range(len(df)):
        while tod[j0] < toa[i] - 0.003 and j0 < i:
            j0 += 1
        for j in range(j0, i):
            if not (alive[i] and alive[j]) or tod[j] <= toa[i] or hi[j] <= lo[i] or lo[j] >= hi[i]:
                continue
            s, w = (i, j) if snr[i] >= snr[j] else (j, i)
            if abs(snr[s] - snr[w]) < 10:  # union blob: labelled like the detector, by the larger t-f area
                area = lambda k: (tod[k] - toa[k]) * (hi[k] - lo[k])  # noqa: E731
                s, w = (s, w) if area(s) >= area(w) else (w, s)
            alive[w] = False
            if abs(snr[s] - snr[w]) < 10:
                toa[s], tod[s] = min(toa[s], toa[w]), max(tod[s], tod[w])
                lo[s], hi[s] = min(lo[s], lo[w]), max(hi[s], hi[w])
                df.loc[s, "merged"] = True
    df["toa_s"], df["pw_s"] = toa, tod - toa
    df["fc_off_hz"], df["bw_hz"] = (lo + hi) / 2, hi - lo
    return df[alive].reset_index(drop=True)


def apply_noise_model(df: pd.DataFrame, nm: NoiseModelCfg, rng: np.random.Generator) -> pd.DataFrame:
    """Measured BDW from ground truth (columns ``*_true`` keep the truth)."""
    snr_lin = 10 ** (df.snr_db.to_numpy() / 10)
    p_miss = 1 / (1 + np.exp((df.snr_db.to_numpy() - nm.miss_snr50_db) / nm.miss_slope_db))
    df = df[rng.uniform(size=len(df)) >= p_miss].copy()
    snr_lin = 10 ** (df.snr_db.to_numpy() / 10)
    g = 1 / np.sqrt(np.maximum(snr_lin, 1e-3))
    n = len(df)
    for c in ["toa_s", "pw_s", "bw_hz", "fc_off_hz", "snr_db"]:
        df[c + "_true"] = df[c].to_numpy()
    df["toa_s"] = df.toa_s + rng.normal(0, 1, n) * (nm.toa_a_s + nm.toa_b_s * g)
    df["pw_s"] = df.pw_s * np.maximum(0.2, 1 + rng.normal(0, 1, n) * (nm.pw_rel_a + nm.pw_rel_b * g))
    df["bw_hz"] = df.bw_hz * np.maximum(0.2, 1 + rng.normal(0, 1, n) * (nm.bw_rel_a + nm.bw_rel_b * g))
    df["fc_off_hz"] = df.fc_off_hz + rng.normal(0, 1, n) * (nm.fc_frac_a + nm.fc_frac_b * g) * df.bw_hz_true
    df["snr_db"] = df.snr_db + rng.normal(0, nm.snr_err_db, n)
    m = rng.uniform(size=n) < nm.merge_prob  # merge with an unseen interferer: bigger box
    df.loc[m, "pw_s"] *= rng.uniform(1.2, 2.0, m.sum())
    df.loc[m, "bw_hz"] *= rng.uniform(1.0, 3.0, m.sum())
    df.loc[m, "merged"] = True
    s = rng.uniform(size=n) < nm.split_prob
    if s.any():
        a = df[s].copy()
        u = rng.uniform(0.3, 0.7, len(a))
        b = a.copy()
        a["pw_s"] = a.pw_s * u
        b["toa_s"] = a.toa_s + a.pw_s + 20e-6
        b["pw_s"] = b.pw_s * (1 - u) - 20e-6
        df = pd.concat([df[~s], a, b])
    return df.sort_values("toa_s").reset_index(drop=True)


def make_snippet(row: pd.Series, cfg: Config, scene_seed: int, snippet_len: int, fs_band: float) -> tuple[np.ndarray, float]:
    """Burst IQ as the detector would output it: rate 1.25·BW_meas, centered on the measured FC."""
    protos = cfg.protos
    f = protos.get(row.protocol)
    fs = min(1.25 * row.bw_hz, fs_band)
    n = int(np.clip(round(row.pw_s * fs), 8, snippet_len))
    rng = burst_rng(scene_seed, int(row.src_idx), int(row.k))
    fs_gen = max(fs, 1.25 * f.bw_hz)  # generate wide enough for the true signal, then filter
    x = synth(f.modulation, int(np.ceil(n * fs_gen / fs)) + 16, fs_gen, f.bw_hz, rng, **wave_kwargs(f, protos))
    x = chan.apply_multipath(x, chan.multipath_taps(rng, fs_gen, cfg.sim.channel))
    x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)  # distortion only: the gain is already in the truth SNR
    x *= np.exp(2j * np.pi * (row.fc_off_hz_true - row.fc_off_hz) * np.arange(len(x)) / fs_gen).astype(np.complex64)
    if fs_gen != fs:
        from vimmune.dsp import resample

        x = resample(x, fs_gen, fs)
    x = x[:n]
    if row.get("merged", False):  # interferer energy inside the merged blob
        x = x + noise_burst(n, fs, 0.5 * fs, rng) * rng.uniform(0.1, 1.0)
    a = chan.amplitude_for_snr(float(row.snr_db_true), f.bw_hz, fs)
    y = a * x + chan.awgn(n, rng)
    return y.astype(np.complex64), fs


def fast_session(scene: SceneCfg, cfg: Config, store: BurstStore | None, rng_seed: int = 0,
                 snippet_len: int | None = None) -> pd.DataFrame:
    """Measured BDW (+ snippets in ``store``) for a scene, without full-band IQ."""
    truth = build_truth(scene, cfg)
    meas = apply_noise_model(resolve_collisions(truth), cfg.sim.noise_model, np.random.default_rng([scene.seed, rng_seed, 5]))
    sl = snippet_len or cfg.sim.corpus.snippet_len
    refs: list[str | None] = []
    fss: list[float | None] = []
    for _, row in meas.iterrows():
        if store is None:
            refs.append(None)
            fss.append(None)
            continue
        y, fs = make_snippet(row, cfg, scene.seed, sl, scene.fs)
        refs.append(store.put(y))
        fss.append(fs)
    sess = SessionMeta(fs=scene.fs, center_freq=scene.center_freq, environment=scene.environment,
                       session_id=scene.session_id or scene.name, node_id=scene.node_id)
    sig = pd.DataFrame({
        "toa_s": meas.toa_s, "tod_s": meas.toa_s + meas.pw_s, "pw_s": meas.pw_s,
        "fc_hz": scene.center_freq + meas.fc_off_hz, "bw_hz": meas.bw_hz, "snr_db": meas.snr_db,
        "burst_iq_ref": refs, "iq_fs": fss, "iq_fc_hz": scene.center_freq + meas.fc_off_hz,
        "protocol": meas.protocol, "source_id": meas.source_id, "signal_type": meas.protocol.map(lambda p: cfg.protos.get(p).kind),
    })
    out = make_bdw(sig, sess)
    out["distance_m"] = meas.distance_m.to_numpy()
    out["channel"] = meas.channel.to_numpy()
    out["toa_true"] = meas.toa_s_true.to_numpy()
    return out


# ------------------------------------------------------------------------------------------- corpus
def corpus_scene(fam: str, i: int, cfg: Config, background: list[str], seed: int) -> SceneCfg:
    """One corpus session: a primary emitter plus 0–2 random background emitters (known families only)."""
    from vimmune.sim.families import mean_interval

    rng = np.random.default_rng([seed, i, sum(map(ord, fam))])
    f = cfg.protos.get(fam)
    c = cfg.sim.corpus
    dur = c.bursts_per_session * mean_interval(f) * 1.25 + 0.05
    lo, hi = np.log(c.distance_m[0]), np.log(c.distance_m[1])
    em = [EmitterCfg(family=fam, distance_m=float(np.exp(rng.uniform(lo, hi))), seed=int(rng.integers(1 << 30)))]
    nb = int(rng.integers(0, 3)) if background else 0
    if f.hop == "adaptive" and "P7" not in background:  # adaptive hopping needs something to avoid
        background = [*background, "P7"]
        nb = max(nb, 1)
    if f.hop == "adaptive":
        em.append(EmitterCfg(family="P7", distance_m=float(np.exp(rng.uniform(lo, np.log(300)))), seed=int(rng.integers(1 << 30))))
        nb -= 1
    for _ in range(max(nb, 0)):
        em.append(EmitterCfg(family=str(rng.choice(background)), distance_m=float(np.exp(rng.uniform(lo, hi))),
                             seed=int(rng.integers(1 << 30))))
    return SceneCfg(name=f"{fam}-{i:04d}", fs=cfg.sim.fs, center_freq=cfg.sim.center_freq, duration_s=dur,
                    seed=int(rng.integers(1 << 30)), environment="sim-fast", session_id=f"{fam}-{i:04d}",
                    node_id="corpus", emitters=em)


def _corpus_job(args: tuple) -> pd.DataFrame:
    fam, i, cfg, background, seed, root, max_per_source = args
    scene = corpus_scene(fam, i, cfg, background, seed)
    store = BurstStore(root, prefix=f"{scene.name}_")
    df = fast_session(scene, cfg, store)
    store.flush()
    df = df.groupby("source_id", group_keys=False).apply(lambda g: g.head(max_per_source))
    return df.reset_index(drop=True)


def make_corpus(cfg: Config, families: list[str], out: str | Path, n_sessions: int | None = None,
                workers: int = 8, seed: int | None = None) -> pd.DataFrame:
    """Simulate a fast-BDW corpus over ``families`` (background drawn from the known P7/P8 only)."""
    out = Path(out)
    (out / "bursts").mkdir(parents=True, exist_ok=True)
    background = [f for f in families if cfg.protos.get(f).kind == "background"]
    ns = n_sessions or cfg.sim.corpus.sessions_per_family
    s = cfg.seed if seed is None else seed
    jobs = [(fam, i, cfg, background, s, out / "bursts", cfg.sim.corpus.bursts_per_session)
            for fam in families for i in range(ns)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_corpus_job, jobs, chunksize=4))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.protocol.isin(families)].reset_index(drop=True)
    return df
