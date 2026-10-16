"""M1: simulator families match config (period, channel count) and PSD bandwidth within 10 %."""

import numpy as np
import pytest

from vimmune.config import load_config
from vimmune.dsp import bandwidth
from vimmune.io.bdw import read_bdw, to_labeling_json, validate, write_bdw
from vimmune.io.iq import BurstStore, read_sigmf
from vimmune.sim.families import schedule
from vimmune.sim.fastbdw import fast_session
from vimmune.sim.scene import EmitterCfg, SceneCfg, simulate_scene, wave_kwargs
from vimmune.sim.waveforms import synth

CFG = load_config("demo")
P = CFG.protos


@pytest.mark.parametrize("fid", ["P1", "P2", "P3", "P4", "P5", "P6"])
def test_family_period_and_channels(fid):
    f = P.get(fid)
    df = schedule(f, P, 0.0, 30.0, np.random.default_rng(0), "s", CFG.sim.center_freq)
    up = df[df.link != "dn"] if fid == "P6" else df
    assert abs(np.median(np.diff(up.toa_s)) - f.frame_s) / f.frame_s < 0.02
    if fid in ("P1", "P2", "P3"):
        assert up.channel.nunique() == f.n_channels
    if fid == "P1":  # deterministic sequence of length 16
        ch = up.channel.to_numpy()
        assert (ch[16:] == ch[:-16]).all()
    if fid == "P6":
        dn = df[df.link == "dn"]
        assert abs(np.median(dn.toa_s.to_numpy()[:50] - up.toa_s.to_numpy()[:50]) - f.pair_delay_s) < 50e-6


@pytest.mark.parametrize("fid", ["P1", "P2", "P4", "P5", "P6", "P7", "P8"])
def test_psd_bandwidth(fid):
    f = P.get(fid)
    x = synth(f.modulation, 200_000, CFG.sim.fs, f.bw_hz, np.random.default_rng(1), **wave_kwargs(f, P))
    bw = bandwidth(x, CFG.sim.fs, nfft=4096)
    assert abs(bw - f.bw_hz) / f.bw_hz < 0.10


def test_adaptive_avoids_wifi():
    wifi = schedule(P.get("P7"), P, 0, 20, np.random.default_rng(2), "w", 2.44e9, channel=2)
    p3 = schedule(P.get("P3"), P, 0, 20, np.random.default_rng(3), "a", 2.44e9, background=wifi)
    from vimmune.sim.families import _busy, channel_centers

    hits = sum(_busy(wifi, t, t + 0.6e-3, channel_centers(P.get("P3")), 1e6)[c] for t, c in zip(p3.toa_s, p3.channel))
    assert hits == 0


def test_scene_iq_and_bdw_roundtrip(tmp_path):
    sc = SceneCfg(name="t", duration_s=0.05, seed=1, emitters=[EmitterCfg(family="P1", distance_m=50)])
    truth = simulate_scene(sc, CFG, tmp_path, workers=2)
    src = read_sigmf(tmp_path / "iq.sigmf-meta")
    assert src.n_samples == int(0.05 * CFG.sim.fs)
    x = src.read(0, 1000)
    assert 0.5 < np.mean(np.abs(x) ** 2) < 2.0  # unit noise floor (plus some signal)
    df = read_bdw(tmp_path / "truth.parquet")
    validate(df)
    assert len(df) == len(truth)
    js = to_labeling_json(df)
    assert js["session"]["session_id"] == "t" and len(js["bdw"]) == len(df)
    write_bdw(df, tmp_path / "x.parquet")


def test_fast_mode_snippets(tmp_path):
    sc = SceneCfg(name="f", duration_s=0.5, seed=3, emitters=[EmitterCfg(family="P4", distance_m=50)])
    store = BurstStore(tmp_path)
    df = fast_session(sc, CFG, store)
    store.flush()
    assert len(df) > 100
    y = store.get(df.burst_iq_ref.iloc[0])
    assert len(y) <= CFG.sim.corpus.snippet_len and np.isfinite(y).all()
