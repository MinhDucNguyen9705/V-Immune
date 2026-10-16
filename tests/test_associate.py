"""M4: SDIF recovers interleaved periods; same-family emitters are separated (purity ≥ 0.9)."""

import numpy as np

from vimmune.associate.assoc import assoc_metrics, associate, released_tracks
from vimmune.associate.sdif import deinterleave
from vimmune.config import load_config
from vimmune.sim.fastbdw import fast_session
from vimmune.sim.scene import EmitterCfg, SceneCfg

CFG = load_config("demo")


def test_sdif_two_periods():
    rng = np.random.default_rng(0)
    a = 0.0013 + np.arange(280) * 7e-3 + rng.uniform(-70e-6, 70e-6, 280)
    b = 0.004 + np.arange(200) * 10e-3 + rng.uniform(-1e-4, 1e-4, 200)
    a = a[rng.uniform(size=280) > 0.05]  # some missed pulses
    toa = np.r_[a, b]
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    trains, res, periods = deinterleave(toa, CFG.assoc)
    assert sorted(round(p * 1e3) for p in periods) == [7, 10]
    for t in trains:
        assert len(set(lab[t])) == 1 and len(t) > 150
    assert len(res) < 10


def test_same_family_purity():
    sc = SceneCfg(name="a", duration_s=6, seed=4, emitters=[
        EmitterCfg(family=f, distance_m=d, seed=i) for i, (f, d) in enumerate([("P1", 100), ("P1", 150), ("P8", 100), ("P7", 200)])])
    df = fast_session(sc, CFG, None)
    rng = np.random.default_rng(0)
    fam = {p: rng.normal(size=64) for p in df.protocol.unique()}  # family-level waveform embedding
    z = np.stack([fam[p] + 0.2 * rng.normal(size=64) for p in df.protocol])
    out = associate(df, z, CFG.assoc)
    m = assoc_metrics(out, CFG.assoc.min_track_bursts)
    assert m["purity"] >= 0.9
    p1 = out[out.protocol == "P1"]
    assert p1.groupby("source_id").track_id.agg(lambda s: s.value_counts().index[0]).nunique() == 2  # two tracks
    assert len(released_tracks(out, 32)) >= 3
