"""M2: reference detector recall / FC / TOA accuracy and consistency with the fast-mode noise model."""

import numpy as np
import pytest

from vimmune.config import load_config
from vimmune.detect.detector import ReferenceDetector, label_from_truth
from vimmune.detect.metrics import error_stats, isolated, match, noise_model_stats
from vimmune.io.iq import BurstStore, read_sigmf
from vimmune.sim.scene import EmitterCfg, SceneCfg, simulate_scene

SPEC = [("P1", 100), ("P1", 250), ("P2", 200), ("P6", 150), ("P8", 120), ("P3", 250), ("P7", 150), ("P4", 180)]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = {}
    for name, taps in [("los", 1), ("mp", 4)]:
        cfg = load_config("demo", {"sim": {"channel": {"taps_min": min(taps, 2) if taps > 1 else 1, "taps_max": taps}}})
        d = tmp_path_factory.mktemp(name)
        sc = SceneCfg(name="t", duration_s=0.6, seed=8, emitters=[EmitterCfg(family=f, distance_m=m, seed=i) for i, (f, m) in enumerate(SPEC)])
        truth = simulate_scene(sc, cfg, d, workers=4)
        det = ReferenceDetector(cfg.detect).detect(read_sigmf(d / "iq.sigmf-meta"), BurstStore(d / "bursts"))
        truth["fc_hz"] = cfg.sim.center_freq + truth.fc_off_hz
        out[name] = (cfg, truth, det, match(det, truth[isolated(truth)]))
    return out


def test_recall_toa(run):
    for name in ("los", "mp"):
        _, _, _, m = run[name]
        hi = m[m.snr_db >= 0]
        assert len(hi) > 100
        assert hi.hit.mean() >= 0.9, name
        assert np.nanmedian(np.abs(hi.toa_err)) < 20e-6


def test_fc_error_los(run):
    _, _, _, m = run["los"]
    hi = m[(m.snr_db >= 0) & m.hit]
    assert np.median(np.abs(hi.fc_err) / hi.bw_hz) < 0.02


def test_noise_model_matches_detector(run):
    cfg, _, _, m = run["mp"]
    st = error_stats(m, (0, 5, 10, 40))
    for b, row in st.iterrows():
        nm = noise_model_stats(cfg.sim.noise_model, b.mid)
        assert 1 / 3 < row.toa_sd_us / nm["toa_sd_us"] < 3
        assert row.bw_rel_sd < 3 * nm["bw_rel_sd"] + 0.05
        assert row.pw_rel_sd < 3 * nm["pw_rel_sd"] + 0.02


def test_label_from_truth(run):
    _, truth, det, _ = run["los"]
    lab = label_from_truth(det, truth)
    assert lab.protocol.notna().mean() > 0.9
