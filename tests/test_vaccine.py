"""M8: adapter training + validation gate (passes a trained adapter, blocks a deliberately broken one),
waveform classifier, quality filter, synthetic sequence grammar, adapter file round trip."""

import numpy as np
import pytest
import torch

from toy import CFG, grammar_tracks, make_world
from vimmune.lm.model import LoRALinear
from vimmune.openset.scorer import OpenSetScorer
from vimmune.pipeline import fit_calibration
from vimmune.sim.waveforms import gfsk, ofdm
from vimmune.vaccine.adapter import adapted_model, build_adapter, evaluate_gate, load_adapter, save_adapter
from vimmune.vaccine.synth import classify_waveform


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_gate_passes_trained_and_blocks_broken(world):
    m, mu, calib, tr = world
    syn, held = grammar_tracks("C", 10, 7), grammar_tracks("C", 4, 8)
    base_sc = OpenSetScorer(m, ["A", "B"], mu, calib)
    before = base_sc.score(held)
    assert before.unknown.mean() > 0.5  # the new grammar is flagged before vaccination
    model, mu_ext, _ = build_adapter(m, ["A", "B"], syn, tr["train"], "C", CFG, 0)
    cal_ad, _ = fit_calibration(CFG, model, ["A", "B", "C"], mu_ext, tr, fixed=calib, extra_class=("C", syn))
    good = evaluate_gate(base_sc, OpenSetScorer(model, ["A", "B", "C"], mu_ext, cal_ad), held, tr["test"], tr["test"][6:], "C", CFG.vaccine.gate, True)
    assert good["checks"]["recall_heldout_real"]["value"] >= 0.9
    assert good["passed"]

    broken, _ = adapted_model(m, CFG.vaccine.lora_rank)
    with torch.no_grad():
        for mod in broken.modules():
            if isinstance(mod, LoRALinear):
                mod.B.normal_(0, 3.0)
    bad = evaluate_gate(base_sc, OpenSetScorer(broken, ["A", "B", "C"], mu_ext, cal_ad), held, tr["test"], tr["test"][6:], "C", CFG.vaccine.gate, True)
    assert not bad["passed"]
    no_approval = evaluate_gate(base_sc, OpenSetScorer(model, ["A", "B", "C"], mu_ext, cal_ad), held, tr["test"], tr["test"][6:], "C", CFG.vaccine.gate, False)
    assert not no_approval["passed"] and no_approval["automatic_pass"]


def test_adapter_file_roundtrip(world, tmp_path):
    m, mu, _, _ = world
    model, _ = adapted_model(m, CFG.vaccine.lora_rank)
    size = save_adapter(tmp_path / "a.pt", model, mu[0], "C", {"rank": CFG.vaccine.lora_rank, "n_new": 1})
    assert size < 200 * 1024
    m2, obj = load_adapter(m, tmp_path / "a.pt")
    assert obj["name"] == "C" and m2.cls.out_features == 3


def test_waveform_classifier():
    rng = np.random.default_rng(0)
    g = np.stack([gfsk(1024, 1.25e6, 1e6, rng, 0.5, 0.5, 0.91) for _ in range(10)])
    o = np.stack([ofdm(1024, 12.5e6, 10e6, rng, 15e3) for _ in range(10)])
    assert classify_waveform(g) == "gfsk-like" and classify_waveform(o) == "ofdm-like"
