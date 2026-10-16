"""M8: adapter training + validation gate (passes a trained adapter, blocks a deliberately broken one),
waveform classifier, quality filter, synthetic sequence grammar, adapter file round trip."""

import numpy as np
import pytest
import torch

from vimmune.config import load_config
from vimmune.lm.data import Track
from vimmune.lm.model import LoRALinear, SpectrumLM
from vimmune.lm.train import centroids, embed_windows, finetune_classifier, pretrain
from vimmune.openset.scorer import OpenSetScorer
from vimmune.pipeline import fit_calibration
from vimmune.sim.waveforms import gfsk, ofdm
from vimmune.vaccine.adapter import (
    adapted_model,
    evaluate_gate,
    load_adapter,
    new_centroid,
    save_adapter,
    train_adapter,
)
from vimmune.vaccine.synth import classify_waveform

CFG = load_config("demo", {"lm": {"d_model": 32, "n_layers": 2, "n_heads": 2, "ctx": 32, "batch": 16, "warmup": 10,
                                  "lr": 3e-3, "ft_lr": 1e-3},
                           "openset": {"cusum_sim_hours": 0.2, "cusum_sim_streams": 20},
                           "vaccine": {"adapter_steps": 120, "adapter_batch": 16, "adapter_lr": 3e-3}})
V = [259, 515, 131, 131, 59, 515]


def grammar_tracks(lab, n, seed, L=160):
    rng = np.random.default_rng(seed)
    spec = {"A": (100, 16, 40, 50, 10, 7e-3), "B": (150, 1, 60, 70, 20, 2.5e-3), "C": (120, 20, 45, 80, 30, 8e-3)}[lab]
    out = []
    for i in range(n):
        dt, nch, bw, pw, wc, per = spec
        tok = np.zeros((L, 6), np.int64)
        tok[:, 0] = dt + rng.integers(-1, 2, L)
        tok[:, 1] = 3 + (rng.integers(0, nch, L) if lab == "C" else (np.arange(L) * 7) % nch) * 10
        tok[:, 2], tok[:, 3] = bw + rng.integers(-1, 2, L), pw + rng.integers(-1, 2, L)
        tok[:, 4] = rng.integers(20, 30, L)
        tok[:, 5] = wc
        out.append(Track(f"{lab}{seed}-{i}/0", tok, np.cumsum(np.full(L, per)), np.full(L, 10.0), lab, f"{lab}{seed}-{i}", np.arange(L)))
    return out


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    tr = {sp: grammar_tracks("A", 6, s) + grammar_tracks("B", 6, s + 50) for sp, s in [("train", 1), ("val", 2), ("cal", 3), ("test", 4)]}
    m = SpectrumLM(V, CFG.lm, 2)
    pretrain(m, tr["train"], tr["val"], CFG.lm, 0, steps=150)
    finetune_classifier(m, tr["train"], ["A", "B"], CFG.lm, 0, steps=40)
    h, _, lab = embed_windows(m, tr["train"], 32, 32, stride=32)
    mu = centroids(h, lab, ["A", "B"])
    calib, _ = fit_calibration(CFG, m, ["A", "B"], mu, tr)
    return m, mu, calib, tr


def test_gate_passes_trained_and_blocks_broken(world):
    m, mu, calib, tr = world
    syn, held = grammar_tracks("C", 10, 7), grammar_tracks("C", 4, 8)
    base_sc = OpenSetScorer(m, ["A", "B"], mu, calib)
    before = base_sc.score(held)
    assert before.unknown.mean() > 0.5  # the new grammar is flagged before vaccination
    model, params = adapted_model(m, CFG.vaccine.lora_rank)
    train_adapter(model, params, syn, tr["train"], ["A", "B", "C"], CFG, 0)
    mu_ext = np.concatenate([mu, new_centroid(model, syn, CFG)[None]])
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
