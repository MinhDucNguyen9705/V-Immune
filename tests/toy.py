"""Shared toy world for tests: three synthetic burst grammars, a tiny SpectrumLM and its calibration."""

import numpy as np
import torch

from vimmune.config import load_config
from vimmune.lm.data import Track
from vimmune.lm.model import SpectrumLM
from vimmune.lm.train import centroids, embed_windows, finetune_classifier, pretrain
from vimmune.pipeline import fit_calibration

CFG = load_config("demo", {"lm": {"d_model": 32, "n_layers": 2, "n_heads": 2, "ctx": 32, "batch": 16, "warmup": 10,
                                  "lr": 3e-3, "ft_lr": 1e-3},
                           "openset": {"cusum_sim_hours": 0.2, "cusum_sim_streams": 20},
                           "vaccine": {"adapter_steps": 120, "adapter_batch": 16, "adapter_lr": 3e-3}})
V = [259, 515, 131, 131, 59, 515]


def grammar_tracks(lab, n, seed, L=160):
    rng = np.random.default_rng(seed)
    dt, nch, bw, pw, wc, per = {"A": (100, 16, 40, 50, 10, 7e-3), "B": (150, 1, 60, 70, 20, 2.5e-3), "C": (120, 20, 45, 80, 30, 8e-3)}[lab]
    out = []
    for i in range(n):
        tok = np.zeros((L, 6), np.int64)
        tok[:, 0] = dt + rng.integers(-1, 2, L)
        tok[:, 1] = 3 + (rng.integers(0, nch, L) if lab == "C" else (np.arange(L) * 7) % nch) * 10
        tok[:, 2], tok[:, 3] = bw + rng.integers(-1, 2, L), pw + rng.integers(-1, 2, L)
        tok[:, 4] = rng.integers(20, 30, L)
        tok[:, 5] = wc
        out.append(Track(f"{lab}{seed}-{i}/0", tok, np.cumsum(np.full(L, per)), np.full(L, 10.0), lab, f"{lab}{seed}-{i}", np.arange(L)))
    return out


def make_world():
    torch.manual_seed(0)
    tr = {sp: grammar_tracks("A", 6, s) + grammar_tracks("B", 6, s + 50) for sp, s in [("train", 1), ("val", 2), ("cal", 3), ("test", 4)]}
    m = SpectrumLM(V, CFG.lm, 2)
    pretrain(m, tr["train"], tr["val"], CFG.lm, 0, steps=150)
    finetune_classifier(m, tr["train"], ["A", "B"], CFG.lm, 0, steps=40)
    h, _, lab = embed_windows(m, tr["train"], 32, 32, stride=32)
    mu = centroids(h, lab, ["A", "B"])
    calib, _ = fit_calibration(CFG, m, ["A", "B"], mu, tr)
    return m, mu, calib, tr
