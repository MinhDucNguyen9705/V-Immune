"""M5: SpectrumLM pretraining loss decreases; classifier head trains; time encoding; LoRA wrapper."""

import numpy as np
import torch

from vimmune.config import load_config
from vimmune.lm.data import Track, sample_batch, split_sessions
from vimmune.lm.model import LoRALinear, SpectrumLM, TimeEncoding, n_params
from vimmune.lm.train import embed_windows, finetune_classifier, pretrain

CFG = load_config("demo", {"lm": {"d_model": 32, "n_layers": 2, "n_heads": 2, "ctx": 32, "batch": 16, "warmup": 10, "lr": 3e-3, "ft_lr": 1e-3}})
V = [259, 515, 131, 131, 59, 515]


def toy_tracks(n=24, L=200, seed=0):
    """Two synthetic 'grammars': periodic hopping vs fixed channel."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lab = "A" if i % 2 else "B"
        tok = np.zeros((L, 6), np.int64)
        tok[:, 0] = 100 if lab == "A" else 150
        tok[:, 1] = (3 + (np.arange(L) * 7) % 16) if lab == "A" else 200
        tok[:, 2], tok[:, 3] = (40, 50) if lab == "A" else (60, 70)
        tok[:, 4] = rng.integers(20, 30, L)
        tok[:, 5] = 10 if lab == "A" else 20
        out.append(Track(f"s{i}/0", tok, np.cumsum(np.full(L, 7e-3)), np.full(L, 10.0), lab, f"s{i}", np.arange(L)))
    return out


def test_pretrain_loss_decreases_and_classifier():
    torch.manual_seed(0)
    tr = toy_tracks()
    m = SpectrumLM(V, CFG.lm, 2)
    h = pretrain(m, tr[:20], tr[20:], CFG.lm, 0, steps=150)
    assert np.mean(h["loss"][-10:]) < 0.7 * np.mean(h["loss"][:5])
    assert h["val_loss"][-1] < h["val_loss"][0]
    finetune_classifier(m, tr[:20], ["A", "B"], CFG.lm, 0, steps=30)
    _, logits, lab = embed_windows(m, tr[20:], 32, 16)
    assert np.mean(np.array(["A", "B"])[logits.argmax(1)] == np.array(lab)) == 1.0


def test_full_profile_size():
    full = load_config("full")
    assert 4e6 < n_params(SpectrumLM(V, full.lm, 6)) < 7e6


def test_mask_and_split():
    rng = np.random.default_rng(0)
    b = sample_batch(toy_tracks(4), 8, 32, rng, mask_prob=0.15)
    frac = (b["inp"][:, 1:] == 1).float().mean().item()
    assert 0.08 < frac < 0.22
    sp = split_sessions([f"s{i}" for i in range(100)], (0.7, 0.1, 0.1, 0.1), 0)
    assert sum(v == "train" for v in sp.values()) == 70


def test_time_encoding_and_lora():
    te = TimeEncoding(16, 32, (10e-6, 10.0))
    a, b = te(torch.tensor([[0.0, 1e-5]])), te(torch.tensor([[0.0, 2e-5]]))
    assert not torch.allclose(a[0, 1], b[0, 1])
    lin = torch.nn.Linear(8, 8)
    lo = LoRALinear(lin, 4)
    x = torch.randn(3, 8)
    assert torch.allclose(lo(x), lin(x))  # B initialised to zero
