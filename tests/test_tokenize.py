"""M3: field quantizer round trip (≤ half a bin) and VQ codebook usage (> 50 %, no collapse)."""

import numpy as np
import torch

from vimmune.config import load_config
from vimmune.sim.fastbdw import make_corpus
from vimmune.tokenize.fields import FIELDS, N_SPECIAL, Tokenizer
from vimmune.tokenize.vq import codebook_usage, encode_waves, prep_bursts, train_vq

CFG = load_config("demo")


def test_quantizer_roundtrip():
    tok = Tokenizer(CFG)
    rng = np.random.default_rng(0)
    for name, q in tok.q.items():
        lo, hi = q.lo, q.hi
        x = 10 ** rng.uniform(np.log10(lo), np.log10(hi), 2000) if q.log else rng.uniform(lo, hi, 2000)
        y = q.decode(q.encode(x))
        err = np.abs(np.log10(y) - np.log10(x)) if q.log else np.abs(y - x)
        assert err.max() <= q.bin_width() / 2 + 1e-9, name
    assert tok.q["snr"].decode(tok.q["snr"].encode(np.array([7.0])))[0] == 7.0  # 1 dB steps
    assert tok.q["bw"].encode(np.array([1.0]))[0] == N_SPECIAL  # clamps into edge bin
    assert len(tok.vocab_sizes) == len(FIELDS) and tok.vocab_sizes[0] == 256 + 3


def test_vq_codebook_usage(tmp_path):
    torch.manual_seed(0)
    df = make_corpus(CFG, ["P1", "P4", "P7", "P8"], tmp_path, n_sessions=3, workers=4)
    W = prep_bursts(df, tmp_path / "bursts", CFG.tokenizer, workers=4)
    assert W.shape[1] == 1024 and np.allclose(np.mean(np.abs(W[:, :100]) ** 2), 1, atol=0.3)
    model, hist = train_vq(W, CFG.tokenizer, steps=300)
    assert np.mean(hist[-30:]) < np.mean(hist[:30])
    z, codes = encode_waves(model, W)
    assert z.shape == (len(W), 64)
    assert codebook_usage(codes, CFG.tokenizer.codebook) > 0.5
