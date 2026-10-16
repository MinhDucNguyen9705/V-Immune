"""M7: hop-type classes, channel-set Jaccard ≥ 0.9 at SNR ≥ 5 dB, frame-period error < 2 %, pairing, outputs."""

import pytest

from vimmune.config import load_config
from vimmune.grammar.card import card_vs_truth, induce_card, true_centers, write_card
from vimmune.sim.fastbdw import fast_session
from vimmune.sim.scene import EmitterCfg, SceneCfg

CFG = load_config("demo")


def make(fam, extra=(), dist=150, dur=60.0):
    em = [EmitterCfg(family=fam, distance_m=dist, seed=1)] + [EmitterCfg(family=f, distance_m=d, seed=9) for f, d in extra]
    df = fast_session(SceneCfg(name="g", duration_s=dur, seed=3, emitters=em), CFG, None)
    sid = sorted(df[df.protocol == fam].source_id.unique())[-1]
    tr = df[df.source_id == sid]
    others = {s: df[df.source_id == s].toa_s.values for s in df.source_id.unique() if s != sid}
    card = induce_card(tr, df[df.source_id != sid], others, CFG.grammar, CFG.assoc)
    f = CFG.protos.get(fam)
    return card, card_vs_truth(card, f, true_centers(df, sid, CFG.sim.center_freq, f)), tr


@pytest.mark.parametrize("fam,hop,adaptive", [("P1", "deterministic", False), ("P2", "pseudo-random", False),
                                              ("P3", None, True)])
def test_hop_type_channels_period(fam, hop, adaptive):
    card, ev, _ = make(fam, [("P7", 100)] if fam == "P3" else [])
    assert card.snr_db >= 5
    if hop:
        assert card.hop_type == hop
    assert ev["hop_type_ok"]
    assert card.adaptive == adaptive
    assert ev["channel_jaccard"] >= 0.9
    assert ev["frame_period_rel_err"] < 0.02
    if fam == "P1":
        assert card.hop_sequence_len == 16


def test_pairing_and_outputs(tmp_path):
    card, ev, tr = make("P6", dur=20.0)
    assert card.paired_track is not None and abs(card.pair_lag_s - 1.2e-3) < 0.1e-3
    write_card(card, tr, tmp_path, truth=ev)
    for f in ("card.json", "card.png", "card.md"):
        assert (tmp_path / f).stat().st_size > 0
