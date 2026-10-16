"""Vaccine orchestration: few real bursts + approved card -> synthetic data -> LoRA adapter -> gate report.

Implements report §2.7 end to end. The flagged track is split per session by time: the first 70 % is the
*learn* part (K real bursts are drawn from it for the waveform model and the PW/BW distributions; the card's
transition matrix and period come from it), the last 30 % is the *real held-out* part used only by the gate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from vimmune.config import Config, seed_everything
from vimmune.grammar.card import Card
from vimmune.lm.data import Track, build_tracks
from vimmune.pipeline import _log, corpus_tracks, fit_calibration, load_lm, read_json, write_json
from vimmune.tokenize.fields import Tokenizer
from vimmune.tokenize.vq import encode_waves, load_vq, prep_bursts, to_tensor
from vimmune.vaccine import synth
from vimmune.vaccine.adapter import adapted_model, evaluate_gate, new_centroid, save_adapter, train_adapter

MIN_KEEP = 20  # fewer filtered DDPM samples than this -> parametric fallback (recorded, never silent)


def split_learn_heldout(track: pd.DataFrame, learn_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per session (stratified), the first ``learn_frac`` of the bursts by time are 'learn', the rest held out."""
    L, H = [], []
    for _, g in track.sort_values("toa_s").groupby("session_id"):
        n = int(round(learn_frac * len(g)))
        L.append(g.iloc[:n])
        H.append(g.iloc[n:])
    return pd.concat(L), pd.concat(H)


def synth_waveforms(cfg: Config, mode: str, card: Card, W_learn: np.ndarray, z_learn: np.ndarray, learn_k: pd.DataFrame,
                    vq, ddpm_path: Path | None, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generated waveform pool (preprocessed domain) + its z, with filter statistics and the mode used."""
    v = cfg.vaccine
    d = v.ddpm
    info: dict = {"requested_mode": mode}
    kind = synth.classify_waveform(W_learn)
    info["waveform_kind"] = kind
    if mode == "ddpm" and ddpm_path is not None and ddpm_path.exists():
        from vimmune.vaccine.ddpm import add_lora, ddim_sample, load_ddpm, make_cond, train_ddpm

        m = load_ddpm(ddpm_path, d, cfg.tokenizer.z_dim)
        params = add_lora(m, d.lora_rank)
        zbar = np.repeat(z_learn.mean(0, keepdims=True), len(W_learn), 0)
        t0 = time.time()
        train_ddpm(m, to_tensor(W_learn), make_cond(zbar, learn_k.bw_hz.to_numpy(), learn_k.pw_s.to_numpy()), d.lora_steps,
                   min(d.batch, max(4, len(W_learn))), d.lora_lr, d.T, seed, params, log=_log)
        cond = make_cond(np.repeat(z_learn.mean(0, keepdims=True), d.n_samples, 0), np.full(d.n_samples, card.bw_hz), np.full(d.n_samples, card.pw_s))
        x = ddim_sample(m, cond, d.ddim_steps, d.T, cfg.tokenizer.wave_len, seed, d.sample_batch).numpy()
        gen = (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
        gen /= np.sqrt(np.mean(np.abs(gen) ** 2, 1, keepdims=True) + 1e-20)
        info["ddpm_s"] = time.time() - t0
        z_gen, _ = encode_waves(vq, gen)
        keep, stats = synth.quality_filter(z_gen, z_learn, gen, card, cfg)
        info["filter"] = stats
        if keep.sum() >= MIN_KEEP:
            info["mode"] = "ddpm"
            return gen[keep], z_gen[keep], info
        info["fallback_reason"] = f"only {int(keep.sum())} DDPM samples passed the quality filter (< {MIN_KEEP})"
    elif mode == "ddpm":
        info["fallback_reason"] = "no pretrained DDPM available"
    gen = synth.param_waveforms(card, kind, d.n_samples, cfg, seed)
    z_gen, _ = encode_waves(vq, gen)
    keep, stats = synth.quality_filter(z_gen, z_learn, gen, card, cfg)
    info["filter_param"] = stats
    info["mode"] = "param"
    if keep.sum() < MIN_KEEP:  # parametric samples are kept regardless; the shortfall is reported
        info["param_filter_note"] = f"{int(keep.sum())} parametric samples passed the filter; all {len(gen)} used"
        keep[:] = True
    return gen[keep], z_gen[keep], info


def run_vaccine(cfg: Config, *, track: pd.DataFrame, store: Path, card: Card, base_lm: Path, base_calib: Path, tok_dir: Path,
                corpus: Path, ddpm_path: Path | None, out: Path, k: int | None = None, waveform_mode: str | None = None,
                approve: bool = False, new_name: str = "NEW", seed: int | None = None, steps: int | None = None) -> dict:
    """Run the vaccine for one flagged track (BDW rows with ``session_id``). Writes everything to ``out``."""
    from vimmune.openset.calib import OpenSetCalib
    from vimmune.openset.scorer import OpenSetScorer

    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed if seed is None else seed
    seed_everything(seed)
    v = cfg.vaccine
    k = k or v.k
    mode = waveform_mode or v.waveform_mode
    tok = Tokenizer(cfg)
    vq = load_vq(tok_dir / "vq.pt", cfg.tokenizer)
    rng = np.random.default_rng(seed)

    # 1. learn / held-out split and the K real bursts
    learn, held = split_learn_heldout(track.reset_index(drop=True), v.learn_frac)
    k_idx = np.sort(rng.choice(len(learn), min(k, len(learn)), replace=False))
    learn_k = learn.iloc[k_idx]
    W_learn = prep_bursts(learn_k, store, cfg.tokenizer)
    z_learn, _ = encode_waves(vq, W_learn)

    # 2. waveforms
    pool, z_pool, winfo = synth_waveforms(cfg, mode, card, W_learn, z_learn, learn_k, vq, ddpm_path, seed)
    card = card.model_copy(update={"waveform": winfo["waveform_kind"]})

    # 3. sequences + channel augmentation -> waveform codes
    seqs = synth.generate_sequences(card, learn, cfg, v.n_seq, v.seq_len, seed)
    seqs = seqs.assign(pw_s=learn_k.pw_s.to_numpy()[rng.integers(0, len(learn_k), len(seqs))],
                       bw_hz=learn_k.bw_hz.to_numpy()[rng.integers(0, len(learn_k), len(seqs))])
    pick = rng.integers(0, len(pool), len(seqs))
    aug = synth.augment(pool[pick], seqs.snr_db.to_numpy(), cfg, rng, card.bw_hz)
    z_syn, codes_syn = encode_waves(vq, aug)
    seqs["session_id"] = "synthetic"
    seqs["source_id"] = seqs["source_id"].astype(str)
    seqs["protocol"] = new_name
    syn_tracks = build_tracks(seqs, codes_syn, tok)

    # 4. real held-out tracks (tokens from the scene's waveform codes; ΔTOA continues from the learn part)
    codes_track = np.asarray(track["wcode"].to_numpy(), np.int64)
    held_tracks = []
    for sess, g in held.groupby("session_id"):
        prev = learn[learn.session_id == sess].toa_s
        first = float(g.toa_s.iloc[0] - prev.iloc[-1]) if len(prev) else None
        tokens = tok.encode_track(g, codes_track[g.index], first)
        held_tracks.append(Track(f"heldout/{sess}", tokens, g.toa_s.to_numpy(), g.snr_db.to_numpy(), new_name, str(sess), g.index.to_numpy()))

    # 5. adapter (frozen SpectrumLM + LoRA on Q/V + extended head), 50/50 synthetic / replay
    base, bundle = load_lm(base_lm)
    classes = bundle.classes + [new_name]
    corp = corpus_tracks(cfg, corpus, tok_dir)
    replay = [t for t in corp["train"] if t.label in bundle.classes]
    model, params = adapted_model(base, v.lora_rank)
    t0 = time.time()
    hist = train_adapter(model, params, syn_tracks, replay, classes, cfg, seed, steps, log=_log)
    t_adapter = time.time() - t0
    mu_new = new_centroid(model, syn_tracks, cfg)
    mu_ext = np.concatenate([bundle.centroids, mu_new[None]])

    # 6. recalibrate with the adapted model (α, κ, h fixed); 7. gate on real data only
    base_cal = OpenSetCalib.from_dict(read_json(base_calib))
    cal_ad, _ = fit_calibration(cfg, model, classes, mu_ext, corp, fixed=base_cal, extra_class=(new_name, syn_tracks))
    base_sc = OpenSetScorer(base, bundle.classes, bundle.centroids, base_cal)
    ad_sc = OpenSetScorer(model, classes, mu_ext, cal_ad)
    known_test = [t for t in corp["test"] if t.label in bundle.classes]
    background = [t for t in known_test if cfg.protos.get(t.label).kind == "background"]
    gate = evaluate_gate(base_sc, ad_sc, held_tracks, known_test, background, new_name, v.gate, approve)

    meta = {"rank": v.lora_rank, "n_new": 1, "base_lm": str(base_lm), "classes": classes, "k": int(len(learn_k)),
            "waveform_mode": winfo["mode"]}
    size = save_adapter(out / "adapter.pt", model, mu_new, new_name, meta)
    write_json(out / "calib_adapted.json", cal_ad.to_dict())
    np.save(out / "mu_ext.npy", mu_ext)
    (out / "card.json").write_text(card.model_dump_json(indent=1))
    np.savez_compressed(out / "samples.npz", real=W_learn[:16], synthetic=pool[:16], augmented=aug[:16])
    seqs.to_parquet(out / "synthetic_sequences.parquet")
    report = {"new_class": new_name, "k_requested": k, "k_used": int(len(learn_k)), "n_learn_bursts": len(learn),
              "n_heldout_bursts": len(held), "waveforms": winfo, "n_synthetic_sequences": len(syn_tracks),
              "n_synthetic_bursts": int(len(seqs)), "adapter_kb": size / 1024, "adapter_train_s": t_adapter,
              "adapter_loss_first": float(np.mean(hist[:10])), "adapter_loss_last": float(np.mean(hist[-10:])),
              "gate": gate, "wall_clock_s": time.time() - t_start}
    write_json(out / "gate_report.json", gate)
    write_json(out / "vaccine_report.json", report)
    return report


def approve_gate(out: Path) -> dict:
    """Set the human-approval flag on an existing gate report (UI button / ``--approve``)."""
    g = read_json(out / "gate_report.json")
    g["checks"]["human_approval"]["pass"] = True
    g["passed"] = all(c["pass"] for c in g["checks"].values())
    write_json(out / "gate_report.json", g)
    r = read_json(out / "vaccine_report.json")
    r["gate"] = g
    write_json(out / "vaccine_report.json", r)
    return g
