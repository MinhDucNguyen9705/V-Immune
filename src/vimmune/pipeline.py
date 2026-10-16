"""Training/inference stages shared by the CLI, the demo orchestrator and the evaluation suite.

Each stage reads its inputs from and writes its outputs to a directory under ``runs/``; nothing is passed
in memory between stages, so every reported number is backed by a file. Covers report §2.1–2.5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from vimmune.config import Config, get_device, seed_everything
from vimmune.io.bdw import read_bdw, write_bdw
from vimmune.lm.data import Track, build_tracks, split_sessions
from vimmune.lm.model import SpectrumLM, n_params
from vimmune.lm.train import LMBundle, centroids, embed_windows, finetune_classifier, pretrain
from vimmune.tokenize.fields import Tokenizer
from vimmune.tokenize.vq import encode_waves, load_vq, prep_bursts, save_vq, train_vq


def _log(msg: str) -> None:
    from rich.console import Console

    Console(stderr=True).log(msg)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------- corpus
def stage_corpus(cfg: Config, families: list[str], out: Path, workers: int = 8) -> pd.DataFrame:
    from vimmune.sim.fastbdw import make_corpus

    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    df = make_corpus(cfg, families, out, workers=workers)
    sessions = df.drop_duplicates("session_id")
    split = split_sessions(sessions.session_id.tolist(), cfg.sim.corpus.split, cfg.seed,
                           strata=[s.split("-")[0] for s in sessions.session_id])
    df["split"] = df.session_id.map(split)
    write_bdw(df, out / "bdw.parquet")
    write_json(out / "meta.json", {"families": families, "n_bursts": len(df), "n_sessions": len(sessions),
                                   "split_counts": df.drop_duplicates("session_id").split.value_counts().to_dict()})
    return df


# ------------------------------------------------------------------------------------------- tokenizer
def stage_tokenizer(cfg: Config, corpus: Path, out: Path, workers: int = 8) -> None:
    """Train the VQ-VAE on corpus *train* bursts, then encode every corpus burst (z, code)."""
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    df = read_bdw(corpus / "bdw.parquet")
    t = cfg.tokenizer
    tr = df[df.split == "train"]
    tr = tr.sample(min(len(tr), t.max_train_bursts), random_state=cfg.seed)
    W = prep_bursts(tr, corpus / "bursts", t, workers)
    t0 = time.time()
    model, hist = train_vq(W, t, cfg.seed, log=_log)
    save_vq(model, out / "vq.pt")
    z, codes = encode_waves(model, prep_bursts(df, corpus / "bursts", t, workers))
    np.save(out / "z.npy", z)
    np.save(out / "codes.npy", codes)
    from vimmune.tokenize.vq import codebook_usage

    write_json(out / "metrics.json", {"train_bursts": len(W), "train_s": time.time() - t0, "loss_first": float(np.mean(hist[:20])),
                                      "loss_last": float(np.mean(hist[-20:])), "codebook_usage": codebook_usage(codes, t.codebook)})
    write_json(out / "history.json", hist)


def encode_bdw(cfg: Config, df: pd.DataFrame, store: Path, tok_dir: Path, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(z, codes) for any BDW frame with snippets in ``store`` using the trained VQ-VAE."""
    model = load_vq(tok_dir / "vq.pt", cfg.tokenizer)
    return encode_waves(model, prep_bursts(df, store, cfg.tokenizer, workers))


def corpus_tracks(cfg: Config, corpus: Path, tok_dir: Path) -> dict[str, list[Track]]:
    """Tokenized corpus tracks by split (idealized association: one track per simulated source)."""
    df = read_bdw(corpus / "bdw.parquet")
    codes = np.load(tok_dir / "codes.npy")
    tok = Tokenizer(cfg)
    out: dict[str, list[Track]] = {}
    for sp in ["train", "val", "cal", "test"]:
        m = (df.split == sp).to_numpy()
        out[sp] = build_tracks(df[m], codes[m], tok)
    return out


# ------------------------------------------------------------------------------------------- SpectrumLM
def stage_lm(cfg: Config, corpus: Path, tok_dir: Path, out: Path, classes: list[str]) -> LMBundle:
    """Pretrain SpectrumLM, fine-tune the classifier head, compute centroids, report known-class accuracy."""
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dev = get_device(cfg)
    tr = corpus_tracks(cfg, corpus, tok_dir)
    tok = Tokenizer(cfg)
    model = SpectrumLM(tok.vocab_sizes, cfg.lm, len(classes))
    t0 = time.time()
    hist = pretrain(model, tr["train"], tr["val"], cfg.lm, cfg.seed, dev, log=_log)
    t_pre = time.time() - t0
    ft = finetune_classifier(model, tr["train"], classes, cfg.lm, cfg.seed, dev, log=_log)
    model.to("cpu")
    h, _, lab = embed_windows(model, tr["train"], cfg.lm.ctx, cfg.openset.score_len, stride=cfg.lm.ctx)
    mu = centroids(h, lab, classes)
    acc = {}
    for sp in ["val", "test"]:
        _, logits, lab = embed_windows(model, [t for t in tr[sp] if t.label in classes], cfg.lm.ctx, cfg.openset.score_len)
        pred = [classes[i] for i in logits.argmax(1)]
        acc[sp] = float(np.mean([p == y for p, y in zip(pred, lab)])) if lab else float("nan")
        cm = pd.crosstab(pd.Series(lab, name="true"), pd.Series(pred, name="pred"))
        cm.to_csv(out / f"confusion_{sp}.csv")
    bundle = LMBundle(tok.vocab_sizes, cfg.lm, classes, model.state_dict(), mu, {"n_params": n_params(model)})
    bundle.save(out / "lm.pt")
    write_json(out / "history.json", {**hist, "ft_ce": ft})
    write_json(out / "metrics.json", {"n_params": n_params(model), "pretrain_s": t_pre, "train_loss_first": float(np.mean(hist["loss"][:20])),
                                      "train_loss_last": float(np.mean(hist["loss"][-20:])), "val_loss": hist["val_loss"],
                                      "classifier_acc": acc, "classes": classes})
    return bundle


def load_lm(path: Path) -> tuple[SpectrumLM, LMBundle]:
    b = LMBundle.load(path)
    return b.build(), b


# ------------------------------------------------------------------------------------------- open set
def stage_calibrate(cfg: Config, corpus: Path, tok_dir: Path, lm_dir: Path, out: Path) -> dict:
    """Fit SNR calibration (cal-A), choose α on val + pseudo-unknowns, conformal scores (cal-B), CUSUM κ/h."""
    from vimmune.lm.score import burst_surprise, field_weights, score_windows
    from vimmune.openset.calib import OpenSetCalib, SNRCalib, choose_alpha, choose_kappa_h, distance
    from vimmune.openset.scorer import OpenSetScorer, pseudo_unknowns

    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    o = cfg.openset
    model, bundle = load_lm(lm_dir / "lm.pt")
    tok = Tokenizer(cfg)
    tr = corpus_tracks(cfg, corpus, tok_dir)
    classes, mu = bundle.classes, bundle.centroids
    known = lambda ts: [t for t in ts if t.label in classes]  # noqa: E731
    cal = known(tr["cal"])
    sess = sorted({t.session for t in cal})
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(sess)
    half = set(sess[: len(sess) // 2])
    cal_a, cal_b = [t for t in cal if t.session in half], [t for t in cal if t.session not in half]
    w = field_weights(tok.n_bins)
    ctx, L = cfg.lm.ctx, o.score_len

    wa = score_windows(model, cal_a, ctx, L, w)
    da, _ = distance(wa.h, mu)
    s_cal = SNRCalib.fit(wa.S, wa.snr, o.snr_bin_db, o.smooth_bandwidth_bins)
    d_cal = SNRCalib.fit(da, wa.snr, o.snr_bin_db, o.smooth_bandwidth_bins)
    bs = [burst_surprise(model, t, ctx, w) for t in cal_a]
    b_cal = SNRCalib.fit(np.concatenate([b[1:] for b in bs]), np.concatenate([t.snr[1:] for t in cal_a]), o.snr_bin_db, o.smooth_bandwidth_bins)
    b_cal_class = {}
    for c in classes:
        ix = [i for i, t in enumerate(cal_a) if t.label == c]
        if ix:
            b_cal_class[c] = SNRCalib.fit(np.concatenate([bs[i][1:] for i in ix]), np.concatenate([cal_a[i].snr[1:] for i in ix]),
                                          o.snr_bin_db, o.smooth_bandwidth_bins)

    # α on validation: known windows vs pseudo-unknown windows
    val = known(tr["val"])
    pu = pseudo_unknowns(val, max(20, len(val)), cfg.seed)
    wk, wu = score_windows(model, val, ctx, L, w), score_windows(model, pu, ctx, L, w)
    dk, du = distance(wk.h, mu)[0], distance(wu.h, mu)[0]
    alpha, a_res = choose_alpha(s_cal.z(wk.S, wk.snr), d_cal.z(dk, wk.snr), s_cal.z(wu.S, wu.snr), d_cal.z(du, wu.snr), o.alpha_grid)

    calib = OpenSetCalib(w.tolist(), s_cal, d_cal, b_cal, alpha, o.epsilon, [], 0.0, 0.0, L, ctx,
                         alpha_auroc={str(k): v for k, v in a_res.items()}, b_cal_class=b_cal_class, ell_clip=o.ell_clip)
    wb = score_windows(model, cal_b, ctx, L, w)
    calib.cal_scores = calib.fuse(wb.S, wb.snr, distance(wb.h, mu)[0]).tolist()

    # CUSUM: block-bootstrap per-burst ℓ of cal-B tracks per family; h for MTBFA ≥ target
    sc = OpenSetScorer(model, classes, mu, calib)
    series: dict[str, list[np.ndarray]] = {}
    dts: dict[str, list[float]] = {}
    for t in cal_b:
        series.setdefault(t.label, []).append(sc.burst_ell(t)[1:])
        dts.setdefault(t.label, []).append((t.toa[-1] - t.toa[0]) / max(1, len(t) - 1))  # mean burst interval
    dt = {f: float(np.median(v)) for f, v in dts.items()}
    pu_shift = float(np.median([sc.burst_ell(t)[1:].mean() for t in pu[:20]]))  # typical unknown drift
    t0 = time.time()
    calib.kappa, calib.h, calib.mtbfa_h, h_table = choose_kappa_h(series, dt, o.cusum_kappa_grid, o.cusum_h_max, o.cusum_target_h,
                                                                  o.cusum_sim_streams, o.cusum_sim_hours, o.cusum_block, cfg.seed, o.cusum_ref_shift)
    t_cusum = time.time() - t0

    # held-out known test: empirical false-unknown rate (should be ≤ ε)
    test = sc.score(known(tr["test"]))
    test_pu = sc.score(pseudo_unknowns(known(tr["test"]), 40, cfg.seed + 1))
    from vimmune.openset.calib import auroc

    metrics = {"alpha": alpha, "alpha_auroc": a_res, "n_cal_a_windows": len(wa.S), "n_cal_b_windows": len(wb.S),
               "epsilon": o.epsilon, "false_unknown_rate_test": float(test.unknown.mean()), "n_test_windows": len(test),
               "pseudo_unknown_auroc_test": auroc(test.U.to_numpy(), test_pu.U.to_numpy()),
               "pseudo_unknown_tpr_test": float(test_pu.unknown.mean()),
               "cusum": {"kappa": calib.kappa, "h": calib.h, "mtbfa_hours": calib.mtbfa_h, "h_by_kappa": h_table,
                         "pseudo_unknown_mean_ell": pu_shift, "design_shift": o.cusum_ref_shift,
                         "method": "block-bootstrap simulation of known-class ℓ + log-linear tail extrapolation",
                         "mean_burst_interval_s": dt, "sim_seconds": t_cusum,
                         "mean_ell_by_family": {f: float(np.mean(np.concatenate(v))) for f, v in series.items()}}}
    write_json(out / "calib.json", calib.to_dict())
    write_json(out / "metrics.json", metrics)
    test.to_parquet(out / "test_windows.parquet")
    return metrics


def load_scorer(lm_path: Path, calib_path: Path):
    from vimmune.openset.calib import OpenSetCalib
    from vimmune.openset.scorer import OpenSetScorer

    model, b = load_lm(lm_path)
    return OpenSetScorer(model, b.classes, b.centroids, OpenSetCalib.from_dict(read_json(calib_path)))
