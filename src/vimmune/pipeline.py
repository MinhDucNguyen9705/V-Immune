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
def fit_calibration(cfg: Config, model: SpectrumLM, classes: list[str], mu: np.ndarray, tr: dict[str, list[Track]],
                    fixed=None, extra_class: tuple[str, list[Track]] | None = None, score_kw: dict | None = None,
                    no_snr: bool = False, weights: np.ndarray | None = None):
    """Fit the open-set calibration on corpus cal/val tracks.

    With ``fixed`` (a base :class:`OpenSetCalib`), α, κ and h are kept and only the score tables and
    conformal scores are refit (used after an adapter changes the model). ``extra_class`` adds per-burst
    CUSUM calibration for a new class from its (synthetic) tracks. ``score_kw`` / ``weights`` / ``no_snr`` are
    the evaluation ablations (masked fields or no context; field weights; a single flat SNR bin).
    """
    from vimmune.lm.score import burst_surprise, field_weights
    from vimmune.lm.score import score_windows as _score_windows
    from vimmune.openset.calib import OpenSetCalib, SNRCalib, choose_alpha, choose_kappa_h, distance
    from vimmune.openset.scorer import OpenSetScorer, pseudo_unknowns

    o = cfg.openset
    tok = Tokenizer(cfg)
    known = lambda ts: [t for t in ts if t.label in classes]  # noqa: E731
    cal = known(tr["cal"])
    sess = sorted({t.session for t in cal})
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(sess)
    half = set(sess[: len(sess) // 2])
    cal_a, cal_b = [t for t in cal if t.session in half], [t for t in cal if t.session not in half]
    w = field_weights(tok.n_bins) if weights is None else weights
    ctx, L = cfg.lm.ctx, o.score_len
    kw = score_kw or {}

    def fitc(v, snr):
        return SNRCalib.fit(v, np.zeros_like(snr) if no_snr else snr, o.snr_bin_db, o.smooth_bandwidth_bins)

    def score_windows(m, ts, c, n, ww):  # noqa: F811 - ablation-aware wrapper
        return _score_windows(m, ts, c, n, ww, **kw)

    wa = score_windows(model, cal_a, ctx, L, w)
    da, _ = distance(wa.h, mu)
    s_cal, d_cal = fitc(wa.S, wa.snr), fitc(da, wa.snr)
    lab_a = np.array(wa.label, dtype=object)
    s_cal_class = {c: fitc(wa.S[lab_a == c], wa.snr[lab_a == c]) for c in classes if (lab_a == c).sum() >= 5}
    if extra_class:
        wx = score_windows(model, extra_class[1], ctx, L, w)
        s_cal_class[extra_class[0]] = fitc(wx.S, wx.snr)
    bs = [burst_surprise(model, t, ctx, w) for t in cal_a]
    b_cal = fitc(np.concatenate([b[1:] for b in bs]), np.concatenate([t.snr[1:] for t in cal_a]))
    b_cal_class = {}
    groups = [(c, [i for i, t in enumerate(cal_a) if t.label == c], cal_a, bs) for c in classes]
    if extra_class:
        xs = extra_class[1]
        groups.append((extra_class[0], list(range(len(xs))), xs, [burst_surprise(model, t, ctx, w) for t in xs]))
    for c, ix, ts, ss in groups:
        if ix:
            b_cal_class[c] = fitc(np.concatenate([ss[i][1:] for i in ix]), np.concatenate([ts[i].snr[1:] for i in ix]))

    info: dict = {"n_cal_a_windows": len(wa.S)}
    if fixed is None:
        val = known(tr["val"])
        pu = pseudo_unknowns(val, max(20, len(val)), cfg.seed)
        wk, wu = score_windows(model, val, ctx, L, w), score_windows(model, pu, ctx, L, w)
        (dk, kk), (du, ku) = distance(wk.h, mu), distance(wu.h, mu)
        tmp = OpenSetCalib(w.tolist(), s_cal, d_cal, b_cal, 0.0, o.epsilon, [], 0.0, 0.0, L, ctx, s_cal_class=s_cal_class)
        zs_k = tmp.z_s(wk.S, wk.snr, [classes[i] for i in kk])
        zs_u = tmp.z_s(wu.S, wu.snr, [classes[i] for i in ku])
        alpha, a_res = choose_alpha(zs_k, d_cal.z(dk, wk.snr), zs_u, d_cal.z(du, wu.snr), o.alpha_grid)
        info.update(alpha_auroc={str(k): v for k, v in a_res.items()}, pseudo=pu)
        kappa, h = 0.0, 0.0
    else:
        alpha, kappa, h = fixed.alpha, fixed.kappa, fixed.h
    calib = OpenSetCalib(w.tolist(), s_cal, d_cal, b_cal, alpha, o.epsilon, [], kappa, h, L, ctx,
                         alpha_auroc=info.get("alpha_auroc", fixed.alpha_auroc if fixed else {}), b_cal_class=b_cal_class,
                         s_cal_class=s_cal_class,
                         ell_clip=o.ell_clip, mtbfa_h=fixed.mtbfa_h if fixed else {})
    wb = score_windows(model, cal_b, ctx, L, w)
    db, kb = distance(wb.h, mu)
    calib.cal_scores = calib.fuse(wb.S, wb.snr, db, [classes[i] for i in kb]).tolist()
    info["n_cal_b_windows"] = len(wb.S)
    if fixed is None:  # CUSUM κ/h on known data (block bootstrap of cal-B ℓ per family)
        sc = OpenSetScorer(model, classes, mu, calib)
        series: dict[str, list[np.ndarray]] = {}
        dts: dict[str, list[float]] = {}
        for t in cal_b:
            series.setdefault(t.label, []).append(sc.burst_ell(t)[1:])
            dts.setdefault(t.label, []).append((t.toa[-1] - t.toa[0]) / max(1, len(t) - 1))  # mean burst interval
        dt = {f: float(np.median(v)) for f, v in dts.items()}
        t0 = time.time()
        calib.kappa, calib.h, calib.mtbfa_h, h_table = choose_kappa_h(
            series, dt, o.cusum_kappa_grid, o.cusum_h_max, o.cusum_target_h, o.cusum_sim_streams, o.cusum_sim_hours,
            o.cusum_block, cfg.seed, o.cusum_ref_shift)
        info["cusum"] = {"kappa": calib.kappa, "h": calib.h, "mtbfa_hours": calib.mtbfa_h, "h_by_kappa": h_table,
                         "pseudo_unknown_mean_ell": float(np.median([sc.burst_ell(t)[1:].mean() for t in info["pseudo"][:20]])),
                         "design_shift": o.cusum_ref_shift,
                         "method": "block-bootstrap simulation of known-class ℓ + log-linear tail extrapolation",
                         "mean_burst_interval_s": dt, "sim_seconds": time.time() - t0,
                         "mean_ell_by_family": {f: float(np.mean(np.concatenate(v))) for f, v in series.items()}}
    return calib, info


def stage_calibrate(cfg: Config, corpus: Path, tok_dir: Path, lm_dir: Path, out: Path) -> dict:
    """Fit SNR calibration (cal-A), choose α on val + pseudo-unknowns, conformal scores (cal-B), CUSUM κ/h."""
    from vimmune.openset.calib import auroc
    from vimmune.openset.scorer import OpenSetScorer, pseudo_unknowns

    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    model, bundle = load_lm(lm_dir / "lm.pt")
    tr = corpus_tracks(cfg, corpus, tok_dir)
    classes, mu = bundle.classes, bundle.centroids
    calib, info = fit_calibration(cfg, model, classes, mu, tr)
    sc = OpenSetScorer(model, classes, mu, calib)
    known = [t for t in tr["test"] if t.label in classes]
    test = sc.score(known)
    test_pu = sc.score(pseudo_unknowns(known, 40, cfg.seed + 1))
    metrics = {"alpha": calib.alpha, "alpha_auroc": calib.alpha_auroc, "n_cal_a_windows": info["n_cal_a_windows"],
               "n_cal_b_windows": info["n_cal_b_windows"], "epsilon": calib.epsilon,
               "false_unknown_rate_test": float(test.unknown.mean()), "n_test_windows": len(test),
               "pseudo_unknown_auroc_test": auroc(test.U.to_numpy(), test_pu.U.to_numpy()),
               "pseudo_unknown_tpr_test": float(test_pu.unknown.mean()), "cusum": info["cusum"]}
    write_json(out / "calib.json", calib.to_dict())
    write_json(out / "metrics.json", metrics)
    test.to_parquet(out / "test_windows.parquet")
    return metrics


def load_scorer(lm_path: Path, calib_path: Path):
    from vimmune.openset.calib import OpenSetCalib
    from vimmune.openset.scorer import OpenSetScorer

    model, b = load_lm(lm_path)
    return OpenSetScorer(model, b.classes, b.centroids, OpenSetCalib.from_dict(read_json(calib_path)))


# ------------------------------------------------------------------------------------------- DDPM pretrain
def stage_ddpm(cfg: Config, corpus: Path, tok_dir: Path, out: Path, workers: int = 8) -> dict:
    """Pretrain the conditional waveform DDPM on known-class corpus bursts (train split)."""
    from vimmune.tokenize.vq import to_tensor
    from vimmune.vaccine.ddpm import UNet1D, make_cond, save_ddpm, train_ddpm

    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    d = cfg.vaccine.ddpm
    df = read_bdw(corpus / "bdw.parquet").reset_index(drop=True)
    z_all = np.load(tok_dir / "z.npy")
    df["row"] = np.arange(len(df))
    tr = df[df.split == "train"]
    zc = pd.DataFrame(z_all[tr.row]).groupby(tr.source_id.to_numpy() + "|" + tr.session_id.to_numpy()).transform("mean").to_numpy()
    sel = np.random.default_rng(cfg.seed).choice(len(tr), min(len(tr), d.max_train_bursts), replace=False)
    sub = tr.iloc[sel]
    W = prep_bursts(sub, corpus / "bursts", cfg.tokenizer, workers)
    X = to_tensor(W)
    cond = make_cond(zc[sel], sub.bw_hz.to_numpy(), sub.pw_s.to_numpy())
    model = UNet1D(d, cfg.tokenizer.z_dim)
    t0 = time.time()
    hist = train_ddpm(model, X, cond, d.steps, d.batch, d.lr, d.T, cfg.seed, log=_log)
    save_ddpm(model, out / "ddpm.pt")
    m = {"train_bursts": len(sub), "steps": d.steps, "train_s": time.time() - t0, "loss_first": float(np.mean(hist[:20])),
         "loss_last": float(np.mean(hist[-20:])), "params": sum(p.numel() for p in model.parameters())}
    write_json(out / "metrics.json", m)
    return m


# ------------------------------------------------------------------------------------------- edge (§2.8)
def synthetic_tracks(cfg: Config, vaccine_dir: Path) -> list[Track]:
    seqs = pd.read_parquet(vaccine_dir / "synthetic_sequences.parquet")
    return build_tracks(seqs, seqs.wcode.to_numpy(), Tokenizer(cfg))


def _eval_scorer(sc, known: list[Track], unk: list[Track]) -> dict:
    from vimmune.openset.calib import auroc, tpr_at_fpr

    k, u = sc.score(known), sc.score(unk)
    return {"accuracy": float((k.pred == k.label).mean()), "auroc": auroc(k.U.to_numpy(), u.U.to_numpy()),
            "tpr_at_fpr5": tpr_at_fpr(k.U.to_numpy(), u.U.to_numpy())}


def _eval_onnx(lm, calib, classes, mu, known: list[Track], unk: list[Track], cfg: Config) -> dict:
    from vimmune.edge.runtime import onnx_decide
    from vimmune.lm.data import scoring_windows
    from vimmune.openset.calib import auroc, tpr_at_fpr

    def run(ts):
        out = []
        for t in ts:
            for s, e in scoring_windows(t, cfg.lm.ctx, cfg.openset.score_len):
                out.append((t.label, onnx_decide(lm, t.slice(s, e), calib, classes, mu)))
        return out

    k, u = run(known), run(unk)
    Uk, Uu = np.array([d["U"] for _, d in k]), np.array([d["U"] for _, d in u])
    return {"accuracy": float(np.mean([d["pred"] == lab for lab, d in k])), "auroc": auroc(Uk, Uu), "tpr_at_fpr5": tpr_at_fpr(Uk, Uu)}


def stage_edge(cfg: Config, corpus: Path, tok_dir: Path, lm_path: Path, vaccine_dir: Path, out: Path,
               scene_meta: Path | None = None, bench_s: float | None = 2.0) -> dict:
    """Distil teacher(+adapter) -> student, export ONNX fp32 + INT8, compare, benchmark, run the async edge pipeline."""
    from vimmune.edge.distill import (
        OnnxLM,
        distill,
        export_onnx,
        quantize_int8,
        tensorrt_available,
        window_feed,
    )
    from vimmune.edge.runtime import bench_lm, run_edge_bench
    from vimmune.openset.calib import OpenSetCalib
    from vimmune.openset.scorer import OpenSetScorer, pseudo_unknowns
    from vimmune.tokenize.vq import load_vq
    from vimmune.vaccine.adapter import load_adapter

    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    base, bundle = load_lm(lm_path)
    teacher, obj = load_adapter(base, vaccine_dir / "adapter.pt")
    classes = obj["meta"]["classes"]
    new = classes[-1]
    t_cal = OpenSetCalib.from_dict(read_json(vaccine_dir / "calib_adapted.json"))
    mu_t = np.load(vaccine_dir / "mu_ext.npy")
    tr = corpus_tracks(cfg, corpus, tok_dir)
    syn = synthetic_tracks(cfg, vaccine_dir)
    train = [t for t in tr["train"] if t.label in classes] + syn
    t0 = time.time()
    student, hist = distill(teacher, train, classes, cfg, cfg.seed, log=_log)
    t_dist = time.time() - t0
    h, _, lab = embed_windows(student, [t for t in tr["train"] if t.label in classes] + syn, cfg.lm.ctx, cfg.openset.score_len, stride=cfg.lm.ctx)
    mu_s = centroids(h, lab, classes)
    s_cal, _ = fit_calibration(cfg, student, classes, mu_s, tr, fixed=t_cal, extra_class=(new, syn))
    known = [t for t in tr["test"] if t.label in bundle.classes]
    unk = pseudo_unknowns(known, 40, cfg.seed + 7)
    res = {"teacher": _eval_scorer(OpenSetScorer(teacher, classes, mu_t, t_cal), known, unk),
           "student_torch": _eval_scorer(OpenSetScorer(student, classes, mu_s, s_cal), known, unk)}
    export_onnx(student, out / "student_fp32.onnx", cfg.lm.ctx)
    feeds = window_feed(train[:: max(1, len(train) // 200)], cfg.lm.ctx, cfg.openset.score_len, cfg.edge.calib_sequences)
    quantize_int8(out / "student_fp32.onnx", out / "student_int8.onnx", feeds)
    lm32, lm8 = OnnxLM(out / "student_fp32.onnx"), OnnxLM(out / "student_int8.onnx")
    res["student_onnx_fp32"] = _eval_onnx(lm32, s_cal, classes, mu_s, known, unk, cfg)
    res["student_onnx_int8"] = _eval_onnx(lm8, s_cal, classes, mu_s, known, unk, cfg)
    bench = {"onnx_fp32": bench_lm(lm32, feeds[:200]), "onnx_int8": bench_lm(lm8, feeds[:200])}
    import torch as _t

    tt = []
    for f in feeds[:100]:
        b = {k: _t.from_numpy(v) for k, v in f.items() if not k.startswith("_")}
        t1 = time.perf_counter()
        with _t.no_grad():
            teacher(b["inp"], b["t"])
        tt.append(time.perf_counter() - t1)
    bench["teacher_torch"] = {"latency_mean_ms": 1e3 * float(np.mean(tt)), "latency_p99_ms": 1e3 * float(np.percentile(tt, 99))}
    pipeline_bench = None
    if scene_meta is not None:
        pipeline_bench = run_edge_bench(scene_meta, cfg, load_vq(tok_dir / "vq.pt", cfg.tokenizer), lm8, s_cal, classes, mu_s, bench_s)
        pd.DataFrame(pipeline_bench.pop("decisions")).to_parquet(out / "edge_decisions.parquet")
    _t.save({"state": student.state_dict(), "cfg": student.cfg.model_dump(), "classes": classes}, out / "student.pt")
    write_json(out / "calib_student.json", s_cal.to_dict())
    np.save(out / "mu_student.npy", mu_s)
    rep = {"student_params": n_params(student), "teacher_params": n_params(teacher), "distill_s": t_dist,
           "distill_loss_first": float(np.mean(hist[:10])), "distill_loss_last": float(np.mean(hist[-10:])),
           "metrics": res, "drop_vs_teacher": {k: {m: res["teacher"][m] - v[m] for m in v} for k, v in res.items() if k != "teacher"},
           "size_bytes": {"fp32": (out / "student_fp32.onnx").stat().st_size, "int8": (out / "student_int8.onnx").stat().st_size},
           "calib_sequences": len(feeds), "tensorrt": "available" if tensorrt_available() else "not installed (skipped)",
           "bench": bench, "pipeline": pipeline_bench}
    write_json(out / "edge_report.json", rep)
    return rep


def stage_release(cfg: Config, vaccine_dir: Path, out: Path, version: str = "1.1.0") -> Path:
    """Sign a .vpkg with the adapter, μ_new, card, calibration tables/conformal scores/CUSUM params."""
    from vimmune.edge.release import build_release, generate_keys

    out.mkdir(parents=True, exist_ok=True)
    keys = out / "keys"
    priv = keys / "release_ed25519.pem"
    if not priv.exists():
        generate_keys(keys)
    mu_ext = np.load(vaccine_dir / "mu_ext.npy")
    np.save(out / "mu_new.npy", mu_ext[-1])
    gate = read_json(vaccine_dir / "gate_report.json")
    if not gate["passed"]:
        raise RuntimeError("gate not passed (approval missing or a check failed): refusing to build a release")
    files = {"adapter.pt": vaccine_dir / "adapter.pt", "mu_new.npy": out / "mu_new.npy", "mu_ext.npy": vaccine_dir / "mu_ext.npy",
             "card.json": vaccine_dir / "card.json", "calib.json": vaccine_dir / "calib_adapted.json",
             "gate_report.json": vaccine_dir / "gate_report.json"}
    calib = read_json(vaccine_dir / "calib_adapted.json")
    pkg = build_release(out / f"vimmune-{version}.vpkg", files, version, priv,
                        {"cusum": {"kappa": calib["kappa"], "h": calib["h"]}, "epsilon": calib["epsilon"], "alpha": calib["alpha"]})
    write_json(out / "release.json", {"package": pkg.name, "version": version, "size_bytes": pkg.stat().st_size,
                                      "public_key": str(keys / "release_ed25519.pub.pem")})
    return pkg


def stage_fleet(cfg: Config, lm_path: Path, calib_path: Path, tok_dir: Path, release: Path, pub: Path, out: Path,
                new_family: str) -> dict:
    from vimmune.edge.fleet import run_fleet

    paths = {"lm": str(lm_path), "calib": str(calib_path), "tok": str(tok_dir), "pub": str(pub)}
    return run_fleet(cfg, paths, release, out, new_family)
