"""Leave-classes-out evaluation: open-set metrics, ablations, blob-level baselines, card metrics, K-shot curve.

Implements report §2.9. Every fold hides 2–3 families from *all* training (tokenizer/VQ, SpectrumLM,
diffusion pretraining, calibration); they are the unknowns at test time.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vimmune.config import Config, seed_everything
from vimmune.io.bdw import read_bdw
from vimmune.lm.data import Track, build_tracks
from vimmune.openset.calib import auroc, tpr_at_fpr
from vimmune.openset.scorer import OpenSetScorer
from vimmune.pipeline import (
    _log,
    corpus_tracks,
    encode_bdw,
    fit_calibration,
    load_lm,
    load_scorer,
    read_json,
    stage_calibrate,
    stage_corpus,
    stage_ddpm,
    stage_lm,
    stage_tokenizer,
    write_json,
)
from vimmune.tokenize.fields import FIELDS, Tokenizer


def window_ttd(scored: pd.DataFrame, tracks: list[Track], thr: float, col: str = "U") -> list[float | None]:
    """Seconds from a track's first burst to its first window whose score exceeds ``thr`` (None if never)."""
    start = {t.key: float(t.toa[0]) for t in tracks}
    out = []
    for key, g in scored.groupby("key", sort=False):
        hit = g[g[col] > thr].sort_values("end")
        out.append(float(hit.t_end.iloc[0] - start[key]) if len(hit) else None)
    return out


def summarize(known: np.ndarray, unk: np.ndarray, ttd: list[float | None]) -> dict:
    det = [t for t in ttd if t is not None]
    return {"auroc": auroc(known, unk), "tpr_at_fpr5": tpr_at_fpr(known, unk),
            "ttd_s_median": float(np.median(det)) if det else None, "detected_tracks": f"{len(det)}/{len(ttd)}"}


def hidden_tracks(cfg: Config, fold: Path, hidden: list[str], n_sessions: int) -> tuple[list[Track], pd.DataFrame, np.ndarray]:
    from vimmune.sim.fastbdw import make_corpus

    d = fold / "hidden"
    if not (d / "bdw.parquet").exists():
        df = make_corpus(cfg, hidden, d, n_sessions, seed=cfg.seed + 999)
        from vimmune.io.bdw import write_bdw

        write_bdw(df, d / "bdw.parquet")
    df = read_bdw(d / "bdw.parquet").reset_index(drop=True)
    z, codes = encode_bdw(cfg, df, d / "bursts", fold / "tok")
    return build_tracks(df, codes, Tokenizer(cfg)), df, codes


def run_baselines(cfg: Config, fold: Path, known_tracks: list[Track], hid: list[Track], hid_df: pd.DataFrame, classes: list[str],
                  score_len: int, ctx: int) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]:
    """Blob CNN trained on known train bursts; per-burst scores averaged over the same scoring windows."""
    from vimmune.eval.baselines import OpenMax, baseline_scores, burst_features, cnn_outputs, train_cnn
    from vimmune.lm.data import scoring_windows
    from vimmune.tokenize.vq import prep_bursts

    corp = read_bdw(fold / "corpus" / "bdw.parquet").reset_index(drop=True)
    tr = corp[(corp.split == "train") & corp.protocol.isin(classes)].sample(n=min(6000, int((corp.split == "train").sum())), random_state=cfg.seed)
    W = prep_bursts(tr, fold / "corpus" / "bursts", cfg.tokenizer)
    img, sc = burst_features(W, tr.bw_hz.to_numpy(), tr.pw_s.to_numpy())
    y = np.array([classes.index(p) for p in tr.protocol])
    cnn = train_cnn(img, sc, y, len(classes), 400, cfg.seed)
    lg_tr, f_tr = cnn_outputs(cnn, img, sc)
    om = OpenMax().fit(lg_tr, y)
    fn = f_tr / np.linalg.norm(f_tr, axis=1, keepdims=True)
    cen = np.stack([fn[y == c].mean(0) for c in range(len(classes))])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)

    def per_burst(df: pd.DataFrame, store: Path) -> dict[str, np.ndarray]:
        Wd = prep_bursts(df, store, cfg.tokenizer)
        i2, s2 = burst_features(Wd, df.bw_hz.to_numpy(), df.pw_s.to_numpy())
        lg, f = cnn_outputs(cnn, i2, s2)
        return baseline_scores(lg, f, cen, om)

    test_rows = corp[(corp.split == "test")]
    kb = per_burst(test_rows, fold / "corpus" / "bursts")
    pos_test = {r: i for i, r in enumerate(test_rows.index.to_numpy())}
    test_tracks = build_tracks(corp[corp.split == "test"], np.zeros(len(corp), np.int64)[corp.split.to_numpy() == "test"], Tokenizer(cfg))
    test_tracks = [t for t in test_tracks if t.label in classes]
    hb = per_burst(hid_df, fold / "hidden" / "bursts")

    def windows(tracks: list[Track], scores: dict[str, np.ndarray], rowmap: dict[int, int] | None, rows_index: np.ndarray | None):
        recs = []
        for t in tracks:
            for _, e in scoring_windows(t, ctx, score_len):
                rows = t.rows[e - score_len : e]
                if rowmap is not None and rows_index is not None:
                    rows = np.array([rowmap[int(rows_index[r])] for r in rows])
                recs.append({"key": t.key, "end": e, "t_end": float(t.toa[e - 1]), **{m: float(v[rows].mean()) for m, v in scores.items()}})
        return pd.DataFrame(recs)

    test_sub = corp[corp.split == "test"].reset_index()
    kw = windows(test_tracks, kb, pos_test, test_sub["index"].to_numpy())
    uw = windows(hid, hb, None, None)
    out = {}
    for m in kb:
        out[m] = (kw[m].to_numpy(), uw[m].to_numpy(), kw.rename(columns={m: "U"}), uw.rename(columns={m: "U"}))
    return out


def card_metrics(cfg: Config, hidden: list[str], dur: float = 20.0) -> list[dict]:
    from vimmune.grammar.card import card_vs_truth, induce_card, true_centers
    from vimmune.sim.fastbdw import fast_session
    from vimmune.sim.scene import EmitterCfg, SceneCfg

    out = []
    for i, fam in enumerate(hidden):
        em = [EmitterCfg(family=fam, distance_m=150, seed=11 + i), EmitterCfg(family="P7", distance_m=100, seed=5),
              EmitterCfg(family="P8", distance_m=120, seed=6)]
        df = fast_session(SceneCfg(name="card", duration_s=dur, seed=cfg.seed + i, emitters=em), cfg, None)
        for sid in sorted(df[df.protocol == fam].source_id.unique()):
            tr = df[df.source_id == sid]
            others = {s: df[df.source_id == s].toa_s.values for s in df.source_id.unique() if s != sid}
            card = induce_card(tr, df[df.source_id != sid], others, cfg.grammar, cfg.assoc, track_id=sid)
            f = cfg.protos.get(fam)
            ev = card_vs_truth(card, f, true_centers(df, sid, cfg.sim.center_freq, f))
            out.append({"family": fam, "source": sid, "hop_inferred": card.hop_type, "adaptive_inferred": card.adaptive,
                        "paired": card.paired_track is not None, **ev})
    return out


def sdif_ablation(cfg: Config, fold: Path, scorer: OpenSetScorer, hidden: list[str], classes: list[str]) -> dict:
    """Association with vs without the same-type SDIF split on multi-emitter scenes (incl. same-family pairs)."""
    from vimmune.associate.assoc import assoc_metrics, associate
    from vimmune.io.iq import BurstStore
    from vimmune.sim.fastbdw import fast_session
    from vimmune.sim.scene import EmitterCfg, SceneCfg

    known_fg = [c for c in classes if cfg.protos.get(c).kind != "background"]
    bg = [c for c in classes if cfg.protos.get(c).kind == "background"]
    em = [EmitterCfg(family=f, distance_m=120 + 20 * i, seed=i) for i, f in enumerate(known_fg[:1] * 2 + known_fg[1:2] + bg + hidden[:1] * 2)]
    sc = SceneCfg(name="assoc", duration_s=6.0, seed=cfg.seed + 5, emitters=em)
    d = fold / "assoc_scene"
    store = BurstStore(d / "bursts")
    df = fast_session(sc, cfg, store)
    store.flush()
    z, codes = encode_bdw(cfg, df, d / "bursts", fold / "tok")
    res = {}
    for name, use in [("with_sdif", True), ("no_sdif", False)]:
        a = associate(df, z, cfg.assoc, use_sdif=use)
        m = assoc_metrics(a, cfg.assoc.min_track_bursts)
        tracks = [t for t in build_tracks(a, codes, Tokenizer(cfg), by="track_id") if len(t) >= cfg.assoc.min_track_bursts]
        s = scorer.score(tracks)
        is_unk = s.label.isin(hidden).to_numpy()
        res[name] = {**m, "auroc_tracks": auroc(s.U.to_numpy()[~is_unk], s.U.to_numpy()[is_unk])}
    return res


def evaluate_fold(cfg: Config, fold: Path, hidden: list[str], k_curve: bool = False, log=_log) -> dict:
    """All §2.9 metrics for one trained fold directory (corpus/, tok/, lm/, cal/ [ddpm/])."""
    t0 = time.time()
    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sc = load_scorer(fold / "lm" / "lm.pt", fold / "cal" / "calib.json")
    model, bundle = load_lm(fold / "lm" / "lm.pt")
    classes = bundle.classes
    tr = corpus_tracks(cfg, fold / "corpus", fold / "tok")
    known = [t for t in tr["test"] if t.label in classes]
    hid, hid_df, _ = hidden_tracks(cfg, fold, hidden, max(4, cfg.eval.n_seq_per_family // 10))
    log(f"fold {hidden}: {len(known)} known test tracks, {len(hid)} hidden tracks")
    ks, us = sc.score(known), sc.score(hid)
    thr = float(np.quantile(ks.U, 0.95))
    methods: dict[str, dict] = {}
    rocs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name: str, kU: np.ndarray, uU: np.ndarray, uw: pd.DataFrame, tracks: list[Track]):
        thr_m = float(np.quantile(kU, 0.95))
        methods[name] = summarize(kU, uU, window_ttd(uw.assign(U=uU), tracks, thr_m))
        rocs[name] = (kU, uU)

    add("V-IMMUNE (SpectrumLM, fused U)", ks.U.to_numpy(), us.U.to_numpy(), us, hid)
    cres = [sc.cusum_track(t) for t in hid]
    cus = [c["ttd_s"] for c in cres]
    kres = [sc.cusum_track(t) for t in known[:: max(1, len(known) // 6)][:6]]
    traces = {}
    for name, ts, rs in (("unknown", hid[:: max(1, len(hid) // 4)][:4], cres[:: max(1, len(hid) // 4)][:4]), ("known", known[:: max(1, len(known) // 6)][:6], kres)):
        for j, (t, r) in enumerate(zip(ts, rs)):
            traces[f"{name}{j}|{t.label}|t"] = t.toa - t.toa[0]
            traces[f"{name}{j}|{t.label}|g"] = r["g"]
    np.savez_compressed(fold / "cusum_traces.npz", h=np.array([sc.cal.h]), **traces)
    det = [x for x in cus if x is not None]
    methods["V-IMMUNE + CUSUM (online)"] = {"auroc": methods["V-IMMUNE (SpectrumLM, fused U)"]["auroc"],
                                             "tpr_at_fpr5": methods["V-IMMUNE (SpectrumLM, fused U)"]["tpr_at_fpr5"],
                                             "ttd_s_median": float(np.median(det)) if det else None, "detected_tracks": f"{len(det)}/{len(cus)}"}
    known_alarms = sum(1 for t in known if sc.cusum_track(t)["alarms"])
    add("Ablation: z_S only (α=1)", ks.zS.to_numpy(), us.zS.to_numpy(), us, hid)
    add("Ablation: z_d only (α=0)", ks.zd.to_numpy(), us.zd.to_numpy(), us, hid)
    open_acc = float(np.mean(np.r_[((~ks.unknown) & (ks.pred == ks.label)).to_numpy(), us.unknown.to_numpy()]))

    from vimmune.lm.score import WCODE, field_weights

    tok = Tokenizer(cfg)
    variants: dict[str, dict] = {"Ablation: no SNR calibration": {"no_snr": True},
                "Ablation: no waveform code field": {"score_kw": {"mask_fields": (WCODE,)}, "weights": field_weights(tok.n_bins, (FIELDS.index("snr"), WCODE))},
                "Ablation: no sequence context (N=1)": {"score_kw": {"no_context": True}}}
    for name, kw in variants.items():
        cal_v, _ = fit_calibration(cfg, model, classes, bundle.centroids, tr, fixed=sc.cal, **kw)
        sv = OpenSetScorer(model, classes, bundle.centroids, cal_v)
        skw = dict(kw.get("score_kw", {}))
        if "weights" in kw:
            skw["weights"] = kw["weights"]
        kv, uv = sv.score(known, **skw), sv.score(hid, **skw)
        add(name, kv.U.to_numpy(), uv.U.to_numpy(), uv, hid)
        log(f"  {name}: AUROC {methods[name]['auroc']:.3f}")

    base = run_baselines(cfg, fold, known, hid, hid_df, classes, cfg.openset.score_len, cfg.lm.ctx)
    for m, (kU, uU, _, uw) in base.items():
        add(m, kU, uU, uw, hid)
    assoc = sdif_ablation(cfg, fold, sc, hidden, classes)
    cards = card_metrics(cfg, hidden)
    res = {"hidden": hidden, "known": classes, "n_known_windows": len(ks), "n_unknown_windows": len(us), "threshold_fpr5": thr,
           "open_set_accuracy": open_acc, "false_unknown_rate_known": float(ks.unknown.mean()),
           "cusum_false_alarm_tracks": f"{known_alarms}/{len(known)}", "methods": methods, "association_sdif_ablation": assoc,
           "cards": cards, "card_summary": {"channel_jaccard_mean": float(np.mean([c["channel_jaccard"] for c in cards])),
                                            "frame_period_rel_err_mean": float(np.mean([c["frame_period_rel_err"] for c in cards if c["frame_period_rel_err"] is not None])),
                                            "hop_type_accuracy": float(np.mean([c["hop_type_ok"] for c in cards]))}}
    if k_curve:
        res["k_curve"] = k_shot_curve(cfg, fold, hidden[0])
    np.savez_compressed(fold / "roc_scores.npz", **{f"{k}|known": v[0] for k, v in rocs.items()}, **{f"{k}|unknown": v[1] for k, v in rocs.items()})
    res["eval_s"] = time.time() - t0
    write_json(fold / "fold_results.json", res)
    return res


def k_shot_curve(cfg: Config, fold: Path, family: str, ks: list[int] | None = None, dur: float = 8.0) -> list[dict]:
    """Gate recall on real held-out bursts vs the number K of real bursts given to the vaccine."""
    from vimmune.grammar.card import induce_card
    from vimmune.io.iq import BurstStore
    from vimmune.sim.fastbdw import fast_session
    from vimmune.sim.scene import EmitterCfg, SceneCfg
    from vimmune.vaccine.run import run_vaccine, split_learn_heldout

    d = fold / "kcurve"
    em = [EmitterCfg(family=family, distance_m=160, seed=1), EmitterCfg(family="P7", distance_m=80, seed=2),
          EmitterCfg(family="P8", distance_m=100, seed=3)]
    store = BurstStore(d / "bursts")
    df = fast_session(SceneCfg(name="kc", duration_s=dur, seed=cfg.seed + 3, session_id="kc", emitters=em), cfg, store)
    store.flush()
    _, codes = encode_bdw(cfg, df, d / "bursts", fold / "tok")
    df["wcode"] = codes
    sid = df[df.protocol == family].source_id.iloc[0]
    tr = df[df.source_id == sid]
    learn, _ = split_learn_heldout(tr, cfg.vaccine.learn_frac)
    others = {s: df[df.source_id == s].toa_s.values for s in df.source_id.unique() if s != sid}
    card = induce_card(learn, df[df.source_id != sid], others, cfg.grammar, cfg.assoc, track_id=sid)
    out = []
    for k in ks or cfg.eval.k_list:
        rep = run_vaccine(cfg, track=tr, store=d / "bursts", card=card, base_lm=fold / "lm" / "lm.pt", base_calib=fold / "cal" / "calib.json",
                          tok_dir=fold / "tok", corpus=fold / "corpus", ddpm_path=fold / "ddpm" / "ddpm.pt", out=d / f"k{k}", k=k,
                          approve=True, new_name=family)
        g = rep["gate"]["checks"]
        out.append({"k": k, "recall": g["recall_heldout_real"]["value"], "gate_passed": rep["gate"]["passed"],
                    "wall_clock_s": rep["wall_clock_s"], "waveform_mode": rep["waveforms"]["mode"]})
        _log(f"K={k}: recall {out[-1]['recall']:.2f}")
    return out


def train_fold(cfg: Config, fold: Path, known: list[str], with_ddpm: bool = False) -> None:
    """Train everything for one fold on the known families only."""
    if not (fold / "corpus" / "bdw.parquet").exists():
        stage_corpus(cfg, known, fold / "corpus")
    if not (fold / "tok" / "vq.pt").exists():
        stage_tokenizer(cfg, fold / "corpus", fold / "tok")
    if not (fold / "lm" / "lm.pt").exists():
        stage_lm(cfg, fold / "corpus", fold / "tok", fold / "lm", known)
    if not (fold / "cal" / "calib.json").exists():
        stage_calibrate(cfg, fold / "corpus", fold / "tok", fold / "lm", fold / "cal")
    if with_ddpm and not (fold / "ddpm" / "ddpm.pt").exists():
        stage_ddpm(cfg, fold / "corpus", fold / "tok", fold / "ddpm")


def run_loco(cfg: Config, out: Path, n_folds: int | None = None, k_curve: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    folds = cfg.eval.folds[: n_folds or len(cfg.eval.folds)]
    results = []
    for i, hidden in enumerate(folds):
        known = [f for f in cfg.protos.ids if f not in hidden]
        fold = out / f"fold{i}_{'-'.join(hidden)}"
        _log(f"=== fold {i}: hidden {hidden}")
        train_fold(cfg, fold, known, with_ddpm=k_curve)
        results.append(evaluate_fold(cfg, fold, hidden, k_curve=k_curve))
    agg = aggregate(results)
    write_json(out / "results.json", {"profile": cfg.profile, "folds": results, "aggregate": agg})
    (out / "table3.md").write_text(table3(agg, results))
    return agg


def aggregate(results: list[dict]) -> dict:
    names = list(results[0]["methods"])
    agg = {}
    for n in names:
        vals = [r["methods"][n] for r in results]
        f = lambda k, vals=vals: [v[k] for v in vals if v[k] is not None and v[k] == v[k]]  # noqa: E731
        agg[n] = {k: (float(np.mean(f(k))) if f(k) else None) for k in ("auroc", "tpr_at_fpr5", "ttd_s_median")}
        agg[n]["auroc_std"] = float(np.std(f("auroc"))) if len(f("auroc")) > 1 else None
        agg[n]["detected_tracks"] = ", ".join(v["detected_tracks"] for v in vals)
    return agg


def table3(agg: dict, results: list[dict]) -> str:
    folds = "; ".join("+".join(r["hidden"]) for r in results)
    lines = [f"Table 3 — open-set detection of unseen protocols (leave-classes-out; hidden per fold: {folds}).", "",
             "| Method | AUROC | TPR@FPR5% | time-to-detect (s) |", "|---|---|---|---|"]

    def fmt(x, nd=3):
        return "—" if x is None else f"{x:.{nd}f}"

    for n, v in agg.items():
        au = fmt(v["auroc"]) + (f" ± {v['auroc_std']:.3f}" if v.get("auroc_std") is not None else "")
        lines.append(f"| {n} | {au} | {fmt(v['tpr_at_fpr5'])} | {fmt(v['ttd_s_median'], 2)} |")
    lines += ["", "Time-to-detect: median over hidden tracks of the time from a track's first burst to the first",
              "flag (window rows: first 32-burst window above the known-class 95th-percentile score; CUSUM row:",
              "first CUSUM alarm). Values computed by `vimmune eval` from the files in this directory."]
    return "\n".join(lines) + "\n"


def evaluate_demo_fold(cfg: Config, demo: Path, hidden: list[str]) -> dict:
    """Evaluation of the demo's own fold (artifacts already trained by the demo); writes table3.md there."""
    fold = demo / "eval_fold"
    fold.mkdir(parents=True, exist_ok=True)
    for name, src in (("corpus", "corpus"), ("tok", "tokenizer"), ("lm", "lm"), ("cal", "calibrate")):
        link = fold / name
        if not link.exists():
            link.symlink_to((demo / src).resolve())
    res = evaluate_fold(cfg, fold, hidden)
    agg = aggregate([res])
    write_json(demo / "results.json", {"profile": cfg.profile, "folds": [res], "aggregate": agg})
    (demo / "table3.md").write_text(table3(agg, [res]))
    return res


def load_results(exp: Path) -> dict:
    return read_json(exp / "results.json")
