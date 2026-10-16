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
