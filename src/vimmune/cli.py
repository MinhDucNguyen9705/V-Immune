"""``vimmune`` command-line entry point (typer). Thin wrappers over the package modules (report §2.1–2.10).

Paths default to a workspace layout under ``runs/`` (corpus/, tok/, lm/, cal/, ddpm/ ...), the same one
the demo uses, so commands can be chained by hand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from vimmune.config import load_config, seed_everything

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
R = Path("runs")


def _cfg(profile: str):
    cfg = load_config(profile)
    seed_everything(cfg.seed)
    return cfg


def _sub(name: str, help: str) -> typer.Typer:
    t = typer.Typer(no_args_is_help=True)
    app.add_typer(t, name=name, help=help)
    return t


sim_app = _sub("sim", "Simulator (§2.1): corpora and full-IQ scenes.")
tok_app = _sub("tokenizer", "Field quantizers + VQ-VAE waveform tokenizer (§2.2).")
lm_app = _sub("lm", "SpectrumLM (§2.4).")
os_app = _sub("openset", "Surprise, calibration, conformal, CUSUM (§2.5).")
gr_app = _sub("grammar", "Grammar induction -> Protocol ID Card (§2.6).")
vac_app = _sub("vaccine", "Data synthesis + adapter + gate (§2.7).")
edge_app = _sub("edge", "Distillation, ONNX INT8, bench, release, nodes, federation (§2.8).")
eval_app = _sub("eval", "Leave-classes-out evaluation (§2.9).")
demo_app = _sub("demo", "End-to-end demo (§2.10).")


# ---------------------------------------------------------------------------------------------- sim / detect
@sim_app.command("corpus")
def sim_corpus(profile: str = "demo", families: str = "P1,P2,P4,P5,P7,P8", out: Path = R / "corpus", workers: int = 8) -> None:
    """Fast-BDW corpus (measured BDW + burst snippets, session split) for the given families."""
    from vimmune.pipeline import stage_corpus

    df = stage_corpus(_cfg(profile), families.split(","), out, workers)
    console.print(f"[green]{len(df)} bursts[/] from {df.session_id.nunique()} sessions -> {out}")


@sim_app.command("scene")
def sim_scene(scene: Path, profile: str = "demo", out: Path = R / "scene1", workers: int = 8) -> None:
    """Full-band IQ scene (SigMF ci16_le) + ground-truth BDW."""
    from vimmune.sim.scene import load_scene, simulate_scene

    truth = simulate_scene(load_scene(scene), _cfg(profile), out, workers=workers)
    console.print(f"[green]{len(truth)} truth bursts[/] -> {out}")


@app.command("detect")
def detect(iq: Path, out: Path = R / "scene1" / "bdw.parquet", profile: str = "demo", truth: Path | None = None) -> None:
    """Reference detector (STFT + CA-CFAR) on a SigMF recording -> BDW parquet + burst store."""
    from vimmune.detect.detector import ReferenceDetector, label_from_truth
    from vimmune.io.bdw import read_bdw, write_bdw
    from vimmune.io.iq import BurstStore, read_sigmf

    cfg = _cfg(profile)
    df = ReferenceDetector(cfg.detect).detect(read_sigmf(iq), BurstStore(out.parent / "bursts"))
    if truth is not None:
        df = label_from_truth(df, read_bdw(truth))
    write_bdw(df, out)
    console.print(f"[green]{len(df)} bursts[/] -> {out}")


# ---------------------------------------------------------------------------------------------- tokenizer
@tok_app.command("train")
def tok_train(corpus: Path = R / "corpus", out: Path = R / "tok", profile: str = "demo") -> None:
    from vimmune.pipeline import stage_tokenizer

    stage_tokenizer(_cfg(profile), corpus, out)
    console.print(f"[green]tokenizer[/] -> {out}")


@tok_app.command("encode")
def tok_encode(bdw: Path, store: Path | None = None, tok: Path = R / "tok", profile: str = "demo") -> None:
    """Add the waveform code (``wcode``) to a BDW parquet and save z next to it (``<bdw>.z.npy``)."""
    import numpy as np

    from vimmune.io.bdw import read_bdw, write_bdw
    from vimmune.pipeline import encode_bdw

    cfg = _cfg(profile)
    df = read_bdw(bdw)
    z, codes = encode_bdw(cfg, df, store or bdw.parent / "bursts", tok)
    write_bdw(df.assign(wcode=codes), bdw)
    np.save(bdw.with_suffix(".z.npy"), z)
    console.print(f"[green]{len(df)} bursts encoded[/] (codes used: {len(set(codes.tolist()))})")


@app.command("associate")
def associate_cmd(bdw: Path, out: Path | None = None, profile: str = "demo", no_sdif: bool = False) -> None:
    """Source association (graph grouping + SDIF split + Hungarian tracking). Needs ``tokenizer encode`` first."""
    import numpy as np

    from vimmune.associate.assoc import assoc_metrics, associate, released_tracks
    from vimmune.io.bdw import read_bdw, write_bdw

    cfg = _cfg(profile)
    df = read_bdw(bdw)
    z_path = bdw.with_suffix(".z.npy") if bdw.with_suffix(".z.npy").exists() else bdw.parent / "z.npy"
    df = associate(df, np.load(z_path), cfg.assoc, use_sdif=not no_sdif)
    write_bdw(df, out or bdw.parent / "tracks.parquet")
    console.print(f"[green]{len(released_tracks(df, cfg.assoc.min_track_bursts))} tracks released[/]")
    if df.source_id.notna().any():
        console.print(assoc_metrics(df, cfg.assoc.min_track_bursts))


# ---------------------------------------------------------------------------------------------- lm
@lm_app.command("pretrain")
def lm_pretrain(corpus: Path = R / "corpus", tok: Path = R / "tok", out: Path = R / "lm", profile: str = "demo",
                classes: str = "") -> None:
    """Pretrain SpectrumLM, fine-tune the classifier head, compute centroids (known classes = corpus families)."""
    from vimmune.pipeline import read_json, stage_lm

    fams = classes.split(",") if classes else read_json(corpus / "meta.json")["families"]
    stage_lm(_cfg(profile), corpus, tok, out, fams)
    console.print(read_json(out / "metrics.json"))


@lm_app.command("finetune")
def lm_finetune(corpus: Path = R / "corpus", tok: Path = R / "tok", lm: Path = R / "lm", profile: str = "demo") -> None:
    """Re-run the classifier fine-tune (+ centroids) on an existing pretrained model."""
    from vimmune.lm.train import LMBundle, centroids, embed_windows, finetune_classifier
    from vimmune.pipeline import corpus_tracks, load_lm

    cfg = _cfg(profile)
    model, b = load_lm(lm / "lm.pt")
    tr = corpus_tracks(cfg, corpus, tok)
    finetune_classifier(model, tr["train"], b.classes, cfg.lm, cfg.seed)
    h, _, lab = embed_windows(model, [t for t in tr["train"] if t.label in b.classes], cfg.lm.ctx, cfg.openset.score_len, stride=cfg.lm.ctx)
    LMBundle(b.vocab_sizes, b.cfg, b.classes, model.state_dict(), centroids(h, lab, b.classes), b.meta).save(lm / "lm.pt")
    console.print("[green]fine-tuned[/]")


@lm_app.command("score")
def lm_score(tracks: Path, lm: Path = R / "lm" / "lm.pt", calib: Path = R / "cal" / "calib.json", out: Path | None = None,
             profile: str = "demo") -> None:
    """Score released tracks of an associated BDW parquet (windows + per-track summary + CUSUM)."""
    from vimmune.demo.live import scene_tracks, score_scene
    from vimmune.pipeline import load_scorer

    cfg = _cfg(profile)
    d = tracks.parent
    if tracks.name != "bdw.parquet":
        import shutil

        shutil.copy(tracks, d / "bdw.parquet")
    det, trs = scene_tracks(cfg, d)
    res = score_scene(load_scorer(lm, calib), trs, det, out or d, "cli")
    console.print(res[["track_id", "n_bursts", "decision", "zS_median", "p_min", "cusum_ttd_s"]])


# ---------------------------------------------------------------------------------------------- open set
@os_app.command("calibrate")
def os_calibrate(corpus: Path = R / "corpus", tok: Path = R / "tok", lm: Path = R / "lm", out: Path = R / "cal", profile: str = "demo") -> None:
    from vimmune.pipeline import stage_calibrate

    m = stage_calibrate(_cfg(profile), corpus, tok, lm, out)
    console.print({k: m[k] for k in ("alpha", "false_unknown_rate_test", "pseudo_unknown_auroc_test")} | {"cusum": m["cusum"]["h"]})


@os_app.command("detect")
def os_detect(tracks: Path, lm: Path = R / "lm" / "lm.pt", calib: Path = R / "cal" / "calib.json", profile: str = "demo") -> None:
    """Flag unknown tracks (conformal p < ε, CUSUM alarms)."""
    lm_score(tracks, lm, calib, None, profile)


# ---------------------------------------------------------------------------------------------- grammar
@gr_app.command("card")
def grammar_card(track_id: int, bdw: Path = R / "scene1" / "bdw.parquet", out: Path = R / "card", profile: str = "demo",
                 learn_frac: float = 1.0) -> None:
    """Protocol ID Card for one associated track (card.json / card.png / card.md)."""
    from vimmune.grammar.card import induce_card, write_card
    from vimmune.io.bdw import read_bdw
    from vimmune.vaccine.run import split_learn_heldout

    cfg = _cfg(profile)
    det = read_bdw(bdw)
    g = det[det.track_id == track_id]
    if learn_frac < 1.0:
        g, _ = split_learn_heldout(g, learn_frac)
    others = {f"track {k}": det[det.track_id == k].toa_s.to_numpy() for k in det.track_id.unique() if k >= 0 and k != track_id}
    card = induce_card(g, det[det.track_id != track_id], others, cfg.grammar, cfg.assoc, track_id=str(track_id))
    write_card(card, g, out)
    console.print((out / "card.md").read_text())


# ---------------------------------------------------------------------------------------------- vaccine
@vac_app.command("run")
def vaccine_run(card: Path = R / "card" / "card.json", bdw: Path = R / "scene1" / "bdw.parquet", track_id: int | None = None,
                store: Path | None = None, k: int = 30, waveform_mode: str = "ddpm", approve: bool = False, name: str = "NEW",
                ws: Path = R, out: Path = R / "vaccine", profile: str = "demo") -> None:
    """Synthesize data from K real bursts + the approved card, train the adapter, write the gate report."""
    from vimmune.grammar.card import Card
    from vimmune.io.bdw import read_bdw
    from vimmune.vaccine.run import run_vaccine

    cfg = _cfg(profile)
    c = Card.model_validate_json(card.read_text())
    det = read_bdw(bdw).reset_index(drop=True)
    tid = int(c.track_id) if track_id is None else track_id
    rep = run_vaccine(cfg, track=det[det.track_id == tid], store=store or bdw.parent / "bursts", card=c, base_lm=ws / "lm" / "lm.pt",
                      base_calib=ws / "cal" / "calib.json", tok_dir=ws / "tok", corpus=ws / "corpus", ddpm_path=ws / "ddpm" / "ddpm.pt",
                      out=out, k=k, waveform_mode=waveform_mode, approve=approve, new_name=name)
    console.print({c: v["pass"] for c, v in rep["gate"]["checks"].items()}, f"passed={rep['gate']['passed']}")


@vac_app.command("ddpm")
def vaccine_ddpm(corpus: Path = R / "corpus", tok: Path = R / "tok", out: Path = R / "ddpm", profile: str = "demo") -> None:
    """Pretrain the conditional waveform DDPM on known-class bursts."""
    from vimmune.pipeline import stage_ddpm

    console.print(stage_ddpm(_cfg(profile), corpus, tok, out))


# ---------------------------------------------------------------------------------------------- edge
@edge_app.command("distill")
def edge_distill(ws: Path = R, vaccine: Path = R / "vaccine", out: Path = R / "edge", iq: Path | None = None, bench_s: float = 2.0,
                 profile: str = "demo") -> None:
    """Distil teacher(+adapter) into the student, export ONNX fp32/INT8, compare and benchmark."""
    from vimmune.pipeline import stage_edge

    rep = stage_edge(_cfg(profile), ws / "corpus", ws / "tok", ws / "lm" / "lm.pt", vaccine, out, iq, bench_s)
    console.print(rep["metrics"])


@edge_app.command("export")
def edge_export(lm: Path = R / "lm" / "lm.pt", out: Path = R / "edge" / "lm_fp32.onnx") -> None:
    """Export any SpectrumLM bundle to ONNX (fixed context, right-padded)."""
    from vimmune.edge.distill import export_onnx
    from vimmune.pipeline import load_lm

    model, b = load_lm(lm)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out, b.cfg.ctx)
    console.print(f"[green]{out}[/]")


@edge_app.command("bench")
def edge_bench(iq: Path = R / "scene1" / "iq.sigmf-meta", edge: Path = R / "edge", tok: Path = R / "tok", seconds: float = 2.0,
               profile: str = "demo") -> None:
    """Async detector -> tokenizer -> association -> INT8 student pipeline on a recording."""
    import numpy as np
    import torch

    from vimmune.edge.distill import OnnxLM
    from vimmune.edge.runtime import run_edge_bench
    from vimmune.openset.calib import OpenSetCalib
    from vimmune.pipeline import read_json
    from vimmune.tokenize.vq import load_vq

    cfg = _cfg(profile)
    classes = torch.load(edge / "student.pt", weights_only=False)["classes"]
    rep = run_edge_bench(iq, cfg, load_vq(tok / "vq.pt", cfg.tokenizer), OnnxLM(edge / "student_int8.onnx"),
                         OpenSetCalib.from_dict(read_json(edge / "calib_student.json")), classes, np.load(edge / "mu_student.npy"), seconds)
    rep.pop("decisions")
    console.print(rep)


@edge_app.command("release")
def edge_release(vaccine: Path = R / "vaccine", out: Path = R / "release", version: str = "1.1.0", profile: str = "demo") -> None:
    """Sign a .vpkg release (requires a fully passed gate, including human approval)."""
    from vimmune.pipeline import stage_release

    console.print(f"[green]{stage_release(_cfg(profile), vaccine, out, version)}[/]")


@edge_app.command("node")
def edge_node(release_dir: Path, pub: Path = R / "release" / "keys" / "release_ed25519.pub.pem", lm: Path = R / "lm" / "lm.pt",
              calib: Path = R / "cal" / "calib.json", seconds: float = 60.0, node_id: str = "node-cli") -> None:
    """Run a node that watches ``release_dir`` and hot-loads new signed packages (no restart)."""
    import time

    from vimmune.edge.node import EdgeNode, NodeState
    from vimmune.pipeline import load_lm, load_scorer

    base, _ = load_lm(lm)
    node = EdgeNode(node_id, base, NodeState("1.0.0", load_scorer(lm, calib)), release_dir, pub, release_dir / "events.jsonl")
    node.watch()
    console.print(f"watching {release_dir} for {seconds:.0f}s (current version 1.0.0)")
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(1.0)
    node.stop()
    console.print(node.events)


@edge_app.command("fed")
def edge_fed(ws: Path = R, release: Path | None = None, out: Path = R / "fleet", family: str = "P3", profile: str = "demo") -> None:
    """Three node processes (configs/nodes.yaml) + hot-load + FedAvg of calibration statistics."""
    from vimmune.pipeline import read_json, stage_fleet

    rel = read_json(ws / "release" / "release.json")
    rep = stage_fleet(_cfg(profile), ws / "lm" / "lm.pt", ws / "cal" / "calib.json", ws / "tok",
                      release or ws / "release" / rel["package"], Path(rel["public_key"]), out, family)
    console.print([(n["node_id"], n["final_version"], n["bytes_per_burst"]) for n in rep["nodes"]])


# ---------------------------------------------------------------------------------------------- eval
@eval_app.command("loco")
def eval_loco(profile: str = "demo", folds: int = 0, out: Path = R / "eval", k_curve: bool = False) -> None:
    """Leave-classes-out folds: train on known families only, evaluate the hidden ones (+ baselines, ablations)."""
    from vimmune.eval.loco import run_loco

    agg = run_loco(_cfg(profile), out, folds or None, k_curve)
    console.print((out / "table3.md").read_text())
    del agg


@eval_app.command("report")
def eval_report(exp: Path = R / "eval") -> None:
    """Compile results.json / table3.md / figures into report.html."""
    from vimmune.eval.report import build_report

    console.print(f"[green]{build_report(exp)}[/]")


# ---------------------------------------------------------------------------------------------- demo
@demo_app.command("run")
def demo_run(profile: str = "demo", approve: bool = False, until: str | None = None) -> None:
    """Run (or resume, using the stage cache) the whole demo story."""
    from vimmune.demo.orchestrator import Demo

    st = Demo(profile, approve=approve).run(until)
    for s, v in st.items():
        if not s.startswith("_"):
            console.print(f"{s:10s} {v.get('status'):18s} {v.get('seconds', 0):7.1f}s  {v.get('error') or v.get('reason') or ''}")
    console.print(f"this run: {st['_total_seconds_this_run'] / 60:.1f} min")
    if any(v.get("status") == "failed" for k, v in st.items() if not k.startswith("_")):
        raise typer.Exit(1)


@demo_app.command("approve")
def demo_approve(profile: str = "demo") -> None:
    """Set the human-approval flag on the demo gate and run the remaining stages."""
    from vimmune.demo.orchestrator import approve_and_continue

    approve_and_continue(profile=profile)


@demo_app.command("ui")
def demo_ui() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "demo" / "app.py")], check=False)


@demo_app.command("clean")
def demo_clean() -> None:
    from vimmune.demo.orchestrator import clean

    clean()
    console.print("[green]runs/demo removed[/]")


if __name__ == "__main__":
    app()
