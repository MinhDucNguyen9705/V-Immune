"""``vimmune`` command-line entry point (typer). Thin wrappers over the package modules (report §2.1–2.10)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from vimmune.config import load_config, seed_everything

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

sim_app = typer.Typer(no_args_is_help=True)
app.add_typer(sim_app, name="sim", help="Simulator (§2.1): corpora and full-IQ scenes.")


def _cfg(profile: str):
    cfg = load_config(profile)
    seed_everything(cfg.seed)
    return cfg


@sim_app.command("corpus")
def sim_corpus(profile: str = "demo", families: str = "P1,P2,P4,P5,P7,P8", out: Path = Path("runs/corpus"),
               sessions: int = 0, workers: int = 8) -> None:
    """Fast-BDW corpus (measured BDW + burst snippets) for the given families."""
    from vimmune.io.bdw import write_bdw
    from vimmune.sim.fastbdw import make_corpus

    cfg = _cfg(profile)
    df = make_corpus(cfg, families.split(","), out, sessions or None, workers)
    write_bdw(df, out / "bdw.parquet")
    console.print(f"[green]{len(df)} bursts[/] from {df.session_id.nunique()} sessions -> {out}")


@sim_app.command("scene")
def sim_scene(scene: Path, profile: str = "demo", out: Path = Path("runs/scene1"), workers: int = 8) -> None:
    """Full-band IQ scene (SigMF ci16_le) + ground-truth BDW."""
    from vimmune.sim.scene import load_scene, simulate_scene

    truth = simulate_scene(load_scene(scene), _cfg(profile), out, workers=workers)
    console.print(f"[green]{len(truth)} truth bursts[/] -> {out}")


@app.command("detect")
def detect(iq: Path, out: Path = Path("runs/scene1/bdw.parquet"), profile: str = "demo", truth: Path | None = None) -> None:
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


if __name__ == "__main__":
    app()
