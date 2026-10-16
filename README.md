# V-IMMUNE

Receive-only RF burst analysis that **detects previously unseen (fictional) UAV-style protocols**, describes
them with a Protocol ID Card, learns them from a few real bursts ("vaccine") and ships a signed adapter to
edge nodes that hot-load it.

> **Scope.** Passive analysis only. No transmit, jamming, spoofing or payload decoding code exists in this repo,
> and no real commercial protocol is modelled. All signals come from the built-in simulator (fictional families
> P1–P8) or from user-supplied recordings.

<!-- RESULTS -->

## Quickstart

```bash
# CPU-only works; Python 3.11 venv recommended
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make check          # ruff + mypy + pytest
make demo           # whole story, demo profile (<= ~20 min on an 8-core CPU)
make ui             # Streamlit app over runs/demo
open runs/demo/demo_report.html   # static, self-contained version of the demo
make eval           # leave-classes-out folds (demo profile, 2 folds) -> runs/eval/{table3.md,report.html}
```

`make demo-clean` wipes the stage cache (`runs/demo`). Re-running `make demo` or the UI never retrains a stage whose inputs, config and code are unchanged.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the Mermaid dataflow diagram.

```
IQ (SigMF / sim) → detect/ (STFT + CA-CFAR) → BDW + burst IQ → tokenize/ (6 fields, VQ-VAE) → associate/ (graph → SDIF → Hungarian)
   → lm/ SpectrumLM → openset/ (z_S, z_d, conformal p, CUSUM) ─┬─ known → label
                                                               └─ unknown → grammar/ card → vaccine/ (synthesis → LoRA adapter → gate)
                                                                    → edge/ (signed .vpkg → hot-loading nodes, FedAvg; distilled INT8 student)
```

## Report section → code

| § | topic | code |
|---|---|---|
| 2.1 | data model, IQ I/O, simulator, reference detector | `src/vimmune/{config.py,io/,sim/,detect/}` |
| 2.2 | tokenization (field quantizers, VQ-VAE, time encoding) | `src/vimmune/tokenize/`, `lm/model.py::TimeEncoding` |
| 2.3 | source association | `src/vimmune/associate/` |
| 2.4 | SpectrumLM | `src/vimmune/lm/` |
| 2.5 | surprise, SNR calibration, fusion, conformal, CUSUM | `src/vimmune/openset/`, `lm/score.py` |
| 2.6 | grammar induction → Protocol ID Card | `src/vimmune/grammar/` |
| 2.7 | vaccine: synthesis, DDPM + LoRA, adapter, gate | `src/vimmune/vaccine/` |
| 2.8 | distillation, ONNX INT8, edge runtime, release, hot-load, FedAvg | `src/vimmune/edge/` |
| 2.9 | evaluation (LOCO, baselines, ablations, reports) | `src/vimmune/eval/` |
| 2.10 | demo orchestrator, UI, static report, CLI | `src/vimmune/demo/`, `cli.py`, `pipeline.py` |

## Using real recordings and a real detector

**Recorded IQ (SigMF).** `vimmune detect --iq my.sigmf-meta --out runs/rec/bdw.parquet` reads `ci16_le` or `cf32_le` SigMF (`io/iq.py::read_sigmf`). The
centre frequency comes from `captures[0].core:frequency`. Headerless interleaved files open with `io.iq.open_raw(path, fmt, fs, fc)`.
Files are memory-mapped and processed in 100 ms blocks. Then:

```bash
vimmune tokenizer encode runs/rec/bdw.parquet --tok runs/demo/tokenizer
vimmune associate runs/rec/bdw.parquet --out runs/rec/bdw.parquet
vimmune openset detect runs/rec/bdw.parquet --lm runs/demo/lm/lm.pt --calib runs/demo/calibrate/calib.json
vimmune grammar card <track_id> --bdw runs/rec/bdw.parquet --out runs/rec/card
```

**Your own detector.** Implement `vimmune.detect.detector.Detector`:

```python
class MyDetector:
    def detect(self, src: IQSource, store: BurstStore, session: SessionMeta) -> pd.DataFrame:
        # return a BDW frame (columns of vimmune.io.bdw.BDW; build it with io.bdw.make_bdw)
        # and put each burst's IQ snippet in `store` (burst_iq_ref = store.put(iq); set iq_fs, iq_fc_hz)
```

Everything downstream consumes only the BDW frame and the burst store. Snippets may have any sample rate and centre; the tokenizer
re-centres, filters and resamples them (`tokenize/vq.py::prep_burst`). Re-measure the fast-mode noise model for your detector
with `detect/metrics.py::error_stats`, so pretraining corpora match its error statistics.

## Profiles: demo → full

`configs/full.yaml` holds the spec numbers (SpectrumLM 6×256, ctx 256, ≈5.6 M params; VQ 20 k steps; DDPM [64,128,256], 2000 samples,
etc.). `configs/demo.yaml` inherits from it and only scales sizes and steps. Every command takes `--profile full`, e.g.
`vimmune demo run --profile full --approve` or `vimmune eval loco --profile full --folds 4`. The full profile is meant for
a GPU or a long CPU run; CUDA is used automatically when available.

## Known limitations (honest list)

<!-- LIMITATIONS -->

See [DECISIONS.md](DECISIONS.md) for every deviation from the build spec and why.
