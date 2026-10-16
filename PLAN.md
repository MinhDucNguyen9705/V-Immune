# V-IMMUNE build plan

Milestones follow spec §9. Each one ends with `make check` green and a git commit.

| M | Scope | Exit test |
|---|---|---|
| M0 | Scaffold: layout, pydantic configs (`full`, `demo`), typer CLI skeleton, `make check`, CLAUDE.md | `make check` passes |
| M1 | `io/` (BDW schema, int16/f32/SigMF IQ, burst store), `sim/` (GFSK/OFDM/noise, P1–P8, channel, streaming scenes, fast BDW + noise model) | family period/channel count match config; PSD BW within 10% |
| M2 | `detect/` STFT + noise floor + 2-D CA-CFAR + morphology + BDW estimation + burst IQ | recall ≥ 0.9 @ SNR ≥ 0 dB, FC err < 2% BW, TOA err < 20 µs; detector stats vs fast-mode noise model |
| M3 | `tokenize/` field quantizers, waveform prep, VQ-VAE, time encoding | round-trip ≤ ½ bin; codebook usage > 50% |
| M4 | `associate/` graph grouping, SDIF split, Hungarian tracking, purity/completeness/ID switches | SDIF recovers 2 periods; same-family purity ≥ 0.9 |
| M5 | `lm/` SpectrumLM, pretraining, classifier fine-tune, centroids | loss decreases; known-class accuracy reported |
| M6 | `openset/` surprise, SNR calibration, distance, fusion, conformal, CUSUM | false-unknown ≤ ε + 0.02; CUSUM ARL test |
| M7 | `grammar/` channels, period, hop type, adaptivity, link pairing, similarity, card json/png/md | hop-type classes; Jaccard ≥ 0.9; period err < 2% |
| M8 | `vaccine/` sequence generator, DDPM + LoRA, param fallback, channel aug, adapter, gate | recall-vs-K curve; gate blocks broken adapter |
| M9 | `edge/` distillation, ONNX INT8, async runtime bench, signed `.vpkg`, hot-load, 3-node FedAvg | hot-load during run with no dropped bursts |
| M10 | `eval/` LOCO folds, baselines (MSP, energy, OpenMax, centroid), ablations, table3, report.html | runs on demo profile |
| M11 | `demo/` orchestrator with stage cache, Streamlit app, static report, README | `make demo` end-to-end ≤ 20 min on 8-core CPU |

Progress is tracked by the commit log; deviations go to `DECISIONS.md`.
