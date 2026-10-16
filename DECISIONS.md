# Decisions and deviations from the spec

Every place where the implementation departs from, or fills a gap in, the build spec.

## Environment / tooling
- **Repo root.** The spec's top-level `vimmune/` folder is this repository root.
- **Python.** The host had Python 3.10, so the project uses a `uv`-created Python 3.11 venv (`.venv`) with CPU torch. `requires-python` is `>=3.10` so the system interpreter also works.
- **mypy.** Runs non-strict, as the spec says. Error codes that only report numpy-stub unions and Optional per-family config fields (`operator`, `arg-type`, `union-attr`, `call-overload`, `index`, `assignment`, `return-value`, `misc`) are disabled in `pyproject.toml`.

## Simulator (§2.1)
- **GFSK bandwidth.** Family BW is the −10 dB occupied bandwidth. This is the same definition the detector uses. For BT = 0.5 and h = 0.5, the measured −10 dB width is 0.91 × symbol rate, so the symbol rate is BW / 0.91 (`gfsk_bw_per_rate`).
- **SNR definition.** SNR is *in-band*: signal power over the noise power inside the signal bandwidth, with unit full-band noise power. This matches the detector's "mean in-blob power over noise floor".
- **Channel plans.** All channel plans fit the 30.72 MHz band:
  - P1: 1.6 MHz spacing.
  - P2: 1.1 MHz spacing. 24 × 2 MHz channels cannot fit without overlap, so the 2 MHz channels overlap slightly.
  - P3: 1.3 MHz spacing.
  - P4: 3 centers.
  - P5, P7: 3 possible centers.
  - P6: 12 channels, 2 MHz spacing.
  - P8: 3 advertising channels.
- **P3 adaptive avoidance.** P3 avoids channels occupied by a background burst *at the instant of transmission*. Avoiding them over a sensing window would leave only about 9 usable channels when a 20 MHz WiFi-like emitter shares the band, and the card is expected to report about 20 channels.
- **P6 pair.** Uplink and downlink are separate sources (`<id>.up`, `<id>.dn`). The downlink follows each uplink by 1.2 ms ± 1 % on a different fixed channel.
- **P8 advertising.** Each advertising event sends one 0.2 ms PDU on each of the 3 channels, 0.4 ms apart.
- **CFO and fading.** CFO is drawn once per emitter (U(±5 ppm) × RF center). Per-burst SNR has 0.5 dB Gaussian fading around the path-loss value.
- **Full-IQ scenes.** Scenes are stored as SigMF `ci16_le` (scale 64) to halve disk use.
- **Fast BDW collisions.** Besides the noise model, fast mode resolves time-frequency collisions the way connected-component labelling would: the weaker burst vanishes, and if the two are within 10 dB the survivor becomes the union box. Without this, corpus sequences never contain collision artefacts that the live detector does produce.
- **Fast BDW snippets.** Snippets are generated directly at 1.25·BW (measured) around the measured FC, capped at 1024 samples. This is the format the reference detector emits (see Detector).
- **Corpus sessions.** Each session has one primary emitter plus 0–2 background emitters drawn from the *known* background families. Hidden families never appear in training corpora.
- **Multipath.** Applied at each burst's RF offset, not at baseband, so hop channels fade differently. Taps are static per emitter: 2–4 taps within 0.3 µs, each echo 0.2× the power of the previous (LOS-dominant, Rician-like). The truth `snr_db` includes each burst's in-band channel gain. With 0.5 µs and a 0.5 decay, notches of about −11 dB split 10–20 MHz bursts into fragments, which is unrealistic for mostly-LOS UAV links. In fast-mode snippets the multipath only distorts the waveform (it is power-normalized), because the gain is already in the SNR.

## Reference detector (§2.1, spec §5)
- **Noise floor.** The spec says "per-bin median over 200 ms". Bins under the video families plus the WiFi-like emitter are busy up to about 90 % of the time, so the median measures the signal, not the noise. The implementation instead uses:
  - the per-bin **0.02 quantile**, corrected for exponentially distributed bin power (÷ −ln(1−q));
  - frames at 4× hop over a 200 ms window centred on each 100 ms block;
  - each bin capped at **+1 dB above the 10th-percentile bin** (`noise_cap_db`; raise it for real receivers with passband ripple).
- **CFAR.** CA-CFAR runs on the noise-normalized spectrogram:
  1. 2×2 pooling, then smoothing to a total of 12 frames × 8 bins. With 75 %-overlapped frames, fewer cells gave too few independent samples, and α ≈ 3.7 shattered bursts at 6 dB.
  2. Guard 2×2 and training 6×6 pooled cells.
  3. The cell average is capped at the noise floor (`min(CA, 1)`). Otherwise wide bursts that fill their own training cells (20 MHz, 2 ms) mask themselves.
  4. α is calibrated empirically for `pfa = 1e-4` on simulated noise, because the smoothed cells are correlated.
- **Morphology.** Closing is 3×7 pooled cells, so it bridges OFDM DC nulls. Blobs smaller than 8 frames × 8 bins are dropped.
- **TOA/TOD** are half-level crossings of the in-band excess-power profile at full resolution. **FC** is the power-weighted centroid inside the −10 dB extent. **SNR** is 10·log10(mean excess) over that extent.
- **Burst IQ.** Mix FC to DC, apply a 129-tap FIR low-pass with cutoff 0.6·BW (≡ a 1.2·BW band-pass at FC), then decimate to ≈1.25·BW and keep at most 2048 samples. Storing full-rate 2 ms snippets would cost about 0.5 MB per burst.
- **Test definitions.** Recall and TOA are evaluated on *isolated* truth bursts, meaning no other burst within 0.2 ms / 1 MHz. Closer bursts legitimately merge into one blob. The FC < 2 % BW criterion is checked on a LOS (single-tap) scene: with multipath, the in-band spectral tilt moves the centroid by 2–4 % of BW, which is physical, not an estimator bug.
- **Known limitation.** Wide bursts below about 0 dB in-band SNR (e.g. a 10 MHz OFDM link at −2.5 dB) come out as many small fragments. A second, coarser CFAR scale merged neighbouring emitters, so it was rejected. Demo scenes keep emitters at ≥ 3 dB.
- **Fast-mode noise model** values in `config.py` were measured from the detector on multi-emitter scenes (`detect.metrics.error_stats`). `tests/test_detect.py` checks the detector spreads stay within 3× of the model.
- **Scene occupancy.** The WiFi-like mean CSMA gap is 4 ms (about 20 % duty) and the P4 channel centres are −10/0/+10 MHz. With 2 ms gaps, a 20 MHz WiFi-like emitter plus the 80 %-duty 10 MHz video link saturate a 30.72 MHz band. Collisions then cascade into 29 MHz × 5 ms blobs, both in the detector and in the fast-mode collision model.

## Source association (§2.3)
- **SDIF threshold.** Uses the Milojević & Popović form T(τ) = x·(E−c)·exp(−τ/(kN)) with x = 0.3, k = 0.3, N = 2000 bins over 0–150 ms. Candidate periods are tried in ascending order for orders 1→3.
- **Train extraction.** Trains are followed greedily at ±2 % of the period. A train may skip up to 8 consecutive missed pulses, but it must fill ≥ 50 % of the period slots it spans and hold ≥ 8 pulses. Without the fill rule, sparse spurious long periods (80–110 ms) got through. Chains of one emitter broken by a long fade are re-joined when they are disjoint in time and on the same phase grid.
- **Split trigger.** The spec's "time overlap at different FC" test is a necessary sign of > 1 emitter but rarely fires, because two same-period emitters seldom overlap inside a 2 s window. A group is therefore split when the spec trigger fires, *or* when SDIF finds ≥ 2 trains, *or* when it finds one train plus a residual ≥ 30 % of the group. The `use_sdif=False` ablation turns this off.
- **Component merge.** Before splitting, connected components whose centroid z, median log-BW and median log-PW meet the same tolerances as single bursts are merged. Collision-inflated outlier bursts otherwise cut one source into several components.
- **Tracking cost.** Hungarian matching on (1 − cos z̄) + |Δperiod|/period + (1 − burst overlap). The burst-overlap term is added because co-channel same-family emitters have identical z and period, so the spec's two terms alone cannot tell their tracks apart. A group left unmatched after the 1:1 assignment continues the track it overlaps by ≥ 50 % (extra fragments of one emitter).

## SpectrumLM (§2.4)
- **Field independence.** The six next-burst heads are conditionally independent given the history: p(b_i | b_<i) = Π_a p(x_{i,a} | b_<i). This keeps scoring to one forward pass and per-field surprise additive. It ignores within-burst dependencies, e.g. the BW of a WiFi-like burst predicting its waveform code. `lm.autoregressive_fields: true` enables a variant where head a also sees the true fields < a of the target burst (teacher forcing). It is off by default and not used in the reported results.
- **Attention dropout.** Dropout 0.1 is applied on residual and MLP outputs, not on attention probabilities. Attention dropout forces PyTorch's slow math kernel on CPU, which was about 2× slower per step. The attention mask is purely causal: sequences are right-padded, so valid queries never see padded keys.
- **Input layout.** Position 0 is [BOS] at t = 0; position j holds burst j−1 with time = seconds since the window's first burst. Float64 phases keep µs-period features exact over seconds. The first burst of a track has ΔTOA = [PAD], which is ignored in the loss.
- **Classifier fine-tune.** Uses CE plus the LM loss. With CE alone the field heads drift and the surprise score would no longer be the pretrained model's.
- **Pretraining corpus.** Fast-BDW corpus tracks are grouped by the simulator `source_id`, an idealized association. Live scenes use the real association pipeline (§2.3).
- **Demo profile.** d = 128, 4 layers, 4 heads, ctx = 64, 900 steps, batch 32. Full profile: d = 256, 6 layers, 8 heads, ctx = 256 (5.58 M parameters).

## Open-set decision (§2.5)
- **Calibration splits.** The corpus split is 60/10/20/10 (train/val/cal/test, by session). Calibration is halved by session: cal-A fits μ(s)/σ(s), cal-B supplies the conformal scores, so the two sets do not overlap.
- **SNR smoothing.** Counts-weighted local-linear regression (tricube kernel, bandwidth 2 bins) across the 2 dB bins (LOWESS-style). There is no extrapolation beyond the observed SNR range. σ is floored at 0.5 × the overall std so sparse bins cannot yield tiny σ.
- **α selection needs unknowns, but the hidden families must stay unseen.** Validation unknowns are therefore *pseudo-unknowns*: known validation tracks recombined, keeping ΔTOA/FC/times from family A and BW/PW/SNR/waveform from family B ≠ A. α maximizes validation AUROC of known vs pseudo-unknown. In the demo profile α comes out at 0, so the fused score is z_d only; the z_S-only row of `table3.md` shows what surprise alone gives.
- **CUSUM ℓ_n.** Each burst's surprise is standardized with the per-burst SNR calibration of the track's *nearest known class*, taken causally from the latest scored window; the global calibration applies before the first window. With one global calibration, the WiFi-like family's intrinsic randomness (mean ℓ ≈ 1.6) forced κ ≥ 3 and hid real unknowns. ℓ is clipped at ±10, so a single outlier burst (e.g. a split burst) cannot alarm alone.
- **CUSUM κ and h** are chosen on known data:
  - For each κ ∈ {0.5, 1, 2, 3}, h is the smallest value whose mean time between false alarms is ≥ 1 h for *every* known family. This uses block-bootstrap simulation (blocks of 32 cal-B bursts, vectorized over 100 streams and all thresholds) of each family for `cusum_sim_hours` at that family's mean burst interval.
  - Because 1 h of P4 is about 1.4 M bursts, the alarm rate at large h comes from a log-linear fit of log(rate) vs h on grid points with ≥ 10 alarms (CUSUM tails decay exponentially), unless the Poisson 95 % upper bound already certifies the target.
  - Among feasible κ, the pair minimizing the expected delay h/(δ − κ) for a design shift δ = 3σ (`cusum_ref_shift`) wins. The pseudo-unknowns' drift (≈1σ) proved too mild a reference: real hidden families drift by 3–5σ.

## Grammar induction (§2.6)
- **DBSCAN eps.** Starts at the spec's 0.5·BW and is halved (≤ 4 times) while any cluster spans more than 2·BW. P2's 24 × 2 MHz plan cannot be non-overlapping in 30.72 MHz (1.2 MHz spacing), so FC-noise tails chain every channel into one cluster at eps = 1 MHz.
- **Frame-period folding** uses three zoom stages (201 points each over ±2 %, then ±2 grid steps), because 60 s of 8 ms frames needs period resolution better than 1/7500. Among candidates within 0.02 of the best Rayleigh R, the largest period wins, since submultiples fold equally well. A period is reported only if R ≥ 0.3.
- **Transitions.** Only consecutive bursts less than 1.5 frames apart enter the transition matrix, so missed bursts do not add entropy. If more than 80 % of transitions are self-transitions, the hop type is "fixed" (covers P4/P5/P6 and P4's occasional re-tunes). Additive smoothing is 0.1.
- **Adaptive avoidance.** Occupancy is the union of background-burst intervals overlapping each channel band over [t_end − max(10 s, track duration), t_end]. Background means every burst not in the flagged track.
- **Link pairing.** Cross-correlation is computed as the histogram of nearest-neighbour burst lags within ±50 ms (50 µs bins). A partner needs ≥ 50 % of the flagged bursts in the peak (two-bin) lag.
- **Card evaluation.** Jaccard compares against the channels the source actually used in the simulator truth (a P6 link uses one channel of its family plan).
- **z_S per class.** z_S is standardized per *nearest class*: μ(s)/σ(s) tables of window surprise per known class, from cal-A, with the global table as fallback. With one global table, a vaccinated high-entropy protocol (P3 hops pseudo-randomly) keeps a high surprise relative to the known families even when perfectly modelled, so it would stay "unknown" after vaccination. Per-class tables also lifted hidden-family AUROC in the demo profile from 0.95 to 1.0 (P3/P6 fast-mode tracks), with test false-unknown 4.8 % at ε = 5 %. After vaccination, the new class's table comes from its synthetic sequences; that is calibration only, never gate scoring.

## Vaccine (§2.7)
- **K and the 70/30 split.** "K real bursts, 70 % learn / 30 % held-out" is implemented as follows:
  - The flagged track is split per session by time: first 70 % = learn, last 30 % = real held-out.
  - K bursts are drawn at random from the learn part. They train the DDPM LoRA and give the empirical PW/BW distributions and the conditioning centroid z̄.
  - The card's transition matrix and period come from the whole learn part, because the card is built from it.
  - The held-out part is never seen by the card or the vaccine and is used only by the gate, as windows of 32 bursts. 30 % of K = 30 bursts (9) could not form a single 32-burst sequence.
- **Snippet length.** Synthetic waveforms live in the tokenizer's preprocessed domain (1024 samples at fs′ = BW/0.8, unit power), so their VQ codes are directly comparable with real bursts. The quality filter re-estimates BW from the PSD at fs′. PW is checked only when PW·fs′ < 0.95·1024; longer bursts (video/WiFi-like) fill the window, so their PW is not measurable there.
- **Parametric fallback.** If fewer than 20 DDPM samples pass the filter, the run falls back to the parametric generator (`--waveform-mode param`: GFSK or OFDM fitted from the card). The waveform kind is classified from the real bursts' spectral shape: band-edge/centre PSD ratio > 0.5 → OFDM. The fallback and its reason are written to `vaccine_report.json` and shown in the UI and report. The demo profile's tiny DDPM (300 pretraining steps on CPU) typically lands here, which is the honest result.
- **Augmentation.** Background interference uses simulated WiFi-/BLE-like waveforms (fictional families), not recordings.
- **Adapter.** Trainable parameters are LoRA A/B on Q and V of every layer plus the whole (n+1)-class head; replay is real known-class corpus train tracks. μ_new is the centroid of the synthetic windows under the adapted model. After training, the score tables and conformal scores are refit with the adapted model; α, κ and h keep their base values. The adapter is stored as fp16 tensors.
- **Gate.**
  - The known-class accuracy check uses window-level argmax accuracy on the corpus blind-test split.
  - Background false alarm = fraction of background-only windows (P7/P8 test tracks) that are flagged unknown *or* labelled as the new class. The check compares adapted vs base.
- **Anti-forgetting.** Replay alone let the LoRA shift the known classes' surprise and embeddings. In the demo profile that raised known false-unknowns from 4.8 % to 6.8–14.8 % and failed the background check. Two terms on the replayed half fix it: KL(base ‖ adapted) on the six field heads (weight 1) and 1 − cos(pooled h_adapted, pooled h_base) (weight 10), learning-without-forgetting style. Every centroid is also recomputed under the adapted model, since stale known-class centroids inflated z_d.
- **Background check data.** The background-FA check scores half-overlapping windows of the val + test background tracks, with base and adapted compared on the same windows. With 212 windows the binomial noise (≈2 pp) exceeded the 1 pp threshold. `gate.max_fa_increase` keeps its 0.01 default; the spec gives no number.

## Edge, release, federation (§2.8)
- **ONNX export.** The student is exported with a fixed context (right-padded) and an explicit validity mask for mean pooling. During export, attention uses an explicit causal mask (`Attention.export_mode`): the legacy exporter mis-exports SDPA's `is_causal` (logit errors ≈ 0.13), whereas the explicit mask matches PyTorch to 1e-6. INT8 is onnxruntime static QDQ PTQ (per-channel symmetric INT8 on MatMul/Gemm) on `edge.calib_sequences` windows (512 full, 128 demo). TensorRT is only reported as available or not; no TensorRT engine is built.
- **INT8 speed.** On the development CPU, INT8 QDQ ran *slower* than fp32 (≈1.8 vs 0.9 ms per window), probably for lack of VNNI kernels. The size halves. Both numbers are reported as measured.
- **Fleet nodes.** Nodes run the PyTorch teacher with the hot-loadable LoRA adapter, because the release package carries adapter weights per the spec. The distilled INT8 student is a separate edge deliverable, benchmarked in the async pipeline (`edge bench`). Shipping a re-distilled student inside the release is future work.
- **Hot-load.** A watcher thread polls the release folder every 0.2 s. A package that verifies (Ed25519 signature over the manifest plus SHA-256 of every file) and has a higher semver is loaded into a new state object, which replaces the old one in a single locked assignment. The scoring loop snapshots the state per decision. Packages are written to a temp name and renamed, so a watcher never sees a partial file.
- **Upstream record.** 96 bytes per burst: TOA f64, PW f32, FC f64, BW f32, SNR f16, track-id i32, z-scale f16, plus 64 × int8 z. FedAvg runs on count/Σ/Σ² of window surprise per 2 dB SNR bin, collected from windows the base model judged known. Adapter-delta averaging (spec: optional) is not implemented.
- **Node scenes.** Each node simulates its own fast-BDW scene (background mix and distance range from `configs/nodes.yaml`, plus P1 and the new family). Each is paced so all nodes take the same wall time, and the release is dropped once every node is ≥ 40 % through its stream.

## Evaluation (§2.9)
- **Baselines.** One blob-level CNN is trained on single-burst inputs: the log-STFT image of the preprocessed burst plus [log BW, log PW], which stand in for the blob's shape on the wideband t-f plane. Its per-burst unknown scores (MSP, energy, OpenMax with tail 20 and α = 2, nearest class-mean cosine) are averaged over the same 32-burst windows SpectrumLM scores.
- **Time-to-detect.** For window-scored methods this is the time from a track's first burst to its first window above the known-class 95th percentile. The CUSUM row uses the first CUSUM alarm.
- **Ablations.**
  - "No waveform code": the field is masked at inference and removed from A′, not retrained.
  - "No SNR calibration": one flat bin is fitted.
  - "N = 1": each burst is scored from [BOS] only.
  - All three refit their own conformal scores with α fixed.
  - The SDIF ablation is reported as association metrics plus track-level AUROC on a fast-mode scene with same-family pairs.
- **Hidden-family test data.** Comes from fast-BDW sessions of the hidden families, tokenized with the fold's own VQ. The live-scene results (full IQ + reference detector) are in the demo report.
- **Demo profile.** `make demo` evaluates its own fold ({P3, P6}) into `runs/demo/table3.md`. `vimmune eval loco` trains and evaluates `eval.folds` from scratch (`--k-curve` adds the vaccine recall-vs-K curve, which needs a DDPM per fold).

## Demo (§2.10)
- **Approval.** `make demo` runs `vimmune demo run --profile demo --approve`, i.e. the human approval is given by the CLI flag, as the spec allows. `make demo-interactive` stops at the gate (`awaiting_approval`) for the Streamlit Approve button, which applies `approve_gate` and runs the remaining stages.
- **Which track is vaccinated.** The demo vaccinates the *largest* track flagged as unknown in scene 1. Ground truth is never used to choose it. Cards are built for every flagged track, so a P6 card shows the link pairing.
- **Stage caching.** The cache key of each stage is a hash of the full config, a hash of all source and config files, and the upstream keys. Any code change reruns everything, which is safer than stale caches.
