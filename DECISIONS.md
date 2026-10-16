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
