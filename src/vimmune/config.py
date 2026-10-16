"""Typed configuration (pydantic v2) loaded from YAML profiles in ``configs/``.

Implements report §2.1 (system configuration). ``full.yaml`` holds the spec numbers; ``demo.yaml``
names ``inherit: full`` and overrides sizes/steps only (algorithms stay identical).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- protocols (§2.1, spec §4.2)
class Family(_M):
    id: str
    name: str
    kind: Literal["control", "video", "telemetry", "background"]
    modulation: Literal["gfsk", "ofdm", "noise"]
    bw_hz: float
    pw_s: float | None = None  # fixed PW
    pw_range_s: tuple[float, float] | None = None  # P7: U(lo, hi)
    timing: Literal["periodic", "csma", "advertising", "pair"]
    frame_s: float | None = None
    hop: Literal["fixed_seq", "prng", "adaptive", "fixed", "retune", "none", "adv"]
    n_channels: int = 1
    channel_spacing_hz: float = 0.0
    channel_centers_hz: list[float] | None = None  # explicit centers (offset from band center)
    hop_seed: int = 0  # family-level seed (P1 fixed sequence)
    hop_seq_len: int = 0
    retune_s: float | None = None
    pair_delay_s: float | None = None
    csma_mean_gap_s: float | None = None
    csma_difs_s: float = 50e-6
    adv_interval_s: float | None = None
    adv_rand_s: float | None = None
    adv_gap_s: float = 0.4e-3
    ofdm_scs_hz: float = 30e3
    ofdm_cp: float = 0.125


class ProtocolsCfg(_M):
    jitter_frac: float = 0.01
    cfo_ppm: float = 5.0
    gfsk_bt: float = 0.5
    gfsk_h: float = 0.5
    gfsk_bw_per_rate: float = 0.91  # -10 dB bandwidth / symbol rate (calibrated, see DECISIONS)
    families: list[Family]

    def get(self, fid: str) -> Family:
        for f in self.families:
            if f.id == fid:
                return f
        raise KeyError(fid)

    @property
    def ids(self) -> list[str]:
        return [f.id for f in self.families]


# ---------------------------------------------------------------- sim (§2.1, spec §4)
class ChannelCfg(_M):
    snr_ref_db: float = 35.0
    d_ref_m: float = 10.0
    pathloss_exp: float = 2.2
    taps_min: int = 2
    taps_max: int = 4
    max_delay_s: float = 0.3e-6
    tap_decay: float = 0.2  # LOS-dominant (Rician-like) echoes


class NoiseModelCfg(_M):
    """Fast-BDW measurement-noise model (spec §4.5). sigma = a + b / sqrt(snr_lin)."""

    # values measured from the reference detector on simulated scenes (see detect.metrics / DECISIONS)
    toa_a_s: float = 0.3e-6
    toa_b_s: float = 5.0e-6
    pw_rel_a: float = 0.002
    pw_rel_b: float = 0.02
    bw_rel_a: float = 0.02
    bw_rel_b: float = 0.13
    fc_frac_a: float = 0.025
    fc_frac_b: float = 0.01
    snr_err_db: float = 0.6
    miss_snr50_db: float = -4.2
    miss_slope_db: float = 1.7
    split_prob: float = 0.005
    merge_prob: float = 0.01


class CorpusCfg(_M):
    families: list[str] = ["P1", "P2", "P4", "P5", "P7", "P8"]
    sessions_per_family: int = 200
    bursts_per_session: int = 1024
    distance_m: tuple[float, float] = (10.0, 600.0)
    snippet_len: int = 1024
    split: tuple[float, float, float, float] = (0.6, 0.1, 0.2, 0.1)  # train/val/cal/test by session


class SimCfg(_M):
    fs: float = 30.72e6
    center_freq: float = 2.44e9
    chunk_s: float = 0.1
    iq_scale: float = 64.0
    channel: ChannelCfg = ChannelCfg()
    noise_model: NoiseModelCfg = NoiseModelCfg()
    corpus: CorpusCfg = CorpusCfg()


# ---------------------------------------------------------------- detect (spec §5)
class DetectCfg(_M):
    nfft: int = 1024
    hop: int = 256
    noise_window_s: float = 0.2
    noise_quantile: float = 0.02  # spec: median; see DECISIONS (union duty cycles approach 90%)
    noise_cap_db: float = 1.0  # per-bin floor capped at this much above the quiet (10th pct) bins
    pool: tuple[int, int] = (2, 2)  # (frames, bins) non-overlapping mean before CFAR
    smooth: tuple[int, int] = (12, 8)  # total (frames, bins) averaging before CFAR, incl. pooling
    cfar_guard: tuple[int, int] = (2, 2)  # pooled cells
    cfar_train: tuple[int, int] = (6, 6)  # pooled cells
    pfa: float = 1e-4
    closing: tuple[int, int] = (3, 7)  # pooled cells (bridges OFDM DC nulls)
    min_frames: int = 8
    min_bins: int = 8
    bw_db: float = -10.0
    iq_bw_factor: float = 1.2
    iq_max_len: int = 2048
    block_s: float = 0.1
    margin_s: float = 0.005
    workers: int = 8


# ---------------------------------------------------------------- tokenizer (§2.2)
class TokenizerCfg(_M):
    dtoa_bins: int = 256
    dtoa_range_s: tuple[float, float] = (1e-6, 1.0)
    fc_bins: int = 512
    bw_bins: int = 128
    bw_min_hz: float = 10e3
    pw_bins: int = 128
    pw_range_s: tuple[float, float] = (1e-6, 0.1)
    snr_range_db: tuple[float, float] = (-15.0, 40.0)
    snr_bins: int = 56
    wave_len: int = 1024
    occupancy: float = 0.8
    lpf_factor: float = 1.2
    z_dim: int = 64
    codebook: int = 512
    channels: list[int] = [32, 64, 64, 64]
    commitment: float = 0.25
    ema_decay: float = 0.99
    steps: int = 20000
    batch: int = 256
    lr: float = 1e-3
    max_train_bursts: int = 100000


# ---------------------------------------------------------------- association (§2.3)
class AssocCfg(_M):
    window_s: float = 2.0
    step_s: float = 0.5
    cos_min: float = 0.85
    dlogbw_max: float = 0.1
    dlogpw_max: float = 0.2
    sdif_x: float = 0.3
    sdif_k: float = 0.3
    sdif_bins: int = 2000
    sdif_max_order: int = 3
    sdif_tol: float = 0.02
    sdif_min_train: int = 8
    sdif_max_missing: int = 8
    sdif_min_fill: float = 0.5  # pulses / period slots spanned by a train
    min_track_bursts: int = 32
    max_missed_windows: int = 2
    match_max_cost: float = 1.5


# ---------------------------------------------------------------- SpectrumLM (§2.4)
class LMCfg(_M):
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ctx: int = 256
    dropout: float = 0.1
    n_time_freqs: int = 32
    time_period_range_s: tuple[float, float] = (10e-6, 10.0)
    mask_prob: float = 0.15
    lr: float = 3e-4
    warmup: int = 2000
    steps: int = 30000
    batch: int = 64
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    ft_steps: int = 3000
    ft_lr: float = 1e-4
    autoregressive_fields: bool = False


# ---------------------------------------------------------------- open set (§2.5)
class OpenSetCfg(_M):
    score_len: int = 32
    snr_bin_db: float = 2.0
    alpha_grid: list[float] = [round(0.1 * i, 1) for i in range(11)]
    epsilon: float = 0.05
    cusum_kappa_grid: list[float] = [0.5, 1.0, 2.0, 3.0]
    cusum_h_max: float = 200.0
    cusum_ref_shift: float = 3.0  # design shift δ (σ units) for which (κ, h) minimise the detection delay
    ell_clip: float = 10.0  # robust CUSUM: |ℓ_n| clipped so one outlier burst cannot alarm alone
    cusum_target_h: float = 1.0  # mean time between false alarms per known source (hours)
    cusum_sim_streams: int = 200
    cusum_sim_hours: float = 4.0
    cusum_block: int = 32
    smooth_bandwidth_bins: float = 2.0


# ---------------------------------------------------------------- grammar (§2.6)
class GrammarCfg(_M):
    max_track_s: float = 60.0
    dbscan_eps_bw: float = 0.5
    dbscan_min_samples: int = 5
    hist_bin_bw: float = 0.25
    trans_smoothing: float = 0.1
    det_entropy: float = 0.2
    rand_entropy: float = 0.85
    adaptive_rho: float = -0.5
    adaptive_window_s: float = 10.0
    fold_search: float = 0.02
    fold_steps: int = 201
    link_max_lag_s: float = 0.05
    link_bin_s: float = 50e-6
    link_min_frac: float = 0.5


# ---------------------------------------------------------------- vaccine (§2.7)
class GateCfg(_M):
    min_recall: float = 0.9
    max_acc_drop: float = 0.01
    max_fa_increase: float = 0.01


class DDPMCfg(_M):
    channels: list[int] = [64, 128, 256]
    cond_dim: int = 128
    T: int = 1000
    steps: int = 20000
    batch: int = 64
    lr: float = 2e-4
    lora_rank: int = 4
    lora_steps: int = 500
    lora_lr: float = 1e-3
    ddim_steps: int = 50
    n_samples: int = 2000
    sample_batch: int = 250
    max_train_bursts: int = 50000


class VaccineCfg(_M):
    k: int = 30
    k_list: list[int] = [5, 10, 20, 50]
    learn_frac: float = 0.7
    n_seq: int = 500
    seq_len: int = 64
    waveform_mode: Literal["ddpm", "param"] = "ddpm"
    filter_cos: float = 0.8
    filter_rel: float = 0.10
    aug_snr_db: tuple[float, float] = (-10.0, 20.0)
    aug_sir_db: tuple[float, float] = (0.0, 20.0)
    aug_overlap_prob: float = 0.3
    lora_rank: int = 8
    adapter_steps: int = 2000
    adapter_lr: float = 1e-3
    adapter_batch: int = 32
    adapter_kd_weight: float = 1.0  # KL(base ‖ adapted) on replayed known data (anti-forgetting)
    adapter_feat_weight: float = 10.0  # 1 − cos(pooled h_adapted, pooled h_base) on replayed known data
    ddpm: DDPMCfg = DDPMCfg()
    gate: GateCfg = GateCfg()


# ---------------------------------------------------------------- edge (§2.8)
class EdgeCfg(_M):
    student_layers: int = 2
    student_d: int = 192
    student_heads: int = 4
    distill_T: float = 2.0
    distill_steps: int = 5000
    distill_lr: float = 5e-4
    distill_batch: int = 32
    hidden_mse_weight: float = 1.0
    calib_sequences: int = 512
    rescore_every: int = 32
    watch_interval_s: float = 0.2
    fed_rounds: int = 3


class EvalCfg(_M):
    folds: list[list[str]] = [["P3", "P6"], ["P2", "P5"], ["P1", "P8"], ["P4", "P7", "P6"]]
    n_seq_per_family: int = 200
    k_list: list[int] = [5, 10, 20, 30, 50]


class DemoCfg(_M):
    hidden: list[str] = ["P3", "P6"]
    scene1: str = "configs/scenes/first_contact.yaml"
    scene2: str = "configs/scenes/immunity.yaml"
    unknown_family: str = "P3"


class Config(_M):
    profile: str = "full"
    seed: int = 1234
    device: str = "auto"
    sim: SimCfg = SimCfg()
    detect: DetectCfg = DetectCfg()
    tokenizer: TokenizerCfg = TokenizerCfg()
    assoc: AssocCfg = AssocCfg()
    lm: LMCfg = LMCfg()
    openset: OpenSetCfg = OpenSetCfg()
    grammar: GrammarCfg = GrammarCfg()
    vaccine: VaccineCfg = VaccineCfg()
    edge: EdgeCfg = EdgeCfg()
    eval: EvalCfg = EvalCfg()
    demo: DemoCfg = DemoCfg()
    protocols: ProtocolsCfg | None = None

    @property
    def protos(self) -> ProtocolsCfg:
        assert self.protocols is not None
        return self.protocols


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_profile(path: Path) -> dict[str, Any]:
    d = yaml.safe_load(path.read_text()) or {}
    parent = d.pop("inherit", None)
    if parent:
        d = _deep_merge(_read_profile(path.parent / f"{parent}.yaml"), d)
    return d


def load_protocols(path: Path | None = None) -> ProtocolsCfg:
    return ProtocolsCfg.model_validate(yaml.safe_load((path or CONFIG_DIR / "protocols.yaml").read_text()))


def load_config(profile: str = "demo", overrides: dict[str, Any] | None = None) -> Config:
    """Load ``configs/<profile>.yaml`` (or a path) plus ``protocols.yaml`` into a :class:`Config`."""
    p = Path(profile)
    path = p if p.suffix == ".yaml" else CONFIG_DIR / f"{profile}.yaml"
    d = _deep_merge(_read_profile(path), overrides or {})
    cfg = Config.model_validate(d)
    if cfg.protocols is None:
        cfg.protocols = load_protocols()
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (spec §1 determinism)."""
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def get_device(cfg: Config) -> str:
    import torch

    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"
