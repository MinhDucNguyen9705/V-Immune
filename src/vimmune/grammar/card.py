"""Grammar induction from a flagged track -> Protocol ID Card (card.json / card.png / card.md).

Implements report §2.6: channel set (FC histogram + DBSCAN), frame period (SDIF candidates refined by
phase folding / Rayleigh R), dwell & duty cycle, hop type (transition-matrix entropy rate), adaptive
avoidance (Spearman ρ of channel usage vs background occupancy), uplink/downlink pairing (burst-time
cross-correlation), and similarity to known families.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN

from vimmune.associate.sdif import sdif_candidates
from vimmune.config import AssocCfg, Family, GrammarCfg


class Confidence(BaseModel):
    channels: float
    period: float
    hop_type: float
    adaptive: float
    pairing: float


class Card(BaseModel):
    """Protocol ID Card. Field names follow the BDW / labeling schema where one exists."""

    protocol: str
    signal_type: str
    track_id: str
    session_id: str
    n_bursts: int
    duration_s: float
    bw_hz: float
    pw_s: float
    pw_iqr_s: tuple[float, float]
    bw_iqr_hz: tuple[float, float]
    snr_db: float
    channels_hz: list[float]  # absolute FC of every channel in C
    n_channels: int
    channel_spacing_hz: float | None
    channel_usage: list[float]
    frame_period_s: float | None
    rayleigh_r: float
    dwell_s: float | None
    duty_cycle: float
    hop_type: str  # fixed | deterministic | pseudo-random | mixed
    entropy_rate: float | None
    hop_sequence_len: int | None
    adaptive: bool
    adaptive_rho: float | None
    paired_track: str | None
    pair_lag_s: float | None
    similar: list[dict]
    confidence: Confidence
    waveform: str  # 'gfsk-like' | 'ofdm-like' | 'unknown' (filled by the vaccine when IQ is available)


# ------------------------------------------------------------------------------------------- pieces
def channel_set(fc: np.ndarray, bw: float, g: GrammarCfg) -> tuple[np.ndarray, np.ndarray]:
    """DBSCAN (eps = 0.5·BW, min_samples = 5) on FC -> channel centers (sorted) and per-burst labels.

    Overlapping channel plans (spacing < BW + FC spread) chain into one cluster spanning several BW; then
    eps is halved (≤ 4 times) until no cluster spans more than 2·BW.
    """
    eps = g.dbscan_eps_bw * bw
    for _ in range(5):
        lab = DBSCAN(eps=eps, min_samples=g.dbscan_min_samples).fit_predict(fc[:, None])
        spans = [np.ptp(fc[lab == k]) for k in np.unique(lab) if k >= 0]
        if not spans or max(spans) <= 2 * bw:
            break
        eps /= 2
    ids = [k for k in np.unique(lab) if k >= 0]
    centers = np.array([np.median(fc[lab == k]) for k in ids])
    order = np.argsort(centers)
    remap = {ids[o]: i for i, o in enumerate(order)}
    return centers[order], np.array([remap.get(k, -1) for k in lab])


def rayleigh(toa: np.ndarray, period: float) -> float:
    return float(np.abs(np.mean(np.exp(2j * np.pi * (toa - toa[0]) / period))))


def fold_refine(toa: np.ndarray, tau: float, g: GrammarCfg) -> tuple[float, float]:
    """Phase folding around τ (±fold_search, two zoom stages). Returns (period, Rayleigh R)."""
    lo, hi = tau * (1 - g.fold_search), tau * (1 + g.fold_search)
    best = tau
    for _ in range(3):
        grid = np.linspace(lo, hi, g.fold_steps)
        r = np.abs(np.exp(2j * np.pi * np.outer(1 / grid, toa - toa[0])).mean(1))
        best = float(grid[int(np.argmax(r))])
        step = grid[1] - grid[0]
        lo, hi = best - 2 * step, best + 2 * step
    return best, rayleigh(toa, best)


def frame_period(toa: np.ndarray, a: AssocCfg, g: GrammarCfg, max_period: float = 0.15) -> tuple[float | None, float]:
    """Best SDIF candidate after folding; ties (within 0.02 R) go to the larger period (submultiples fold too)."""
    cands = sorted({c for o in (1, 2, 3) for c in sdif_candidates(np.sort(toa), o, a, max_period)})
    if not cands:
        d = np.diff(np.sort(toa))
        cands = [float(np.median(d))] if len(d) else []
    res = [fold_refine(np.sort(toa), c, g) for c in cands]
    if not res:
        return None, 0.0
    rmax = max(r for _, r in res)
    p, r = max((x for x in res if x[1] >= rmax - 0.02), key=lambda x: x[0])
    return (p if r >= 0.3 else None), r


def transition_entropy(ch: np.ndarray, ok_pair: np.ndarray, n: int, smoothing: float) -> tuple[float, np.ndarray]:
    """Normalized entropy rate Ĥ of the smoothed transition matrix (stationary-distribution weighted)."""
    C = np.full((n, n), smoothing)
    np.add.at(C, (ch[:-1][ok_pair], ch[1:][ok_pair]), 1.0)
    P = C / C.sum(1, keepdims=True)
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = np.abs(pi) / np.abs(pi).sum()
    H = -np.sum(pi[:, None] * P * np.log(P)) / np.log(n)
    return float(H), P


def sequence_length(toa: np.ndarray, ch: np.ndarray, period: float, max_len: int = 128) -> int | None:
    """Smallest lag L (in frame slots) at which the channel sequence repeats (≥ 90 % agreement)."""
    slot = np.round((toa - toa[0]) / period).astype(int)
    m = dict(zip(slot.tolist(), ch.tolist()))
    for L in range(1, max_len + 1):
        pairs = [(m[s], m[s + L]) for s in m if s + L in m and m[s] >= 0 and m[s + L] >= 0]
        if len(pairs) >= 10 and np.mean([a == b for a, b in pairs]) >= 0.9:
            return L
    return None


def occupancy(bg: pd.DataFrame, centers: np.ndarray, bw: float, t0: float, t1: float) -> np.ndarray:
    """Fraction of [t0, t1] during which background bursts overlap each channel band."""
    occ = np.zeros(len(centers))
    if len(bg) == 0:
        return occ
    b = bg[(bg.tod_s > t0) & (bg.toa_s < t1)]
    lo, hi = (b.fc_hz - b.bw_hz / 2).to_numpy(), (b.fc_hz + b.bw_hz / 2).to_numpy()
    for k, c in enumerate(centers):
        sel = (hi > c - bw / 2) & (lo < c + bw / 2)
        iv = np.sort(np.stack([b.toa_s.to_numpy()[sel].clip(t0, t1), b.tod_s.to_numpy()[sel].clip(t0, t1)], 1), 0)
        tot, cur_s, cur_e = 0.0, None, None
        for s, e in iv:  # union of intervals
            if cur_e is None or s > cur_e:
                tot += (cur_e - cur_s) if cur_e is not None else 0.0
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        tot += (cur_e - cur_s) if cur_e is not None else 0.0
        occ[k] = tot / max(t1 - t0, 1e-9)
    return occ


def pairing(toa: np.ndarray, others: dict[str, np.ndarray], g: GrammarCfg) -> tuple[str | None, float | None, float]:
    """Partner track with a stable burst-time lag (peak of the lag histogram ≥ link_min_frac of bursts)."""
    best: tuple[str | None, float | None, float] = (None, None, 0.0)
    nb = int(2 * g.link_max_lag_s / g.link_bin_s)
    for key, ot in others.items():
        ot = np.sort(ot)
        j = np.searchsorted(ot, toa)
        lags = []
        for i, t in zip(j, toa):
            for k in (i - 1, i):
                if 0 <= k < len(ot) and abs(ot[k] - t) <= g.link_max_lag_s and ot[k] != t:
                    lags.append(ot[k] - t)
        if not lags:
            continue
        lags = np.asarray(lags)
        h, e = np.histogram(lags, bins=nb, range=(-g.link_max_lag_s, g.link_max_lag_s))
        k = int(np.argmax(h + np.r_[h[1:], 0]))
        frac = (h[k] + (h[k + 1] if k + 1 < nb else 0)) / len(toa)
        if frac > best[2]:
            sel = lags[(lags >= e[k]) & (lags < e[min(k + 2, nb)])]
            best = (key, float(np.median(sel)), float(frac))
    return best if best[2] >= g.link_min_frac else (None, None, best[2])


def card_vector(c: Card) -> np.ndarray:
    """Normalized card-parameter vector used for similarity."""
    return np.array([np.log10(c.bw_hz), np.log10(c.pw_s), np.log10(c.frame_period_s or 1.0),
                     np.log2(max(c.n_channels, 1)), c.entropy_rate or 0.0, c.duty_cycle])


CARD_SCALE = np.array([0.3, 0.3, 0.3, 1.0, 0.3, 0.2])


# ------------------------------------------------------------------------------------------- main entry
def induce_card(track: pd.DataFrame, background: pd.DataFrame, others: dict[str, np.ndarray], g: GrammarCfg,
                a: AssocCfg, track_id: str = "t0", h: np.ndarray | None = None,
                refs: dict[str, dict] | None = None) -> Card:
    """Build a Protocol ID Card from a flagged track's BDW rows (≤ ``max_track_s`` from its start)."""
    tr = track.sort_values("toa_s")
    tr = tr[tr.toa_s <= tr.toa_s.iloc[0] + g.max_track_s]
    toa, fc, bw_all, pw_all = tr.toa_s.to_numpy(), tr.fc_hz.to_numpy(), tr.bw_hz.to_numpy(), tr.pw_s.to_numpy()
    bw, pw = float(np.median(bw_all)), float(np.median(pw_all))
    dur = float(toa[-1] - toa[0] + pw)
    centers, ch = channel_set(fc, bw, g)
    n_ch = len(centers)
    usage = np.bincount(ch[ch >= 0], minlength=n_ch) / max(1, (ch >= 0).sum())
    spacing = float(np.median(np.diff(centers))) if n_ch >= 2 else None
    period, R = frame_period(toa, a, g)
    duty = float(pw_all.sum() / max(dur, 1e-9))

    # hop type
    H, seq_len, dwell, hop = None, None, None, "fixed"
    conf_hop = 1.0
    if n_ch >= 2:
        if period:  # skip transitions across missed bursts (gap ≥ 1.5 frames)
            gap = np.diff(toa) / period
            ok = (gap < 1.5) & (ch[:-1] >= 0) & (ch[1:] >= 0)
        else:
            ok = (ch[:-1] >= 0) & (ch[1:] >= 0)
        H, _ = transition_entropy(np.where(ch >= 0, ch, 0), ok, n_ch, g.trans_smoothing)
        stay = float(np.mean(ch[:-1][ok] == ch[1:][ok])) if ok.any() else 0.0
        if stay > 0.8:  # mostly self-transitions: a fixed channel that occasionally re-tunes
            hop = "fixed"
            conf_hop = stay
        elif H < g.det_entropy:
            hop = "deterministic"
            seq_len = sequence_length(toa, ch, period) if period else None
            conf_hop = (g.det_entropy - H) / g.det_entropy
        elif H > g.rand_entropy:
            hop = "pseudo-random"
            conf_hop = (H - g.rand_entropy) / (1 - g.rand_entropy)
        else:
            hop = "mixed"
            conf_hop = min(H - g.det_entropy, g.rand_entropy - H) / ((g.rand_entropy - g.det_entropy) / 2)
        runs = np.diff(np.flatnonzero(np.r_[True, ch[1:] != ch[:-1], True]))
        dwell = float(np.mean(runs) * period) if period else None
    elif period:
        dwell = dur

    # adaptive avoidance: usage vs background occupancy over the preceding 10 s (and the track span)
    t1 = float(toa[-1])
    t0 = t1 - max(g.adaptive_window_s, dur)
    occ = occupancy(background, centers, bw, t0, t1) if n_ch >= 3 else np.zeros(n_ch)
    rho = None
    if n_ch >= 3 and np.ptp(occ) > 0 and np.ptp(usage) > 0:
        rho = float(spearmanr(usage, occ).statistic)
    adaptive = rho is not None and rho < g.adaptive_rho

    partner, lag, pfrac = pairing(toa, others, g)

    snr = float(np.median(tr.snr_db))
    kind = "video" if bw >= 5e6 and duty > 0.3 else ("telemetry" if partner else "control")
    card = Card(
        protocol=f"UNK-{abs(hash((round(bw, -4), round(pw, 5), n_ch))) % 10000:04d}", signal_type=kind,
        track_id=str(track_id), session_id=str(tr.session_id.iloc[0]) if "session_id" in tr else "",
        n_bursts=len(tr), duration_s=dur, bw_hz=bw, pw_s=pw,
        pw_iqr_s=tuple(np.percentile(pw_all, [25, 75]).tolist()), bw_iqr_hz=tuple(np.percentile(bw_all, [25, 75]).tolist()),
        snr_db=snr, channels_hz=centers.tolist(), n_channels=n_ch, channel_spacing_hz=spacing, channel_usage=usage.tolist(),
        frame_period_s=period, rayleigh_r=R, dwell_s=dwell, duty_cycle=duty, hop_type=hop, entropy_rate=H,
        hop_sequence_len=seq_len, adaptive=adaptive, adaptive_rho=rho, paired_track=partner, pair_lag_s=lag,
        similar=[], waveform="unknown",
        confidence=Confidence(channels=float((ch >= 0).mean()), period=R, hop_type=float(np.clip(conf_hop, 0, 1)),
                              adaptive=float(abs(rho)) if rho is not None else 0.0, pairing=pfrac),
    )
    if refs:
        card.similar = similarity(card, h, refs)
    return card


def similarity(card: Card, h: np.ndarray | None, refs: dict[str, dict], k: int = 3) -> list[dict]:
    """Top-k known families by normalized card-parameter distance + (1 − cos(h, μ_k))."""
    v = card_vector(card)
    out = []
    for fam, r in refs.items():
        dv = float(np.linalg.norm((v - np.asarray(r["vector"])) / CARD_SCALE) / np.sqrt(len(v)))
        dh = 0.0
        if h is not None and r.get("mu") is not None:
            hn = h / (np.linalg.norm(h) + 1e-12)
            dh = float(1 - hn @ np.asarray(r["mu"]))
        out.append({"family": fam, "distance": dv + dh, "card_distance": dv, "embedding_distance": dh})
    return sorted(out, key=lambda x: x["distance"])[:k]


# ------------------------------------------------------------------------------------------- outputs
def write_card(card: Card, track: pd.DataFrame, out: Path, truth: dict | None = None, P: np.ndarray | None = None) -> None:
    """card.json, card.md and card.png (waterfall, channel usage, transition matrix, folded phase)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    (out / "card.json").write_text(card.model_dump_json(indent=1))
    tr = track.sort_values("toa_s")
    toa, fc = tr.toa_s.to_numpy(), tr.fc_hz.to_numpy()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    c0 = np.mean(card.channels_hz) if card.channels_hz else fc.mean()
    for t, f, p, b in zip(toa, fc, tr.pw_s, tr.bw_hz):
        ax[0, 0].add_patch(plt.Rectangle((t - toa[0], (f - c0 - b / 2) / 1e6), p, b / 1e6, color="C0", alpha=0.7))
    ax[0, 0].set_xlim(0, min(toa[-1] - toa[0] + 0.01, 0.5))
    ax[0, 0].set_ylim((fc.min() - c0) / 1e6 - 1, (fc.max() - c0) / 1e6 + 1)
    ax[0, 0].set(title="Track waterfall (first 0.5 s)", xlabel="time since first burst (s)", ylabel="FC offset (MHz)")
    ax[0, 1].bar(range(card.n_channels), card.channel_usage, color="C1")
    ax[0, 1].set(title=f"Channel usage ({card.n_channels} channels, Δf={(card.channel_spacing_hz or 0) / 1e6:.2f} MHz)", xlabel="channel", ylabel="fraction")
    if card.n_channels >= 2:
        g = GrammarCfg()
        centers = np.asarray(card.channels_hz)
        ch = np.argmin(np.abs(fc[:, None] - centers[None]), 1)
        _, Pm = transition_entropy(ch, np.ones(len(ch) - 1, bool), card.n_channels, g.trans_smoothing)
        im = ax[1, 0].imshow(Pm if P is None else P, cmap="viridis")
        fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set(title=f"Transition matrix  Ĥ={card.entropy_rate if card.entropy_rate is not None else float('nan'):.2f} ({card.hop_type})", xlabel="to", ylabel="from")
    if card.frame_period_s:
        ph = ((toa - toa[0]) / card.frame_period_s) % 1.0
        ax[1, 1].hist(ph, bins=50, color="C2")
    ax[1, 1].set(title=f"Folded phase  T={1e3 * (card.frame_period_s or 0):.3f} ms  R={card.rayleigh_r:.3f}", xlabel="phase (cycles)")
    fig.suptitle(f"Protocol ID Card {card.protocol}  (track {card.track_id})")
    fig.tight_layout()
    fig.savefig(out / "card.png", dpi=90)
    plt.close(fig)
    (out / "card.md").write_text(card_markdown(card, truth))


def card_markdown(c: Card, truth: dict | None = None) -> str:
    def f(x, unit="", scale=1.0, nd=3):
        return "—" if x is None else f"{x * scale:.{nd}f}{unit}"

    lines = [f"# Protocol ID Card `{c.protocol}`", "",
             f"Inferred from track `{c.track_id}` ({c.n_bursts} bursts, {c.duration_s:.2f} s, median SNR {c.snr_db:.1f} dB).", "",
             "| parameter | inferred | confidence |", "|---|---|---|",
             f"| signal type | {c.signal_type} | — |",
             f"| bandwidth | {f(c.bw_hz, ' MHz', 1e-6)} | — |",
             f"| pulse width | {f(c.pw_s, ' ms', 1e3)} | — |",
             f"| channels | {c.n_channels} (spacing {f(c.channel_spacing_hz, ' MHz', 1e-6)}) | {c.confidence.channels:.2f} |",
             f"| frame period | {f(c.frame_period_s, ' ms', 1e3, 4)} | R = {c.rayleigh_r:.3f} |",
             f"| dwell per channel | {f(c.dwell_s, ' ms', 1e3)} | — |",
             f"| duty cycle | {c.duty_cycle:.3f} | — |",
             f"| hop type | {c.hop_type} (Ĥ = {f(c.entropy_rate, '', 1, 2)}{', length ' + str(c.hop_sequence_len) if c.hop_sequence_len else ''}) | {c.confidence.hop_type:.2f} |",
             f"| adaptive avoidance | {'yes' if c.adaptive else 'no'} (ρ = {f(c.adaptive_rho, '', 1, 2)}) | {c.confidence.adaptive:.2f} |",
             f"| link pairing | {c.paired_track or 'none'}{' at lag ' + f(c.pair_lag_s, ' ms', 1e3) if c.pair_lag_s is not None else ''} | {c.confidence.pairing:.2f} |",
             f"| waveform | {c.waveform} | — |", "",
             "Most similar known families: " + (", ".join(f"{s['family']} ({s['distance']:.2f})" for s in c.similar) or "—")]
    if truth:
        lines += ["", "## Inferred vs simulator ground truth", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in truth.items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- evaluation
EXPECTED_HOP = {"fixed_seq": {"deterministic"}, "prng": {"pseudo-random"}, "adaptive": {"pseudo-random", "mixed"},
                "fixed": {"fixed"}, "retune": {"fixed"}, "adv": {"deterministic"}}


def true_centers(truth: pd.DataFrame, source_id: str, center_freq: float, fam: Family) -> np.ndarray:
    """Absolute channel centers used by ``source_id`` in the simulator truth."""
    from vimmune.sim.families import channel_centers

    cc = channel_centers(fam)
    used = [c for c in sorted(truth[truth.source_id == source_id].channel.unique()) if c < len(cc)]
    return center_freq + cc[used]


def card_vs_truth(card: Card, fam: Family, centers_true_hz: np.ndarray) -> dict:
    """Channel-set Jaccard (centers matched within BW/2), frame-period relative error, hop-type correctness.

    ``centers_true_hz`` are the channels the source actually used (from the simulator truth).
    """
    inf = np.asarray(card.channels_hz)
    used = set()
    match = 0
    for c in inf:
        d = np.abs(centers_true_hz - c)
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= fam.bw_hz / 2 and j not in used:
            used.add(j)
            match += 1
    jac = match / max(1, len(inf) + len(centers_true_hz) - match)
    per_err = abs(card.frame_period_s - fam.frame_s) / fam.frame_s if card.frame_period_s and fam.frame_s else None
    return {"channel_jaccard": jac, "n_channels_true": int(len(centers_true_hz)), "n_channels_inferred": card.n_channels,
            "frame_period_true_s": fam.frame_s, "frame_period_rel_err": per_err,
            "hop_type_ok": card.hop_type in EXPECTED_HOP[fam.hop], "hop_true": fam.hop,
            "adaptive_true": fam.hop == "adaptive", "adaptive_ok": card.adaptive == (fam.hop == "adaptive")}


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))
