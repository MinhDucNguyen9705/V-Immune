"""LoRA adapter for a new protocol + validation gate.

Implements report §2.7 (adapter training, gate). SpectrumLM is frozen; LoRA (rank 8) wraps Q and V of
every layer; the classifier head grows by one class; batches are 50 % synthetic new-class sequences and
50 % replayed real known-class sequences; loss = next-burst LM loss + CE. The gate scores only *real*
data: held-out real bursts of the new protocol, the known-class blind test, background-only tracks.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vimmune.config import Config, GateCfg
from vimmune.lm.data import Track, sample_batch
from vimmune.lm.model import LoRALinear, SpectrumLM
from vimmune.lm.train import centroids, embed_windows, lm_loss


def attach_lora(model: SpectrumLM, r: int) -> list[nn.Parameter]:
    """Freeze the model; LoRA on Q and V of all layers. Returns the LoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for blk in model.blocks:
        for name in ("q", "v"):
            base = getattr(blk.attn, name)
            if not isinstance(base, LoRALinear):
                lo = LoRALinear(base, r)
                setattr(blk.attn, name, lo)
                params += [lo.A, lo.B]
    return params


def extend_head(model: SpectrumLM, n_new: int = 1) -> nn.Linear:
    old = model.cls
    assert old is not None
    new = nn.Linear(old.in_features, old.out_features + n_new)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        new.bias[old.out_features :] = old.bias.mean()
    model.cls = new
    return new


def adapted_model(base: SpectrumLM, r: int, n_new: int = 1) -> tuple[SpectrumLM, list[nn.Parameter]]:
    m = copy.deepcopy(base)
    params = attach_lora(m, r)
    head = extend_head(m, n_new)
    return m, params + list(head.parameters())


def train_adapter(model: SpectrumLM, params: list[nn.Parameter], syn: list[Track], replay: list[Track], classes: list[str],
                  cfg: Config, seed: int, steps: int | None = None, log=None, base: SpectrumLM | None = None) -> list[float]:
    """50/50 synthetic-new / replayed-known batches; LM + CE loss; only LoRA + head train.

    If ``base`` is given, a KL(base ‖ adapted) term on the field heads and a cosine term on the pooled hidden
    state of the replayed half keep the known classes' surprise and embedding distributions (hence their
    calibration) unchanged (learning-without-forgetting / feature distillation).
    """
    v = cfg.vaccine
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=v.adapter_lr, weight_decay=0.0)
    model.train()
    hist = []
    n = steps or v.adapter_steps
    half = max(1, v.adapter_batch // 2)
    for step in range(n):
        bs = sample_batch(syn, half, cfg.lm.ctx, rng, cfg.lm.mask_prob, classes)
        br = sample_batch(replay, half, cfg.lm.ctx, rng, cfg.lm.mask_prob, classes)
        b = {k: torch.cat([bs[k], br[k]]) for k in bs}
        logits, h = model(b["inp"], b["t"], b["valid"], b["tgt"])
        ce = F.cross_entropy(model.classify(h, b["valid"]), b["label"])
        loss = lm_loss(logits, b["tgt"]) + ce
        if base is not None and v.adapter_kd_weight > 0:
            with torch.no_grad():
                bl, bh = base(br["inp"], br["t"], br["valid"], br["tgt"])
            m = br["valid"].float()[..., None]
            ha, hb = model.pool(h[half:], br["valid"]), model.pool(bh, br["valid"])
            loss = loss + v.adapter_feat_weight * (1 - F.cosine_similarity(ha, hb, dim=-1)).mean()
            kd = sum((F.kl_div(F.log_softmax(a[half:], -1), F.log_softmax(t, -1), log_target=True, reduction="none").sum(-1, keepdim=True) * m).sum() / m.sum()
                     for a, t in zip(logits, bl))
            loss = loss + v.adapter_kd_weight * kd
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.lm.grad_clip)
        opt.step()
        hist.append(loss.item())
        if log and (step + 1) % max(1, n // 5) == 0:
            log(f"adapter step {step + 1}/{n} loss {np.mean(hist[-20:]):.3f} ce {ce.item():.3f}")
    model.eval()
    return hist


def new_centroid(model: SpectrumLM, syn: list[Track], cfg: Config) -> np.ndarray:
    h, _, lab = embed_windows(model, syn, cfg.lm.ctx, cfg.openset.score_len, stride=cfg.lm.ctx)
    return centroids(h, lab, [str(lab[0])])[0]


# ------------------------------------------------------------------------------------------- adapter file
def adapter_state(model: SpectrumLM) -> dict[str, torch.Tensor]:
    """Only what the adapter adds: LoRA A/B of every Q/V and the extended classifier head."""
    sd = model.state_dict()
    return {k: v.clone() for k, v in sd.items() if k.endswith((".A", ".B")) or k.startswith("cls.")}


def save_adapter(path: str | Path, model: SpectrumLM, mu_new: np.ndarray, name: str, meta: dict[str, Any]) -> int:
    """Standalone adapter file (fp16 tensors). Returns its size in bytes."""
    obj = {"state": {k: v.half() for k, v in adapter_state(model).items()}, "mu_new": mu_new.astype(np.float32),
           "name": name, "meta": meta}
    buf = io.BytesIO()
    torch.save(obj, buf)
    Path(path).write_bytes(buf.getvalue())
    return len(buf.getvalue())


def load_adapter(base: SpectrumLM, path_or_obj: str | Path | dict) -> tuple[SpectrumLM, dict]:
    """A copy of ``base`` with the adapter's LoRA + head loaded."""
    obj = path_or_obj if isinstance(path_or_obj, dict) else torch.load(path_or_obj, map_location="cpu", weights_only=False)
    r = obj["meta"]["rank"]
    m, _ = adapted_model(base, r, obj["meta"].get("n_new", 1))
    missing = m.load_state_dict({k: v.float() for k, v in obj["state"].items()}, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return m.eval(), obj


# ------------------------------------------------------------------------------------------- gate
def window_accuracy(scored, classes_ok: list[str] | None = None) -> float:
    d = scored if classes_ok is None else scored[scored.label.isin(classes_ok)]
    return float((d.pred == d.label).mean()) if len(d) else float("nan")


def evaluate_gate(base_sc, ad_sc, heldout: list[Track], known_test: list[Track], background: list[Track], new_name: str,
                  gate: GateCfg, approved: bool) -> dict:
    """The four gate checks (real data only). ``*_sc`` are :class:`OpenSetScorer`s."""
    ho = ad_sc.score(heldout)
    recall = float(((~ho.unknown) & (ho.pred == new_name)).mean()) if len(ho) else 0.0
    kb, ka = base_sc.score(known_test), ad_sc.score(known_test)
    acc_b, acc_a = window_accuracy(kb), window_accuracy(ka)
    st = max(1, base_sc.cal.score_len // 2)  # half-overlapping windows: more background evidence
    bb, ba = base_sc.score(background, stride=st), ad_sc.score(background, stride=st)
    fa_b = float((bb.unknown | (bb.pred == new_name)).mean()) if len(bb) else 0.0
    fa_a = float((ba.unknown | (ba.pred == new_name)).mean()) if len(ba) else 0.0
    checks = {
        "recall_heldout_real": {"value": recall, "threshold": gate.min_recall, "pass": recall >= gate.min_recall, "n_windows": int(len(ho))},
        "known_accuracy_drop": {"base": acc_b, "adapted": acc_a, "drop": acc_b - acc_a, "threshold": gate.max_acc_drop,
                                "pass": (acc_b - acc_a) <= gate.max_acc_drop, "n_windows": int(len(ka))},
        "background_fa_increase": {"base": fa_b, "adapted": fa_a, "increase": fa_a - fa_b, "threshold": gate.max_fa_increase,
                                   "pass": (fa_a - fa_b) <= gate.max_fa_increase, "n_windows": int(len(ba))},
        "human_approval": {"pass": bool(approved)},
    }
    return {"checks": checks, "automatic_pass": all(c["pass"] for k, c in checks.items() if k != "human_approval"),
            "passed": all(c["pass"] for c in checks.values()), "synthetic_used_for_scoring": False}
