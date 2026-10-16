"""Knowledge distillation of SpectrumLM into a small edge student, ONNX export and INT8 PTQ.

Implements report §2.8 (distillation, export). Loss = Σ_heads KL(teacher_T ‖ student_T)·T² over the six
field heads and the class head, + MSE between the projected student hidden state and the teacher's
(Hinton et al. 2015). The student is exported to ONNX (fixed context, right-padded) and quantized with
onnxruntime static INT8 PTQ on calibration sequences; TensorRT is used only if importable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vimmune.config import Config
from vimmune.lm.data import Track, sample_batch, scoring_windows, to_batch, window_arrays
from vimmune.lm.model import Attention, SpectrumLM


def student_cfg(cfg: Config):
    e = cfg.edge
    return cfg.lm.model_copy(update={"d_model": e.student_d, "n_layers": e.student_layers, "n_heads": e.student_heads,
                                     "autoregressive_fields": False})


def distill(teacher: SpectrumLM, tracks: list[Track], classes: list[str], cfg: Config, seed: int, steps: int | None = None,
            log=None) -> tuple[SpectrumLM, list[float]]:
    """Train a student (edge.student_* sizes) to match the teacher's field heads, class head and hidden state."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    e = cfg.edge
    T = e.distill_T
    student = SpectrumLM(teacher.vocab_sizes, student_cfg(cfg), len(classes))
    proj = nn.Linear(e.student_d, teacher.cfg.d_model)
    opt = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=e.distill_lr)
    teacher.eval()
    student.train()
    hist = []
    n = steps or e.distill_steps
    for step in range(n):
        b = sample_batch(tracks, e.distill_batch, cfg.lm.ctx, rng, 0.0)
        with torch.no_grad():
            tl, th = teacher(b["inp"], b["t"], b["valid"])
            tc = teacher.classify(th, b["valid"])
        sl, sh = student(b["inp"], b["t"], b["valid"])
        m = b["valid"].float()[..., None]
        kl = sum((F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="none").sum(-1, keepdim=True) * m).sum() / m.sum()
                 for s, t in zip(sl, tl)) * T * T
        kc = F.kl_div(F.log_softmax(student.classify(sh, b["valid"]) / T, -1), F.log_softmax(tc / T, -1), log_target=True, reduction="batchmean") * T * T
        mse = ((proj(sh) - th) ** 2 * m).sum() / (m.sum() * th.shape[-1])
        loss = kl + kc + e.hidden_mse_weight * mse
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.lm.grad_clip)
        opt.step()
        hist.append(loss.item())
        if log and (step + 1) % max(1, n // 5) == 0:
            log(f"distill step {step + 1}/{n} loss {np.mean(hist[-20:]):.3f}")
    student.eval()
    return student, hist


class EdgeWrapper(nn.Module):
    """ONNX signature: (inp int64 [B, L, 6], t float64 [B, L], valid float32 [B, L]) ->
    six field log-probs, class logits, pooled hidden state."""

    def __init__(self, m: SpectrumLM):
        super().__init__()
        self.m = m

    def forward(self, inp: torch.Tensor, t: torch.Tensor, valid: torch.Tensor):
        lg, h = self.m(inp, t)
        w = valid[..., None]
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        assert self.m.cls is not None
        return (*[F.log_softmax(x, -1) for x in lg], self.m.cls(pooled), pooled)


OUT_NAMES = [f"logp_{i}" for i in range(6)] + ["cls", "h"]


def export_onnx(model: SpectrumLM, path: Path, ctx: int) -> None:
    w = EdgeWrapper(copy.deepcopy(model).eval())
    inp = torch.full((1, ctx, 6), 3, dtype=torch.long)
    t = torch.zeros(1, ctx, dtype=torch.float64)
    valid = torch.ones(1, ctx)
    Attention.export_mode = True
    try:
        torch.onnx.export(w, (inp, t, valid), str(path), input_names=["inp", "t", "valid"], output_names=OUT_NAMES,
                          dynamic_axes={k: {0: "batch"} for k in ["inp", "t", "valid", *OUT_NAMES]}, opset_version=17, dynamo=False)
    finally:
        Attention.export_mode = False


def window_feed(tracks: list[Track], ctx: int, score_len: int, limit: int | None = None) -> list[dict[str, np.ndarray]]:
    feeds = []
    for tr in tracks:
        for s, e in scoring_windows(tr, ctx, score_len):
            b = to_batch([window_arrays(tr, s, e, ctx)])
            feeds.append({"inp": b["inp"].numpy(), "t": b["t"].numpy(), "valid": b["valid"].float().numpy(), "_n": e - s})
            if limit and len(feeds) >= limit:
                return feeds
    return feeds


def quantize_int8(fp32: Path, int8: Path, calib: list[dict[str, np.ndarray]]) -> None:
    """onnxruntime static PTQ (QDQ, symmetric INT8 weights and activations on MatMul/Gemm)."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{k: v for k, v in f.items() if not k.startswith("_")} for f in calib])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"], per_channel=True)


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


class OnnxLM:
    """Minimal SpectrumLM-like scorer backed by an onnxruntime session (used by the edge runtime/bench)."""

    def __init__(self, path: Path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.sess.run(None, {k: v for k, v in feed.items() if not k.startswith("_")})
