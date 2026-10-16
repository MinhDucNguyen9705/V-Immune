"""Static, self-contained demo report (``runs/demo/demo_report.html``) built from the demo's output files.

Implements report §2.10 (static report). Mirrors the Streamlit pages so the demo can be shown without a
server. If a stage did not finish, the section says which stage and why.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vimmune.config import Config
from vimmune.demo import figs


def _tbl(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def f(v):
        if isinstance(v, float):
            return "—" if v != v else floatfmt.format(v)
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{head}</tr>{body}</table>"


def _img(p: Path) -> str:
    return f"<img src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}' style='max-width:100%'>"


def _md(md: str) -> str:
    out, rows = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if rows:
            out.append(_tbl(pd.DataFrame(rows[1:], columns=rows[0])))
            rows = []
        if line.startswith("#"):
            out.append(f"<h4>{html.escape(line.lstrip('# '))}</h4>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    if rows:
        out.append(_tbl(pd.DataFrame(rows[1:], columns=rows[0])))
    return "".join(out)


def build_demo_report(root: Path, cfg: Config) -> Path:
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    parts: list[str] = ["<h1>V-IMMUNE demo — first contact to immunity</h1>",
                        "<p>Receive-only analysis of simulated, fictional protocol families. Every number on this page is read from "
                        f"files written by the demo run in <code>{html.escape(str(root))}</code>.</p>"]
    first = [True]

    def plot(fig) -> str:
        s = fig.to_html(full_html=False, include_plotlyjs="inline" if first[0] else False)
        first[0] = False
        return s

    def missing(stage: str) -> str:
        st = status.get(stage, {})
        return f"<p class='warn'>Stage <b>{stage}</b> did not complete: {html.escape(str(st.get('status', 'not run')))} — {html.escape(str(st.get('error') or st.get('reason') or ''))}</p>"

    # stage table
    rows = [{"stage": s, "status": v.get("status"), "seconds": v.get("seconds", float("nan"))} for s, v in status.items() if not s.startswith("_")]
    parts += ["<h2>Pipeline stages</h2>", _tbl(pd.DataFrame(rows), "{:.1f}")]
    total = sum(r["seconds"] for r in rows if r["seconds"] == r["seconds"])
    parts.append(f"<p>Total compute time of all stages: {total / 60:.1f} min.</p>")

    s1 = root / "scene1"
    if (s1 / "tracks_base.parquet").exists():
        info = json.loads((s1 / "detect_info.json").read_text())
        tr = pd.read_parquet(s1 / "tracks_base.parquet")
        parts += ["<h2>1. Scene 1 — first contact</h2>",
                  f"<p>{info['n_truth']} truth bursts, {info['n_detected']} detected (recall on isolated bursts at SNR ≥ 0 dB: "
                  f"{info['recall_isolated_snr_ge_0']:.3f}); association purity {info['association']['purity']:.3f}, "
                  f"completeness {info['association']['completeness']:.3f}; {info['released_tracks']} tracks released to SpectrumLM.</p>",
                  plot(figs.waterfall(s1)),
                  "<h2>2. Tracks</h2>",
                  _tbl(tr[["track_id", "n_bursts", "truth_family", "bw_mhz", "pw_ms", "snr_db", "decision", "zS_median", "p_min", "cusum_ttd_s"]]),
                  "<p>truth_family is simulator ground truth shown for evaluation only.</p>",
                  "<h2>3. Surprise monitor</h2>", plot(figs.cusum_fig(s1, "base", tr))]
    else:
        parts.append(missing("scene1"))

    cards = root / "cards" / "cards.json"
    if cards.exists():
        parts.append("<h2>4. Protocol ID Cards</h2>")
        for c in json.loads(cards.read_text()):
            d = root / "cards" / c["dir"]
            parts += [f"<h3>Track {c['track_id']}</h3>", _img(d / "card.png"), _md((d / "card.md").read_text())]
    else:
        parts.append(missing("cards"))

    vac = root / "vaccine"
    if (vac / "vaccine_report.json").exists():
        rep = json.loads((vac / "vaccine_report.json").read_text())
        gate = json.loads((vac / "gate_report.json").read_text())
        w = rep["waveforms"]
        mode = w["mode"]
        label = "DDPM (LoRA-fine-tuned)" if mode == "ddpm" else "PARAMETRIC FALLBACK (not DDPM)"
        parts += ["<h2>5. Vaccine &amp; gate</h2>",
                  f"<p>K = {rep['k_used']} real bursts; waveform synthesis: <b>{label}</b>"
                  + (f" — reason: {html.escape(w.get('fallback_reason', ''))}" if mode != "ddpm" else "") + "; "
                  f"{rep['n_synthetic_sequences']} synthetic sequences ({rep['n_synthetic_bursts']} bursts); adapter {rep['adapter_kb']:.1f} KB; "
                  f"wall-clock flag→adapter {rep['wall_clock_s']:.0f} s.</p>"]
        smp = np.load(vac / "samples.npz")
        for i in range(2):
            parts.append(plot(figs.iq_spec(smp["real"][i], f"real burst {i}")))
            parts.append(plot(figs.iq_spec(smp["synthetic"][i], f"synthetic burst {i} ({label})")))
        for k in ("filter", "filter_param"):
            if k in w:
                parts.append(f"<p>Quality filter ({k}): " + html.escape(json.dumps({a: b for a, b in w[k].items()}, default=float)) + "</p>")
        gt = figs.gate_table(gate)
        parts += [_tbl(gt.drop(columns=["details"])), f"<p>Gate passed: <b>{gate['passed']}</b> (synthetic data used for scoring: {gate['synthetic_used_for_scoring']}).</p>"]
    else:
        parts.append(missing("vaccine"))

    fl = root / "fleet" / "fleet_report.json"
    if fl.exists():
        rep = json.loads(fl.read_text())
        nodes = pd.DataFrame([{"node": n["node_id"], "pid": n["pid"], "bursts": n["n_bursts"], "bytes/burst": n["bytes_per_burst"],
                               "upstream B/s": n["upstream_bytes_per_s"], "final version": n["final_version"],
                               "hot-loads": "; ".join(f"{e['from']}→{e['to']} ({e['load_s'] * 1e3:.0f} ms, restart={e['restart']})" for e in n["hot_loads"])}
                              for n in rep["nodes"]])
        parts += ["<h2>6. Fleet</h2>", _tbl(nodes, "{:.1f}"), plot(figs.fleet_fig(rep))]
    else:
        parts.append(missing("fleet"))

    s2 = root / "scene2" / "before_after.parquet"
    if s2.exists():
        ba = pd.read_parquet(s2)
        parts += ["<h2>7. Scene 2 — immunity (before / after)</h2>", _tbl(ba),
                  "<p>before = base SpectrumLM; after = base + released adapter. first_known_decision_s = time from the track's first burst to its first window classified (not flagged).</p>"]
    else:
        parts.append(missing("scene2"))
    if (root / "table3.md").exists():
        parts += ["<h2>Summary metrics (demo fold: hidden " + ", ".join(cfg.demo.hidden) + ")</h2>", _md((root / "table3.md").read_text())]
    edge = root / "edge" / "edge_report.json"
    if edge.exists():
        e = json.loads(edge.read_text())
        m = pd.DataFrame(e["metrics"]).T.reset_index().rename(columns={"index": "model"})
        parts += ["<h2>Edge</h2>", _tbl(m), f"<p>Student {e['student_params']:,} params vs teacher {e['teacher_params']:,}; ONNX fp32 {e['size_bytes']['fp32'] / 1e6:.2f} MB, INT8 {e['size_bytes']['int8'] / 1e6:.2f} MB; TensorRT: {e['tensorrt']}.</p>",
                  _tbl(pd.DataFrame(e["bench"]).T.reset_index().rename(columns={"index": "runtime"}), "{:.2f}")]
        if e.get("pipeline"):
            p = e["pipeline"]
            parts.append(f"<p>Async edge pipeline on {p['signal_s']:.1f} s of scene-1 IQ: {p['bursts']} bursts in {p['wall_s']:.1f} s wall "
                         f"({p['throughput_bursts_per_s']:.0f} bursts/s, real-time factor {p['realtime_factor']:.2f}); decision latency mean "
                         f"{(p['latency_mean_s'] or float('nan')):.2f} s, p99 {(p['latency_p99_s'] or float('nan')):.2f} s; peak RSS {p['peak_rss_mb']:.0f} MB.</p>")
    style = ("body{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:16px}table{border-collapse:collapse;margin:8px 0}"
             "td,th{border:1px solid #ccc;padding:3px 7px;font-size:12px}.warn{color:#b00}")
    out = root / "demo_report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>V-IMMUNE demo report</title><style>{style}</style></head><body>{''.join(parts)}</body></html>")
    return out
