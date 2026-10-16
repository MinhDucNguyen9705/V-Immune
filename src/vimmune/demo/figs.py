"""Plotly figures shared by the Streamlit app and the static demo report (report §2.10).

Every figure is built from files under ``runs/demo``; nothing is recomputed or invented here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = {"P1": "#1f77b4", "P2": "#aec7e8", "P3": "#d62728", "P4": "#2ca02c", "P5": "#98df8a", "P6": "#ff7f0e",
           "P7": "#7f7f7f", "P8": "#9467bd", None: "#bcbd22"}


def waterfall(scene: Path, seconds: float = 0.4) -> go.Figure:
    w = np.load(scene / "waterfall.npz")
    det = pd.read_parquet(scene / "bdw.parquet")
    c = float(w["center"][0])
    fig = go.Figure(go.Heatmap(z=w["db"].T, x=w["t"], y=w["f"] / 1e6, colorscale="Viridis", zmin=np.percentile(w["db"], 5),
                               zmax=np.percentile(w["db"], 99.7), colorbar={"title": "dB"}, hoverinfo="skip"))
    d = det[det.toa_s < seconds]
    for fam, g in d.groupby(d.truth_protocol.fillna("?")):
        xs, ys = [], []
        for r in g.itertuples():
            f0, f1 = (r.fc_hz - c - r.bw_hz / 2) / 1e6, (r.fc_hz - c + r.bw_hz / 2) / 1e6
            xs += [r.toa_s, r.tod_s, r.tod_s, r.toa_s, r.toa_s, None]
            ys += [f0, f0, f1, f1, f0, None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line={"color": PALETTE.get(fam, "#fff"), "width": 1.5},
                        name=f"detected blobs — truth {fam}")
    fig.update_layout(title="Waterfall with detected blobs (legend = simulator ground truth, display only)", xaxis_title="time (s)",
                      yaxis_title="offset from center (MHz)", height=520, legend={"orientation": "h", "y": -0.2})
    return fig


def cusum_fig(scene: Path, tag: str, tracks: pd.DataFrame) -> go.Figure:
    z = np.load(scene / f"cusum_{tag}.npz")
    h = float(z["h"][0])
    fig = go.Figure()
    for r in tracks.itertuples():
        t, g = z[f"{r.track}|t"], z[f"{r.track}|g"]
        lab = f"track {r.track_id} ({r.truth_family})"
        fig.add_scatter(x=t, y=np.minimum(g, 3 * h), mode="lines", name=lab, line={"color": PALETTE.get(r.truth_family, "#333")})
        if r.cusum_alarm and r.cusum_ttd_s is not None:
            fig.add_scatter(x=[r.cusum_ttd_s], y=[h], mode="markers", marker={"symbol": "x", "size": 12, "color": "black"},
                            name=f"alarm {lab} @ {r.cusum_ttd_s:.2f}s", showlegend=True)
    fig.add_hline(y=h, line_dash="dash", annotation_text=f"h = {h:.1f}")
    fig.update_layout(title="Surprise monitor — CUSUM g_n per track (display capped at 3h)", xaxis_title="time since track start (s)",
                      yaxis_title="g_n", height=480)
    return fig


def iq_spec(x: np.ndarray, title: str) -> go.Figure:
    S = np.abs(np.fft.fftshift(np.fft.fft(x.reshape(-1, 64) * np.hanning(64), axis=1), axes=1)) ** 2
    fig = go.Figure(go.Heatmap(z=10 * np.log10(S.T + 1e-9), colorscale="Viridis", showscale=False))
    fig.update_layout(title=title, height=220, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="time block", yaxis_title="freq bin")
    return fig


def gate_table(gate: dict) -> pd.DataFrame:
    rows = []
    for k, c in gate["checks"].items():
        val = c.get("value", c.get("drop", c.get("increase", "")))
        rows.append({"check": k, "value": val, "threshold": c.get("threshold", "human"), "pass": "PASS" if c["pass"] else "FAIL",
                     "details": {kk: vv for kk, vv in c.items() if kk not in ("pass", "threshold", "value")}})
    return pd.DataFrame(rows)


def fleet_fig(rep: dict) -> go.Figure:
    names = [n["node_id"] for n in rep["nodes"]]
    up = [n["upstream_bytes_per_s"] for n in rep["nodes"]]
    raw = list(rep["raw_iq_bytes_per_s"].items())
    fig = go.Figure([go.Bar(x=names, y=up, name="upstream (BDW + int8 embedding)")]
                    + [go.Bar(x=names, y=[v] * len(names), name=f"raw IQ {k}") for k, v in raw])
    fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="bytes / s", title="Upstream bandwidth vs raw IQ", height=380)
    return fig
