"""Streamlit demo UI (``make ui``). Reads everything from ``runs/demo``; never retrains.

Implements report §2.10 (interactive demo): Scene, Tracks, Surprise monitor, Protocol ID Card,
Vaccine & gate (with the human-approval button), Fleet, Before/after.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from vimmune.config import load_config
from vimmune.demo import figs
from vimmune.demo.orchestrator import ROOT, STAGES

st.set_page_config(page_title="V-IMMUNE demo", layout="wide")
cfg = load_config("demo")


def j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


status = j(ROOT / "status.json") or {}
st.sidebar.title("V-IMMUNE")
st.sidebar.caption("Receive-only analysis of simulated, fictional protocols.")
page = st.sidebar.radio("Page", ["Scene", "Tracks", "Surprise monitor", "Protocol ID Card", "Vaccine & gate", "Fleet", "Before / after"])
st.sidebar.subheader("Stages")
for s in STAGES:
    v = status.get(s, {})
    icon = {"done": "✅", "failed": "❌", "running": "⏳", "awaiting_approval": "✋", "blocked": "⛔"}.get(v.get("status"), "·")
    secs = f"({v['seconds']:.0f}s)" if "seconds" in v else ""
    st.sidebar.write(f"{icon} {s} {secs}")
    if v.get("status") == "failed":
        st.sidebar.error(f"{s}: {v.get('error')}")
        with st.sidebar.expander(f"traceback ({s})"):
            st.code(v.get("traceback", ""))
    elif v.get("status") in ("awaiting_approval", "blocked"):
        st.sidebar.caption(f"{s}: {v.get('reason', '')}")
if not status:
    st.warning("No demo run found. Run `make demo` (or `vimmune demo run --profile demo`) first.")
    st.stop()

s1, s2 = ROOT / "scene1", ROOT / "scene2"


def need(p: Path, stage: str) -> bool:
    if not p.exists():
        v = status.get(stage, {})
        st.error(f"Stage '{stage}' has no output yet: {v.get('status', 'not run')} — {v.get('error') or v.get('reason') or ''}")
        return False
    return True


if page == "Scene":
    st.header("Scene 1 — first contact")
    if need(s1 / "waterfall.npz", "scene1"):
        info = j(s1 / "detect_info.json")
        c = st.columns(4)
        c[0].metric("truth bursts", info["n_truth"])
        c[1].metric("detected", info["n_detected"])
        c[2].metric("recall (isolated, SNR≥0)", f"{info['recall_isolated_snr_ge_0']:.3f}")
        c[3].metric("tracks → SpectrumLM", info["released_tracks"])
        st.plotly_chart(figs.waterfall(s1), use_container_width=True)
        sc = j(s1 / "scene.json")
        st.subheader("Emitters (simulator ground truth)")
        st.dataframe(pd.DataFrame(sc["emitters"]).assign(name=lambda d: d.family.map(lambda f: cfg.protos.get(f).name)))

elif page == "Tracks":
    st.header("Tracks (scene 1)")
    if need(s1 / "tracks_base.parquet", "scene1"):
        tr = pd.read_parquet(s1 / "tracks_base.parquet")
        st.dataframe(tr[["track_id", "n_bursts", "truth_family", "bw_mhz", "pw_ms", "snr_db", "decision", "zS_median", "p_min", "cusum_ttd_s"]],
                     use_container_width=True)
        tid = st.selectbox("Burst timeline of track", tr.track_id.tolist())
        det = pd.read_parquet(s1 / "bdw.parquet")
        g = det[det.track_id == tid]
        st.plotly_chart(figs.go.Figure(figs.go.Scatter(x=g.toa_s, y=(g.fc_hz - cfg.sim.center_freq) / 1e6, mode="markers",
                                                       marker={"size": 5})).update_layout(xaxis_title="TOA (s)", yaxis_title="FC offset (MHz)",
                                                                                          height=380), use_container_width=True)

elif page == "Surprise monitor":
    st.header("Surprise monitor — CUSUM per track")
    if need(s1 / "cusum_base.npz", "scene1"):
        tr = pd.read_parquet(s1 / "tracks_base.parquet")
        st.plotly_chart(figs.cusum_fig(s1, "base", tr), use_container_width=True)
        st.dataframe(tr[["track_id", "truth_family", "cusum_alarm", "cusum_ttd_s", "decision"]])

elif page == "Protocol ID Card":
    st.header("Protocol ID Card")
    cards = j(ROOT / "cards" / "cards.json")
    if cards is None:
        need(ROOT / "cards" / "cards.json", "cards")
    else:
        c = st.selectbox("Flagged track", [x["track_id"] for x in cards])
        d = ROOT / "cards" / next(x["dir"] for x in cards if x["track_id"] == c)
        st.image(str(d / "card.png"))
        st.markdown((d / "card.md").read_text())

elif page == "Vaccine & gate":
    st.header("Vaccine & validation gate")
    rep, gate = j(ROOT / "vaccine" / "vaccine_report.json"), j(ROOT / "vaccine" / "gate_report.json")
    if rep is None:
        need(ROOT / "vaccine" / "vaccine_report.json", "vaccine")
    else:
        mode = rep["waveforms"]["mode"]
        if mode != "ddpm":
            st.warning(f"Waveforms: PARAMETRIC FALLBACK (not DDPM) — {rep['waveforms'].get('fallback_reason', '')}")
        smp = np.load(ROOT / "vaccine" / "samples.npz")
        cols = st.columns(2)
        for i in range(2):
            cols[0].plotly_chart(figs.iq_spec(smp["real"][i], f"real burst {i}"), use_container_width=True)
            cols[1].plotly_chart(figs.iq_spec(smp["synthetic"][i], f"synthetic burst {i} ({mode})"), use_container_width=True)
        st.json({k: v for k, v in rep["waveforms"].items()})
        st.subheader("Gate")
        st.dataframe(figs.gate_table(gate), use_container_width=True)
        st.metric("gate passed", str(gate["passed"]))
        if not gate["checks"]["human_approval"]["pass"]:
            if st.button("Approve", type="primary"):
                log = open(ROOT / "approve.log", "w")
                subprocess.Popen([sys.executable, "-m", "vimmune.cli", "demo", "approve"], stdout=log, stderr=subprocess.STDOUT)
                st.success("Approved. Release, fleet and scene 2 are running in the background (runs/demo/approve.log). Refresh in a few minutes.")
        else:
            st.success("Human approval recorded.")

elif page == "Fleet":
    st.header("Fleet — signed release, hot-load, federation")
    rep = j(ROOT / "fleet" / "fleet_report.json")
    if rep is None:
        need(ROOT / "fleet" / "fleet_report.json", "fleet")
    else:
        st.dataframe(pd.DataFrame([{k: v for k, v in n.items() if k != "hot_loads"} for n in rep["nodes"]]))
        for n in rep["nodes"]:
            ev = [json.loads(x) for x in (ROOT / "fleet" / n["node_id"] / "events.jsonl").read_text().splitlines()]
            st.caption(n["node_id"])
            st.dataframe(pd.DataFrame(ev))
        st.plotly_chart(figs.fleet_fig(rep), use_container_width=True)
        fed = rep["fedavg"]
        st.subheader("FedAvg of calibration statistics (window surprise per SNR bin)")
        st.dataframe(pd.DataFrame({"snr_bin_lo": fed["edges"][:-1], "n": fed["n"], "mu": fed["mu"], "sigma": fed["sigma"]}).query("n > 0"))

elif page == "Before / after":
    st.header("Scene 2 — immunity")
    if need(s2 / "before_after.parquet", "scene2"):
        st.dataframe(pd.read_parquet(s2 / "before_after.parquet"), use_container_width=True)
    if (ROOT / "table3.md").exists():
        st.subheader("Summary metrics (demo fold)")
        st.markdown((ROOT / "table3.md").read_text())
    if (ROOT / "demo_report.html").exists():
        st.caption(f"Static report: {ROOT / 'demo_report.html'}")
