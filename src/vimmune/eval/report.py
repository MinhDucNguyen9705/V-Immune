"""Figures (ROC, K-shot, CUSUM traces) and the self-contained HTML evaluation report.

Implements report §2.9 (outputs). Everything is read from files under the experiment directory.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


def _folds(exp: Path) -> list[Path]:
    fs = sorted(p.parent for p in exp.glob("*/fold_results.json"))
    return fs or ([exp / "eval_fold"] if (exp / "eval_fold" / "fold_results.json").exists() else [])


def make_figures(exp: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = exp / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in _folds(exp):
        tag = fold.name
        z = np.load(fold / "roc_scores.npz")
        names = sorted({k.split("|")[0] for k in z.files})
        fig, ax = plt.subplots(figsize=(7, 6))
        for n in names:
            k, u = z[f"{n}|known"], z[f"{n}|unknown"]
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(k)), np.ones(len(u))], np.r_[k, u])
            ax.plot(fpr, tpr, label=n, lw=2 if n.startswith("V-IMMUNE") else 1)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set(xlabel="FPR (known windows flagged)", ylabel="TPR (unknown windows flagged)", title=f"ROC — {tag}")
        ax.legend(fontsize=7)
        p = out / f"roc_{tag}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        if (fold / "cusum_traces.npz").exists():
            c = np.load(fold / "cusum_traces.npz")
            fig, ax = plt.subplots(figsize=(8, 4))
            for k in c.files:
                if k.endswith("|g"):
                    base = k[:-2]
                    ax.plot(c[base + "|t"], c[k], color="C3" if base.startswith("unknown") else "C0", lw=1,
                            label=base.split("|")[1] + (" (hidden)" if base.startswith("unknown") else ""))
            ax.axhline(float(c["h"][0]), color="k", ls="--", label="threshold h")
            ax.set(xlabel="time since first burst (s)", ylabel="CUSUM g_n", title=f"CUSUM traces — {tag}", yscale="symlog")
            ax.legend(fontsize=7, ncol=2)
            p = out / f"cusum_{tag}.png"
            fig.savefig(p, dpi=90, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        res = json.loads((fold / "fold_results.json").read_text())
        if res.get("k_curve"):
            kc = res["k_curve"]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot([r["k"] for r in kc], [r["recall"] for r in kc], "o-")
            ax.set(xscale="log", xlabel="K real bursts", ylabel="recall on real held-out", ylim=(0, 1.05), title=f"K-shot curve — {tag}")
            p = out / f"kshot_{tag}.png"
            fig.savefig(p, dpi=90, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    return paths


def _md_table(md: str) -> str:
    rows = [r for r in md.splitlines() if r.startswith("|")]
    out = ["<table>"]
    for i, r in enumerate(rows):
        if set(r.replace("|", "").strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in r.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
    return "\n".join(out + ["</table>"])


def build_report(exp: Path, title: str = "V-IMMUNE evaluation") -> Path:
    import plotly.graph_objects as go

    res = json.loads((exp / "results.json").read_text())
    parts = [f"<h1>{html.escape(title)}</h1>", f"<p>Profile <b>{res['profile']}</b>. All numbers below are read from files written by the code in <code>{html.escape(str(exp))}</code>.</p>"]
    if (exp / "table3.md").exists():
        parts += ["<h2>Table 3</h2>", _md_table((exp / "table3.md").read_text())]
    first = True
    for fold in _folds(exp):
        r = json.loads((fold / "fold_results.json").read_text())
        parts.append(f"<h2>Fold {html.escape(fold.name)} — hidden {', '.join(r['hidden'])}</h2>")
        parts.append(f"<p>Open-set accuracy {r['open_set_accuracy']:.3f}; known false-unknown rate {r['false_unknown_rate_known']:.3f}; "
                     f"CUSUM false alarms on known test tracks {r['cusum_false_alarm_tracks']}.</p>")
        z = np.load(fold / "roc_scores.npz")
        fig = go.Figure()
        for n in sorted({k.split("|")[0] for k in z.files}):
            k, u = z[f"{n}|known"], z[f"{n}|unknown"]
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(k)), np.ones(len(u))], np.r_[k, u])
            fig.add_scatter(x=fpr, y=tpr, name=n, mode="lines")
        fig.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=450)
        parts.append(fig.to_html(full_html=False, include_plotlyjs="inline" if first else False))
        first = False
        a = r["association_sdif_ablation"]
        parts.append("<h3>Association: same-type SDIF split ablation</h3><table><tr><th></th><th>purity</th><th>completeness</th><th>ID switches</th><th>tracks</th><th>track AUROC</th></tr>"
                     + "".join(f"<tr><td>{k}</td><td>{v['purity']:.3f}</td><td>{v['completeness']:.3f}</td><td>{v['id_switches']}</td><td>{v['n_tracks']}</td><td>{v['auroc_tracks']:.3f}</td></tr>" for k, v in a.items()) + "</table>")
        cs = r["card_summary"]
        parts.append(f"<h3>Protocol ID Cards (hidden families)</h3><p>Channel-set Jaccard {cs['channel_jaccard_mean']:.3f}; frame-period relative error {cs['frame_period_rel_err_mean']:.2e}; hop-type accuracy {cs['hop_type_accuracy']:.2f}.</p>")
        parts.append("<table><tr><th>family</th><th>source</th><th>hop (inferred)</th><th>hop ok</th><th>Jaccard</th><th>period err</th><th>adaptive</th><th>paired</th></tr>"
                     + "".join(f"<tr><td>{c['family']}</td><td>{c['source']}</td><td>{c['hop_inferred']}</td><td>{c['hop_type_ok']}</td><td>{c['channel_jaccard']:.2f}</td><td>{(c['frame_period_rel_err'] or 0):.1e}</td><td>{c['adaptive_inferred']}</td><td>{c['paired']}</td></tr>" for c in r["cards"]) + "</table>")
        if r.get("k_curve"):
            kc = r["k_curve"]
            f2 = go.Figure(go.Scatter(x=[x["k"] for x in kc], y=[x["recall"] for x in kc], mode="lines+markers"))
            f2.update_layout(title="Vaccine: recall on real held-out vs K", xaxis_type="log", xaxis_title="K", yaxis_title="recall", height=350)
            parts.append(f2.to_html(full_html=False, include_plotlyjs=False))
        if (fold / "cusum_traces.npz").exists():
            c = np.load(fold / "cusum_traces.npz")
            f3 = go.Figure()
            for k in c.files:
                if k.endswith("|g"):
                    b = k[:-2]
                    f3.add_scatter(x=c[b + "|t"], y=c[k], name=b, mode="lines", line={"color": "crimson" if b.startswith("unknown") else "steelblue"})
            f3.add_hline(y=float(c["h"][0]), line_dash="dash")
            f3.update_layout(title="CUSUM traces (red: hidden families)", xaxis_title="s", yaxis_type="log", height=400)
            parts.append(f3.to_html(full_html=False, include_plotlyjs=False))
    style = "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
    out = exp / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(parts)}</body></html>")
    make_figures(exp)
    return out
