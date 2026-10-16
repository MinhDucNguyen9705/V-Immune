"""Demo orchestrator: every stage of the story, cached under ``runs/demo/<stage>/`` by content hash.

Implements report §2.10. A stage reruns only if its key (config + code hash + upstream keys) changes.
``runs/demo/status.json`` records each stage's status, timing and, on failure, the error and traceback,
which the UI shows (nothing is skipped silently). Human approval is applied to the gate report after the
vaccine stage; release and later stages run only once the gate has fully passed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vimmune.config import REPO_ROOT, Config, load_config, seed_everything
from vimmune.pipeline import _log, read_json, write_json

ROOT = REPO_ROOT / "runs" / "demo"
STAGES = ["corpus", "tokenizer", "lm", "calibrate", "ddpm", "scene1", "cards", "vaccine", "edge", "release", "fleet",
          "scene2", "eval", "report"]
DEPS = {"tokenizer": ["corpus"], "lm": ["tokenizer"], "calibrate": ["lm"], "ddpm": ["tokenizer"], "scene1": ["calibrate"],
        "cards": ["scene1"], "vaccine": ["cards", "ddpm"], "edge": ["vaccine"], "release": ["vaccine"], "fleet": ["release"],
        "scene2": ["release"], "eval": ["calibrate"], "report": ["scene2", "fleet", "edge", "eval"]}


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO_ROOT / "src" / "vimmune").rglob("*.py")):
        if "demo/app.py" not in str(p):
            h.update(p.read_bytes())
    for p in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Demo:
    def __init__(self, profile: str = "demo", root: Path = ROOT, approve: bool = False, workers: int = 8):
        self.cfg: Config = load_config(profile)
        self.root = root
        self.approve = approve
        self.workers = workers
        root.mkdir(parents=True, exist_ok=True)
        self.status_path = root / "status.json"
        self.status = read_json(self.status_path) if self.status_path.exists() else {}
        self.chash = code_hash()
        self.known = [f for f in self.cfg.protos.ids if f not in self.cfg.demo.hidden]

    # -------------------------------------------------------------- bookkeeping
    def d(self, s: str) -> Path:
        return self.root / s

    def key(self, s: str) -> str:
        up = [self.status.get(u, {}).get("key", "") for u in DEPS.get(s, [])]
        payload = json.dumps([s, self.cfg.model_dump(mode="json"), self.chash, up], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self) -> None:
        write_json(self.status_path, self.status)

    def run_stage(self, s: str, fn: Callable[[], dict | None]) -> bool:
        k = self.key(s)
        st = self.status.get(s, {})
        if st.get("status") == "done" and st.get("key") == k:
            _log(f"[cached] {s}")
            return True
        self.status[s] = {"status": "running", "key": k, "started": time.time()}
        self.save()
        _log(f"=== stage {s}")
        t0 = time.time()
        try:
            seed_everything(self.cfg.seed)
            info = fn() or {}
            self.status[s] = {"status": "done", "key": k, "seconds": time.time() - t0, "info": info}
            ok = True
        except Exception as e:  # recorded for the UI; later stages are marked blocked
            self.status[s] = {"status": "failed", "key": k, "seconds": time.time() - t0, "error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()}
            _log(f"[red]stage {s} failed: {e}")
            ok = False
        self.save()
        return ok

    # -------------------------------------------------------------- stages
    def s_corpus(self):
        from vimmune.pipeline import stage_corpus

        if self.d("corpus").exists():
            shutil.rmtree(self.d("corpus"))
        df = stage_corpus(self.cfg, self.known, self.d("corpus"), self.workers)
        return {"bursts": len(df), "families": self.known}

    def s_tokenizer(self):
        from vimmune.pipeline import stage_tokenizer

        stage_tokenizer(self.cfg, self.d("corpus"), self.d("tokenizer"), self.workers)
        return read_json(self.d("tokenizer") / "metrics.json")

    def s_lm(self):
        from vimmune.pipeline import stage_lm

        stage_lm(self.cfg, self.d("corpus"), self.d("tokenizer"), self.d("lm"), self.known)
        m = read_json(self.d("lm") / "metrics.json")
        return {"classifier_acc": m["classifier_acc"], "n_params": m["n_params"]}

    def s_calibrate(self):
        from vimmune.pipeline import stage_calibrate

        m = stage_calibrate(self.cfg, self.d("corpus"), self.d("tokenizer"), self.d("lm"), self.d("calibrate"))
        return {"false_unknown_rate_test": m["false_unknown_rate_test"], "kappa": m["cusum"]["kappa"], "h": m["cusum"]["h"]}

    def s_ddpm(self):
        from vimmune.pipeline import stage_ddpm

        return stage_ddpm(self.cfg, self.d("corpus"), self.d("tokenizer"), self.d("ddpm"), self.workers)

    def scorer_base(self):
        from vimmune.pipeline import load_scorer

        return load_scorer(self.d("lm") / "lm.pt", self.d("calibrate") / "calib.json")

    def scorer_adapted(self):
        from vimmune.openset.calib import OpenSetCalib
        from vimmune.openset.scorer import OpenSetScorer
        from vimmune.pipeline import load_lm
        from vimmune.vaccine.adapter import load_adapter

        base, _ = load_lm(self.d("lm") / "lm.pt")
        model, obj = load_adapter(base, self.d("vaccine") / "adapter.pt")
        cal = OpenSetCalib.from_dict(read_json(self.d("vaccine") / "calib_adapted.json"))
        return OpenSetScorer(model, obj["meta"]["classes"], np.load(self.d("vaccine") / "mu_ext.npy"), cal)

    def _scene(self, name: str, scene_path: str, tag: str):
        from vimmune.demo.live import scene_tracks, score_scene, simulate_and_detect
        from vimmune.sim.scene import load_scene

        out = self.d(name)
        if out.exists():
            shutil.rmtree(out)
        info = simulate_and_detect(self.cfg, load_scene(REPO_ROOT / scene_path), out, self.d("tokenizer"))
        det, tracks = scene_tracks(self.cfg, out)
        base = score_scene(self.scorer_base(), tracks, det, out, "base")
        info["tracks"] = len(tracks)
        info["flagged"] = base[base.decision == "UNKNOWN"].track.tolist()
        return info, det, tracks, base

    def s_scene1(self):
        info, _, _, _ = self._scene("scene1", self.cfg.demo.scene1, "base")
        return info

    def s_cards(self):
        import pandas as pd

        from vimmune.demo.live import scene_tracks
        from vimmune.grammar.card import card_vs_truth, induce_card, true_centers, write_card
        from vimmune.lm.score import score_windows
        from vimmune.vaccine.run import split_learn_heldout

        det, tracks = scene_tracks(self.cfg, self.d("scene1"))
        tr = pd.read_parquet(self.d("scene1") / "tracks_base.parquet")
        flagged = tr[tr.decision == "UNKNOWN"].sort_values("n_bursts", ascending=False)
        if flagged.empty:
            raise RuntimeError("no track was flagged as unknown in scene 1")
        truth = pd.read_parquet(self.d("scene1") / "truth_full.parquet")
        out = self.d("cards")
        if out.exists():
            shutil.rmtree(out)
        cards = []
        sc = self.scorer_base()
        for _, row in flagged.iterrows():
            g = det[det.track_id == row.track_id]
            learn, _ = split_learn_heldout(g, self.cfg.vaccine.learn_frac)
            others = {f"track {k}": det[det.track_id == k].toa_s.to_numpy() for k in det.track_id.unique() if k >= 0 and k != row.track_id
                      and (det.track_id == k).sum() >= self.cfg.assoc.min_track_bursts}
            t = next(x for x in tracks if x.key.endswith(f"/{row.track_id}"))
            ws = score_windows(sc.model, [t], sc.cal.ctx, sc.cal.score_len, np.asarray(sc.cal.weights))
            refs = self._reference_cards(sc)
            card = induce_card(learn, det[det.track_id != row.track_id], others, self.cfg.grammar, self.cfg.assoc,
                               track_id=str(row.track_id), h=ws.h[-1], refs=refs)
            fam = row.truth_family
            ev = None
            if fam in self.cfg.protos.ids:
                srcs = det[(det.track_id == row.track_id)].truth_source.mode()
                f = self.cfg.protos.get(fam)
                ev = card_vs_truth(card, f, true_centers(truth, srcs.iloc[0], self.cfg.sim.center_freq, f)) if len(srcs) else None
                if ev:
                    ev = {"truth_family (simulator, display only)": fam, **ev,
                          "truth_adaptive": f.hop == "adaptive", "truth_pair_delay_ms": (f.pair_delay_s or 0) * 1e3 if f.pair_delay_s else None}
            write_card(card, learn, out / f"track{row.track_id}", ev)
            cards.append({"track_id": int(row.track_id), "dir": f"track{row.track_id}", "protocol": card.protocol,
                          "n_channels": card.n_channels, "frame_ms": (card.frame_period_s or 0) * 1e3, "hop_type": card.hop_type,
                          "adaptive": card.adaptive, "paired": card.paired_track, "truth": ev})
        write_json(out / "cards.json", cards)
        return {"cards": len(cards), "vaccinate_track": cards[0]["track_id"]}

    def _reference_cards(self, sc) -> dict:
        """Card vectors of the known families (from corpus tracks) + their SpectrumLM centroids, for similarity."""
        p = self.d("cards_refs.json")
        if p.exists():
            return read_json(p)
        from vimmune.grammar.card import card_vector, induce_card
        from vimmune.io.bdw import read_bdw

        corp = read_bdw(self.d("corpus") / "bdw.parquet")
        refs = {}
        for i, fam in enumerate(sc.classes):
            src = corp[(corp.protocol == fam) & (corp.split == "train")].groupby(["session_id", "source_id"]).size().sort_values().index[-1]
            g = corp[(corp.session_id == src[0]) & (corp.source_id == src[1])]
            c = induce_card(g, corp.iloc[:0], {}, self.cfg.grammar, self.cfg.assoc, track_id=fam)
            refs[fam] = {"vector": card_vector(c).tolist(), "mu": np.asarray(sc.mu[i]).tolist()}
        write_json(p, refs)
        return refs

    def s_vaccine(self):
        from vimmune.grammar.card import Card
        from vimmune.io.bdw import read_bdw
        from vimmune.vaccine.run import approve_gate, run_vaccine

        info = read_json(self.d("cards") / "cards.json")[0]
        det = read_bdw(self.d("scene1") / "bdw.parquet").reset_index(drop=True)
        track = det[det.track_id == info["track_id"]]
        card = Card.model_validate_json((self.d("cards") / info["dir"] / "card.json").read_text())
        out = self.d("vaccine")
        if out.exists():
            shutil.rmtree(out)
        rep = run_vaccine(self.cfg, track=track, store=self.d("scene1") / "bursts", card=card, base_lm=self.d("lm") / "lm.pt",
                          base_calib=self.d("calibrate") / "calib.json", tok_dir=self.d("tokenizer"), corpus=self.d("corpus"),
                          ddpm_path=self.d("ddpm") / "ddpm.pt", out=out, new_name=self.cfg.demo.unknown_family, approve=False)
        if self.approve:
            approve_gate(out)
        g = read_json(out / "gate_report.json")
        return {"automatic_pass": rep["gate"]["automatic_pass"], "approved": g["checks"]["human_approval"]["pass"],
                "waveform_mode": rep["waveforms"]["mode"], "adapter_kb": rep["adapter_kb"], "wall_clock_s": rep["wall_clock_s"]}

    def gate_passed(self) -> bool:
        p = self.d("vaccine") / "gate_report.json"
        return p.exists() and read_json(p)["passed"]

    def s_edge(self):
        from vimmune.pipeline import stage_edge

        rep = stage_edge(self.cfg, self.d("corpus"), self.d("tokenizer"), self.d("lm") / "lm.pt", self.d("vaccine"), self.d("edge"),
                         self.d("scene1") / "iq.sigmf-meta", self.cfg.demo.edge_bench_s)
        return {"drop_vs_teacher": rep["drop_vs_teacher"], "pipeline": rep["pipeline"]}

    def s_release(self):
        from vimmune.pipeline import stage_release

        out = self.d("release")
        if out.exists():
            shutil.rmtree(out)
        pkg = stage_release(self.cfg, self.d("vaccine"), out)
        return read_json(out / "release.json") | {"package": pkg.name}

    def s_fleet(self):
        from vimmune.pipeline import stage_fleet

        rel = read_json(self.d("release") / "release.json")
        rep = stage_fleet(self.cfg, self.d("lm") / "lm.pt", self.d("calibrate") / "calib.json", self.d("tokenizer"),
                          self.d("release") / rel["package"], Path(rel["public_key"]), self.d("fleet"), self.cfg.demo.unknown_family)
        return {"nodes": [(n["node_id"], n["final_version"], len(n["hot_loads"])) for n in rep["nodes"]]}

    def s_scene2(self):
        from vimmune.demo.live import score_scene

        info, det, tracks, base = self._scene("scene2", self.cfg.demo.scene2, "base")
        after = score_scene(self.scorer_adapted(), tracks, det, self.d("scene2"), "adapted")
        cmp = base[["track", "n_bursts", "truth_family", "decision", "zS_median", "cusum_ttd_s"]].merge(
            after[["track", "decision", "zS_median", "first_known_decision_s", "pred_last"]], on="track", suffixes=("_before", "_after"))
        cmp.to_parquet(self.d("scene2") / "before_after.parquet")
        info["before_after"] = cmp.to_dict("records")
        return info

    def s_eval(self):
        from vimmune.eval.loco import evaluate_demo_fold

        res = evaluate_demo_fold(self.cfg, self.root, self.cfg.demo.hidden)
        return {"open_set_accuracy": res["open_set_accuracy"]}

    def s_report(self):
        from vimmune.demo.report import build_demo_report

        p = build_demo_report(self.root, self.cfg)
        return {"report": str(p)}

    # -------------------------------------------------------------- driver
    def run(self, until: str | None = None) -> dict:
        t0 = time.time()
        for s in STAGES:
            blocked = [u for u in DEPS.get(s, []) if self.status.get(u, {}).get("status") != "done"]
            if blocked:
                self.status[s] = {"status": "blocked", "reason": f"waiting for {', '.join(blocked)}"}
                self.save()
                continue
            if s in ("release", "fleet", "scene2", "edge", "report") and not self.gate_passed():
                self.status[s] = {"status": "awaiting_approval",
                                  "reason": "gate not passed: press Approve in the UI or rerun with --approve (or a check failed)"}
                self.save()
                continue
            self.run_stage(s, getattr(self, f"s_{s}"))
            if until == s:
                break
        self.status["_total_seconds_this_run"] = time.time() - t0
        self.save()
        return self.status


def approve_and_continue(root: Path = ROOT, profile: str = "demo") -> dict:
    """UI button: set the human-approval flag, then run the remaining stages."""
    from vimmune.vaccine.run import approve_gate

    approve_gate(root / "vaccine")
    return Demo(profile, root, approve=True).run()


def clean(root: Path = ROOT) -> None:
    if root.exists():
        shutil.rmtree(root)
