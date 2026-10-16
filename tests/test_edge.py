"""M9: signed releases (verify/tamper), hot-load during an active run with no dropped bursts,
upstream record size (~100 B/burst), FedAvg of calibration statistics, ONNX export parity."""

import tarfile
import threading
import time

import numpy as np
import pytest
import torch

from toy import CFG, grammar_tracks, make_world
from vimmune.edge.distill import OnnxLM, export_onnx, window_feed
from vimmune.edge.node import (
    EdgeNode,
    NodeState,
    calib_stats,
    fedavg_stats,
    pack_burst,
    score_stream,
    unpack_burst,
)
from vimmune.edge.release import ReleaseError, build_release, generate_keys, verify_release
from vimmune.lm.data import to_batch, window_arrays
from vimmune.openset.scorer import OpenSetScorer
from vimmune.pipeline import write_json
from vimmune.vaccine.adapter import adapted_model, new_centroid, save_adapter, train_adapter


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def release(world, tmp_path_factory):
    m, mu, calib, tr = world
    d = tmp_path_factory.mktemp("rel")
    syn = grammar_tracks("C", 8, 7)
    model, params = adapted_model(m, 4)
    train_adapter(model, params, syn, tr["train"], ["A", "B", "C"], CFG, 0, steps=30)
    mu_ext = np.concatenate([mu, new_centroid(model, syn, CFG)[None]])
    save_adapter(d / "adapter.pt", model, mu_ext[-1], "C", {"rank": 4, "n_new": 1, "classes": ["A", "B", "C"]})
    np.save(d / "mu_ext.npy", mu_ext)
    write_json(d / "calib.json", calib.to_dict())
    priv, pub = generate_keys(d / "keys")
    pkg = build_release(d / "out" / "v-1.1.0.vpkg", {n: d / n for n in ("adapter.pt", "mu_ext.npy", "calib.json")}, "1.1.0", priv)
    return pkg, pub, d


def test_sign_verify_and_tamper(release, tmp_path):
    pkg, pub, d = release
    man, blobs = verify_release(pkg, pub)
    assert man["version"] == "1.1.0" and set(blobs) == {"adapter.pt", "mu_ext.npy", "calib.json"}
    bad = tmp_path / "bad.vpkg"  # replace a payload file, keep the manifest + signature
    with tarfile.open(pkg) as src, tarfile.open(bad, "w") as dst:
        for mem in src.getmembers():
            f = src.extractfile(mem)
            data = f.read() if f else b""
            if mem.name == "calib.json":
                data = data.replace(b"alpha", b"alphA")
                mem.size = len(data)
            import io

            dst.addfile(mem, io.BytesIO(data))
    with pytest.raises(ReleaseError):
        verify_release(bad, pub)
    _, other_pub = generate_keys(tmp_path / "k2")
    with pytest.raises(ReleaseError):
        verify_release(pkg, other_pub)


def test_hot_load_during_run_no_dropped_bursts(world, release, tmp_path):
    m, mu, calib, tr = world
    pkg, pub, _ = release
    node = EdgeNode("n0", m, NodeState("1.0.0", OpenSetScorer(m, ["A", "B"], mu, calib)), tmp_path / "rel", pub, interval=0.05)
    node.watch()
    tracks = grammar_tracks("A", 2, 30) + grammar_tracks("C", 2, 31)
    total = sum(len(t) for t in tracks)
    seen: list[float] = []

    def drop():
        while not seen or seen[-1] < 0.3:
            time.sleep(0.01)
        import shutil

        shutil.copy(pkg, tmp_path / "rel" / pkg.name)

    th = threading.Thread(target=drop)
    th.start()
    ev = score_stream(node, tracks, 32, pace_s=0.002, progress=seen.append)
    th.join()
    node.stop()
    versions = [e["version"] for e in ev]
    assert versions[0] == "1.0.0" and versions[-1] == "1.1.0"  # swapped mid-run, no restart
    assert sum(1 for _ in ev) == sum(len(t) // 32 for t in tracks)  # every scheduled re-score happened
    assert seen[-1] == 1.0 and total == sum(len(t) for t in tracks)
    hl = [e for e in node.events if e["event"] == "hot_load"]
    assert len(hl) == 1 and hl[0]["restart"] is False
    after = [e for e in ev if e["version"] == "1.1.0" and e["label"] == "C"]
    assert after and all(e["pred"] in ("A", "B", "C") for e in after)


def test_upstream_record_and_fedavg():
    z = np.random.default_rng(0).normal(size=64).astype(np.float32)
    b = pack_burst(1.25, 4e-4, 2.44e9, 1e6, 12.3, 7, z)
    assert 90 <= len(b) <= 110
    head, z2 = unpack_burst(b)
    assert head[5] == 7 and np.corrcoef(z, z2)[0, 1] > 0.999
    rng = np.random.default_rng(1)
    S, snr = rng.normal(5, 1, 3000), rng.uniform(0, 20, 3000)
    parts = [calib_stats(S[i::3], snr[i::3]) for i in range(3)]
    fed, cen = fedavg_stats(parts), calib_stats(S, snr)
    n = np.array(cen["n"])
    mu_c = np.array(cen["s1"])[n > 0] / n[n > 0]
    assert np.allclose(np.array(fed["mu"])[n > 0], mu_c)


def test_onnx_parity(world, tmp_path):
    m, _, _, tr = world
    export_onnx(m, tmp_path / "s.onnx", 32)
    lm = OnnxLM(tmp_path / "s.onnx")
    f = window_feed(tr["test"][:2], 32, 32, 3)
    out = lm.run(f[0])
    b = to_batch([window_arrays(tr["test"][0], 0, 32, 32)])
    with torch.no_grad():
        lg, _ = m(b["inp"], b["t"])
    assert np.abs(out[0] - torch.log_softmax(lg[0], -1).numpy()).max() < 1e-3
