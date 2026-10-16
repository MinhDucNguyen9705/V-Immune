"""M6: conformal false-unknown rate ≤ ε + 0.02; SNR calibration; CUSUM ARL vs Siegmund; κ/h choice."""

import numpy as np

from vimmune.openset.calib import (
    SNRCalib,
    auroc,
    choose_alpha,
    choose_h,
    cusum,
    p_value,
    siegmund_arl,
    simulate_alarm_rate,
)


def test_conformal_false_unknown_rate():
    rng = np.random.default_rng(0)
    eps = 0.05
    rates = []
    for r in range(20):
        cal = rng.standard_t(4, 300)  # heavy-tailed known-class scores
        test = rng.standard_t(4, 2000)
        rates.append(np.mean(p_value(test, cal) < eps))
    assert np.mean(rates) <= eps + 0.02
    unk = rng.standard_t(4, 500) + 4
    assert np.mean(p_value(unk, cal) < eps) > 0.5


def test_snr_calibration_removes_trend():
    rng = np.random.default_rng(1)
    snr = rng.uniform(-5, 30, 5000)
    v = 10 - 0.2 * snr + rng.normal(0, 1 + 0.05 * snr.clip(0), 5000)
    c = SNRCalib.fit(v, snr)
    z = c.z(v, snr)
    for lo in range(-5, 30, 5):
        m = (snr >= lo) & (snr < lo + 5)
        assert abs(z[m].mean()) < 0.2 and 0.7 < z[m].std() < 1.3
    # sparse bin does not explode
    assert np.isfinite(c.z(np.array([5.0]), np.array([45.0]))).all()


def test_alpha_choice():
    rng = np.random.default_rng(2)
    zs_k, zd_k = rng.normal(size=500), rng.normal(size=500)
    zs_u, zd_u = rng.normal(size=500), rng.normal(2, 1, 500)  # only z_d is informative
    a, res = choose_alpha(zs_k, zd_k, zs_u, zd_u, [round(0.1 * i, 1) for i in range(11)])
    assert a <= 0.2 and res[a] > 0.85
    assert abs(auroc(zs_k, zs_u) - 0.5) < 0.06


def test_cusum_arl_matches_siegmund():
    rng = np.random.default_rng(3)
    kappa, h = 0.5, 4.0
    series = [rng.standard_normal(5000) for _ in range(40)]
    rate = simulate_alarm_rate(series, kappa, np.array([h]), n_streams=200, n_steps=20000, block=32, rng=rng)[0]
    arl_sim, arl_th = 1 / rate, siegmund_arl(kappa, h)
    assert abs(arl_sim / arl_th - 1) < 0.25
    _, al = cusum(rng.standard_normal(1000) + 3.0, kappa, h)
    assert al and al[0] < 5  # a 3σ shift is detected within a few samples


def test_choose_h_meets_target():
    rng = np.random.default_rng(4)
    series = {"A": [rng.standard_normal(2000) for _ in range(20)], "B": [rng.standard_normal(2000) for _ in range(20)]}
    h, mtbfa = choose_h(series, {"A": 0.01, "B": 0.002}, 1.0, 1.0, 100, 2.0, 32, 0, 30.0)
    assert np.isfinite(h)
    assert all(v >= 1.0 for v in mtbfa.values())
    # check by simulation with an independent seed: alarms per hour of B-rate stream below 1
    r = simulate_alarm_rate(series["B"], 1.0, np.array([h]), 100, 40000, 32, np.random.default_rng(9))[0]
    assert r / 0.002 * 3600 <= 1.5
