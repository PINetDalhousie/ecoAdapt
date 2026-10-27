"""CPU-only tests of the control components (no GPU or vLLM needed).

  python -m pytest tests -q        or        python tests/test_ecoadapt.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoadapt.acceptance import AcceptancePosterior, fit_prior
from ecoadapt.config import SLO, ControlConfig
from ecoadapt.controller import EcoAdaptController, JointSelector
from ecoadapt.estimator import PerfEnergyEstimator, energy, mean_tbt, service_time
from ecoadapt.horizon import HorizonPredictor
from ecoadapt.ledger import Ledger
from ecoadapt.projector import mu, project

FREQS = (630, 1050, 1410)


class FakeEstimator:
    """Latency falls with f and grows with batch/gamma; power grows with f."""

    def predict(self, X):
        B, K, phi, gbar, g, f = X.T
        lat_ms = (12.0 + 0.3 * B + 0.004 * K + 0.02 * phi) * (1 + 0.12 * gbar) * (1410.0 / f)
        return lat_ms / 1000.0, 60.0 + 180.0 * (f / 1410.0) ** 2


def horizon(med=100, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return HorizonPredictor.fit(rng.integers(32, 2048, n), rng.integers(med // 2, 2 * med, n))


def ledger_with(n=4, prompt=256, committed=0, deadline=60.0, prior=(8.0, 2.0)):
    L = Ledger(horizon(), prior=prior)
    for i in range(n):
        e = L.new_entry(f"r{i}", prompt, 0.0, deadline, 4096)
        e.committed = committed
        L.add(e)
    return L


def selector(tbt=100.0, e2e=60.0, cap=100_000, cfg=None):
    cfg = cfg or ControlConfig(freqs=FREQS)
    return JointSelector(cfg, SLO(tbt, e2e), FakeEstimator(), cap, 16)


# -- expected committed tokens ---------------------------------------------
def test_mu():
    assert float(mu(0, 0.8)) == 1.0
    assert abs(float(mu(3, 0.0)) - 1.0) < 1e-9
    assert abs(float(mu(3, 1.0)) - 4.0) < 1e-6
    assert abs(float(mu(2, 0.5)) - 1.75) < 1e-9
    assert all(np.diff([float(mu(g, 0.7)) for g in range(6)]) > 0)


# -- acceptance posterior ------------------------------------------------
def test_posterior_counts_first_mismatch_only():
    p = AcceptancePosterior(1.0, 1.0)
    p.update(2, 5)
    assert (p.A, p.V) == (2, 1)
    p.update(5, 5)
    assert (p.A, p.V) == (7, 1)
    p.update(0, 0)
    assert p.n_obs == 2


def test_posterior_tails_bracket_mean():
    p = AcceptancePosterior()
    for _ in range(50):
        p.update(3, 4)
    assert p.quantile(0.05) < p.mean() < p.quantile(0.95)


def test_fit_prior_mean():
    a0, b0 = fit_prior([3, 1, 4, 0], [4, 4, 4, 4])
    assert abs(a0 / (a0 + b0) - 8 / 11) < 1e-9


# -- generation horizon ----------------------------------------------------
def test_horizon_quantiles():
    hp = horizon()
    med, p95 = hp.predict(512)
    assert 1 <= med <= p95
    med2, p952 = hp.predict(512, generated=p95)
    assert med2 > p95 and p952 >= med2


def test_horizon_roundtrip(tmp_path=None):
    import tempfile
    hp = horizon()
    path = Path(tempfile.mkdtemp()) / "h.json"
    hp.save(path)
    assert HorizonPredictor.load(path).predict(300) == hp.predict(300)


# -- ledger ------------------------------------------------------------------
def test_ledger_feedback_and_reestimate():
    L = ledger_with(n=1)
    e = L.entries["r0"]
    L.observe("r0", 4, 3, 5)
    assert e.committed == 4 and e.post.A == 3 and e.post.V == 1
    h = e.h_p95
    L.observe("r0", h, 0, 0)
    assert e.h_p95 > e.committed


def test_stale_acceptance_uses_prior():
    L = ledger_with(n=1, prior=(2.0, 1.0))
    for _ in range(30):
        L.observe("r0", 1, 0, 4)
    assert L.view().alpha()[0] < 0.2
    for _ in range(4):
        L.mark_epoch()
    assert abs(L.view().alpha()[0] - 2 / 3) < 1e-9


# -- projector ------------------------------------------------------------------
def test_projection_shrinks_with_gamma():
    v = ledger_with().view()
    H = [project(v, g, 0, v.alpha(), v.h_med, 16).H for g in (0, 2, 5)]
    assert H[0] > H[1] > H[2]


def test_projection_drains_and_counts_iterations():
    v = ledger_with(n=3, committed=10).view()
    t = project(v, 3, 3, v.alpha(), v.h_med, 16)
    assert t.batch[0] == 3 and t.batch[-1] >= 1
    m = float(mu(3, v.alpha()[0]))
    assert t.R[0] == 1 + np.ceil((v.h_med[0] - 10 - m) / m)


def test_peak_kv():
    v = ledger_with(prompt=256).view()
    t0 = project(v, 0, 0, v.alpha(), v.h_med, 16)
    assert np.array_equal(t0.kv_peak, t0.kv)
    t5 = project(v, 5, 5, v.alpha(), v.h_med, 16)
    assert np.all(t5.kv_peak >= t5.kv) and t5.kv_peak[0] > t5.kv[0]


# -- estimator ---------------------------------------------------------------------
def test_estimator_quantities():
    v = ledger_with(n=2, committed=5).view()
    t = project(v, 0, 0, v.alpha(), v.h_med, 16, max_iters=10_000)
    lat = np.full(t.H, 0.02)
    assert np.allclose(mean_tbt(t, lat), 0.02)
    assert np.allclose(service_time(t, lat), 0.02 * t.R)
    assert abs(energy(t, lat, np.full(t.H, 100.0)) - 2.0 * t.H) < 1e-9


def test_capped_horizon_is_extrapolated():
    v = ledger_with(n=2, committed=5).view()
    full = project(v, 0, 0, v.alpha(), v.h_med, 16, max_iters=10_000)
    cap = project(v, 0, 0, v.alpha(), v.h_med, 16, max_iters=8)
    lat = np.full(full.H, 0.02)
    assert np.allclose(service_time(cap, lat[:8]), service_time(full, lat))
    assert abs(energy(cap, lat[:8], np.full(8, 50.0)) - energy(full, lat, np.full(full.H, 50.0))) < 1e-9


def test_estimator_fit():
    rng = np.random.default_rng(0)
    rows = [dict(batch=b, kv_blocks=k, phi=0, gamma_bar=g, gamma=g, freq_mhz=f,
                 iter_ms=10 + b * 0.5 + 1410 / f, power_w=50 + f / 10)
            for b, k, g, f in zip(rng.integers(1, 64, 400), rng.integers(10, 5000, 400),
                                  rng.integers(0, 6, 400), rng.choice(FREQS, 400))]
    est = PerfEnergyEstimator.fit(rows)
    lat, pw = est.predict(np.array([[8, 500, 0, 2, 2, 1050]]))
    assert 0.005 < lat[0] < 0.05 and 100 < pw[0] < 200
    assert est.info["iter_ms"]["cv_median_ape"] < 0.1


# -- joint selection ------------------------------------------------------------------
def test_large_slack_avoids_fmax():
    d = selector(tbt=500.0).select(ledger_with().view(), 0.0, 0, 0.0, FREQS)
    assert not d.best_effort and d.freq < 1410 and d.n_feasible > 0


def test_tight_tbt_needs_higher_frequency():
    v = ledger_with().view()
    cfg = ControlConfig(gammas=(0,), freqs=FREQS)
    loose = selector(tbt=500.0, cfg=cfg).select(v, 0.0, 0, 0.0, FREQS)
    tight = selector(tbt=15.0, cfg=cfg).select(v, 0.0, 0, 0.0, FREQS)
    assert tight.freq > loose.freq and tight.n_feasible < loose.n_feasible


def test_kv_capacity_prunes_lookaheads():
    sel = selector(tbt=500.0)
    for q in range(1000, 1016):            # find a prompt where lookaheads differ
        v = ledger_with(n=8, prompt=q).view()
        peaks = {g: sel._project(v, g, 0, 0.0)[2] for g in sel.cfg.gammas}
        if max(peaks.values()) > min(peaks.values()):
            break
    sel.K = min(peaks.values())
    d = sel.select(v, 0.0, 0, 0.0, FREQS)
    assert d.n_kv_pruned > 0 and peaks[d.gamma] <= sel.K


def test_best_effort_when_infeasible():
    d = selector(tbt=1.0).select(ledger_with().view(), 0.0, 3, 0.0, FREQS)
    assert d.best_effort and d.gamma == 0 and d.freq == 1410


def test_admission():
    L = ledger_with()
    sel = selector(e2e=60.0)
    late = L.new_entry("x", 256, 0.0, 0.5, 4096)
    assert not sel.admissible(L.view(extra=late), 0.0, 0, 256.0, 1410)
    ok = L.new_entry("y", 256, 0.0, 60.0, 4096)
    assert sel.admissible(L.view(extra=ok), 0.0, 0, 256.0, 1410)


# -- controller -------------------------------------------------------------------------
class FakeEngine:
    gamma = None

    def set_gamma(self, g):
        self.gamma = g


class FakeActuator:
    def __init__(self, dwell=False):
        self.current, self.dwell, self.calls = 1410, dwell, []

    def can_set(self):
        return not self.dwell

    def set(self, f, force=False):
        if self.dwell and not force:
            return False
        self.calls.append(f)
        self.current = f
        return True


def make_controller(dwell=False, tbt=500.0):
    cfg = ControlConfig(freqs=FREQS)
    L = ledger_with()
    c = EcoAdaptController(cfg, selector(tbt=tbt, cfg=cfg), L, FakeEngine(), FakeActuator(dwell))
    c.t0 = time.monotonic()
    return c


def test_epoch_applies_both_knobs():
    c = make_controller()
    c._epoch("test")
    rec = c.log[-1]
    assert c.engine.gamma == rec["gamma"] and c.freq.current == rec["freq_selected"] < 1410


def test_dwell_restricts_frequencies():
    c = make_controller(dwell=True)
    c._epoch("test")
    assert c.log[-1]["freq_selected"] == 1410 and c.freq.calls == []


def test_best_effort_stops_admission():
    c = make_controller(tbt=1.0)
    c._epoch("test")
    assert c.best_effort and c.engine.gamma == 0
    e = c.ledger.new_entry("new", 128, 0.0, 60.0, 64)
    assert c.try_admit(e, 0.0) == (False, "best-effort")


def test_admission_on_control_thread():
    c = make_controller()
    c.start()
    e = c.ledger.new_entry("new", 128, 0.0, 60.0, 64)
    c.offer(e)
    for _ in range(200):
        if "new" in c.decided:
            break
        time.sleep(0.01)
    c.stop()
    assert c.decided["new"] == (True, "") and "new" in c.ledger.entries and c.n_epochs >= 1


def test_admission_during_dwell():
    c = make_controller(dwell=True)
    c.freq.current = 630
    e = c.ledger.new_entry("new", 128, 0.0, 60.0, 64)
    ok, why = c.try_admit(e, 0.0)
    assert ok or why in ("dwell", "infeasible")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
