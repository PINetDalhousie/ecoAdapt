"""SLO-aware joint controller."""
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .estimator import energy, mean_tbt, service_time, state_rows
from .projector import project


@dataclass
class Decision:
    gamma: int
    freq: int
    energy_j: float = None
    best_effort: bool = False
    n_feasible: int = 0
    n_kv_pruned: int = 0


class Timer:
    def __init__(self):
        self.ms = defaultdict(list)

    @contextmanager
    def section(self, name):
        t0 = time.perf_counter()
        yield
        self.ms[name].append(1000.0 * (time.perf_counter() - t0))

    def report(self):
        return {k: dict(n=len(v), mean_ms=float(np.mean(v)),
                        p95_ms=float(np.percentile(v, 95)),
                        p99_ms=float(np.percentile(v, 99)))
                for k, v in self.ms.items() if v}


class JointSelector:
    """Bounded search over Gamma x F_e."""

    def __init__(self, cfg, slo, estimator, kv_capacity, block_size, kv_bin=4):
        self.cfg, self.slo, self.est = cfg, slo, estimator
        self.K = kv_capacity
        self.N = block_size
        self.kv_bin = kv_bin

    def _project(self, view, gamma, gamma_bar_now, phi_now):
        """Representative (posterior mean, median horizon) and conservative
        (acceptance tails, p95 horizon) projections for one lookahead."""
        kw = dict(block_size=self.N, phi_now=phi_now, max_iters=self.cfg.max_proj_iters)
        rep = project(view, gamma, gamma_bar_now, view.alpha(), view.h_med, **kw)
        lo = project(view, gamma, gamma_bar_now, view.alpha(self.cfg.alpha_lo), view.h_p95, **kw)
        hi = project(view, gamma, gamma_bar_now, view.alpha(self.cfg.alpha_hi), view.h_p95, **kw)
        # KV feasibility keeps the larger peak of both acceptance tails
        return rep, lo, max(lo.peak, hi.peak)

    def _plans(self, view, gamma_bar_now, phi_now, gammas):
        plans, pruned = [], 0
        for g in gammas:
            rep, cons, peak = self._project(view, g, gamma_bar_now, phi_now)
            if peak > self.K:                     # KVInfeasible
                pruned += 1
                continue
            plans.append((g, rep, cons))
        return plans, pruned

    def _predict(self, plans, freqs):
        """One estimator call for every (state, gamma, f); repeated projected
        states are collapsed first."""
        rows, index, n = [], [], 0
        for g, rep, cons in plans:
            S = np.vstack([state_rows(rep), state_rows(cons)])
            S[:, 1] = np.round(S[:, 1] / self.kv_bin) * self.kv_bin
            U, inv = np.unique(S, axis=0, return_inverse=True)
            for f in freqs:
                rows.append(np.column_stack([U, np.full(len(U), g), np.full(len(U), f)]))
                index.append((g, f, n, n + len(U), inv.ravel(), rep.H))
                n += len(U)
        if not rows:
            return {}
        lat, pw = self.est.predict(np.vstack(rows))
        out = {}
        for g, f, a, b, inv, h in index:
            l, p = lat[a:b][inv], pw[a:b][inv]
            out[(g, f)] = (l[:h], p[:h], l[h:])
        return out

    def _slo_feasible(self, cons, lat, slack):
        if cons.H == 0:
            return True
        if np.any(mean_tbt(cons, lat) * 1000.0 > self.slo.tbt_ms):
            return False
        return bool(np.all(service_time(cons, lat) <= slack))

    def _slack(self, view, now):
        return view.deadline_s - (now - view.arrival_s)

    def select(self, view, now, gamma_bar_now, phi_now, freqs):
        if len(view) == 0:
            return Decision(0, max(freqs))
        slack = self._slack(view, now)
        plans, pruned = self._plans(view, gamma_bar_now, phi_now, self.cfg.gammas)
        pred = self._predict(plans, freqs)
        best, n_ok = None, 0
        for g, rep, cons in plans:
            for f in freqs:
                lat_r, pw_r, lat_c = pred[(g, f)]
                if not self._slo_feasible(cons, lat_c, slack):
                    continue
                n_ok += 1
                E = energy(rep, lat_r, pw_r)
                if best is None or E < best.energy_j:
                    best = Decision(g, f, E)
        if best is None:
            # BestEffort: no speculation, f_max
            return Decision(0, self.cfg.fmax, best_effort=True, n_kv_pruned=pruned)
        best.n_feasible, best.n_kv_pruned = n_ok, pruned
        return best

    def admissible(self, view, now, gamma_bar_now, phi_now, freq):
        """Is there any feasible lookahead at `freq` for the tentative workload?"""
        slack = self._slack(view, now)
        plans, _ = self._plans(view, gamma_bar_now, phi_now, self.cfg.gammas)
        pred = self._predict(plans, [freq])
        return any(self._slo_feasible(cons, pred[(g, freq)][2], slack)
                   for g, rep, cons in plans)


class EcoAdaptController:
    """Runs control epochs on a CPU thread and applies (gamma*, f*)."""

    def __init__(self, cfg, selector, ledger, engine, actuator, timer=None):
        self.cfg, self.selector, self.ledger = cfg, selector, ledger
        self.engine, self.freq = engine, actuator
        self.timer = timer or Timer()
        self.batch, self.kv, self.phi, self.gamma_bar = 0, 0, 0.0, 0
        self.best_effort = False
        self.n_epochs = 0
        self.log = []
        self.decided = {}          # req_id -> (admitted, reason)
        self.t0 = time.monotonic()
        self._iters = 0
        self._reason = None
        self._offer = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def trigger(self, reason):
        with self._lock:
            self._reason = self._reason or reason
        self._wake.set()

    def offer(self, entry):
        """Queue an arriving request for the admission test on the control thread;
        the outcome appears in self.decided."""
        with self._lock:
            self._offer = entry
        self._wake.set()

    # -- called from the serving engine ---------------------------------
    def on_iteration(self, batch, kv, phi, gamma_bar):
        with self.timer.section("state_sampling"):
            self.batch, self.kv, self.phi, self.gamma_bar = batch, kv, phi, gamma_bar
            self._iters += 1
        if self._iters >= self.cfg.epoch_iters:
            self.trigger("periodic")

    def on_verify(self, req_id, n_committed, u, gamma_bar):
        with self.timer.section("acceptance_update"):
            self.ledger.observe(req_id, n_committed, u, gamma_bar)

    def on_complete(self, req_id):
        self.ledger.remove(req_id)
        self.trigger("completion")

    # -- admission ---------------------------------------------------------
    def dwell_active(self):
        return self.freq.current is not None and not self.freq.can_set()

    def try_admit(self, entry, now):
        """Returns (admitted, reason)."""
        if self.best_effort:
            return False, "best-effort"
        with self.timer.section("admission"):
            view = self.ledger.view(extra=entry)
            args = (view, now, self.gamma_bar, float(entry.prompt_len))
            if not self.dwell_active():
                ok = self.selector.admissible(*args, self.cfg.fmax)
                why = "" if ok else "infeasible"
            elif self.selector.admissible(*args, self.freq.current):
                ok, why = True, ""
            else:
                ok = False
                why = "dwell" if self.selector.admissible(*args, self.cfg.fmax) else "infeasible"
        if ok:
            self.ledger.add(entry)
            self.trigger("arrival")
        return ok, why

    # -- control epochs ---------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            if not self._wake.wait(0.05):
                continue
            self._wake.clear()
            with self._lock:
                entry, self._offer = self._offer, None
            if entry is not None:
                self.decided[entry.req_id] = self.try_admit(entry, time.monotonic() - self.t0)
            with self._lock:
                reason, self._reason = self._reason, None
                if reason:
                    self._iters = 0
            if reason:
                self._epoch(reason)

    def _epoch(self, reason):
        t = time.perf_counter()
        self.ledger.mark_epoch()
        freqs = [self.freq.current] if self.dwell_active() else list(self.cfg.freqs)
        with self.timer.section("projection_selection"):
            d = self.selector.select(self.ledger.view(), time.monotonic() - self.t0,
                                     self.gamma_bar, self.phi, freqs)
        with self.timer.section("actuation"):
            self.best_effort = d.best_effort
            self.engine.set_gamma(d.gamma)          # next decoding boundary
            if d.freq != self.freq.current:
                self.freq.set(d.freq)               # refused while dwelling
        self.n_epochs += 1
        self.log.append(dict(
            t_s=round(time.monotonic() - self.t0, 4), reason=reason,
            batch=self.batch, kv_blocks=self.kv, gamma_bar=self.gamma_bar,
            gamma=d.gamma, freq_selected=d.freq, freq_applied=self.freq.current,
            best_effort=d.best_effort, n_feasible=d.n_feasible,
            n_kv_pruned=d.n_kv_pruned, energy_j=d.energy_j,
            ms=round(1000.0 * (time.perf_counter() - t), 3)))
