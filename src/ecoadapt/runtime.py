"""Closed-loop serving runtime.

Policies
  ecoadapt   joint (gamma, f) control
  freqonly   same controller, Gamma = {0}
  speconly   same controller, F = {f_max}
  fullperf   autoregressive decoding (no draft model) at f_max, no controller
"""
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .controller import EcoAdaptController, JointSelector, Timer
from .engine import ServingEngine
from .gpu import FrequencyActuator, PowerMonitor
from .ledger import Ledger

POLICIES = ("ecoadapt", "freqonly", "speconly", "fullperf")


@dataclass
class ReqStats:
    req_id: str
    arrival_s: float
    prompt_len: int
    max_tokens: int
    admit_s: float = None
    first_s: float = None
    last_s: float = None
    finish_s: float = None
    n_out: int = 0
    deferrals: int = 0
    forced: bool = False

    def as_dict(self):
        d = dict(self.__dict__)
        d["e2e_s"] = None if self.finish_s is None else self.finish_s - self.arrival_s
        d["ttft_s"] = None if self.first_s is None else self.first_s - self.arrival_s
        d["tbt_ms"] = (1000.0 * (self.last_s - self.first_s) / (self.n_out - 1)
                       if self.n_out > 1 else None)
        return d


class Runtime:

    def __init__(self, engine_cfg, control_cfg, slo, policy="ecoadapt",
                 estimator=None, horizon=None, gpu_index=0, ignore_eos=False,
                 spec_k=5):
        assert policy in POLICIES, policy
        if policy == "freqonly":
            control_cfg = replace(control_cfg, gammas=(0,))
        elif policy == "speconly":
            control_cfg = replace(control_cfg, freqs=(control_cfg.fmax,))
        self.engine_cfg, self.cfg, self.slo = engine_cfg, control_cfg, slo
        self.policy = policy
        self.estimator, self.horizon = estimator, horizon
        self.gpu = gpu_index
        self.ignore_eos = ignore_eos
        self.spec_k = 0 if policy == "fullperf" else spec_k
        self.timer = Timer()

    def run(self, requests, warmup=4):
        eng = ServingEngine(self.engine_cfg, self.spec_k)
        freq = FrequencyActuator(self.gpu, self.cfg.dwell_ms)
        ctrl = ledger = None
        if self.policy != "fullperf":
            ledger = Ledger(self.horizon, self.cfg.prior, self.cfg.stale_epochs)
            sel = JointSelector(self.cfg, self.slo, self.estimator,
                                eng.kv_capacity, eng.block_size)
            ctrl = EcoAdaptController(self.cfg, sel, ledger, eng, freq, self.timer)
            eng.on_iteration, eng.on_verify = ctrl.on_iteration, ctrl.on_verify

        for i in range(warmup):
            eng.add(f"warmup-{i}", requests[i % len(requests)].prompt_ids[:256], 32, True)
        eng.drain()

        stats = {r.req_id: ReqStats(r.req_id, r.arrival_s, r.prompt_len, r.max_tokens)
                 for r in requests}
        mon = PowerMonitor(self.gpu)
        mon.start()
        freq.set(self.cfg.fmax, force=True)
        eng.set_gamma(self.spec_k if ctrl is None else 0)
        t0 = time.monotonic()
        if ctrl:
            ctrl.t0 = t0
            ctrl.start()

        pending = deque(requests)
        entry, deferred_at = None, None
        try:
            while pending or eng.busy():
                now = time.monotonic() - t0
                while pending and pending[0].arrival_s <= now:
                    r = pending[0]
                    st = stats[r.req_id]
                    if ctrl is None:
                        self._submit(eng, r, st, now)
                        pending.popleft()
                        continue
                    if entry is None or entry.req_id != r.req_id:
                        with self.timer.section("horizon_prediction"):
                            entry = ledger.new_entry(r.req_id, r.prompt_len, r.arrival_s,
                                                     self.slo.e2e_s, r.max_tokens)
                        ctrl.offer(entry)
                        break
                    res = ctrl.decided.pop(r.req_id, None)
                    if res is None:
                        # deferred requests are re-tested once the serving state changes
                        if deferred_at is not None and (ctrl.n_epochs != deferred_at
                                                        or not eng.busy()):
                            deferred_at = None
                            ctrl.offer(entry)
                        break
                    ok = res[0]
                    if not ok and not eng.busy():
                        # nothing is running, so waiting cannot make it feasible
                        ledger.add(entry)
                        ctrl.trigger("arrival")
                        st.forced = ok = True
                    if not ok:
                        st.deferrals += 1
                        deferred_at = ctrl.n_epochs
                        break
                    deferred_at = None
                    self._submit(eng, r, st, time.monotonic() - t0)
                    pending.popleft()

                if not eng.busy():
                    time.sleep(0.001)
                    continue
                for out in eng.step():
                    st = stats.get(out.request_id)
                    if st is None:
                        continue
                    t = time.monotonic() - t0
                    n = len(out.outputs[0].token_ids) if out.outputs else 0
                    if n:
                        st.first_s = t if st.first_s is None else st.first_s
                        st.last_s = t
                        st.n_out += n
                    if out.finished:
                        st.finish_s = t
                        if ctrl:
                            ctrl.on_complete(out.request_id)
        finally:
            duration = time.monotonic() - t0
            if ctrl:
                ctrl.stop()
            mon.stop()
            freq.reset()
            iters = eng.iterations
            eng.shutdown()

        recs = [s.as_dict() for s in stats.values()]
        return dict(summary=self._summary(recs, mon, freq, ctrl, duration, iters, eng),
                    requests=recs, epochs=ctrl.log if ctrl else [],
                    overhead=self.timer.report(),
                    power=[(round(t - t0, 3), round(w, 1), c) for t, w, c in mon.samples])

    def _submit(self, eng, r, st, now):
        eng.add(r.req_id, r.prompt_ids, r.max_tokens, self.ignore_eos)
        st.admit_s = now

    def _summary(self, recs, mon, freq, ctrl, duration, iters, eng):
        e2e = np.array([r["e2e_s"] for r in recs if r["e2e_s"] is not None])
        tbt = np.array([r["tbt_ms"] for r in recs if r["tbt_ms"] is not None])
        tokens = sum(r["n_out"] for r in recs)
        energy = mon.energy_j()
        out = dict(policy=self.policy, n_requests=len(recs), n_completed=len(e2e),
                   duration_s=round(duration, 2), out_tokens=tokens, iterations=iters,
                   energy_j=round(energy, 1), mean_power_w=mon.mean_power(),
                   tokens_per_joule=tokens / energy if energy else None,
                   throughput_tps=tokens / duration,
                   p99_e2e_s=float(np.percentile(e2e, 99)) if len(e2e) else None,
                   mean_tbt_ms=float(tbt.mean()) if len(tbt) else None,
                   tbt_slo_ms=self.slo.tbt_ms, e2e_slo_s=self.slo.e2e_s,
                   n_preempted=eng.n_preempted,
                   n_deferred=sum(r["deferrals"] > 0 for r in recs),
                   n_forced=sum(r["forced"] for r in recs),
                   mean_wait_s=float(np.mean([r["admit_s"] - r["arrival_s"] for r in recs
                                              if r["admit_s"] is not None])),
                   mean_sm_clock_mhz=mon.mean_clock(),
                   clock_lock_ok=freq.works)
        if ctrl:
            g = [e["gamma"] for e in ctrl.log]
            out.update(n_epochs=len(ctrl.log),
                       n_best_effort=sum(e["best_effort"] for e in ctrl.log),
                       mean_gamma=float(np.mean(g)) if g else None)
        return out
