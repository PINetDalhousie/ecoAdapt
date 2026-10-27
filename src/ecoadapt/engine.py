"""Serving-engine integration with vLLM V1 (0.25.0), without modifying vLLM.

The synchronous LLMEngine runs EngineCore in-process, so the scheduler can be
instrumented at model-execution boundaries:
  * schedule()            -> batch, committed KV, scheduled prefill, gamma_bar
  * update_from_output()  -> per-request committed tokens and accepted drafts
The lookahead is actuated through the scheduler's dynamic speculative-decoding
table (batch size -> K), which is read once per scheduling step.
"""
import time
from dataclasses import dataclass


@dataclass
class IterSample:
    t: float
    iter_ms: float
    batch: int
    kv_blocks: int
    phi: int
    gamma_bar: int
    drafted: int
    accepted: int = 0


class ServingEngine:

    def __init__(self, engine_cfg, gamma_max, record=False):
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine

        self.llm = LLMEngine.from_engine_args(
            EngineArgs(**engine_cfg.engine_kwargs(gamma_max)),
            enable_multiprocessing=False)
        core = self.llm.engine_core
        if not hasattr(core, "engine_core"):
            raise RuntimeError("EngineCore is not in-process; call ecoadapt.env.setup_env() first")
        self.sched = core.engine_core.scheduler
        self.block_size = int(self.sched.block_size)
        self.pool = self.sched.kv_cache_manager.block_pool
        self.kv_capacity = int(self.pool.num_gpu_blocks) - 1     # K

        self.gamma_max = gamma_max
        self.gamma = gamma_max
        self._pending = None
        self.iterations = 0
        self.n_preempted = 0
        self.record = record
        self.samples = []
        self.on_iteration = None       # f(batch, kv, phi, gamma_bar)
        self.on_verify = None          # f(req_id, n_committed, u, gamma_bar)
        self._t_last = None
        self._last = None
        self._hook()

    def set_gamma(self, gamma):
        """Queue a lookahead for the next scheduling step (0 = no drafts)."""
        self._pending = max(0, min(self.gamma_max, int(gamma)))

    def kv_used(self):
        return self.kv_capacity - int(self.pool.get_num_free_blocks())

    def _apply_gamma(self):
        g = self._pending
        if g is None or self.sched.dynamic_sd_lookup is None:
            return
        if g != self.gamma:
            self.sched.dynamic_sd_lookup = [g] * len(self.sched.dynamic_sd_lookup)
            self.gamma = g
        self._pending = None

    def _hook(self):
        s = self.sched
        schedule, update, preempt = s.schedule, s.update_from_output, s._preempt_request

        def _schedule(*a, **kw):
            self._apply_gamma()
            out = schedule(*a, **kw)
            self._sample(out)
            return out

        def _update(so, mro):
            self._observe(so, mro)
            return update(so, mro)

        def _preempt(*a, **kw):
            self.n_preempted += 1
            return preempt(*a, **kw)

        s.schedule, s.update_from_output, s._preempt_request = _schedule, _update, _preempt

    def _sample(self, out):
        sched = out.num_scheduled_tokens
        if not sched:
            return
        spec = out.scheduled_spec_decode_tokens
        gbar = phi = drafted = 0
        for rid, ntok in sched.items():
            g = len(spec.get(rid, ()))
            drafted += g
            gbar = max(gbar, g)
            phi += max(0, ntok - 1 - g)
        now = time.perf_counter()
        dt = 0.0 if self._t_last is None else 1000.0 * (now - self._t_last)
        self._t_last = now
        self.iterations += 1
        kv = self.kv_used()
        self._last = IterSample(time.monotonic(), dt, len(sched), kv, phi, gbar, drafted)
        if self.on_iteration:
            self.on_iteration(len(sched), kv, float(phi), gbar)

    def _observe(self, so, mro):
        sampled = getattr(mro, "sampled_token_ids", None)
        if not sampled:
            return
        spec = so.scheduled_spec_decode_tokens
        acc = 0
        for rid, i in mro.req_id_to_index.items():
            toks = sampled[i]
            if not len(toks):
                continue
            g = len(spec.get(rid, ()))
            u = len(toks) - 1                  # accepted drafts (+1 corrected/bonus)
            acc += u
            if self.on_verify:
                self.on_verify(rid, len(toks), u, g)
        if self._last is not None:
            self._last.accepted = acc
            if self.record and self._last.iter_ms > 0:
                self.samples.append(self._last)
            self._last = None

    def add(self, req_id, prompt_ids, max_tokens, ignore_eos=False):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos,
                            output_kind=RequestOutputKind.DELTA)
        self.llm.add_request(req_id, {"prompt_token_ids": list(prompt_ids)}, sp)

    def step(self):
        return self.llm.step()

    def busy(self):
        return self.llm.has_unfinished_requests()

    def abort(self, ids):
        self.llm.abort_request(list(ids))

    def drain(self):
        while self.busy():
            self.step()

    def shutdown(self):
        try:
            self.llm.engine_core.shutdown()
        except Exception:
            pass
