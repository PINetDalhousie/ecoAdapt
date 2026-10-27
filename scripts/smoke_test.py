"""Checks the vLLM integration on a GPU: state sampling, lookahead actuation and
per-request verification feedback.

  python scripts/smoke_test.py --gpu 0
"""
import time

from common import engine_config, parser, setup

p = parser(__doc__)
p.add_argument("--n-requests", type=int, default=8)
p.add_argument("--prompt-len", type=int, default=256)
p.add_argument("--gen-len", type=int, default=64)
args = p.parse_args()
setup(args)

import numpy as np                                    # noqa: E402
from ecoadapt.engine import ServingEngine             # noqa: E402

eng = ServingEngine(engine_config(args, max_model_len=4096), gamma_max=5, record=True)
vocab = eng.llm.get_tokenizer().vocab_size
rng = np.random.default_rng(args.seed)
print(f"KV capacity {eng.kv_capacity} blocks of {eng.block_size} tokens")

seen = set()
eng.on_verify = lambda rid, n, u, g: seen.add(rid)


def run(gamma):
    eng.set_gamma(gamma)
    eng.samples.clear()
    seen.clear()
    ids = [f"g{gamma}-{i}" for i in range(args.n_requests)]
    for rid in ids:
        eng.add(rid, rng.integers(100, vocab - 100, args.prompt_len).tolist(),
                args.gen_len, ignore_eos=True)
    t0 = time.monotonic()
    eng.drain()
    s = [x for x in eng.samples if x.phi == 0]
    return dict(gamma=gamma, wall_s=round(time.monotonic() - t0, 2),
                max_batch=max(x.batch for x in eng.samples),
                max_kv=max(x.kv_blocks for x in eng.samples),
                prefill_iters=sum(x.phi > 0 for x in eng.samples),
                max_gamma_bar=max(x.gamma_bar for x in s),
                drafted=sum(x.drafted for x in s),
                accepted=sum(x.accepted for x in s),
                feedback_ids_ok=set(ids) <= seen)


run(5)                      # warm up kernels for speculative shapes
res = [run(g) for g in (0, 2, 5)]
for r in res:
    print(r)
ok = (res[0]["drafted"] == 0 and res[1]["max_gamma_bar"] == 2
      and res[2]["max_gamma_bar"] == 5 and res[2]["accepted"] > 0
      and all(r["feedback_ids_ok"] and r["max_kv"] > 0 and r["prefill_iters"] > 0
              for r in res))
print("SMOKE TEST", "PASSED" if ok else "FAILED")
eng.shutdown()
raise SystemExit(0 if ok else 1)
