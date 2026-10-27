"""Offline profiling for the performance-energy estimator.

For each (batch size, KV level) cell the batch is preloaded with prompts that
reach the target KV occupancy, then every (f, gamma) pair is measured over a
short window. Cells in --arrival-levels keep replacing finished requests, which
adds scheduled-prefill (Phi) samples. Each output row is one window:
batch, kv_blocks, phi, gamma_bar, gamma, freq_mhz -> iter_ms, power_w.

  sudo -E python scripts/profile_estimator.py --gpu 0 --kv-blocks 5504 \
      --out results/profile.jsonl
"""
import json
import time
from pathlib import Path

from common import PromptStream, engine_config, parser, setup

p = parser(__doc__)
p.add_argument("--sharegpt", default="data/sharegpt.json")
p.add_argument("--batches", default="1,2,4,8,16,32,64")
p.add_argument("--kv-levels", default="0.10,0.30,0.50,0.75,0.95")
p.add_argument("--arrival-levels", default="0.30,0.75")
p.add_argument("--gammas", default="0,1,2,3,4,5")
p.add_argument("--freqs", default="210,420,630,840,1050,1260,1410")
p.add_argument("--window-s", type=float, default=0.7)
p.add_argument("--out", default="results/profile.jsonl")
args = p.parse_args()
setup(args)

from ecoadapt.env import chown_to_user                # noqa: E402
from ecoadapt.engine import ServingEngine             # noqa: E402
from ecoadapt.gpu import FrequencyActuator, PowerMonitor  # noqa: E402

batches = [int(x) for x in args.batches.split(",")]
levels = [float(x) for x in args.kv_levels.split(",")]
arrival_levels = {float(x) for x in args.arrival_levels.split(",") if x}
gammas = [int(x) for x in args.gammas.split(",")]
freqs = [int(x) for x in args.freqs.split(",")]

eng = ServingEngine(engine_config(args, max_num_seqs=max(batches)), max(gammas), record=True)
prompts = PromptStream(args.sharegpt, eng.llm.get_tokenizer(), seed=args.seed)
act = FrequencyActuator(args.gpu, dwell_ms=0)
mon = PowerMonitor(args.gpu)
mon.start()
cap, bs, ctx = eng.kv_capacity, eng.block_size, args.max_model_len

live, counter = set(), 0


def spawn(n, plen, gen):
    global counter
    for _ in range(n):
        rid = f"p{counter}"
        counter += 1
        eng.add(rid, prompts.get(plen), gen, ignore_eos=True)
        live.add(rid)


def pump(refill=0, plen=64, gen=4096):
    for o in eng.step():
        if o.finished:
            live.discard(o.request_id)
    if refill and len(live) < refill:
        spawn(refill - len(live), plen, gen)


def clear():
    if live:
        eng.abort(live)
        live.clear()
    eng.drain()


eng.set_gamma(max(gammas))            # warm up the speculative kernels
spawn(max(batches), 128, 64)
eng.drain()
live.clear()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fh = open(out, "w")
cells = [(B, lv) for B in batches for lv in levels]
t_begin = time.time()
for ci, (B, level) in enumerate(cells, 1):
    target = max(B, int(level * min(cap, B * ctx // bs)))
    plen = min(max(4, target // B) * bs, ctx - 512)
    gen = max(256, min(4096, ctx - plen - 8))
    refill = B if level in arrival_levels else 0
    clear()
    spawn(B, plen, gen)
    for _ in range(6):
        pump()
    if len(live) < B:
        print(f"[{ci}/{len(cells)}] B={B} level={level}: batch not reached, skipped")
        continue
    full = False
    for f in freqs:
        locked = act.set(f, force=True)
        time.sleep(0.3)
        for g in gammas:
            eng.set_gamma(g)
            for _ in range(3):
                pump(refill, plen, gen)
            eng.samples.clear()
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.window_s:
                pump(refill, plen, gen)
            t1 = time.monotonic()
            rows, pw = list(eng.samples), mon.mean_power(t0, t1)
            if len(rows) >= 2 and pw is not None:
                n = len(rows)
                avg = lambda k: sum(getattr(r, k) for r in rows) / n
                fh.write(json.dumps(dict(
                    batch=avg("batch"), kv_blocks=avg("kv_blocks"), phi=avg("phi"),
                    gamma_bar=avg("gamma_bar"), gamma=g, freq_mhz=f,
                    iter_ms=avg("iter_ms"), power_w=pw, n_iters=n,
                    drafted=sum(r.drafted for r in rows),
                    accepted=sum(r.accepted for r in rows),
                    sm_clock_mhz=mon.mean_clock(t0, t1), clock_locked=locked)) + "\n")
            if eng.kv_used() > 0.96 * cap:
                full = True
                break
        if full:
            break
    fh.flush()
    print(f"[{ci}/{len(cells)}] B={B} level={level:.2f} plen={plen} kv={eng.kv_used()}/{cap} "
          f"elapsed={time.time() - t_begin:.0f}s", flush=True)

clear()
fh.close()
mon.stop()
act.reset()
eng.shutdown()
chown_to_user(out.parent)
print(f"wrote {out}")
