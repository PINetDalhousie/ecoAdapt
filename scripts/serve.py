"""Replay a workload under ecoAdapt or a baseline and record energy and latency.

  sudo -E python scripts/serve.py --gpu 0 --policy ecoadapt \
      --workload results/workloads/conv.jsonl --rps 2.5 \
      --estimator results/estimator.joblib --horizon results/horizon_conv.json \
      --prior results/acceptance_prior.json --tbt-slo 100 --e2e-slo 54.1 \
      --out-dir results/runs/conv_ecoadapt
"""
import csv
import json
from pathlib import Path

from common import engine_config, parser, setup, write_json

p = parser(__doc__)
p.add_argument("--policy", default="ecoadapt",
               choices=["ecoadapt", "freqonly", "speconly", "fullperf"])
p.add_argument("--workload", required=True)
p.add_argument("--rps", type=float, default=None, help="mean offered rate (default: trace rate)")
p.add_argument("--n-requests", type=int, default=None)
p.add_argument("--estimator", default="results/estimator.joblib")
p.add_argument("--horizon", default="results/horizon_conv.json")
p.add_argument("--prior", default="results/acceptance_prior.json")
p.add_argument("--tbt-slo", type=float, default=100.0, help="ms")
p.add_argument("--e2e-slo", type=float, default=54.1, help="s")
p.add_argument("--freqs", default="210,420,630,840,1050,1260,1410")
p.add_argument("--gammas", default="0,1,2,3,4,5")
p.add_argument("--epoch-iters", type=int, default=4)
p.add_argument("--ignore-eos", action="store_true",
               help="generate exactly the trace output length")
p.add_argument("--out-dir", default="results/runs/latest")
args = p.parse_args()
setup(args)

from ecoadapt.config import SLO, ControlConfig        # noqa: E402
from ecoadapt.env import chown_to_user                # noqa: E402
from ecoadapt.estimator import PerfEnergyEstimator    # noqa: E402
from ecoadapt.horizon import HorizonPredictor         # noqa: E402
from ecoadapt.runtime import Runtime                  # noqa: E402
from ecoadapt.workload import load_workload           # noqa: E402

prior = (2.0, 1.0)
if Path(args.prior).exists():
    d = json.loads(Path(args.prior).read_text())
    prior = (d["a0"], d["b0"])
cfg = ControlConfig(gammas=tuple(int(x) for x in args.gammas.split(",")),
                    freqs=tuple(int(x) for x in args.freqs.split(",")),
                    epoch_iters=args.epoch_iters, prior=prior)
slo = SLO(args.tbt_slo, args.e2e_slo)
est = hp = None
if args.policy != "fullperf":
    est = PerfEnergyEstimator.load(args.estimator)
    hp = HorizonPredictor.load(args.horizon)

reqs = load_workload(args.workload, args.rps, args.n_requests)
rt = Runtime(engine_config(args), cfg, slo, args.policy, est, hp, args.gpu, args.ignore_eos)
res = rt.run(reqs)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
for name in ("requests", "epochs", "power"):
    rows = res[name]
    with open(out / f"{name}.csv", "w", newline="") as fh:
        if name == "power":
            fh.write("t_s,power_w,sm_clock_mhz\n")
            csv.writer(fh).writerows(rows)
        elif rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
write_json(out / "summary.json", dict(summary=res["summary"], overhead_ms=res["overhead"],
                                      args=vars(args), engine=rt.engine_cfg.to_dict(),
                                      control=rt.cfg.to_dict()))
chown_to_user(out)
print(json.dumps(res["summary"], indent=2))
print(f"wrote {out}")
