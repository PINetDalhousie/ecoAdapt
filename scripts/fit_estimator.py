"""Fit the GBDT latency and power models from profiling rows (CPU only).

  python scripts/fit_estimator.py --profile results/profile.jsonl \
      --out results/estimator.joblib
"""
import argparse
import json

import common  # noqa: F401
from ecoadapt.estimator import PerfEnergyEstimator, load_profile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--profile", default="results/profile.jsonl")
p.add_argument("--out", default="results/estimator.joblib")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rows = load_profile(args.profile)
est = PerfEnergyEstimator.fit(rows, seed=args.seed)
est.save(args.out)
print(json.dumps(est.info, indent=2))
print(f"wrote {args.out}")
