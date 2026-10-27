"""Fit the generation-horizon predictor from an Azure 2023 trace (CPU only).

  python scripts/fit_horizon.py --trace data/AzureLLMInferenceTrace_conv.csv \
      --out results/horizon_conv.json
"""
import argparse

import numpy as np

import common  # noqa: F401
from ecoadapt.horizon import HorizonPredictor
from ecoadapt.workload import read_lengths

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--trace", default="data/AzureLLMInferenceTrace_conv.csv")
p.add_argument("--out", default="results/horizon_conv.json")
p.add_argument("--n-buckets", type=int, default=8)
p.add_argument("--max-len", type=int, default=16384)
args = p.parse_args()

li, lo = read_lengths(args.trace)
hp = HorizonPredictor.fit(li, lo, n_buckets=args.n_buckets, max_len=args.max_len)
hp.save(args.out)
print(f"{len(li)} requests, output median {np.median(lo):.0f}, p95 {np.percentile(lo, 95):.0f}")
for q in (64, int(np.median(li)), 4096):
    print(f"prompt {q:5d} -> (median, p95) = {hp.predict(q)}")
print(f"wrote {args.out}")
