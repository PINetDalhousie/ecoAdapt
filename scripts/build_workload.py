"""Build a replay workload from a contiguous window of an Azure 2024 trace,
with prompt text matched from ShareGPT (conv) or InstructCoder (code).

  python scripts/build_workload.py --workload conv \
      --trace data/AzureLLMInferenceTrace_conv_1week.csv \
      --corpus data/sharegpt.json --out results/workloads/conv.jsonl
"""
import argparse

import numpy as np

import common  # noqa: F401
from ecoadapt import workload as W

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--workload", choices=["conv", "code"], default="conv")
p.add_argument("--trace", required=True)
p.add_argument("--corpus", required=True, help="sharegpt.json or instructcoder train.json")
p.add_argument("--n-requests", type=int, default=1200)
p.add_argument("--start-row", type=int, default=500_000)
p.add_argument("--max-total-tokens", type=int, default=16376)
p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
p.add_argument("--out", required=True)
args = p.parse_args()

from transformers import AutoTokenizer                # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
corpus = "sharegpt" if args.workload == "conv" else "instructcoder"
# coding outputs are far shorter than InstructCoder references: match input only
w_out = 1.0 if args.workload == "conv" else 0.0

arrival, li, lo = W.read_trace(args.trace, args.n_requests, args.start_row,
                               args.max_total_tokens)
print(f"{len(li)} requests over {arrival[-1]:.0f} s; input median {np.median(li):.0f}, "
      f"output median {np.median(lo):.0f}")
cands = W.index_corpus(corpus, args.corpus, tok)
print(f"{len(cands)} {corpus} candidates")
picked, err = W.match(li, lo, cands, w_out=w_out)
ids = W.tokenize_selected(corpus, args.corpus, tok, picked)
prompts = [ids[e] for e in picked]
lo = np.minimum(lo, [args.max_total_tokens - len(x) for x in prompts])
W.save_workload(args.out, arrival, lo, prompts)
print(f"input-length error median {np.median(err):.3f}, p95 {np.percentile(err, 95):.3f}; "
      f"{len(set(picked))} distinct examples -> {args.out}")
