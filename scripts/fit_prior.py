"""Fit the Beta acceptance prior from speculative verifications on ShareGPT
prompts (prior split, disjoint from evaluation prompts).

  python scripts/fit_prior.py --gpu 0 --gamma 3 --out results/acceptance_prior.json
"""
from common import PromptStream, engine_config, parser, setup, write_json

p = parser(__doc__)
p.add_argument("--sharegpt", default="data/sharegpt.json")
p.add_argument("--gamma", type=int, default=3)
p.add_argument("--n-prompts", type=int, default=64)
p.add_argument("--prompt-len", type=int, default=512)
p.add_argument("--gen-len", type=int, default=192)
p.add_argument("--out", default="results/acceptance_prior.json")
args = p.parse_args()
setup(args)

from ecoadapt.acceptance import fit_prior             # noqa: E402
from ecoadapt.engine import ServingEngine             # noqa: E402

eng = ServingEngine(engine_config(args), gamma_max=5)
prompts = PromptStream(args.sharegpt, eng.llm.get_tokenizer(), seed=args.seed)
accepted, attempted = [], []


def collect(rid, n, u, g):
    if g > 0:
        accepted.append(u)
        attempted.append(g)


eng.on_verify = collect
eng.set_gamma(args.gamma)
for i in range(args.n_prompts):
    eng.add(f"a{i}", prompts.get(args.prompt_len), args.gen_len, ignore_eos=True)
eng.drain()
eng.shutdown()

a0, b0 = fit_prior(accepted, attempted)
write_json(args.out, dict(a0=a0, b0=b0, alpha_mean=a0 / (a0 + b0), gamma=args.gamma,
                          n_verifications=len(accepted),
                          target_model=args.target_model, draft_model=args.draft_model))
print(f"Beta({a0:.3f}, {b0:.3f}), mean alpha {a0 / (a0 + b0):.3f}, "
      f"{len(accepted)} verifications -> {args.out}")
