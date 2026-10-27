# ecoAdapt

This repository contains the source code of **ecoAdapt**, an energy-aware LLM serving
framework that jointly adapts GPU frequency (system-level knob) and the speculative
decoding lookahead γ (inference-level knob) under TBT, E2E and KV-capacity constraints.

At each control epoch, ecoAdapt projects how the active batch and its peak KV demand evolve
under every candidate lookahead, estimates the iteration latency and GPU power of each
(γ, f) pair with offline-trained models, and applies the minimum-energy pair that remains
feasible. Energy is ranked with representative estimates (median generation length,
posterior-mean acceptance) and feasibility is checked with conservative ones (95th-percentile
generation length, acceptance tails).

ecoAdapt is a control layer on top of vLLM V1 (v0.25.0); it does not modify the vLLM source.

## Components

| Component | Code |
|---|---|
| State estimator: generation horizon | `src/ecoadapt/horizon.py` |
| State estimator: Beta–Bernoulli acceptance | `src/ecoadapt/acceptance.py`, `src/ecoadapt/ledger.py` |
| Speculation-aware batch and KV trajectory projector | `src/ecoadapt/projector.py` |
| Performance–energy estimator | `src/ecoadapt/estimator.py` |
| SLO-aware controller: joint (γ, f) selection, admission, best-effort mode | `src/ecoadapt/controller.py` |
| Serving engine hooks and lookahead actuation | `src/ecoadapt/engine.py` |
| GPU-frequency actuation (NVML, 250 ms dwell) and power telemetry | `src/ecoadapt/gpu.py` |
| Closed-loop serving runtime | `src/ecoadapt/runtime.py` |
| Trace-driven workloads | `src/ecoadapt/workload.py` |

Default parameters (`src/ecoadapt/config.py`): Γ = {0,…,5}, F = {210, 420, …, 1410} MHz,
control epoch every C = 4 iterations (plus on every arrival and completion), 250 ms
frequency dwell, acceptance tails at the 5th/95th percentiles, prior fallback after four
epochs without verification evidence.

## Repository structure

```
.
├── src/ecoadapt/          ecoAdapt framework (see the table above)
├── scripts/
│   ├── smoke_test.py      checks the vLLM hooks and lookahead actuation on a GPU
│   ├── profile_estimator.py  offline profiling over (B, KV, Phi, gamma, f)
│   ├── fit_estimator.py   fits the GBDT latency and power models
│   ├── fit_horizon.py     fits the generation-horizon predictor
│   ├── fit_prior.py       fits the Beta acceptance prior
│   ├── build_workload.py  Azure trace window + ShareGPT / InstructCoder prompts
│   └── serve.py           replays a workload under ecoAdapt or a baseline
├── tests/                 CPU-only tests of the control logic
├── requirements.txt
└── LICENSE
```

## Requirements

* One NVIDIA GPU with NVML. We used an A100 40 GB PCIe, Ubuntu 24.04, Python 3.12.
* Root access to lock GPU clocks (`sudo -E`). Without it everything still runs, but frequency
  commands are not applied and a warning is printed.
* Access to the gated Llama models on Hugging Face (`huggingface-cli login`):
  `meta-llama/Llama-3.1-8B` (target) and `meta-llama/Llama-3.2-1B` (draft).

## Installation

```bash
git clone https://github.com/PINetDalhousie/ecoAdapt.git
cd ecoAdapt
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

The PyPI `vllm==0.25.0` wheel is built for CUDA 13 (driver r580+). With a CUDA 12.x driver,
install the CUDA 12.9 build instead:

```bash
uv pip install https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
    --torch-backend=cu129
uv pip install -r requirements.txt --torch-backend=cu129
```

No package install is needed; the scripts add `src/` to the Python path.

## Datasets

Put the files under `data/`:

| File | Source | Used for |
|---|---|---|
| `AzureLLMInferenceTrace_conv.csv`, `AzureLLMInferenceTrace_code.csv` | [Azure LLM inference trace 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) | generation-horizon predictor |
| `AzureLLMInferenceTrace_conv_1week.csv`, `AzureLLMInferenceTrace_code_1week.csv` | [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | arrival times and request lengths |
| `sharegpt.json` | [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) (`ShareGPT_V3_unfiltered_cleaned_split.json`) | Conversation prompts, acceptance prior, profiling prompts |
| `instructcoder/train.json` | [InstructCoder](https://huggingface.co/datasets/likaixin/InstructCoder) | Coding prompts |

The first 5,000 ShareGPT conversations are used only for the acceptance prior and
profiling; workload prompts come from the rest.

## Getting started

```bash
# 1. control logic, CPU only (about 30 s)
python -m pytest tests -q

# 2. vLLM integration on GPU 0: state sampling, gamma actuation, verification feedback
python scripts/smoke_test.py --gpu 0
```

### Offline profiling and predictors

The estimator is tied to the hardware, models and engine configuration, so it has to be
profiled again for a different setup. `--kv-blocks` fixes the KV pool size so that
profiling and serving see the same capacity (5,504 blocks on our A100 40 GB).

```bash
# offline profiling and model fitting
sudo -E $(which python) scripts/profile_estimator.py --gpu 0 --kv-blocks 5504 \
    --out results/profile.jsonl
python scripts/fit_estimator.py --profile results/profile.jsonl --out results/estimator.joblib

# request-level predictors
python scripts/fit_prior.py --gpu 0 --kv-blocks 5504 --out results/acceptance_prior.json
python scripts/fit_horizon.py --trace data/AzureLLMInferenceTrace_conv.csv \
    --out results/horizon_conv.json
python scripts/fit_horizon.py --trace data/AzureLLMInferenceTrace_code.csv \
    --out results/horizon_code.json

# workloads (1,200 consecutive requests each)
python scripts/build_workload.py --workload conv \
    --trace data/AzureLLMInferenceTrace_conv_1week.csv \
    --corpus data/sharegpt.json --out results/workloads/conv.jsonl
python scripts/build_workload.py --workload code \
    --trace data/AzureLLMInferenceTrace_code_1week.csv \
    --corpus data/instructcoder/train.json --out results/workloads/code.jsonl
```

### Serving

```bash
sudo -E $(which python) scripts/serve.py --gpu 0 --kv-blocks 5504 --policy ecoadapt \
    --workload results/workloads/conv.jsonl --rps 2.5 \
    --estimator results/estimator.joblib --horizon results/horizon_conv.json \
    --prior results/acceptance_prior.json --tbt-slo 100 --e2e-slo 54.1 \
    --out-dir results/runs/conv_ecoadapt
```

`--policy` selects the controller:

| Policy | Description |
|---|---|
| `ecoadapt` | joint (γ, f) control |
| `freqonly` | same controller restricted to Γ = {0} |
| `speconly` | same controller restricted to F = {f_max} |
| `fullperf` | autoregressive decoding (no draft model) at f_max, no controller |

Example for Coding, with the trace output lengths enforced:

```bash
sudo -E $(which python) scripts/serve.py --gpu 0 --kv-blocks 5504 --policy fullperf \
    --workload results/workloads/code.jsonl --rps 3.0 --ignore-eos \
    --tbt-slo 200 --e2e-slo 29.3 --out-dir results/runs/code_fullperf
```

`--rps` scales all inter-arrival gaps of the trace window by one factor, which keeps its
burst pattern. Each run directory contains:

* `summary.json`: GPU energy, tokens per joule, throughput, p99 E2E, mean TBT, deferrals,
  per-component control overhead
* `requests.csv`: per-request arrival, admission, first/last token, E2E and TBT
* `epochs.csv`: every control epoch (state, selected γ and f, applied f, feasible pairs, decision time)
* `power.csv`: GPU power and SM clock sampled every 50 ms

## Notes

* ecoAdapt targets a single GPU with a fixed draft–target pair.
* chunked prefill, continuous batching and PagedAttention stay enabled.
* vLLM is driven through the synchronous `LLMEngine` with in-process EngineCore
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). The lookahead is applied by rewriting the
  scheduler's dynamic speculative-decoding table, so it takes effect at the next scheduling
  step. These settings are made in `src/ecoadapt/env.py`.
* Energy is measured with NVML power readings, so other processes on the same GPU affect it.

## Citation

The paper describing ecoAdapt is currently under review at a conference. Until it is
published, please cite it as:

```bibtex
@unpublished{ifath2026ecoadapt,
  title  = {ecoAdapt: State-Aware Joint System and Inference Control for Energy-Efficient LLM Serving},
  author = {Ifath, Md. Monzurul Amin and Abu-Ghazaleh, Nael B. and Haque, Israat},
  note   = {Under review},
  year   = {2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).

## Contact

Md. Monzurul Amin Ifath, PINet Lab, Dalhousie University (monzurula.min@dal.ca).
Questions and bug reports are welcome through GitHub issues.
