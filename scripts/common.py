import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--gpu", type=int, default=0, help="GPU index (NVML order)")
    p.add_argument("--target-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--kv-blocks", type=int, default=None,
                   help="fix the KV pool size in blocks")
    p.add_argument("--seed", type=int, default=0)
    return p


def setup(args):
    from ecoadapt.env import setup_env
    setup_env(args.gpu)


def engine_config(args, **over):
    from ecoadapt.config import EngineConfig
    kw = dict(target_model=args.target_model, draft_model=args.draft_model,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, kv_blocks=args.kv_blocks,
              seed=args.seed)
    kw.update(over)
    return EngineConfig(**kw)


def write_json(path, obj):
    from ecoadapt.env import chown_to_user
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    chown_to_user(path)


class PromptStream:
    """Prompts of any length cut from the ShareGPT prior split, so profiling and
    prior fitting never see evaluation prompts."""

    def __init__(self, path, tokenizer, max_tokens=1_000_000, seed=0):
        import numpy as np
        from ecoadapt.workload import PRIOR_SPLIT
        self.rng = np.random.default_rng(seed)
        self.ids = []
        for conv in json.loads(Path(path).read_text())[:PRIOR_SPLIT]:
            text = "\n\n".join(t.get("value", "") for t in conv.get("conversations", []))
            self.ids += tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(self.ids) >= max_tokens:
                break

    def get(self, length):
        s = int(self.rng.integers(0, len(self.ids) - length))
        return self.ids[s:s + length]
