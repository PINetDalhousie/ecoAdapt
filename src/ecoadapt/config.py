from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EngineConfig:
    target_model: str = "meta-llama/Llama-3.1-8B"
    draft_model: str = "meta-llama/Llama-3.2-1B"
    max_num_seqs: int = 64
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    kv_blocks: Optional[int] = None
    enable_chunked_prefill: bool = True
    # prefix caching shares KV blocks across requests, which breaks the KV projection
    enable_prefix_caching: bool = False
    seed: int = 0

    def engine_kwargs(self, gamma_max: int) -> dict:
        kw = dict(model=self.target_model,
                  max_num_seqs=self.max_num_seqs,
                  max_model_len=self.max_model_len,
                  gpu_memory_utilization=self.gpu_memory_utilization,
                  block_size=self.block_size,
                  enable_chunked_prefill=self.enable_chunked_prefill,
                  enable_prefix_caching=self.enable_prefix_caching,
                  seed=self.seed)
        if self.kv_blocks:
            kw["num_gpu_blocks_override"] = self.kv_blocks
        if gamma_max > 0:
            # the per-batch-size table is rewritten at runtime to set gamma
            kw["speculative_config"] = {
                "method": "draft_model",
                "model": self.draft_model,
                "num_speculative_tokens": gamma_max,
                "num_speculative_tokens_per_batch_size": [[1, 1 << 20, gamma_max]],
            }
        return kw

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SLO:
    tbt_ms: float = 100.0       # tau, mean TBT
    e2e_s: float = 54.1         # D_i, E2E budget


@dataclass(frozen=True)
class ControlConfig:
    gammas: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    freqs: Tuple[int, ...] = (210, 420, 630, 840, 1050, 1260, 1410)
    epoch_iters: int = 4                # C
    dwell_ms: float = 250.0
    alpha_lo: float = 0.05              # acceptance tails
    alpha_hi: float = 0.95
    stale_epochs: int = 4
    prior: Tuple[float, float] = (2.0, 1.0)
    max_proj_iters: int = 512           # cap on |H_e|, tail is extrapolated

    @property
    def gamma_max(self) -> int:
        return max(self.gammas)

    @property
    def fmax(self) -> int:
        return max(self.freqs)

    def to_dict(self) -> dict:
        return asdict(self)
