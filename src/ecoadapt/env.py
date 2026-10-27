import os
import pwd
from pathlib import Path


def setup_env(gpu_index: int) -> None:
    """Call before importing torch/vllm."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"       # CUDA index == NVML index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # keep EngineCore (and its scheduler) in this process
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # vLLM appends a random suffix to request ids; we need the original ids
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _pin_caches()


def _sudo_user():
    name = os.environ.get("SUDO_USER")
    try:
        return pwd.getpwnam(name) if name else None
    except KeyError:
        return None


def _pin_caches() -> None:
    # sudo runs would otherwise leave root-owned compile caches behind
    user = _sudo_user()
    if user is None:
        return
    root = Path(user.pw_dir) / ".cache" / "ecoadapt"
    for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                     ("TRITON_CACHE_DIR", "triton"),
                     ("VLLM_CACHE_ROOT", "vllm")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(var, str(d))
    os.environ.setdefault("HF_HOME", str(Path(user.pw_dir) / ".cache" / "huggingface"))
    chown_to_user(root)


def chown_to_user(path) -> None:
    """Give files written under sudo back to the invoking user."""
    user = _sudo_user()
    if user is None or not Path(path).exists():
        return
    for top, dirs, files in os.walk(path):
        for p in [top] + [os.path.join(top, n) for n in dirs + files]:
            try:
                os.chown(p, user.pw_uid, user.pw_gid)
            except OSError:
                pass
