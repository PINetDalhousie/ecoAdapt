"""Speculation-aware batch and KV trajectory projector."""
from dataclasses import dataclass

import numpy as np


def mu(gamma, alpha):
    """Expected committed tokens of one verification."""
    a = np.asarray(alpha, dtype=float)
    if gamma <= 0:
        return np.ones_like(a)
    a = np.clip(a, 0.0, 1.0 - 1e-9)
    return (1.0 - a ** (gamma + 1)) / (1.0 - a)


@dataclass
class Trajectory:
    gamma: int
    gamma_bar: np.ndarray     # effective lookahead per projected iteration j
    batch: np.ndarray         # |B_j|
    kv: np.ndarray            # committed KV blocks
    kv_peak: np.ndarray       # K_j^peak
    phi: np.ndarray           # scheduled prefill tokens
    progress: np.ndarray      # (H, n) expected committed tokens, 0 once finished
    decode: np.ndarray        # (H, n) iterations that count towards TBT_i
    R: np.ndarray             # remaining iterations per request
    mu_last: np.ndarray       # per-request progress after the first iteration
    H_full: int               # uncapped |H_e|

    @property
    def H(self):
        return len(self.batch)

    @property
    def peak(self):
        return float(self.kv_peak.max()) if self.H else 0.0


def project(view, gamma, gamma_bar_now, alpha, horizon, block_size,
            phi_now=0.0, max_iters=512):
    """Advance every request in the ledger view under lookahead `gamma`.

    The first projected iteration still runs at the lookahead currently in
    effect (gamma_bar_now); `gamma` applies from the next decoding boundary."""
    n = len(view)
    c0, q = view.committed, view.prompt_len
    mu0 = mu(gamma_bar_now, alpha)
    mu1 = mu(gamma, alpha)
    need = np.maximum(horizon - c0, 1.0)
    R = np.where(need <= mu0, 1.0, 1.0 + np.ceil((need - mu0) / mu1)) if n else np.zeros(0)
    H_full = int(R.max()) if n else 0
    H = min(H_full, max_iters)

    j = np.arange(H, dtype=float)[:, None]
    C = c0 + np.minimum(j, 1.0) * mu0 + np.maximum(j - 1.0, 0.0) * mu1
    active = j < R
    gbar = np.full(H, float(gamma))
    phi = np.zeros(H)
    if H:
        gbar[0] = gamma_bar_now
        phi[0] = phi_now

    tokens = q + C
    kv = np.where(active, np.ceil(tokens / block_size), 0.0).sum(axis=1)
    kv_peak = np.where(active, np.ceil((tokens + gbar[:, None]) / block_size),
                       0.0).sum(axis=1)

    prog = np.where(j == 0, mu0, mu1) * active
    # a request that has not produced its first token is still in prefill at j=0
    decode = active & ~((j == 0) & (c0 == 0))
    return Trajectory(gamma, gbar, active.sum(axis=1).astype(float), kv,
                      kv_peak, phi, prog, decode, R, mu1, H_full)
