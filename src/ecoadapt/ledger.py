"""Runtime ledger L_e: per-request state used by the state estimator."""
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta

from .acceptance import AcceptancePosterior


@dataclass
class Entry:
    req_id: str
    prompt_len: int
    arrival_s: float
    deadline_s: float              # D_i
    max_tokens: int
    h_med: int                     # representative horizon
    h_p95: int                     # conservative horizon
    post: AcceptancePosterior = field(default_factory=AcceptancePosterior)
    committed: int = 0             # c_i
    no_evidence_epochs: int = 0


class Ledger:

    def __init__(self, horizon, prior=(2.0, 1.0), stale_epochs=4):
        self.horizon = horizon
        self.prior = prior
        self.stale_epochs = stale_epochs
        self.entries = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.entries)

    def new_entry(self, req_id, prompt_len, arrival_s, deadline_s, max_tokens):
        med, p95 = self.horizon.predict(prompt_len)
        return Entry(req_id, prompt_len, arrival_s, deadline_s, max_tokens,
                     min(med, max_tokens), min(p95, max_tokens),
                     AcceptancePosterior(*self.prior))

    def add(self, e):
        with self._lock:
            self.entries[e.req_id] = e

    def remove(self, req_id):
        with self._lock:
            self.entries.pop(req_id, None)

    def observe(self, req_id, n_committed, u, gamma_bar):
        """Feedback from one serving iteration of a request."""
        e = self.entries.get(req_id)
        if e is None:
            return
        e.committed += n_committed
        if gamma_bar > 0:
            e.post.update(u, gamma_bar)
            e.no_evidence_epochs = 0
        if e.committed >= e.h_med or e.committed >= e.h_p95:
            # outlived its horizon: re-estimate given what it has produced
            med, p95 = self.horizon.predict(e.prompt_len, e.committed)
            e.h_med = max(e.h_med, min(med, e.max_tokens))
            e.h_p95 = max(e.h_p95, min(p95, e.max_tokens))

    def mark_epoch(self):
        for e in list(self.entries.values()):
            e.no_evidence_epochs += 1

    def view(self, extra=None):
        with self._lock:
            items = list(self.entries.values())
        if extra is not None:
            items.append(extra)
        return LedgerView(items, self.prior, self.stale_epochs)


class LedgerView:
    """Numpy snapshot of the ledger taken at a control epoch."""

    def __init__(self, items, prior, stale_epochs):
        self.req_ids = [e.req_id for e in items]
        self.prompt_len = np.array([e.prompt_len for e in items], dtype=float)
        self.committed = np.array([e.committed for e in items], dtype=float)
        self.h_med = np.maximum([e.h_med for e in items], self.committed + 1)
        self.h_p95 = np.maximum([e.h_p95 for e in items], self.committed + 1)
        self.arrival_s = np.array([e.arrival_s for e in items], dtype=float)
        self.deadline_s = np.array([e.deadline_s for e in items], dtype=float)
        # stale evidence -> fall back to the prior
        ab = [prior if e.no_evidence_epochs >= stale_epochs else e.post.params
              for e in items]
        self.a = np.array([x[0] for x in ab], dtype=float)
        self.b = np.array([x[1] for x in ab], dtype=float)
        self._alpha = {}

    def __len__(self):
        return len(self.req_ids)

    def alpha(self, q=None):
        """Posterior mean when q is None, otherwise the q-quantile."""
        if q not in self._alpha:
            if len(self) == 0:
                self._alpha[q] = np.zeros(0)
            elif q is None:
                self._alpha[q] = self.a / (self.a + self.b)
            else:
                self._alpha[q] = beta.ppf(q, self.a, self.b)
        return self._alpha[q]
