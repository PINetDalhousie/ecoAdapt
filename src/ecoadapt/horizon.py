"""Generation-horizon predictor: prompt-length buckets, each holding
an empirical distribution of the total generation length."""
import json
from pathlib import Path

import numpy as np


class HorizonPredictor:

    def __init__(self, edges, grids, max_len=16384):
        self.edges = np.asarray(edges, dtype=float)
        self.grids = [np.asarray(g, dtype=float) for g in grids]
        self.all = np.sort(np.concatenate(self.grids))
        self.max_len = int(max_len)

    def predict(self, prompt_len, generated=0):
        """(median, p95) of the generation length, conditioned on the request
        having already produced `generated` tokens."""
        b = int(np.searchsorted(self.edges, prompt_len, side="left"))
        g = self.grids[min(b, len(self.grids) - 1)]
        g = g[g > generated]
        if g.size == 0:
            g = self.all[self.all > generated]
        if g.size == 0:
            med, p95 = 1.5 * generated, 2.0 * generated
        else:
            med, p95 = np.quantile(g, 0.5), np.quantile(g, 0.95)
        lo = generated + 1
        return (int(min(self.max_len, max(lo, round(med)))),
                int(min(self.max_len, max(lo, round(p95)))))

    @staticmethod
    def fit(prompt_lens, gen_lens, n_buckets=8, n_points=101, max_len=16384):
        p = np.asarray(prompt_lens, dtype=float)
        g = np.asarray(gen_lens, dtype=float)
        qs = np.linspace(0, 1, n_points)
        edges = np.quantile(p, np.linspace(0, 1, n_buckets + 1)[1:-1])
        idx = np.searchsorted(edges, p, side="left")
        glob = np.quantile(g, qs)
        grids = [np.quantile(g[idx == b], qs) if (idx == b).sum() >= 20 else glob
                 for b in range(n_buckets)]
        return HorizonPredictor(edges, grids, max_len)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(
            edges=self.edges.round(2).tolist(),
            grids=[g.round(2).tolist() for g in self.grids],
            max_len=self.max_len)))

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text())
        return HorizonPredictor(d["edges"], d["grids"], d["max_len"])
