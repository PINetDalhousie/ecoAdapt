"""Performance-energy estimator: offline GBDT models of iteration
latency and GPU power, and the per-request TBT / service time / energy they
imply over a projected trajectory."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

FEATURES = ("batch", "kv_blocks", "phi", "gamma_bar", "gamma", "freq_mhz")
GBDT = dict(max_iter=96, learning_rate=0.15, max_leaf_nodes=63,
            min_samples_leaf=5, l2_regularization=1.0, early_stopping=False)


class PerfEnergyEstimator:

    def __init__(self, latency, power, lo, hi, info=None):
        self.latency, self.power = latency, power
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.info = info or {}

    def predict(self, X):
        """Returns iteration latency [s] and GPU power [W]. States outside the
        profiled range are clamped, not extrapolated."""
        X = np.clip(np.asarray(X, dtype=float), self.lo, self.hi)
        lat = np.maximum(self.latency.predict(X), 1e-3) / 1000.0
        pw = np.maximum(self.power.predict(X), 1.0)
        return lat, pw

    @staticmethod
    def fit(rows, seed=0):
        X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
        info = dict(n=len(rows), features=list(FEATURES))
        models = []
        for target in ("iter_ms", "power_w"):
            y = np.array([float(r[target]) for r in rows])
            errs = []
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                m = HistGradientBoostingRegressor(**GBDT, random_state=seed).fit(X[tr], y[tr])
                errs.append(np.abs(m.predict(X[te]) - y[te]) / np.abs(y[te]))
            errs = np.concatenate(errs)
            info[target] = dict(cv_median_ape=float(np.median(errs)),
                                cv_p95_ape=float(np.percentile(errs, 95)))
            models.append(HistGradientBoostingRegressor(**GBDT, random_state=seed).fit(X, y))
        return PerfEnergyEstimator(models[0], models[1], X.min(0), X.max(0), info)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(latency=self.latency, power=self.power,
                         lo=self.lo, hi=self.hi, info=self.info), path)

    @staticmethod
    def load(path):
        d = joblib.load(path)
        return PerfEnergyEstimator(d["latency"], d["power"], d["lo"], d["hi"], d["info"])


def load_profile(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def state_rows(traj):
    """(B, K, Phi, gamma_bar) per projected iteration; gamma and f are appended
    by the caller."""
    return np.column_stack([traj.batch, traj.kv, traj.phi, traj.gamma_bar])


def mean_tbt(traj, lat):
    """Per-iteration TBT averaged over the remaining decode trajectory of each request,
    weighted by expected progress: sum_j l_j / sum_j mu(gamma_bar_j, alpha_i)."""
    num = (lat[:, None] * traj.decode).sum(axis=0)
    den = (traj.progress * traj.decode).sum(axis=0)
    tail = np.maximum(traj.R - traj.H, 0.0)
    if traj.H:
        num = num + tail * lat[-1]
        den = den + tail * traj.mu_last
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def service_time(traj, lat):
    """T_i(e): predicted latency accumulated until each request completes."""
    if traj.H == 0:
        return np.zeros(len(traj.R))
    cum = np.cumsum(lat)
    idx = np.minimum(traj.R, traj.H).astype(int) - 1
    return cum[idx] + np.maximum(traj.R - traj.H, 0.0) * lat[-1]


def energy(traj, lat, pw):
    """Energy objective: sum_j p_j * l_j over the projected horizon."""
    if traj.H == 0:
        return 0.0
    e = pw * lat
    return float(e.sum() + (traj.H_full - traj.H) * e[-1])
