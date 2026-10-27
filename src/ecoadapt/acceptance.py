"""Beta-Bernoulli acceptance posterior."""
from scipy.stats import beta


class AcceptancePosterior:

    def __init__(self, a0=2.0, b0=1.0):
        self.a0, self.b0 = a0, b0
        self.A = 0.0       # accepted draft tokens
        self.V = 0.0       # rejections (first mismatch only)
        self.n_obs = 0

    def update(self, u, gamma_bar):
        """u draft tokens accepted before the first mismatch, out of gamma_bar."""
        if gamma_bar <= 0:
            return
        self.A += u
        self.V += 1.0 if u < gamma_bar else 0.0
        self.n_obs += 1

    @property
    def params(self):
        return self.a0 + self.A, self.b0 + self.V

    def mean(self):
        a, b = self.params
        return a / (a + b)

    def quantile(self, q):
        return float(beta.ppf(q, *self.params))


def fit_prior(accepted, attempted, max_strength=20.0):
    """Method-of-moments Beta prior from offline verifications.

    A verification gives u successes and at most one
    failure, so alpha is estimated as A / (A + V)."""
    A = float(sum(accepted))
    V = float(sum(1 for u, g in zip(accepted, attempted) if u < g))
    if A + V == 0:
        return 2.0, 1.0
    p = A / (A + V)
    strength = max(0.5, min(max_strength, (A + V) / len(accepted)))
    return p * strength, (1.0 - p) * strength
