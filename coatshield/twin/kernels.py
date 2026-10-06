"""Fast pass draw for the twin's hot loop.

One pass count per pellet per step is the dominant cost of a batch, so it is drawn by
table lookup (the alias method) instead of a Poisson sampler:

* pass counts follow Poisson(lam * L), with L a mean-one log-normal (the step-to-step
  spread of cycle time); that mixed distribution is tabulated once per pass-rate class;
* pellets are grouped into log-spaced pass-rate classes, and each pellet's weight is
  rescaled by its own rate over the class rate, so its expected share stays exact;
* the random stream is counter-based: every (key, step, pellet) triple has its own
  SplitMix64 value. The key comes from coatshield.seeds.rng, and the output does not
  depend on thread count or scheduling, so results stay bit-identical across machines.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.special import gammaln
from scipy.stats import norm

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_STEP_MUL = np.uint64(0xD6E8FEB86659FD93)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_TO_UNIT = 1.0 / 9007199254740992.0  # 2**-53
_GRID_POINTS_PER_WIDTH = 8  # integration nodes per standard deviation of the narrowest term


def mixed_poisson_pmf(lam: np.ndarray, sigma_ln: float, tail: float) -> np.ndarray:
    """pmf[c, k] of Poisson(lam[c] * L), L log-normal with mean one and log-sd sigma_ln.

    The table stops where the upper tail falls below `tail` and is renormalised.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    z_tail = float(norm.isf(tail))
    top = lam.max() * np.exp(sigma_ln * z_tail - 0.5 * sigma_ln**2)
    k = np.arange(int(np.ceil(top + z_tail * np.sqrt(top) + 10.0)) + 1)

    def poisson_rows(log_rate: np.ndarray) -> np.ndarray:
        return np.exp(np.outer(log_rate, k) - np.exp(log_rate)[:, None] - gammaln(k + 1.0))

    if sigma_ln <= 0:
        pmf = poisson_rows(np.log(lam))
    else:
        # Integrate over the log pass rate y on a grid fine enough for both factors.
        centre = np.log(lam) - 0.5 * sigma_ln**2
        dy = min(sigma_ln, 1.0 / np.sqrt(k[-1] + 1.0)) / _GRID_POINTS_PER_WIDTH
        y = np.arange(centre.min() - sigma_ln * z_tail, centre.max() + sigma_ln * z_tail + dy, dy)
        w = np.exp(-0.5 * ((y[None, :] - centre[:, None]) / sigma_ln) ** 2)
        pmf = (w / w.sum(axis=1, keepdims=True)) @ poisson_rows(y)
    return pmf / pmf.sum(axis=1, keepdims=True)


@njit(cache=True)
def build_alias(pmf):
    """Vose alias tables (prob, alias), one row per class."""
    n_classes, n = pmf.shape
    prob = np.ones((n_classes, n))
    alias = np.zeros((n_classes, n), dtype=np.int32)
    small = np.empty(n, dtype=np.int32)
    large = np.empty(n, dtype=np.int32)
    scaled = np.empty(n)
    for c in range(n_classes):
        n_small = n_large = 0
        for j in range(n):
            alias[c, j] = j
            scaled[j] = pmf[c, j] * n
            if scaled[j] < 1.0:
                small[n_small] = j
                n_small += 1
            else:
                large[n_large] = j
                n_large += 1
        while n_small > 0 and n_large > 0:
            n_small -= 1
            s = small[n_small]
            g = large[n_large - 1]
            prob[c, s] = scaled[s]
            alias[c, s] = g
            scaled[g] -= 1.0 - scaled[s]
            if scaled[g] < 1.0:
                n_large -= 1
                small[n_small] = g
                n_small += 1
    return prob, alias


@njit(inline="always")
def _mix(z):
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


@njit(parallel=True, cache=True)
def draw_passes(key, step, cls, prob, alias, share, cum_passes, weight, bounds):
    """Draw every pellet's passes for one step and its weight in the split of sprayed volume.

    Writes weight = passes * share, adds passes to cum_passes, returns sum(weight).
    Chunk totals are added in chunk order, so the sum is independent of threading.
    """
    n_chunks = bounds.size - 1
    n_table = prob.shape[1]
    sums = np.zeros(n_chunks)
    base = _mix(np.uint64(key) ^ (np.uint64(step + 1) * _STEP_MUL))
    for c in prange(n_chunks):
        s = 0.0
        for i in range(bounds[c], bounds[c + 1]):
            u = np.float64(_mix(base + np.uint64(i + 1) * _GAMMA) >> _S11) * _TO_UNIT * n_table
            k = np.int64(u)
            row = cls[i]
            if u - k >= prob[row, k]:
                k = alias[row, k]
            cum_passes[i] += k
            w = k * share[i]
            weight[i] = w
            s += w
        sums[c] = s
    total = 0.0
    for c in range(n_chunks):
        total += sums[c]
    return total


@njit(parallel=True, cache=True)
def apply_growth(coated3, coated, weight, coef):
    """Add each pellet's share as an exact shell: D^3 += coef * weight; D = cbrt(D^3)."""
    for i in prange(coated3.size):
        if weight[i] > 0.0:
            c3 = coated3[i] + coef * weight[i]
            coated3[i] = c3
            coated[i] = np.cbrt(c3)
