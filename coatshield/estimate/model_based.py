"""Model-based estimation: a size trend fitted on the window, integrated over f_true.

Every function works on R replicates at once (R = 1 for the point estimate, R =
bootstrap_n for the bootstrap), so the bootstrap costs a few array operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from coatshield.estimate.weights import SizeClasses

_TINY = 1e-12


@dataclass(frozen=True)
class TrendFit:
    """Power-law size trend h = exp(alpha) * D^beta and the spread around it, per replicate."""

    alpha: np.ndarray  # [R]
    beta: np.ndarray  # [R]
    cv2: np.ndarray  # [R] squared relative spread at equal size, measurement noise removed
    cv2_unclipped: np.ndarray  # [R] the same before clipping at zero (unbiased, may be negative)

    def predict(self, log_size: np.ndarray) -> np.ndarray:
        """Trend thickness at log_size; [R, ...] for an array of sizes."""
        shape = (-1,) + (1,) * np.ndim(log_size)
        return np.exp(self.alpha.reshape(shape) + self.beta.reshape(shape) * log_size)


def class_sums(
    counts: np.ndarray, sample_bin: np.ndarray, thickness: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class count, sum of thickness and sum of squared thickness for each replicate.

    counts[r, i] is how many times replicate r contains window object i.
    """
    n_rep = counts.shape[0]
    n_b = np.zeros((n_rep, n_bins))
    s1 = np.zeros((n_rep, n_bins))
    s2 = np.zeros((n_rep, n_bins))
    order = np.argsort(sample_bin, kind="stable")
    by_bin, h = counts[:, order], thickness[order]
    bounds = np.searchsorted(sample_bin[order], np.arange(n_bins + 1))
    for b in range(n_bins):
        lo, hi = bounds[b], bounds[b + 1]
        if hi > lo:
            c = by_bin[:, lo:hi]
            n_b[:, b] = c.sum(axis=1)
            s1[:, b] = np.einsum("rn,n->r", c, h[lo:hi])
            s2[:, b] = np.einsum("rn,n->r", c, h[lo:hi] ** 2)
    return n_b, s1, s2


def fit_trend(
    n_b: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    classes: SizeClasses,
    noise_sigma_um: float,
    size_noise_um: float = 0.0,
) -> TrendFit:
    """Least-squares fit of log thickness against log size on the class means.

    Class means (not single readings) are logged, so additive measurement noise and
    readings clipped at zero do not bias the fit. Three known contributions are removed
    from the residual: the measurement variance noise_sigma_um^2, the size spread inside
    a class, and the scatter that camera size noise (size_noise_um) adds through the trend.
    """
    x = classes.x_obs[None, :]
    ok = (n_b > 0) & (s1 > 0)
    w = np.where(ok, n_b, 0.0)
    y = np.log(np.where(ok, s1 / np.where(ok, n_b, 1.0), 1.0))
    sw = w.sum(axis=1)
    mx = (w * x).sum(axis=1) / sw
    my = (w * y).sum(axis=1) / sw
    dx = x - mx[:, None]
    sxx = (w * dx * dx).sum(axis=1)
    sxy = (w * dx * (y - my[:, None])).sum(axis=1)
    beta = np.where(sxx > 0, sxy / np.where(sxx > 0, sxx, 1.0), 0.0)
    alpha = my - beta * mx

    trend = np.exp(alpha[:, None] + beta[:, None] * x)
    sse = (s2 - 2.0 * trend * s1 + n_b * trend * trend).sum(axis=1)
    scale = (n_b * trend * trend).sum(axis=1)
    within_class = beta * beta * classes.width**2 / 12.0
    size_noise = beta * beta * size_noise_um**2 * (
        (n_b * trend * trend * np.exp(-2.0 * x)).sum(axis=1) / scale
    )
    cv2 = (sse - n_b.sum(axis=1) * noise_sigma_um**2) / scale - within_class - size_noise
    return TrendFit(alpha=alpha, beta=beta, cv2=np.maximum(cv2, 0.0), cv2_unclipped=cv2)


def shrink_to_trend(
    thickness: np.ndarray, log_size: np.ndarray, fit: TrendFit, noise_sigma_um: float
) -> np.ndarray:
    """Pull each reading toward the size trend by the share of its scatter that is noise.

    After shrinking, the scatter around the trend has the variance of the true
    pellet-to-pellet spread, so quantiles are not widened by measurement noise.
    """
    trend = fit.predict(log_size)[0]
    if noise_sigma_um <= 0:
        return thickness.astype(float)
    true_var = fit.cv2[0] * trend * trend
    keep = np.sqrt(true_var / (true_var + noise_sigma_um**2))
    return trend + keep * (thickness - trend)


def solve_mixture(
    counts_sorted: np.ndarray,
    h_sorted: np.ndarray,
    bin_sorted: np.ndarray,
    f_true: np.ndarray,
    n_b: np.ndarray,
    empirical_share: np.ndarray,
    fit: TrendFit,
    classes: SizeClasses,
    quantiles: tuple[float, ...],
    tol_um: float,
    moments: bool = True,
) -> dict[str, np.ndarray]:
    """Quantiles and moments of the batch thickness distribution, per replicate.

    The batch distribution is a mixture over size classes b with weights f_true[b]:
    a share empirical_share[b] comes from the window objects seen in that class, the
    rest from a normal around the fitted trend. Quantiles are roots of the mixture
    CDF, found by bisection, so no random sampling is involved.
    """
    n_rep = counts_sorted.shape[0]
    rows = np.arange(n_rep)
    has = n_b > 0
    lam = np.where(has, empirical_share, 0.0)
    per_object = f_true * lam / np.where(has, n_b, 1.0)
    model_w = f_true * (1.0 - lam)

    w = counts_sorted * per_object[:, bin_sorted]
    cum_w = np.cumsum(w, axis=1)
    h_ref = fit.predict(classes.x_ref)
    sd_ref = h_ref * np.sqrt(fit.cv2[:, None] + (fit.beta[:, None] * classes.width) ** 2 / 12.0)
    sd_safe = np.maximum(sd_ref, _TINY)

    def cdf(h: np.ndarray) -> np.ndarray:
        k = np.searchsorted(h_sorted, h, side="right")
        empirical = np.where(k > 0, cum_w[rows, np.maximum(k - 1, 0)], 0.0)
        return empirical + (model_w * ndtr((h[:, None] - h_ref) / sd_safe)).sum(axis=1)

    lo0 = min(h_sorted[0], float((h_ref - 8.0 * sd_ref).min()))
    hi0 = max(h_sorted[-1], float((h_ref + 8.0 * sd_ref).max()))
    n_iter = int(np.ceil(np.log2(max(hi0 - lo0, tol_um) / tol_um))) + 1
    out = np.empty((n_rep, len(quantiles)))
    for j, p in enumerate(quantiles):
        lo, hi = np.full(n_rep, lo0), np.full(n_rep, hi0)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            above = cdf(mid) >= p
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        out[:, j] = 0.5 * (lo + hi)

    result = {"quantiles": out}
    if moments:
        m1 = np.einsum("rn,n->r", w, h_sorted) + (model_w * h_ref).sum(axis=1)
        m2 = np.einsum("rn,n->r", w, h_sorted**2) + (model_w * (h_ref**2 + sd_ref**2)).sum(axis=1)
        result["mean"] = m1
        result["var"] = np.maximum(m2 - m1 * m1, 0.0)
    return result
