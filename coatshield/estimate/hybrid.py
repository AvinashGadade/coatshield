"""All four estimators on one window, plus the seeded bootstrap for P(d10 >= spec)."""

from __future__ import annotations

import numpy as np

from coatshield.config import Config
from coatshield.estimate.model_based import (
    class_sums,
    fit_trend,
    shrink_to_trend,
    solve_mixture,
)
from coatshield.estimate.quantiles import QUANTILES, raw_estimate
from coatshield.estimate.weights import Reference, SizeClasses, classify, ipw_estimate

FIELDS = ("mean", "d10", "d50", "d90", "cv")


def _empirical_share(n_b: np.ndarray, method: str, min_bin_count: int) -> np.ndarray:
    """Share of each size class taken from its own window objects (empirical Bayes pooling).

    hybrid: n / (n + min_bin_count), so well-sampled classes speak for themselves and
    sparse ones lean on the size trend. model: the trend alone.
    """
    if method == "model":
        return np.zeros_like(n_b)
    return n_b / (n_b + min_bin_count)


def _mixture_summary(sol: dict[str, np.ndarray]) -> dict[str, float]:
    mean = float(sol["mean"][0])
    d10, d50, d90 = (float(v) for v in sol["quantiles"][0])
    cv = float(np.sqrt(sol["var"][0]) / mean) if mean > 0 else float("nan")
    return {"mean": mean, "d10": d10, "d50": d50, "d90": d90, "cv": cv}


def estimate_window(
    size_um: np.ndarray,
    thickness_um: np.ndarray,
    ref: Reference,
    cfg: Config,
    boot_gen: np.random.Generator | None = None,
    boot_from_d10: float | None = None,
) -> dict[str, float]:
    """Raw, IPW, model-based and hybrid estimates of the batch thickness distribution.

    size_um, thickness_um: camera size and measured thickness of the accepted single
    pellets in the window. With boot_gen, a bootstrap of the window (and of the at-line
    reference) adds a d10 interval and P(d10 >= spec) for cfg.estimator.method; with
    boot_from_d10 it runs only once that method's point estimate of d10 has reached the value.
    """
    ec = cfg.estimator
    sigma = cfg.measurement.sigma_um if ec.deconvolve_noise else 0.0
    size_sigma = cfg.camera.size_noise_um if ec.deconvolve_noise else 0.0
    log_size = np.log(size_um)
    h = np.asarray(thickness_um, dtype=float)
    n = h.size
    classes = classify(log_size, ref, ec.n_size_bins)

    out: dict[str, float] = {"n": float(n)}
    out.update({f"raw_{k}": v for k, v in raw_estimate(h).items()})
    ipw = ipw_estimate(h, classes, cfg)
    out.update({f"ipw_{k}": ipw[k] for k in (*FIELDS, "ess", "uncovered")})

    ones = np.ones((1, n))
    n_b, s1, s2 = class_sums(ones, classes.sample_bin, h, classes.n_bins)
    fit = fit_trend(n_b, s1, s2, classes, sigma, size_sigma)
    out["fit_beta"] = float(fit.beta[0])
    out["cv_within"] = float(np.sqrt(fit.cv2[0]))
    out["cv2_within_unclipped"] = float(fit.cv2_unclipped[0])

    shrunk = shrink_to_trend(h, log_size, fit, sigma)
    order = np.argsort(shrunk, kind="stable")
    h_sorted, bin_sorted = shrunk[order], classes.sample_bin[order]
    f_true = classes.f_true[None, :]
    for method in ("model", "hybrid"):
        share = _empirical_share(n_b, method, ec.min_bin_count)
        sol = solve_mixture(ones, h_sorted, bin_sorted, f_true, n_b, share, fit, classes,
                            QUANTILES, ec.quantile_tol_um)
        out.update({f"{method}_{k}": v for k, v in _mixture_summary(sol).items()})

    out.update(d10_lo=np.nan, d10_hi=np.nan, p_spec=np.nan)
    if boot_gen is not None and (
        boot_from_d10 is None or out[f"{ec.method}_d10"] >= boot_from_d10
    ):
        d10 = _bootstrap_d10(h, classes, ref, order, h_sorted, bin_sorted, cfg, sigma, size_sigma,
                             boot_gen)
        tail = 0.5 * (1.0 - ec.interval)
        out["d10_lo"], out["d10_hi"] = (float(v) for v in np.quantile(d10, [tail, 1.0 - tail]))
        out["p_spec"] = float(np.mean(d10 >= cfg.spec.d10_min_um))
    return out


def _bootstrap_d10(
    h: np.ndarray,
    classes: SizeClasses,
    ref: Reference,
    order: np.ndarray,
    h_sorted: np.ndarray,
    bin_sorted: np.ndarray,
    cfg: Config,
    sigma: float,
    size_sigma: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """d10 of cfg.estimator.method on bootstrap_n resamples of the window."""
    ec = cfg.estimator
    n, n_rep = h.size, ec.bootstrap_n
    idx = gen.integers(0, n, size=(n_rep, n))
    if ec.method == "raw":
        return np.quantile(h[idx], QUANTILES[0], axis=1)

    offsets = (np.arange(n_rep) * n)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n_rep * n).reshape(n_rep, n)
    counts = counts.astype(float)
    # The reference is a finite sample too: resample its class shares.
    f_true = np.broadcast_to(classes.f_true, (n_rep, classes.n_bins))
    if ref.n_effective is not None:
        f_true = gen.multinomial(ref.n_effective, classes.f_true, size=n_rep) / ref.n_effective

    n_b, s1, s2 = class_sums(counts, classes.sample_bin, h, classes.n_bins)
    if ec.method == "ipw":
        # Untrimmed weights; d10 is the first reading at which the weighted CDF reaches 10%.
        by_h = np.argsort(h, kind="stable")
        ratio = np.where(n_b > 0, f_true * n / np.where(n_b > 0, n_b, 1.0), 0.0)
        cdf = np.cumsum(counts[:, by_h] * ratio[:, classes.sample_bin[by_h]], axis=1)
        return h[by_h][np.argmax(cdf >= QUANTILES[0] * cdf[:, -1:], axis=1)]

    fit = fit_trend(n_b, s1, s2, classes, sigma, size_sigma)
    share = _empirical_share(n_b, ec.method, ec.min_bin_count)
    sol = solve_mixture(counts[:, order], h_sorted, bin_sorted, f_true, n_b, share, fit, classes,
                        (QUANTILES[0],), ec.quantile_tol_um, moments=False)
    return sol["quantiles"][:, 0]


def corrected_histogram(
    size_um: np.ndarray,
    thickness_um: np.ndarray,
    ref: Reference,
    cfg: Config,
    edges_um: np.ndarray,
) -> np.ndarray:
    """The hybrid estimate of the batch thickness distribution as shares per thickness bin."""
    from scipy.special import ndtr

    ec = cfg.estimator
    sigma = cfg.measurement.sigma_um if ec.deconvolve_noise else 0.0
    size_sigma = cfg.camera.size_noise_um if ec.deconvolve_noise else 0.0
    log_size = np.log(size_um)
    h = np.asarray(thickness_um, dtype=float)
    classes = classify(log_size, ref, ec.n_size_bins)
    n_b, s1, s2 = class_sums(np.ones((1, h.size)), classes.sample_bin, h, classes.n_bins)
    fit = fit_trend(n_b, s1, s2, classes, sigma, size_sigma)
    shrunk = shrink_to_trend(h, log_size, fit, sigma)

    share = _empirical_share(n_b, "hybrid", ec.min_bin_count)[0]
    has = n_b[0] > 0
    lam = np.where(has, share, 0.0)
    per_object = classes.f_true * lam / np.where(has, n_b[0], 1.0)
    out = np.histogram(shrunk, bins=edges_um, weights=per_object[classes.sample_bin])[0]

    h_ref = fit.predict(classes.x_ref)[0]
    sd_ref = h_ref * np.sqrt(fit.cv2[0] + (fit.beta[0] * classes.width) ** 2 / 12.0)
    cdf = ndtr((edges_um[None, :] - h_ref[:, None]) / np.maximum(sd_ref, 1e-12)[:, None])
    out = out + ((classes.f_true * (1.0 - lam))[:, None] * np.diff(cdf, axis=1)).sum(axis=0)
    return out / out.sum()
