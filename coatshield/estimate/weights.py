"""Reference size distribution f_true, size classes, and inverse-probability weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coatshield.config import Config
from coatshield.estimate.quantiles import summarise
from coatshield.seeds import rng
from coatshield.twin.batch import BatchResult


@dataclass(frozen=True)
class Reference:
    """What the whole bed's size distribution is believed to be."""

    log_size: np.ndarray  # reference points (log of coated diameter in um)
    weight: np.ndarray  # weight of each point, summing to one
    n_effective: int | None  # pellets behind the reference; None when it is exact


@dataclass(frozen=True)
class SizeClasses:
    """Log-spaced size classes shared by the window sample and the reference."""

    lo: float
    width: float
    n_bins: int
    sample_bin: np.ndarray  # class of each window object
    f_true: np.ndarray  # reference share of each class
    x_ref: np.ndarray  # mean log size of the reference in each class
    x_obs: np.ndarray  # mean log size of the window objects in each class
    n_obs: np.ndarray  # window objects per class


def _hist_points(counts: np.ndarray, edges_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = counts > 0
    centres = 0.5 * (np.log(edges_um[:-1]) + np.log(edges_um[1:]))
    return centres[keep], counts[keep] / counts.sum()


def oracle_reference(result: BatchResult, step: int) -> Reference:
    """Exact size distribution of the bed's single pellets (debugging only)."""
    log_size, weight = _hist_points(result.size_hist[step], result.size_edges)
    return Reference(log_size, weight, None)


def atline_reference(result: BatchResult, step: int, cfg: Config) -> Reference:
    """An at-line sizing of atline_n pellets drawn evenly from the bed, with camera noise."""
    gen = rng(f"estimate.atline.{step}", cfg.seed)
    counts = result.size_hist[step]
    log_edges = np.log(result.size_edges)
    drawn = gen.multinomial(cfg.estimator.atline_n, counts / counts.sum())
    bins = np.repeat(np.arange(counts.size), drawn)
    log_true = log_edges[bins] + gen.random(bins.size) * (log_edges[bins + 1] - log_edges[bins])
    size = np.exp(log_true) + cfg.camera.size_noise_um * gen.standard_normal(bins.size)
    n = size.size
    return Reference(np.log(size), np.full(n, 1.0 / n), n)


def coa_reference(result: BatchResult, coat_of_core) -> Reference:
    """Certificate of analysis: core sizes only, shifted by the current coating estimate.

    coat_of_core(log_core) returns the estimated thickness for a core of that size.
    """
    log_core, weight = _hist_points(result.core_size_hist, result.size_edges)
    return Reference(np.log(np.exp(log_core) + 2.0 * coat_of_core(log_core)), weight, None)


def classify(log_size: np.ndarray, ref: Reference, n_bins: int) -> SizeClasses:
    lo = min(log_size.min(), ref.log_size.min())
    hi = max(log_size.max(), ref.log_size.max())
    width = max((hi - lo) / n_bins, np.finfo(float).eps)

    def to_bin(x):
        return np.clip(((x - lo) / width).astype(np.intp), 0, n_bins - 1)

    centres = lo + (np.arange(n_bins) + 0.5) * width
    sample_bin, ref_bin = to_bin(log_size), to_bin(ref.log_size)
    f_true = np.bincount(ref_bin, weights=ref.weight, minlength=n_bins)
    x_ref_sum = np.bincount(ref_bin, weights=ref.weight * ref.log_size, minlength=n_bins)
    n_obs = np.bincount(sample_bin, minlength=n_bins).astype(float)
    x_obs_sum = np.bincount(sample_bin, weights=log_size, minlength=n_bins)
    return SizeClasses(
        lo=lo,
        width=width,
        n_bins=n_bins,
        sample_bin=sample_bin,
        f_true=f_true / f_true.sum(),
        x_ref=np.where(f_true > 0, x_ref_sum / np.where(f_true > 0, f_true, 1.0), centres),
        x_obs=np.where(n_obs > 0, x_obs_sum / np.where(n_obs > 0, n_obs, 1.0), centres),
        n_obs=n_obs,
    )


def ipw_weights(classes: SizeClasses, trim_percentile: float) -> np.ndarray:
    """Weight of each window object: f_true(class) / f_obs(class), trimmed at a percentile."""
    f_obs = classes.n_obs / classes.n_obs.sum()
    ratio = np.where(f_obs > 0, classes.f_true / np.where(f_obs > 0, f_obs, 1.0), 0.0)
    w = ratio[classes.sample_bin]
    return np.minimum(w, np.percentile(w, trim_percentile))


def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size."""
    return float(weights.sum() ** 2 / np.sum(weights**2))


def ipw_estimate(thickness_um: np.ndarray, classes: SizeClasses, cfg: Config) -> dict[str, float]:
    w = ipw_weights(classes, cfg.estimator.weight_trim_percentile)
    out = summarise(thickness_um, w)
    out["ess"] = kish_ess(w)
    # Share of the bed in size classes the window never saw: IPW cannot speak for it.
    out["uncovered"] = float(classes.f_true[classes.n_obs == 0].sum())
    return out
