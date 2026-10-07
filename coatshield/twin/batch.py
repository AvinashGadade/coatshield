"""Streaming batch engine: run_batch(config) -> BatchResult, cached by config hash."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from coatshield.config import BATCH_SECTIONS, REPO_ROOT, Config
from coatshield.seeds import rng
from coatshield.twin.faults import fault_state
from coatshield.twin.growth import PassSampler, Recipe, fuse_step, grow_step, make_recipe
from coatshield.twin.population import SINGLE, TWIN, UM3_PER_CM3, Population, make_population
from coatshield.twin.sampling import sample_step

TWIN_VERSION = "1"  # bump when the simulation changes, so old cache entries are not reused
CACHE_DIR = REPO_ROOT / ".cache" / "batches"
_MEMORY: dict[str, BatchResult] = {}


@dataclass
class BatchResult:
    config_hash: str
    config: dict
    truth: pd.DataFrame  # one row per time step: what the whole bed truly looks like
    samples: pd.DataFrame  # one row per object the window saw
    thickness_edges: np.ndarray  # bin edges (um) of thickness_hist
    thickness_hist: np.ndarray  # [step, bin] true thickness counts, single pellets
    size_edges: np.ndarray  # bin edges (um) of size_hist
    size_hist: np.ndarray  # [step, bin] true coated-size counts, single pellets
    core_size_hist: np.ndarray  # [bin] core-size counts at the start (certificate of analysis)
    recipe: Recipe
    meta: dict = field(default_factory=dict)


def _quantiles(x: np.ndarray, n: int, qs: tuple[float, ...]) -> list[float]:
    """Nearest-rank quantiles (ascending qs) of the n smallest-ranked entries.

    One copy, partitioned in place: after each split only the part above it is touched.
    """
    part = x.copy()
    out, start = [], 0
    for k in (min(n - 1, int(q * n)) for q in qs):
        view = part[start:]
        view.partition(k - start)
        out.append(float(view[k - start]))
        start = k + 1
    return out


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Inner product without BLAS: threaded BLAS is slow for short vectors on a busy machine."""
    return float(np.einsum("i,i->", a, b))


def _hist(values: np.ndarray, lo: float, inv_width: float, n_bins: int) -> np.ndarray:
    idx = ((values - lo) * inv_width).astype(np.intp)
    return np.bincount(idx[(idx >= 0) & (idx < n_bins)], minlength=n_bins)


@dataclass(frozen=True)
class _Subsample:
    """Fixed pellets used for the size-trend fit and the stored histograms."""

    idx: np.ndarray
    log_core: np.ndarray
    log_d90_over_d10: float  # log of the large-to-small core size ratio (d90 / d10)


def _truth_row(pop: Population, coated: np.ndarray, cfg: Config, sub: _Subsample) -> dict:
    h_all = 0.5 * (coated - pop.core_um)
    other = np.flatnonzero(pop.kind != SINGLE)
    n_single = pop.n - other.size
    # Statistics over single pellets without copying them out: subtract the few others
    # from the sums, then push them to +inf so they sit above every quantile.
    h_other = h_all[other]
    mean = float((h_all.sum() - h_other.sum()) / n_single)
    var = float((_dot(h_all, h_all) - _dot(h_other, h_other)) / n_single - mean * mean)
    h = h_all
    if other.size:
        h = h_all.copy()
        h[other] = np.inf
    d10, d50, d90 = _quantiles(h, n_single, (0.1, 0.5, 0.9))

    # Size trend h ~ D^k and the spread left once it is removed, on a fixed subsample.
    hs = h_all[sub.idx]
    ok = (pop.kind[sub.idx] == SINGLE) & (hs > 0)
    exponent, cv_within, ratio = np.nan, np.nan, np.nan
    n_ok = int(np.count_nonzero(ok))
    if n_ok > 10:
        w = ok.astype(float)
        y = np.log(np.where(ok, hs, 1.0))
        x = sub.log_core
        mx, my = _dot(w, x) / n_ok, _dot(w, y) / n_ok
        dx = (x - mx) * w
        exponent = float(_dot(dx, y - my) / _dot(dx, dx))
        rel = np.exp(y - my - exponent * (x - mx))
        m1 = _dot(w, rel) / n_ok
        m2 = _dot(w, rel * rel) / n_ok
        cv_within = float(np.sqrt(max(m2 - m1 * m1, 0.0)) / m1)
        ratio = float(np.exp(exponent * sub.log_d90_over_d10))

    n_twin = int(np.count_nonzero(pop.kind == TWIN))
    return {
        "mean_um": mean,
        "d10_um": d10,
        "d50_um": d50,
        "d90_um": d90,
        "cv": float(np.sqrt(max(var, 0.0)) / mean) if mean > 0 else np.nan,
        "cv_within_size": cv_within,
        "growth_exponent": float(exponent),
        "growth_ratio": ratio,
        "frac_below_spec": float(np.count_nonzero(h < cfg.spec.d10_min_um) / n_single),
        "n_single": n_single,
        "n_twin": n_twin,
        "agglomerate_pct": 100.0 * n_twin / (n_single + n_twin),
    }


def _simulate(cfg: Config) -> BatchResult:
    pop = make_population(cfg)
    recipe = make_recipe(cfg)
    dt = cfg.batch.step_s
    n_steps = int(round(cfg.batch.duration_h * 3600.0 / dt))

    key_grow = int(rng("twin.growth", cfg.seed).integers(0, 2**63))
    bounds = np.linspace(0, pop.n, cfg.twin.rng_chunks + 1).astype(np.int64)
    weight = np.empty(pop.n)
    sampler = PassSampler(pop, cfg, dt)
    gen_fuse = rng("twin.fusion", cfg.seed)
    gen_window = rng("twin.window", cfg.seed)
    gen_measure = rng("twin.measure", cfg.seed)

    tc = cfg.twin
    idx = np.arange(min(tc.stats_subsample, pop.n))  # pellets are i.i.d., so a prefix is random
    log_core_sub = np.log(pop.core_um[idx])
    s10, s90 = np.quantile(log_core_sub, [0.1, 0.9])
    sub = _Subsample(idx=idx, log_core=log_core_sub, log_d90_over_d10=float(s90 - s10))
    t_edges = np.linspace(0.0, tc.thickness_hist_max_factor * cfg.batch.target_mean_um,
                          tc.thickness_hist_bins + 1)
    t_inv = 1.0 / (t_edges[1] - t_edges[0])
    log_lo, log_hi = np.log(np.array(tc.size_hist_range) * cfg.pellet.core_median_um)
    s_edges = np.exp(np.linspace(log_lo, log_hi, tc.size_hist_bins + 1))
    s_inv = tc.size_hist_bins / (log_hi - log_lo)
    thickness_hist = np.zeros((n_steps, tc.thickness_hist_bins), dtype=np.int32)
    size_hist = np.zeros((n_steps, tc.size_hist_bins), dtype=np.int32)
    core_size_hist = _hist(log_core_sub, log_lo, s_inv, tc.size_hist_bins)

    core_volume = pop.core_volume_um3()
    density_ratio = cfg.coating.density_g_cm3 / cfg.pellet.core_density_g_cm3
    to_real_cm3 = 1.0 / (pop.scale * UM3_PER_CM3)
    sprayed_g = deposited_um3 = prev_mean = 0.0
    rows: list[dict] = []
    samples: list[dict[str, np.ndarray]] = []

    for step in range(n_steps):
        fault = fault_state(cfg, step * dt)
        t_s = (step + 1) * dt
        d_sprayed, d_volume = grow_step(pop, cfg, fault, recipe, sampler, step, key_grow, bounds,
                                         weight)
        fuse_step(pop, cfg, fault, sampler, gen_fuse)
        sprayed_g += d_sprayed
        deposited_um3 += d_volume
        coated = pop.coated

        row = _truth_row(pop, coated, cfg, sub)
        coating_volume = pop.coating_volume_um3()
        row.update(
            t_s=t_s,
            t_h=t_s / 3600.0,
            growth_um_per_h=(row["mean_um"] - prev_mean) * 3600.0 / dt,
            fines_per_min=cfg.wurster.fines_per_min * fault.fines_factor,
            weight_gain_pct=100.0 * density_ratio * coating_volume / core_volume,
            sprayed_solids_g=sprayed_g,
            deposited_volume_cm3=deposited_um3 * to_real_cm3,
            coating_volume_cm3=coating_volume * to_real_cm3,
            fouling=fault.fouling,
        )
        prev_mean = row["mean_um"]
        rows.append(row)

        sub_single = pop.kind[idx] == SINGLE
        coated_sub = coated[idx][sub_single]
        thickness_hist[step] = _hist(0.5 * (coated_sub - pop.core_um[idx][sub_single]),
                                     0.0, t_inv, tc.thickness_hist_bins)
        size_hist[step] = _hist(np.log(coated_sub), log_lo, s_inv, tc.size_hist_bins)

        samples.append(
            sample_step(pop, coated, cfg, fault, t_s, dt, gen_window, gen_measure)
        )

    first = ["t_s", "t_h"]
    truth = pd.DataFrame(rows)
    truth = truth[first + [c for c in truth.columns if c not in first]]
    sample_df = pd.DataFrame({k: np.concatenate([s[k] for s in samples]) for k in samples[0]})
    return BatchResult(
        config_hash=twin_hash(cfg),
        config=cfg.to_dict(),
        truth=truth,
        samples=sample_df,
        thickness_edges=t_edges,
        thickness_hist=thickness_hist,
        size_edges=s_edges,
        size_hist=size_hist,
        core_size_hist=core_size_hist,
        recipe=recipe,
        meta={
            "twin_version": TWIN_VERSION,
            "n_pellets": pop.n,
            "n_real": pop.n_real,
            "core_median_um": pop.median_um,
            "n_steps": n_steps,
        },
    )


def twin_hash(cfg: Config) -> str:
    """Config hash over the sections that shape the batch (estimator settings excluded)."""
    return cfg.hash(include=BATCH_SECTIONS)


def run_batch(cfg: Config, cache: bool = True, cache_dir: Path | None = None) -> BatchResult:
    """Simulate one batch. Results are cached in memory and on disk by config hash."""
    if not cache and cache_dir is None:
        return _simulate(cfg)
    key = f"{twin_hash(cfg)}-v{TWIN_VERSION}"
    directory = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    path = directory / f"{key}.pkl"
    mem_key = str(path)
    if mem_key in _MEMORY:
        return _MEMORY[mem_key]
    if path.exists():
        with open(path, "rb") as fh:
            result = pickle.load(fh)
    else:
        result = _simulate(cfg)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    _MEMORY[mem_key] = result
    return result
