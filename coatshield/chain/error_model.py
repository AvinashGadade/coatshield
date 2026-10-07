"""A transparent error model of the measurement chain, fitted once and used by the twin.

The full chain is too slow for a whole batch, so its behaviour is summarised as:
* bias and spread of (measured - true) thickness in lookup tables over fouling, SNR and
  thickness;
* the probability that a scan ends undecided, as a logistic function of fouling, pigment
  and SNR;
* the gate's leakage (twins passed as single) and its rejection of true singles.
Saved as JSON with its own SHA-256, so a batch can name the error model it used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from coatshield.config import REPO_ROOT, Config
from coatshield.twin.population import SINGLE, TWIN


def _bin(values, edges) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


@dataclass(frozen=True)
class ErrorModel:
    fouling_edges: tuple[float, ...]
    snr_edges: tuple[float, ...]
    thickness_edges: tuple[float, ...]
    bias_um: tuple  # [fouling][snr][thickness]
    spread_um: tuple
    undecided: dict  # intercept, fouling, pigment, snr_per_10db (logistic coefficients)
    twin_leak: float  # share of twins the gate passes as single
    single_reject: float  # share of single pellets the gate holds back
    operating_point: dict  # snr_db and pigment of a normal batch
    meta: dict

    # --- use ---------------------------------------------------------------------
    def bias_spread(self, fouling, snr_db, thickness_um) -> tuple[np.ndarray, np.ndarray]:
        f = _bin(np.asarray(fouling), self.fouling_edges)
        s = _bin(np.asarray(snr_db), self.snr_edges)
        t = _bin(np.asarray(thickness_um), self.thickness_edges)
        return np.asarray(self.bias_um)[f, s, t], np.asarray(self.spread_um)[f, s, t]

    def p_undecided(self, fouling, pigment, snr_db) -> np.ndarray:
        u = self.undecided
        z = (u["intercept"] + u["fouling"] * np.asarray(fouling, dtype=float)
             + u["pigment"] * np.asarray(pigment, dtype=float)
             + u["snr_per_10db"] * np.asarray(snr_db, dtype=float) / 10.0)
        return 1.0 / (1.0 + np.exp(-z))

    def measure(self, cfg: Config, fouling: float, true_class: np.ndarray,
                true_thickness_um: np.ndarray, gen: np.random.Generator) -> dict:
        """What the chain would report for the objects of one time step of the twin."""
        mc = cfg.measurement
        n = true_class.size
        is_twin = true_class == TWIN
        gate_class = true_class.copy()
        gate_class[is_twin & (gen.random(n) < self.twin_leak)] = SINGLE
        gate_class[(true_class == SINGLE) & (gen.random(n) < self.single_reject)] = TWIN

        if mc.twin_misread_model == "random":
            factor = gen.uniform(*mc.twin_misread_random, size=n)
        else:
            factor = np.full(n, mc.twin_misread_thin if mc.twin_misread_model == "thin"
                             else mc.twin_misread_thick)
        seen = np.where(is_twin, true_thickness_um * factor, true_thickness_um)
        snr, pigment = self.operating_point["snr_db"], self.operating_point["pigment"]
        bias, spread = self.bias_spread(np.full(n, fouling), np.full(n, snr), seen)
        thickness = (seen + bias + spread * gen.standard_normal(n)) * (cfg.coating.n / mc.n_assumed)
        np.maximum(thickness, 0.0, out=thickness)

        p_und = float(self.p_undecided(fouling, pigment, snr))
        to_oct = gate_class == SINGLE
        accepted = to_oct & (gen.random(n) >= p_und)
        thickness[~accepted] = np.nan
        confidence = np.where(accepted, 1.0 - p_und, 0.0)
        confidence[~to_oct] = np.nan
        return {"thickness_um": thickness, "gate_class": gate_class, "accepted": accepted,
                "confidence": confidence}

    # --- storage -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def save(self, path: Path) -> str:
        digest = self.sha256()
        Path(path).write_text(json.dumps({**self.to_dict(), "sha256": digest}, indent=1))
        return digest


def load_error_model(path: Path | str) -> ErrorModel:
    """Load an error model, checking the stored hash against the content."""
    return _load(str(_resolve(path)))


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=8)
def _load(path: str) -> ErrorModel:
    data = json.loads(Path(path).read_text())
    stored = data.pop("sha256", None)
    model = ErrorModel(**{k: (tuple(v) if k.endswith("_edges") else v) for k, v in data.items()})
    if stored != model.sha256():
        raise ValueError(f"error model {path} does not match its stored hash")
    return model


def _robust_spread(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def fit_error_model(rows: pd.DataFrame, cfg: Config, meta: dict | None = None) -> ErrorModel:
    """Fit the tables and the logistic curve from calibration records
    (one row per object: ChainRecord.summary())."""
    from sklearn.linear_model import LogisticRegression

    cc = cfg.chain
    singles = rows[rows.true_true_class == SINGLE]
    twins = rows[rows.true_true_class == TWIN]
    scanned = singles[singles.status != "gated"]
    measured = scanned[scanned.status == "measured"]
    error = (measured.thickness_um - measured.true_thickness_um).to_numpy()

    f = _bin(measured.true_fouling.to_numpy(), cc.fouling_edges)
    s = _bin(measured.true_snr_db.to_numpy(), cc.snr_edges)
    t = _bin(measured.true_thickness_um.to_numpy(), cc.thickness_edges)
    shape = (len(cc.fouling_edges) - 1, len(cc.snr_edges) - 1, len(cc.thickness_edges) - 1)
    bias, spread = np.zeros(shape), np.zeros(shape)
    overall = (float(np.median(error)), _robust_spread(error))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # Fall back to coarser averages where a cell holds too few objects.
                for sel in ((f == i) & (s == j) & (t == k), (f == i) & (s == j), f == i):
                    if sel.sum() >= cc.min_cell_count:
                        bias[i, j, k] = np.median(error[sel])
                        spread[i, j, k] = _robust_spread(error[sel])
                        break
                else:
                    bias[i, j, k], spread[i, j, k] = overall

    x = np.column_stack([scanned.true_fouling, scanned.true_pigment, scanned.true_snr_db / 10.0])
    y = (scanned.status != "measured").to_numpy().astype(int)
    if 0 < y.sum() < y.size:
        fit = LogisticRegression(C=100.0, max_iter=1000).fit(x, y)
        coef = {"intercept": float(fit.intercept_[0]), "fouling": float(fit.coef_[0, 0]),
                "pigment": float(fit.coef_[0, 1]), "snr_per_10db": float(fit.coef_[0, 2])}
    else:  # no (or only) undecided scans: a constant
        p = min(max(y.mean(), 1e-4), 1 - 1e-4)
        coef = {"intercept": float(np.log(p / (1 - p))), "fouling": 0.0, "pigment": 0.0,
                "snr_per_10db": 0.0}
    return ErrorModel(
        fouling_edges=tuple(cc.fouling_edges), snr_edges=tuple(cc.snr_edges),
        thickness_edges=tuple(cc.thickness_edges),
        bias_um=bias.round(4).tolist(), spread_um=spread.round(4).tolist(), undecided=coef,
        twin_leak=float((twins.status != "gated").mean()) if len(twins) else 0.0,
        single_reject=float((singles.status == "gated").mean()),
        operating_point={"snr_db": cc.operating_snr_db, "pigment": cc.operating_pigment},
        meta={"n_objects": int(len(rows)), "n_measured": int(len(measured)), **(meta or {})},
    )
