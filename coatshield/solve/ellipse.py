"""Fit a circle or an ellipse to the outer surface of a pellet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Circle:
    xc: float
    zc: float
    radius: float
    residual_rms: float  # rms distance of the points from the circle


def fit_circle(x: np.ndarray, z: np.ndarray) -> Circle:
    """Algebraic circle fit of Pratt (1987), solved in closed form (Chernov's formulation)."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    mx, mz = x.mean(), z.mean()
    u, v = x - mx, z - mz
    w = u * u + v * v
    muu, mvv, muv = (u * u).mean(), (v * v).mean(), (u * v).mean()
    muw, mvw, mww = (u * w).mean(), (v * w).mean(), (w * w).mean()
    mw = muu + mvv
    cov = muu * mvv - muv * muv
    a3, a2 = 4.0 * mw, -3.0 * mw * mw - mww
    a1 = mww * mw + 4.0 * cov * mw - muw * muw - mvw * mvw - mw**3
    a0 = muw * muw * mvv + mvw * mvw * muu - 2.0 * muw * mvw * muv - mww * cov + mw * mw * cov
    eta = 0.0  # Newton iteration on the characteristic polynomial, from zero
    for _ in range(50):
        f = a0 + eta * (a1 + eta * (a2 + eta * (a3 + 4.0 * eta)))
        df = a1 + eta * (2.0 * a2 + eta * (3.0 * a3 + 16.0 * eta))
        step = f / df
        eta -= step
        if abs(step) < 1e-14 * max(1.0, abs(eta)):
            break
    det = eta * eta - eta * mw + cov
    cu = (muw * (mvv - eta) - mvw * muv) / det / 2.0
    cv = (mvw * (muu - eta) - muw * muv) / det / 2.0
    radius = float(np.sqrt(cu * cu + cv * cv + mw + 2.0 * eta))
    dist = np.hypot(u - cu, v - cv) - radius
    return Circle(float(cu + mx), float(cv + mz), radius, float(np.sqrt(np.mean(dist**2))))


@dataclass(frozen=True)
class Ellipse:
    xc: float
    zc: float
    semi_major: float
    semi_minor: float
    angle: float  # rotation of the major axis from the x axis (rad)
    conic: np.ndarray  # a x^2 + b x z + c z^2 + d x + e z + f = 0, in centred coordinates

    @property
    def aspect(self) -> float:
        return self.semi_major / self.semi_minor


def fit_ellipse(x: np.ndarray, z: np.ndarray) -> Ellipse:
    """Direct least-squares ellipse fit (Fitzgibbon et al. 1999), in the numerically
    stable form of Halir and Flusser."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    mx, mz = x.mean(), z.mean()
    scale = max(np.abs(x - mx).max(), np.abs(z - mz).max())
    u, v = (x - mx) / scale, (z - mz) / scale
    d1 = np.column_stack([u * u, u * v, v * v])
    d2 = np.column_stack([u, v, np.ones_like(u)])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    t = -np.linalg.solve(s3, s2.T)
    m = s1 + s2 @ t
    m = np.array([m[2] / 2.0, -m[1], m[0] / 2.0])
    _, vecs = np.linalg.eig(m)
    vecs = np.real(vecs)
    cond = 4.0 * vecs[0] * vecs[2] - vecs[1] ** 2
    a1 = vecs[:, np.argmax(cond)]
    a, b, c, d, e, f = np.concatenate([a1, t @ a1])

    den = b * b - 4.0 * a * c
    cu, cv = (2.0 * c * d - b * e) / den, (2.0 * a * e - b * d) / den
    num = 2.0 * (a * e * e + c * d * d - b * d * e + den * f)
    root = np.sqrt((a - c) ** 2 + b * b)
    axes = sorted((np.sqrt(num * (a + c + root)) / -den, np.sqrt(num * (a + c - root)) / -den),
                  reverse=True)
    angle = 0.5 * np.arctan2(-b, c - a)
    return Ellipse(float(cu * scale + mx), float(cv * scale + mz), float(axes[0] * scale),
                   float(axes[1] * scale), float(angle), np.array([a, b, c, d, e, f]))
