"""Graph search: class probabilities -> two clean, ordered, non-crossing surfaces.

The outer surface is found column by column with a smoothness limit, then the inner
surface below it with a minimum gap. Each is the shortest path through a cost image
built from the log-probabilities, so the hard rules are enforced, not learned.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_EPS = 1e-6
_INF = 1e30


def surface_costs(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cost of placing each surface at each row, per column.

    prob: [3, depth, columns] probabilities of (above surface, coating, core).
    outer[r, c]: cost of "rows < r are above, rows >= r are not".
    inner[r, c]: cost of "rows < r are not core, rows >= r are core".
    Both are negative log-likelihoods summed over the column.
    """
    log_above = np.log(prob[0] + _EPS)
    log_below = np.log(prob[1] + prob[2] + _EPS)
    log_not_core = np.log(prob[0] + prob[1] + _EPS)
    log_core = np.log(prob[2] + _EPS)

    def split_cost(log_top: np.ndarray, log_bottom: np.ndarray) -> np.ndarray:
        top = np.cumsum(log_top, axis=0) - log_top  # sum over rows above r
        bottom = np.cumsum(log_bottom[::-1], axis=0)[::-1]  # sum over rows r and below
        return -(top + bottom)

    return split_cost(log_above, log_below), split_cost(log_not_core, log_core)


@njit(cache=True)
def shortest_path(cost, max_jump, lower):
    """Row per column minimising total cost, with |row change| <= max_jump between columns
    and row[c] >= lower[c]. Dynamic programming, left to right."""
    n_rows, n_cols = cost.shape
    total = np.full((n_rows, n_cols), _INF)
    back = np.zeros((n_rows, n_cols), dtype=np.int32)
    for r in range(lower[0], n_rows):
        total[r, 0] = cost[r, 0]
    for c in range(1, n_cols):
        for r in range(lower[c], n_rows):
            best = _INF
            arg = r
            for p in range(max(r - max_jump, 0), min(r + max_jump, n_rows - 1) + 1):
                if total[p, c - 1] < best:
                    best = total[p, c - 1]
                    arg = p
            if best < _INF:
                total[r, c] = best + cost[r, c]
                back[r, c] = arg
    path = np.zeros(n_cols, dtype=np.int32)
    end = 0
    best = _INF
    for r in range(n_rows):
        if total[r, n_cols - 1] < best:
            best = total[r, n_cols - 1]
            end = r
    path[n_cols - 1] = end
    for c in range(n_cols - 1, 0, -1):
        path[c - 1] = back[path[c], c]
    return path


def find_surfaces(prob: np.ndarray, max_jump_px: int, min_gap_px: int,
                  valid_min_prob: float) -> dict[str, np.ndarray]:
    """Two ordered surfaces from class probabilities [3, depth, columns].

    Returns outer and inner rows per column (inner >= outer + min_gap_px everywhere),
    a valid flag per column (columns without signal are flagged, not guessed) and the
    probability margin along each path.
    """
    outer_cost, inner_cost = surface_costs(prob)
    n_rows, n_cols = outer_cost.shape
    outer = shortest_path(outer_cost, max_jump_px, np.zeros(n_cols, dtype=np.int32))
    lower = np.minimum(outer + min_gap_px, n_rows - 1).astype(np.int32)
    inner = shortest_path(inner_cost, max_jump_px, lower)

    cols = np.arange(n_cols)
    valid = (prob[1] + prob[2]).max(axis=0) >= valid_min_prob
    # Margin: how decisively the class changes across each surface.
    above_o = prob[0][np.maximum(outer - 1, 0), cols]
    below_o = (prob[1] + prob[2])[outer, cols]
    above_i = (prob[0] + prob[1])[np.maximum(inner - 1, 0), cols]
    below_i = prob[2][inner, cols]
    return {
        "outer": outer,
        "inner": inner,
        "valid": valid,
        "margin_outer": np.minimum(above_o, below_o),
        "margin_inner": np.minimum(above_i, below_i),
    }
