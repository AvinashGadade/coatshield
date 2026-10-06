"""Shared chart style: one palette for the report figures and the dashboard.

Colours follow the entity, never its rank: a controller or a curve keeps its colour
in every figure. Hues are the first slots of a colour-blind-validated categorical
palette, in their validated order.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#8a8983"
CRITICAL = "#d03b3b"

CONTROLLER_COLORS = {"C0": BLUE, "C1": ORANGE, "C2": AQUA, "C3": YELLOW}
CURVE_COLORS = {"truth": NEUTRAL, "raw": ORANGE, "corrected": BLUE}
# Sequential ramp (one hue, light to dark) for magnitude on a grid.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_matplotlib() -> None:
    """Recessive axes and grid, thin marks, text in ink colours."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("coatshield_blue", SEQUENTIAL)
