"""Shared matplotlib style for report figures (light theme, print-friendly).

Categorical colors are assigned in fixed order and follow the entity (a storm
type keeps its color in every figure). Ordinal SVI quartiles use one blue
ramp, light -> dark.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
NO_DATA = "#e1e0d9"  # hatched gray for counties without data
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Storm-type colors: fixed mapping so a type is the same color everywhere.
# Rare types (lightning, heat, wildfire) fold into gray "other".
STORM_COLORS = {
    "wind": CATEGORICAL[0], "winter": CATEGORICAL[1], "hail": CATEGORICAL[2],
    "flood": CATEGORICAL[3], "tornado": CATEGORICAL[4], "tropical": CATEGORICAL[5],
    "ice": CATEGORICAL[6], "other": MUTED,
}
SVI_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # Q1 (least) -> Q4 (most)
SEQUENTIAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def storm_color(t: str) -> str:
    return STORM_COLORS.get(t, MUTED)


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "font.size": 10, "legend.frameon": False,
        "legend.labelcolor": INK_2, "lines.linewidth": 2, "figure.dpi": 110,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)
