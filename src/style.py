import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def setup_style(fontsize: int = 11) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": fontsize,
            "axes.labelsize": fontsize + 1,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "legend.title_fontsize": fontsize - 1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.constrained_layout.use": True,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.major.size": 5.0,
            "ytick.minor.size": 2.5,
            "xtick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.major.width": 0.8,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.8,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.framealpha": 1,
            "legend.fancybox": False,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    "#2166ac",
                    "#d6604d",
                    "#4d9221",
                    "#762a83",
                    "#b35806",
                    "#1a9850",
                ]
            ),
        }
    )


def set_map_axes_style(ax, facecolor: str = "white") -> None:
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")


def add_cavity_circle(ax, radius: float = 80.0, **kwargs) -> Circle:
    defaults = dict(
        linestyle="--",
        linewidth=1.2,
        edgecolor="white",
        facecolor="none",
        alpha=0.8,
        zorder=5,
    )
    defaults.update(kwargs)
    circle = Circle((0, 0), radius, **defaults)
    ax.add_patch(circle)
    return circle


def mark_sun(ax, **kwargs) -> None:
    defaults = dict(
        marker="*",
        color="white",
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=6,
        linestyle="none",
    )
    defaults.update(kwargs)
    ax.plot(0, 0, **defaults)


def add_colorbar(
    fig, ax, im, label: str, fontsize: int | None = None, pad: float = 0.01
) -> mpl.colorbar.Colorbar:
    cbar = fig.colorbar(im, ax=ax, pad=pad, fraction=0.046)
    cbar.ax.tick_params(labelsize=fontsize or mpl.rcParams["xtick.labelsize"])
    cbar.set_label(label, fontsize=fontsize or mpl.rcParams["axes.labelsize"])
    return cbar
