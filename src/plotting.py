import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from .style import add_cavity_circle, mark_sun, add_colorbar, set_map_axes_style
from .spatial_p5 import (
    interpolate_map,
    smooth_masked_field,
    axis_labels,
    plane_coords,
    make_grid_edges,
)
from .statistics_p5 import vsini_error, radial_vsini_profile, radial_density_profile

COLORS = {
    "F": "#2166ac",
    "G": "#d68d4d",
    "ALL": "#404040",
}


def _draw_footprint_outline(
    ax,
    footprint_field: np.ndarray,
    ul: float,
    level: float = 0.6,
    extent_based: bool = True,
    **kwargs,
) -> None:
    """Draw the (smoothed) data-footprint boundary as a thin contour.

    ``footprint_field`` is the up-sampled, blurred valid-cell mask returned by
    ``smooth_masked_field`` (second element). The contour at ``level`` traces the
    same boundary used to clip the displayed field, making the "where there is
    data" region visually explicit. Silently no-ops if scikit-image is absent.
    """
    try:
        from skimage import measure
    except Exception:
        return
    defaults = dict(color="0.3", lw=0.7, alpha=0.7, zorder=4)
    defaults.update(kwargs)
    ny, nx = footprint_field.shape
    for contour in measure.find_contours(footprint_field, level):
        cy, cx = contour[:, 0], contour[:, 1]
        if extent_based:
            xs = -ul + cx / (nx - 1) * 2 * ul
            ys = -ul + cy / (ny - 1) * 2 * ul
        else:
            xs, ys = cx, cy
        ax.plot(xs, ys, **defaults)


def plot_sky_distribution(
    df_F,
    df_G,
    vsini_col: str = "vsini",
    lon_col: str = "GLON",
    lat_col: str = "GLAT",
    figsize: tuple = (9, 6.5),
    cmap="plasma",
) -> tuple:
    """
    Bubble chart of vsini in Galactic (l, b) coordinates.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    datasets = [
        (df_F, "F-type", COLORS["F"]),
        (df_G, "G-type", COLORS["G"]),
    ]

    for ax, (df, label, _) in zip(axes, datasets):
        v = df[vsini_col].values
        l = df[lon_col].values
        b = df[lat_col].values

        vmin, vmax = v.min(), v.max()
        sizes = 3 + 100 * (v - vmin) / (vmax - vmin + 1e-9)
        sc = ax.scatter(
            l,
            b,
            s=sizes,
            c=v,
            cmap=cmap,
            # alpha=0.65,
            linewidths=0.2,
            rasterized=True,
            edgecolors="white",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
        cbar.set_label(r"$v\sin i$ [km s$^{-1}$]")

        ax.set_ylabel(r"$b$ [deg]")
        ax.set_ylim(-90, 90)
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.text(
            0.04,
            0.93,
            label,
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="square", fc="white", ec="black"),
        )

    axes[-1].set_xlabel(r"$\ell$ [deg]")
    axes[-1].set_xlim(360, 0)
    axes[-1].set_xticks([360, 300, 240, 180, 120, 60, 0])

    return fig, axes


def plot_density_maps(
    df_all,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    figsize: tuple = (13, 4.2),
    cmap: str = "viridis",
    interpolate: bool = False,
    show_contours: bool = False,
    nan_color: str = "0.95",
    view_limit: float | None = 110.0,
    tick_step: float = 40.0,
    mask_empty: bool = True,
    vmin: float = 1.0,
    vmax: float | None = None,
) -> tuple:
    """
    Three-panel star-count map: XY, XZ, ZY.

    By default the maps are rendered as raw 20 pc cells with no interpolation,
    consistent with the v sin i and residual maps (referee Major Comment 3):
    interpolation is avoided so that the display does not create the visual
    impression of filled structure beyond the populated cells. Empty cells are
    shown in neutral gray when ``mask_empty`` is True. All three panels share a
    common colour scale (``vmin`` to ``vmax``); ``vmin`` defaults to 1 because
    populated cells contain at least one star and empty cells are masked, so the
    scale does not waste range on the unreachable 0-1 interval. ``vmax`` defaults
    to the global maximum across the three projections. Percentile contours are
    available via ``show_contours`` but are off by default.
    """

    planes = ["XY", "XZ", "ZY"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    edges = make_grid_edges(ul, tc, interval)
    Nc = len(edges) - 1
    extent = [-ul, ul, -ul, ul]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    vlim = ul if view_limit is None else view_limit
    ticks = np.arange(-np.floor(vlim / tick_step) * tick_step,
                      np.floor(vlim / tick_step) * tick_step + 1,
                      tick_step).astype(int)

    # Pass 1: build all three count maps and find the shared maximum.
    count_maps = {}
    for plane in planes:
        x, y = plane_coords(df_all, plane)
        count_map = np.zeros((Nc, Nc), dtype=float)
        for iy in range(Nc):
            for ix in range(Nc):
                mask = (
                    (x >= edges[ix])
                    & (x < edges[ix + 1])
                    & (y >= edges[iy])
                    & (y < edges[iy + 1])
                )
                count_map[iy, ix] = mask.sum()
        count_maps[plane] = count_map

    if vmax is None:
        vmax = float(max(m.max() for m in count_maps.values()))

    # Pass 2: render with the shared (vmin, vmax) scale.
    for ax, plane in zip(axes, planes):
        count_map = count_maps[plane]

        ax.set_facecolor(nan_color)
        if interpolate:
            disp = interpolate_map(count_map)
            im = ax.imshow(
                disp, origin="lower", extent=extent, cmap=cmap_obj,
                vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="bilinear", rasterized=True,
            )
            contour_src = disp
        else:
            # Empty cells -> gray "no data" rather than the colormap minimum,
            # for consistency with the masked v sin i / residual maps.
            disp = np.ma.masked_where(count_map <= 0, count_map) if mask_empty \
                else count_map
            im = ax.imshow(
                disp, origin="lower", extent=extent, cmap=cmap_obj,
                vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="none", rasterized=True,
            )
            contour_src = count_map

        cbar = add_colorbar(fig, ax, im, label="Number of stars", pad=0.01)
        # Explicitly mark the colour-scale endpoints (vmin and vmax) so the
        # range is unambiguous, plus a few evenly spaced interior ticks.
        endpoint_ticks = np.unique(
            np.concatenate([[vmin, vmax], np.linspace(vmin, vmax, 5)])
        )
        cbar.set_ticks(endpoint_ticks)
        cbar.ax.set_yticklabels([f"{int(round(t))}" for t in endpoint_ticks])

        if show_contours:
            pos_vals = contour_src[np.asarray(contour_src) > 0]
            if pos_vals.size:
                levels = np.percentile(pos_vals, [25, 50, 75, 90])
                xs = np.linspace(-ul, ul, np.asarray(contour_src).shape[1])
                ys = np.linspace(-ul, ul, np.asarray(contour_src).shape[0])
                ax.contour(
                    xs, ys, np.asarray(contour_src), levels=levels,
                    colors="white", linewidths=0.5, alpha=0.5,
                )

        xl, yl = axis_labels(plane)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-vlim, vlim)
        ax.set_ylim(-vlim, vlim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    return fig, axes


def plot_vsini_bootstrap_panel(
    original: dict,
    bootstrap: dict,
    plane: str,
    ul: float = 150.0,
    method: str = "spline36",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    figsize: tuple = (20, 8),
    nan_color: str = "0.95",
    factor: int = 5,
    smooth_edges: bool = False,
    mask_sigma: float = 0.5,
    mask_level: float = 0.6,
    show_outline: bool = True,
    outline_kw: dict | None = None,
    raw_cells: bool = False,
    view_limit: float | None = None,
    tick_step: float = 40.0,
) -> tuple:
    """Eight-panel v sin i map (mean + quartiles, bootstrap mean + CIs).

    ``view_limit`` sets the displayed half-range in pc (e.g. 110 shows
    -110..110); it only crops the view and does not change the data or the
    ``ul`` grid extent. Defaults to ``ul`` (no cropping). ``tick_step`` sets the
    axis tick spacing in pc; it should divide ``view_limit`` for clean edges.
    """
    if {"mean", "q25", "q50", "q75"}.issubset(original.keys()):
        original_dict = {
            r"$\langle v\sin i \rangle$ [km/s]": original["mean"],
            r"$v\sin i$ [km/s] (q = 1/4)": original["q25"],
            r"$v\sin i$ [km/s] (q = 1/2)": original["q50"],
            r"$v\sin i$ [km/s] (q = 3/4)": original["q75"],
        }
    else:
        original_dict = original

    if {"mean", "q25", "q50", "q75"}.issubset(bootstrap.keys()):
        bootstrap_dict = {
            r"$\langle v\sin i \rangle$ Bootstrap Mean": bootstrap["mean"],
            r"$\langle v\sin i \rangle$ CI Lower (25%)": bootstrap["q25"],
            r"$\langle v\sin i \rangle$ CI Median (50%)": bootstrap["q50"],
            r"$\langle v\sin i \rangle$ CI Upper (75%)": bootstrap["q75"],
        }
    else:
        bootstrap_dict = bootstrap

    fig, axs = plt.subplots(2, 4, figsize=figsize)

    Ny, Nx = next(iter(original_dict.values())).shape
    x_grid = np.arange(Nx)
    y_grid = np.arange(Ny)
    xnew = np.linspace(0, Nx - 1, Nx)
    ynew = np.linspace(0, Ny - 1, Ny)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    if plane == "XY":
        xlabel = "X [pc]"
        ylabel = "Y [pc]"
    elif plane == "XZ":
        xlabel = "X [pc]"
        ylabel = "Z [pc]"
    elif plane == "ZY":
        xlabel = "Z [pc]"
        ylabel = "Y [pc]"
    else:
        raise ValueError("Invalid plane selection. Choose 'XY', 'XZ', or 'ZY'.")

    all_titles = list(original_dict.keys()) + list(bootstrap_dict.keys())
    all_data = list(original_dict.values()) + list(bootstrap_dict.values())

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanmax([np.nanmax(arr) for arr in all_data])

    for idx, ax in enumerate(axs.flat):
        if idx >= len(all_data):
            fig.delaxes(ax)
            continue

        data_array = all_data[idx]
        footprint = None
        if raw_cells:
            # No interpolation at all: each coarse cell shown as a flat block of
            # its actual value. Most transparent rendering; verifiable cell-by-
            # cell against the data. Masked cells stay gray.
            ima = np.ma.masked_invalid(data_array)
            interp = "none"
        elif smooth_edges:
            # De-pixelated boundary: interpolate the interior, then clip to a
            # morphologically-smoothed footprint. Does NOT extrapolate values
            # into empty cells (see smooth_masked_field docstring).
            field, footprint = smooth_masked_field(
                data_array,
                factor=max(factor, 8),
                mask_sigma=mask_sigma,
                mask_level=mask_level,
            )
            ima = np.ma.masked_invalid(field)
            interp = method
        else:
            # NaN-aware up-sampling: empty / low-count cells stay masked instead
            # of being smeared to 0 across the cavity (referee Major Comment 3).
            ima = np.ma.masked_invalid(
                interpolate_map(data_array, factor=factor, remask=True)
            )
            interp = method

        ax.set_facecolor(nan_color)
        cax = ax.imshow(
            ima,
            cmap=cmap_obj,
            aspect="auto",
            origin="lower",
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
        )

        if smooth_edges and show_outline and footprint is not None:
            _draw_footprint_outline(
                ax,
                footprint,
                ul=ul,
                level=mask_level,
                extent_based=False,
                **(outline_kw or {}),
            )

        cbar = add_colorbar(fig, ax, cax, label=all_titles[idx], pad=0.01, fontsize=16)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)

        # Axes are in pixel-index space; map physical pc -> pixel index.
        # pc value c sits at pixel (c + ul)/(2*ul) * (N - 1).
        def _pc_to_px(c, n):
            return (np.asarray(c, dtype=float) + ul) / (2.0 * ul) * (n - 1)

        vlim = ul if view_limit is None else view_limit
        ticklabels = np.arange(-np.floor(vlim / tick_step) * tick_step,
                               np.floor(vlim / tick_step) * tick_step + 1,
                               tick_step).astype(int)
        nx, ny = ima.shape[1], ima.shape[0]
        ax.set_xticks(_pc_to_px(ticklabels, nx))
        ax.set_xticklabels(ticklabels, fontsize=16)
        ax.set_yticks(_pc_to_px(ticklabels, ny))
        ax.set_yticklabels(ticklabels, fontsize=16)

        # Crop the view to +/- view_limit (in pixel units), data untouched.
        ax.set_xlim(_pc_to_px(-vlim, nx), _pc_to_px(vlim, nx))
        ax.set_ylim(_pc_to_px(-vlim, ny), _pc_to_px(vlim, ny))

    return fig, axs


def plot_vmag_bootstrap_panel(
    original: dict,
    bootstrap: dict,
    plane: str,
    ul: float = 150.0,
    method: str = "spline36",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
    figsize: tuple = (20, 8),
) -> tuple:
    if {"mean", "q25", "q50", "q75"}.issubset(original.keys()):
        original = {
            r"$\langle V_{mag} \rangle$ (Apparent) [Unweighted]": original["mean"],
            r"$V_{mag}$ (q = 1/4)": original["q25"],
            r"$V_{mag}$ (q = 1/2)": original["q50"],
            r"$V_{mag}$ (q = 3/4)": original["q75"],
        }

    if {"mean", "q25", "q50", "q75"}.issubset(bootstrap.keys()):
        bootstrap = {
            r"$\langle V_{mag} \rangle$ Bootstrap Mean": bootstrap["mean"],
            r"$\langle V_{mag} \rangle$ CI Lower": bootstrap["q25"],
            r"$\langle V_{mag} \rangle$ CI Median": bootstrap["q50"],
            r"$\langle V_{mag} \rangle$ CI Upper": bootstrap["q75"],
        }

    return plot_vsini_bootstrap_panel(
        original=original,
        bootstrap=bootstrap,
        plane=plane,
        ul=ul,
        method=method,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        figsize=figsize,
    )


def plot_significance_maps(
    df,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    min_count: int = 20,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    z_range: float = 3.5,
    cmap: str = "RdBu_r",
    figsize: tuple = (13, 4.2),
) -> tuple:
    planes = ["XY", "XZ", "ZY"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    extent = [-ul, ul, -ul, ul]

    vals_all = df[vsini_col].values
    wts_all = df[weight_col].values
    g_mean = float(np.average(vals_all, weights=wts_all))
    norm = TwoSlopeNorm(vmin=-z_range, vcenter=0.0, vmax=z_range)
    nan_color = "0.95"
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    edges = make_grid_edges(ul, tc, interval)
    Nc = len(edges) - 1

    for ax, plane in zip(axes, planes):
        x, y = plane_coords(df, plane)
        vsini = df[vsini_col].values
        weights = df[weight_col].values

        zscore_map = np.full((Nc, Nc), np.nan)
        for iy in range(Nc):
            for ix in range(Nc):
                mask = (
                    (x >= edges[ix])
                    & (x < edges[ix + 1])
                    & (y >= edges[iy])
                    & (y < edges[iy + 1])
                )
                n = mask.sum()
                if n < min_count:
                    continue
                v = vsini[mask]
                w = weights[mask]
                local_mean = float(np.average(v, weights=w))
                local_std = float(np.sqrt(np.average((v - local_mean) ** 2, weights=w)))
                local_se = local_std / np.sqrt(n) if local_std > 0 else np.nan
                if np.isfinite(local_se) and local_se > 0:
                    zscore_map[iy, ix] = (local_mean - g_mean) / local_se

        sm = interpolate_map(zscore_map, remask=True)
        sm = np.ma.masked_invalid(sm)
        set_map_axes_style(ax, facecolor=nan_color)
        ax.set_facecolor(nan_color)
        im = ax.imshow(
            sm,
            origin="lower",
            extent=extent,
            cmap=cmap_obj,
            norm=norm,
            aspect="equal",
            interpolation="bilinear",
            rasterized=True,
        )

        cbar = add_colorbar(fig, ax, im, label=r"$Z$-score")
        cbar.set_ticks(np.linspace(-z_range, z_range, 7))
        cbar.ax.tick_params(labelsize=8)

        ax.contour(
            sm,
            levels=[-2, -1, 1, 2],
            extent=extent,
            colors=["0.2", "0.5", "0.5", "0.2"],
            linewidths=[0.8, 0.5, 0.5, 0.8],
            linestyles="--",
            origin="lower",
        )

        xl, yl = axis_labels(plane)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-ul, ul)
        ax.set_ylim(-ul, ul)

        add_cavity_circle(ax, edgecolor="black")
        mark_sun(ax, color="black")

    return fig, axes


def plot_residual_maps(
    residual_results: dict,
    ul: float = 150.0,
    r_cut: float = 80.0,
    figsize: tuple = (13, 4.2),
    clim: float | None = None,
    cmap: str = "RdBu_r",
    field: str = "residual",
    nan_color: str = "0.95",
    title_prefix: str | None = None,
    smooth_edges: bool = False,
    mask_sigma: float = 0.5,
    mask_level: float = 0.6,
    show_outline: bool = True,
    outline_kw: dict | None = None,
    raw_cells: bool = False,
    view_limit: float | None = None,
    tick_step: float = 40.0,
) -> tuple:
    """Plot cell-wise maps for the requested *field*.

    ``field`` selects which map to draw: ``"residual"`` (default, the
    obs - null difference shown in Fig. 8), ``"expected"`` (the Teff-stratified
    null field requested by the referee), or ``"observed"``. Masked / low-count
    cells are drawn in ``nan_color`` rather than as a value on the colour scale.
    """
    planes = ["XY", "XZ", "ZY"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    extent = [-ul, ul, -ul, ul]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    diverging = field in ("residual",)

    if clim is None:
        all_res = np.concatenate(
            [
                residual_results[p][field].ravel()
                for p in planes
                if p in residual_results
            ]
        )
        if diverging:
            clim = float(np.nanpercentile(np.abs(all_res), 98))
        else:
            vlo = float(np.nanpercentile(all_res, 2))
            vhi = float(np.nanpercentile(all_res, 98))

    if diverging:
        norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)
    else:
        norm = None

    for ax, plane in zip(axes, planes):
        if plane not in residual_results:
            ax.set_visible(False)
            continue

        res = residual_results[plane][field]
        footprint = None
        if raw_cells:
            sm = np.ma.masked_invalid(res)
            interp = "none"
        elif smooth_edges:
            sm_field, footprint = smooth_masked_field(
                res, factor=8, mask_sigma=mask_sigma, mask_level=mask_level
            )
            sm = np.ma.masked_invalid(sm_field)
            interp = "bilinear"
        else:
            sm = np.ma.masked_invalid(interpolate_map(res, remask=True))
            interp = "bilinear"

        set_map_axes_style(ax, facecolor=nan_color)
        ax.set_facecolor(nan_color)
        if diverging:
            im = ax.imshow(
                sm,
                origin="lower",
                extent=extent,
                cmap=cmap_obj,
                norm=norm,
                aspect="equal",
                interpolation=interp,
                rasterized=True,
            )
        else:
            im = ax.imshow(
                sm,
                origin="lower",
                extent=extent,
                cmap=cmap_obj,
                vmin=vlo,
                vmax=vhi,
                aspect="equal",
                interpolation=interp,
                rasterized=True,
            )

        if diverging and not raw_cells:
            ax.contour(
                sm,
                levels=[-2, -1, 1, 2],
                extent=extent,
                colors=["0.2", "0.5", "0.5", "0.2"],
                linewidths=[0.8, 0.5, 0.5, 0.8],
                linestyles="--",
                origin="lower",
            )

        if smooth_edges and show_outline and footprint is not None:
            _draw_footprint_outline(
                ax,
                footprint,
                ul=ul,
                level=mask_level,
                extent_based=True,
                **(outline_kw or {}),
            )

        label = (
            r"$\Delta\langle v\sin i\rangle$ [km s$^{-1}$]"
            if diverging
            else r"$\mathbb{E}[\langle v\sin i\rangle_{\rm null}]$ [km s$^{-1}$]"
            if field == "expected"
            else r"$\langle v\sin i\rangle_{\rm obs}$ [km s$^{-1}$]"
        )
        cbar = add_colorbar(fig, ax, im, label=label)
        if diverging:
            cbar.set_ticks(np.linspace(-clim, clim, 7))
        cbar.ax.tick_params(labelsize=8)

        xl, yl = axis_labels(plane)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

        # Crop view; guard against clipping the dashed r_cut cavity circle.
        vlim = ul if view_limit is None else max(view_limit, r_cut + 10.0)
        ax.set_xlim(-vlim, vlim)
        ax.set_ylim(-vlim, vlim)
        ticks = np.arange(-np.floor(vlim / tick_step) * tick_step,
                          np.floor(vlim / tick_step) * tick_step + 1,
                          tick_step).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        add_cavity_circle(
            ax, radius=r_cut, edgecolor="black", linewidth=1.2, linestyle="--"
        )
        mark_sun(ax, color="black")
        prefix = title_prefix if title_prefix is not None else field.capitalize()
        ax.set_title(f"{prefix} ({plane})")

    return fig, axes


def _infer_completeness_limit(
    centers: np.ndarray,
    counts: np.ndarray,
    threshold: float = 0.85,
    min_consecutive: int = 1,
    interpolate_crossing: bool = True,
) -> tuple[float, np.ndarray]:
    peak_idx = int(np.argmax(counts))

    # Use bright-end bins (before the peak) with enough stars to be reliable.
    bright_mask = (np.arange(len(centers)) < peak_idx) & (counts > 5)

    if bright_mask.sum() >= 2:
        # Fit intercept only; slope is fixed at 0.6 in log space.
        log_counts = np.log10(counts[bright_mask])
        b = np.mean(log_counts - 0.6 * centers[bright_mask])
    else:
        b = np.log10(max(counts[peak_idx], 1)) - 0.6 * centers[peak_idx]

    slope_n = 10.0 ** (0.6 * centers + b)
    ratio = np.divide(
        counts,
        slope_n,
        out=np.full_like(counts, np.nan, dtype=float),
        where=slope_n > 0,
    )

    # Identify the ratio peak on reliable bins, then search for the first
    # sustained drop below the threshold on the descending branch.
    finite_mask = np.isfinite(ratio)
    reliable_mask = finite_mask & (counts > 5)

    if np.any(reliable_mask):
        ratio_peak_idx = int(np.nanargmax(np.where(reliable_mask, ratio, -np.inf)))
    else:
        ratio_peak_idx = peak_idx

    valid_mask = finite_mask & (counts > 0)

    v_lim = centers[ratio_peak_idx]
    run = 0
    first_deficit = None
    for i in range(max(ratio_peak_idx + 1, 1), len(centers)):
        if valid_mask[i] and ratio[i] < threshold:
            run += 1
            if run >= max(1, int(min_consecutive)):
                first_deficit = i - run + 1
                break
        else:
            run = 0

    if first_deficit is not None:
        v_lim = centers[first_deficit]
        if interpolate_crossing and first_deficit > 0:
            left = first_deficit - 1
            while left >= 0 and not valid_mask[left]:
                left -= 1
            if left >= 0 and ratio[left] >= threshold:
                dr = ratio[first_deficit] - ratio[left]
                if dr != 0:
                    frac = (threshold - ratio[left]) / dr
                    if 0.0 <= frac <= 1.0:
                        v_lim = centers[left] + frac * (
                            centers[first_deficit] - centers[left]
                        )

    return float(v_lim), slope_n


def plot_magnitude_completeness(
    df_F,
    df_G,
    v_lim_override: dict | None = None,
    completeness_threshold: float = 0.85,
    min_consecutive: int = 1,
    interpolate_vlim: bool = True,
    bin_width: float = 0.2,
    figsize: tuple = (10, 5.5),
) -> tuple:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 0.5], "hspace": 0.02, "wspace": 0.04},
    )
    inferred_v_lim = {}

    vmag_all = np.concatenate([df_F["Vmag"].to_numpy(), df_G["Vmag"].to_numpy()])
    vmin = np.floor(vmag_all.min() / bin_width) * bin_width
    vmax = np.ceil(vmag_all.max() / bin_width) * bin_width
    bins = np.arange(vmin, vmax + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    datasets = [
        ("F", df_F, COLORS["F"], axes[0, 0], axes[1, 0]),
        ("G", df_G, COLORS["G"], axes[0, 1], axes[1, 1]),
    ]

    for idx, (label, df, color, ax_main, ax_ratio) in enumerate(datasets):
        vmag = df["Vmag"].values
        counts, _ = np.histogram(vmag, bins=bins)
        counts = counts.astype(float)

        v_inf, slope_n = _infer_completeness_limit(
            centers,
            counts,
            threshold=completeness_threshold,
            min_consecutive=min_consecutive,
            interpolate_crossing=interpolate_vlim,
        )
        inferred_v_lim[label] = v_inf

        ratio = np.divide(
            counts,
            slope_n,
            out=np.full_like(counts, np.nan, dtype=float),
            where=slope_n > 0,
        )
        peak_idx = int(np.argmax(counts))
        incomplete_mask = (
            (np.arange(len(centers)) >= peak_idx)
            & (counts > 0)
            & (ratio < completeness_threshold)
        )

        ax_main.stairs(
            counts,
            bins,
            fill=True,
            color=color,
            alpha=0.24,
            linewidth=0,
        )
        ax_main.stairs(
            counts,
            bins,
            color=color,
            linewidth=1.4,
            label=f"{label}-type stars",
        )
        ax_main.plot(
            centers,
            slope_n,
            color="black",
            lw=1.2,
            ls="--",
            label=r"Euclidean model",
        )
        ax_main.axvspan(v_inf, bins[-1], alpha=0.09, color="0.4", lw=0)
        ax_main.axvline(
            v_inf,
            color="0.35",
            lw=1.3,
            ls=":",
            label=rf"$V_{{\rm lim}}^{{\rm infer}}={v_inf:.2f}$",
        )

        v_user = None
        if v_lim_override is not None:
            v_user = v_lim_override.get(label, None)
            if v_user is not None:
                ax_main.axvline(
                    float(v_user),
                    color=color,
                    lw=1.2,
                    ls=(0, (5, 2)),
                    alpha=0.9,
                    label=rf"$V_{{\rm lim}}^{{\rm manual}}={float(v_user):.2f}$",
                )

        ax_main.set_yscale("log")
        if idx == 0:
            ax_main.set_ylabel(r"$dN/dm$")
        ax_main.legend(loc="upper left", frameon=True)

        positive_vals = np.concatenate([counts[counts > 0], slope_n[slope_n > 0]])
        if positive_vals.size > 0:
            ymin = max(1.0, positive_vals.min() * 0.8)
            ymax = positive_vals.max() * 1.35
            ax_main.set_ylim(ymin, ymax)

        # Completeness-ratio panel.
        ax_ratio.plot(
            centers,
            ratio,
            color=color,
            lw=1.4,
            marker="o",
            ms=2.2,
            mfc=color,
            mec="white",
            mew=0.2,
        )
        if np.any(incomplete_mask):
            ax_ratio.scatter(
                centers[incomplete_mask],
                ratio[incomplete_mask],
                s=15,
                color="crimson",
                zorder=3,
            )

        ax_ratio.axhline(1.0, color="0.35", lw=1.0, ls="--")
        ax_ratio.axhline(
            completeness_threshold,
            color="crimson",
            lw=1.0,
            ls=":",
            label=rf"Threshold = {completeness_threshold:.2f}",
        )
        ax_ratio.axvspan(v_inf, bins[-1], alpha=0.09, color="0.4", lw=0)
        ax_ratio.axvline(v_inf, color="0.35", lw=1.2, ls=":")

        if v_user is not None:
            ax_ratio.axvline(
                float(v_user),
                color=color,
                lw=1.2,
                ls=(0, (5, 2)),
                alpha=0.9,
            )

        finite_ratio = ratio[np.isfinite(ratio) & (counts > 0)]
        if finite_ratio.size > 0:
            ratio_max = min(2.5, max(1.2, np.percentile(finite_ratio, 98) * 1.1))
        else:
            ratio_max = 1.2
        ax_ratio.set_ylim(0.0, ratio_max)
        ax_ratio.set_xlim(bins[0], bins[-1])
        if idx == 0:
            ax_ratio.set_ylabel(r"$N_{\rm obs}/N_{\rm model}$")
            ax_ratio.legend(loc="lower left", frameon=True)
        ax_ratio.set_xlabel(r"$V$ [mag]")

    return fig, axes, inferred_v_lim


def _draw_permutation_histogram(
    ax,
    null_dist: np.ndarray,
    obs_diff: float,
) -> None:
    bins = 60
    ax.hist(
        null_dist,
        bins=bins,
        color="0.7",
        alpha=0.45,
        density=True,
        histtype="stepfilled",
        label="Null distribution",
    )
    ax.hist(
        null_dist,
        bins=bins,
        color="0.35",
        linewidth=1.1,
        density=True,
        histtype="step",
    )
    ax.axvline(
        obs_diff,
        color="crimson",
        lw=2,
        label=rf"Observed: $\Delta\langle v\sin i\rangle = {obs_diff:.2f}$ km/s",
    )
    ax.axvline(
        np.percentile(null_dist, 95),
        color="steelblue",
        lw=1.2,
        linestyle="--",
        label="95th percentile of null",
    )

    ax.set_xlabel(
        r"$\Delta\langle v\sin i\rangle_{\rm shell} - "
        r"\langle v\sin i\rangle_{\rm cavity}$ [km s$^{-1}$]"
    )
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8.5, loc="center", bbox_to_anchor=(0.64, 0.86))


def plot_permutation_result(
    null_dist: np.ndarray,
    obs_diff: float,
    figsize: tuple = (7, 4),
) -> tuple:
    fig, ax = plt.subplots(figsize=figsize)
    _draw_permutation_histogram(ax, null_dist, obs_diff)
    return fig, ax


def plot_radial_scan(df_scan, r_cut: float = 80.0, figsize: tuple = (8, 6)) -> tuple:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    r = df_scan["Radius"].values
    ob = df_scan["Observed"].values
    ex = df_scan["Expected"].values
    uc = df_scan["Uncertainty"].values
    zs = df_scan["Z_score"].values

    ax1.plot(
        r, ob, "o-", lw=1.5, color=COLORS["ALL"], label=r"Observed (shell - cavity)"
    )
    ax1.errorbar(
        r,
        ex,
        yerr=uc,
        fmt="s--",
        color="0.55",
        lw=1.2,
        label=r"$T_{\rm eff}$-stratified null",
    )
    ax1.fill_between(r, ex, ob, alpha=0.12, color=COLORS["ALL"])
    ax1.axvline(
        r_cut, color="crimson", lw=1.2, ls=":", label=rf"$R_{{cut}}={r_cut:.0f}$ pc"
    )
    ax1.set_ylabel(r"$\Delta\langle v\sin i\rangle$ [km s$^{-1}$]")
    ax1.legend()

    ax2.plot(r, zs, "D-", lw=1.5, color=COLORS["ALL"])
    ax2.axhline(0.0, color="0.4", lw=0.7)
    ax2.axhline(3.0, color="steelblue", ls="--", lw=1.2, label=r"$\pm 3\sigma$")
    ax2.axhline(-3.0, color="steelblue", ls="--", lw=1.2)
    ax2.axvline(r_cut, color="crimson", lw=1.2, ls=":")
    ax2.set_xlabel(r"Cavity radius $R_{\rm cut}$ [pc]")
    ax2.set_ylabel(r"$Z$-score")
    ax2.legend()
    return fig, (ax1, ax2)


def plot_vsini_distance(
    df, cmap: str = "viridis", figsize: tuple = (10, 4.5), ax=None, label="",
    ylim=None, vmin=None, vmax=None,
) -> tuple:
    v = df["vsini"].values
    d = df["Dist"].values
    e = vsini_error(v)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc = ax.scatter(
        d, v, c=e, cmap=cmap, s=15, edgecolor="k", rasterized=True, label=label,
        vmin=vmin, vmax=vmax,
    )

    ax.errorbar(
        d, v, yerr=e, fmt="none", ecolor="k", elinewidth=0.4, alpha=0.5, zorder=0
    )

    # colorbar per subplot
    cbar = add_colorbar(fig, ax, sc, label=r"$\sigma(v\sin i)$ [km s$^{-1}$]", pad=0.01)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(r"Distance [pc]")
    ax.set_ylabel(r"$v\sin i$ [km s$^{-1}$]")

    # legend INSIDE
    if label:
        ax.legend(loc="upper right", frameon=True)

    return fig, ax


def plot_mv_weight_distributions(
    df_F: "pd.DataFrame",
    df_G: "pd.DataFrame",
    df_all: "pd.DataFrame",
    mv_col: str = "VMAG",
    weight_col: str = "w_vmax",
    figsize: tuple = (9, 7),
) -> tuple:
    fig, (ax_mv, ax_w) = plt.subplots(2, 1, figsize=figsize)

    datasets = [
        ("F stars", df_F, COLORS["F"], dict(lw=1.5)),
        ("G stars", df_G, COLORS["G"], dict(lw=1.5)),
        ("All Stars", df_all, COLORS["ALL"], dict(lw=1.5, ls="--")),
    ]

    for label, df, color, lkw in datasets:
        mv = df[mv_col].dropna().values
        logw = np.log10(df[weight_col].dropna().values)

        ax_mv.hist(
            mv, bins=30, color=color, alpha=0.25, histtype="stepfilled", density=True
        )
        ax_mv.hist(
            mv,
            bins=30,
            color=color,
            alpha=0.9,
            histtype="step",
            density=True,
            label=label,
            **lkw,
        )

        ax_w.hist(
            logw, bins=30, color=color, alpha=0.25, histtype="stepfilled", density=True
        )
        ax_w.hist(
            logw,
            bins=30,
            color=color,
            alpha=0.9,
            histtype="step",
            density=True,
            label=label,
            **lkw,
        )

    ax_mv.set_xlabel(r"Absolute Magnitude ($V_{\rm MAG}$)")
    ax_mv.set_ylabel("Density")
    ax_mv.legend()

    ax_w.set_xlabel(r"$\log_{10}(1/V_{\rm max}\ \rm weight)$")
    ax_w.set_ylabel("Density")
    ax_w.legend()
    return fig, (ax_mv, ax_w)


def plot_teff_strata_boxplots(
    df,
    n_bins=10,
    teff_col="Teff",
    type_label="",
    figsize=(9, 4.5),
    ax=None,
    offset=0.0,
    color="white",
):
    df = df.copy()
    df["_bin"] = pd.qcut(df[teff_col], q=n_bins, labels=False, duplicates="drop")

    bin_groups = [
        df.loc[df["_bin"] == b, teff_col].dropna().values
        for b in range(int(df["_bin"].max()) + 1)
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    positions = [i + offset for i in range(len(bin_groups))]

    ax.boxplot(
        bin_groups,
        positions=positions,
        patch_artist=True,
        medianprops=dict(color="tab:red", lw=1),
        boxprops=dict(facecolor=color, edgecolor="0.3", lw=0.9),
        whiskerprops=dict(color="0.3", lw=0.9),
        capprops=dict(color="0.3", lw=0.9),
        flierprops=dict(
            marker="o",
            markersize=2.5,
            markerfacecolor="0.5",
            markeredgecolor="none",
            alpha=0.4,
        ),
        widths=0.35,
    )

    ax.set_xlabel(r"$T_{\rm eff}$ bin (quantiles)")
    ax.set_ylabel(r"$T_{\rm eff}$ [K]")
    ax.set_xticks(range(len(bin_groups)))
    ax.set_xticklabels((np.arange(len(bin_groups)) + 1).astype(int))

    # create legend handle for this dataset
    legend_handle = Patch(facecolor=color, edgecolor="0.3", label=type_label)

    return fig, ax, legend_handle


def plot_robustness_comparison(
    null_dist: np.ndarray,
    obs_mc_dist: np.ndarray,
    mu_null: float | None = None,
    mu_obs: float | None = None,
    figsize: tuple = (8, 5),
) -> tuple:
    if mu_null is None:
        mu_null = float(null_dist.mean())
    if mu_obs is None:
        mu_obs = float(obs_mc_dist.mean())

    sigma_null = float(null_dist.std())
    sigma_obs = float(obs_mc_dist.std())

    delta_v = mu_obs - mu_null
    z_robust = delta_v / np.sqrt(sigma_obs**2 + sigma_null**2)

    fig, ax = plt.subplots(figsize=figsize)

    # Demographic null (light)
    x_null = np.linspace(null_dist.min() - 1, null_dist.max() + 1, 400)
    ax.hist(
        null_dist,
        bins=60,
        density=True,
        color="0.7",
        histtype="stepfilled",
        alpha=0.85,
        label=rf"Demographic Bias Only ($\mu={mu_null:.2f},\ \sigma={sigma_null:.2f}$)",
    )
    ax.plot(x_null, norm.pdf(x_null, mu_null, sigma_null), color="0.3", lw=1.2, ls="--")

    # Observed + instrumental errors (dark)
    x_obs = np.linspace(obs_mc_dist.min() - 0.5, obs_mc_dist.max() + 0.5, 400)
    ax.hist(
        obs_mc_dist,
        bins=60,
        density=True,
        color="0.25",
        histtype="stepfilled",
        alpha=0.85,
        label=rf"Observed + Instr. Errors ($\mu={mu_obs:.2f},\ \sigma={sigma_obs:.2f}$)",
    )
    ax.plot(x_obs, norm.pdf(x_obs, mu_obs, sigma_obs), color="0.05", lw=1.2, ls="--")

    ax.set_xlabel(r"Velocity Contrast $\Delta v$ (Shell $-$ Cavity) [km s$^{-1}$]")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper left")
    return fig, ax


def plot_morans_I(
    result: dict, label: str = "All stars", figsize: tuple = (7, 4)
) -> tuple:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        result["null_dist"],
        bins=50,
        color="0.7",
        alpha=0.45,
        density=True,
        histtype="stepfilled",
        label="Null distribution",
    )
    ax.hist(
        result["null_dist"],
        bins=50,
        color="0.35",
        linewidth=1.1,
        density=True,
        histtype="step",
    )
    ax.axvline(
        result["I"],
        color="crimson",
        lw=2,
        label=rf"$I = {result['I']:.4f}$  ($Z = {result['Z_I']:.2f}$)",
    )
    ax.axvline(
        result["E_I"],
        color="steelblue",
        lw=1.2,
        ls="--",
        label=rf"$E[I] = {result['E_I']:.4f}$",
    )

    ax.set_xlabel("Moran's $I$")
    ax.set_ylabel("Probability density")
    ax.legend()
    return fig, ax


def plot_error_maps(
    df_all: "pd.DataFrame",
    df_F: "pd.DataFrame",
    df_G: "pd.DataFrame",
    ul: float = 150.0,
    tc: float = 20.0,
    min_count: int = 20,
    r_cut: float = 80.0,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    figsize: tuple = (14, 12),
    raw_cells: bool = True,
    nan_color: str = "0.95",
    view_limit: float | None = 110.0,
    tick_step: float = 40.0,
    shared_scale: bool = True,
) -> tuple:
    planes = ["XY", "XZ", "ZY"]
    # Per-row sequential colormaps are intentional (one per subsample); the
    # masked-cell colour is now a single neutral grey for all rows, consistent
    # with Figs. 3, 4, and 8 (referee Major Comment 3), instead of a tinted
    # per-row "bad" colour that could read as a low value on the scale.
    row_data = [
        (df_all, "All stars (F+G)", "Grays"),
        (df_F, "F-type", "Blues"),
        (df_G, "G-type", "Oranges"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    edges = make_grid_edges(ul, tc)
    Nc = len(edges) - 1
    extent = [-ul, ul, -ul, ul]

    vlim = ul if view_limit is None else max(view_limit, r_cut + 10.0)
    ticks = np.arange(-np.floor(vlim / tick_step) * tick_step,
                      np.floor(vlim / tick_step) * tick_step + 1,
                      tick_step).astype(int)
    interp = "none" if raw_cells else "spline16"

    # --- Pass 1: build all 9 maps and collect values for a shared range ---
    all_maps = {}  # (row, col) -> err_map
    cmaps = []
    global_vals = []
    for row, (df, row_label, cmap_name) in enumerate(row_data):
        sigma_v = vsini_error(df[vsini_col].to_numpy())
        weights = df[weight_col].to_numpy()
        cmap_obj = plt.get_cmap(cmap_name).copy()
        cmap_obj.set_bad(nan_color)
        cmaps.append(cmap_obj)

        for col, plane in enumerate(planes):
            x, y = plane_coords(df, plane)
            err_map = np.full((Nc, Nc), np.nan)
            for iy in range(Nc):
                for ix in range(Nc):
                    mask = (
                        (x >= edges[ix])
                        & (x < edges[ix + 1])
                        & (y >= edges[iy])
                        & (y < edges[iy + 1])
                    )
                    n = mask.sum()
                    if n < min_count:
                        continue
                    w = weights[mask]
                    e = sigma_v[mask]
                    total_w = w.sum()
                    if total_w > 0:
                        err_map[iy, ix] = float(np.average(e, weights=w))
            all_maps[(row, col)] = err_map
            global_vals.append(err_map[np.isfinite(err_map)])

    global_vals = np.concatenate(global_vals) if global_vals else np.array([1.0])
    g_vmin = float(global_vals.min())
    g_vmax = float(global_vals.max())

    # --- Pass 2: render ---
    for row, (df, row_label, cmap_name) in enumerate(row_data):
        cmap_obj = cmaps[row]
        if shared_scale:
            vmin_r, vmax_r = g_vmin, g_vmax
        else:
            rvals = np.concatenate(
                [all_maps[(row, c)][np.isfinite(all_maps[(row, c)])]
                 for c in range(len(planes))]
                or [np.array([1.0])]
            )
            vmin_r = float(rvals.min()) if rvals.size else 0.0
            vmax_r = float(rvals.max()) if rvals.size else 1.0

        for col, plane in enumerate(planes):
            ax = axes[row, col]
            err_map = all_maps[(row, col)]

            sm = np.ma.masked_invalid(
                err_map if raw_cells else interpolate_map(err_map, factor=2)
            )
            ax.set_facecolor(nan_color)
            for spine in ax.spines.values():
                spine.set_edgecolor("0.5")

            im = ax.imshow(
                sm,
                origin="lower",
                extent=extent,
                cmap=cmap_obj,
                vmin=vmin_r,
                vmax=vmax_r,
                aspect="equal",
                interpolation=interp,
                rasterized=True,
            )

            cbar = add_colorbar(
                fig,
                ax,
                im,
                label=r"$\langle\sigma_v\rangle$ [km s$^{-1}$]",
            )
            cbar.ax.tick_params()

            xl, yl = axis_labels(plane)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl if col == 0 else "")
            ax.set_xlim(-vlim, vlim)
            ax.set_ylim(-vlim, vlim)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params()

            add_cavity_circle(ax, radius=r_cut, edgecolor="0.3")
            mark_sun(ax, color="0.3", markeredgecolor="none", markersize=10)

            if col == 0:
                ax.set_ylabel(f"{row_label}\n{yl}")
            else:
                ax.set_ylabel(f"{yl}")

    return fig, axes


def plot_radial_profile(
    df,
    radius: str = "R3D",
    bin_edges=None,
    r_cut: float = 80.0,
    compare_weighting: bool = True,
    figsize: tuple = (6.4, 4.6),
    show_quartiles: bool = True,
    show_density: bool = False,
    show_unweighted_band: bool = False,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    seed: int | None = 42,
    n_bootstrap: int = 1000,
):
    """Per-star weighted v sin i vs. galactocentric (heliocentric) radius.

    Quantitative anchor for the contrast-boundary discussion: the profile is
    built from individual stellar radii, independent of the 2-D cell grid. Plots
    the 1/Vmax-weighted mean with a bootstrap 1-sigma band and (optionally) the
    25-75 percent weighted quartile envelope. A vertical line marks ``r_cut``.
    If ``compare_weighting`` is True, the unweighted mean is overplotted.

    If ``show_density`` is True, a stacked lower panel shows the stellar number
    density n(R) on the same radial axis, demonstrating the two-population
    separation (density peaks at small R and declines outward) that motivates
    the choice of contrast boundary.
    """
    import numpy as np

    prof = radial_vsini_profile(
        df, radius=radius, bin_edges=bin_edges, vsini_col=vsini_col,
        weight_col=weight_col, use_weights=True, n_bootstrap=n_bootstrap, seed=seed,
    )
    c = prof["centers"]
    good = prof["count"] > 0

    if show_density:
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 1.4),
            sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axd = None

    if show_quartiles:
        ax.fill_between(
            c[good], prof["q25"][good], prof["q75"][good],
            color=COLORS["F"], alpha=0.15, lw=0,
            label="25-75% (weighted)",
        )

    se = prof["se"].copy()
    se[~np.isfinite(se)] = 0.0
    ax.fill_between(
        c[good], (prof["mean"] - se)[good], (prof["mean"] + se)[good],
        color=COLORS["F"], alpha=0.35, lw=0,
    )
    ax.plot(
        c[good], prof["mean"][good], "-o", color=COLORS["F"],
        lw=1.8, ms=5, label=r"weighted $\langle v\sin i\rangle$",
    )

    if compare_weighting:
        prof_u = radial_vsini_profile(
            df, radius=radius, bin_edges=bin_edges, vsini_col=vsini_col,
            weight_col=weight_col, use_weights=False,
            n_bootstrap=(n_bootstrap if show_unweighted_band else 0), seed=seed,
        )
        gu = prof_u["count"] > 0
        if show_unweighted_band:
            se_u = prof_u["se"].copy()
            se_u[~np.isfinite(se_u)] = 0.0
            ax.fill_between(
                prof_u["centers"][gu],
                (prof_u["mean"] - se_u)[gu],
                (prof_u["mean"] + se_u)[gu],
                color=COLORS["ALL"], alpha=0.18, lw=0,
            )
        ax.plot(
            prof_u["centers"][gu], prof_u["mean"][gu], "--s",
            color=COLORS["ALL"], lw=1.3, ms=4, mfc="white",
            label=r"unweighted $\langle v\sin i\rangle$",
        )

    ax.axvline(r_cut, color="0.4", ls=":", lw=1.2)
    ax.annotate(
        rf"$R \sim {r_cut:.0f}$ pc", xy=(r_cut, ax.get_ylim()[1]),
        xytext=(3, -3), textcoords="offset points", va="top", ha="left",
        color="0.3", fontsize=10,
    )

    ax.set_ylabel(r"$\langle v\sin i\rangle$ [km s$^{-1}$]")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(prof["bin_edges"][0], prof["bin_edges"][-1])

    # annotate per-bin counts along the top
    ymax = ax.get_ylim()[1]
    for ci, ni in zip(c[good], prof["count"][good]):
        ax.annotate(
            f"{ni}", xy=(ci, ymax), xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.5, color="0.5",
        )

    rlabel = r"$R = \sqrt{X^2+Y^2+Z^2}$ [pc]" if radius == "R3D" else r"$R_{xy}$ [pc]"

    if show_density:
        geom = "sphere" if radius == "R3D" else "annulus"
        dens = radial_density_profile(
            df, radius=radius, bin_edges=bin_edges, geometry=geom
        )
        gd = dens["count"] > 0
        axd.plot(
            dens["centers"][gd], dens["density"][gd] * 1e5, "-D",
            color=COLORS["G"], lw=1.6, ms=4,
        )
        axd.axvline(r_cut, color="0.4", ls=":", lw=1.2)
        axd.set_yscale("log")
        axd.set_ylabel(r"$n(R)$ [$10^{-5}\,\mathrm{pc}^{-3}$]"
                       if radius == "R3D"
                       else r"$n(R)$ [$10^{-5}\,\mathrm{pc}^{-2}$]")
        axd.set_xlabel(rlabel)
    else:
        ax.set_xlabel(rlabel)

    return fig, ax, prof



def plot_residual_decomposition(
    residual_results: dict,
    three_rows: bool = False,
    ul: float = 150.0,
    r_cut: float = 80.0,
    planes: tuple = ("XY", "XZ", "ZY"),
    vsini_cmap: str = "viridis",
    residual_cmap: str = "RdBu_r",
    vsini_vmin: float = 0.0,
    vsini_vmax: float = 20.0,
    clim: float | None = None,
    nan_color: str = "0.95",
    raw_cells: bool = True,
    view_limit: float | None = 110.0,
    tick_step: float = 40.0,
    figsize: tuple | None = None,
) -> tuple:
    """Stacked decomposition of the residual maps (companion to Fig. 8).

    Two layouts, selected by ``three_rows``:

    - ``three_rows=False`` (default): 2 rows x len(planes). Row 1 is the
      Teff-stratified null/expected field E[<vsini>_null]; row 2 is the residual
      <vsini>_obs - E[<vsini>_null].
    - ``three_rows=True``: 3 rows. Row 1 observed <vsini>_obs, row 2 expected
      (null), row 3 residual -- i.e. the full obs - null = residual stack.

    The observed and expected rows use the *standard* sequential v sin i scale
    (same vmin/vmax and colormap as Figs. 3, 4, 13, 14); only the residual row
    uses the diverging, zero-symmetric scale. Masked cells are gray; cells are
    rendered raw (no interpolation) by default.
    """
    if three_rows:
        row_fields = [("observed", "vsini"), ("expected", "vsini"),
                      ("residual", "diverging")]
        row_labels = [
            r"$\langle v\sin i\rangle_{\rm obs}$",
            r"$\mathbb{E}[\langle v\sin i\rangle_{\rm null}]$",
            r"$\Delta\langle v\sin i\rangle$",
        ]
    else:
        row_fields = [("expected", "vsini"), ("residual", "diverging")]
        row_labels = [
            r"$\mathbb{E}[\langle v\sin i\rangle_{\rm null}]$",
            r"$\Delta\langle v\sin i\rangle$",
        ]

    nrow = len(row_fields)
    ncol = len(planes)
    if figsize is None:
        figsize = (4.3 * ncol, 4.0 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_2d(axes)
    extent = [-ul, ul, -ul, ul]

    # sequential colormap (observed / expected rows)
    seq = plt.get_cmap(vsini_cmap).copy()
    seq.set_bad(nan_color)
    # diverging colormap (residual row)
    div = plt.get_cmap(residual_cmap).copy()
    div.set_bad(nan_color)

    # symmetric clim for the residual row, from the residual field itself
    if clim is None:
        allres = np.concatenate(
            [residual_results[p]["residual"].ravel()
             for p in planes if p in residual_results]
        )
        clim = float(np.nanpercentile(np.abs(allres), 98))
    res_norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    vlim = ul if view_limit is None else max(view_limit, r_cut + 10.0)
    ticks = np.arange(-np.floor(vlim / tick_step) * tick_step,
                      np.floor(vlim / tick_step) * tick_step + 1,
                      tick_step).astype(int)
    interp = "none" if raw_cells else "bilinear"

    for i, (field, kind) in enumerate(row_fields):
        for j, plane in enumerate(planes):
            ax = axes[i, j]
            if plane not in residual_results:
                ax.set_visible(False)
                continue
            data = np.ma.masked_invalid(residual_results[plane][field])
            set_map_axes_style(ax, facecolor=nan_color)
            ax.set_facecolor(nan_color)

            if kind == "diverging":
                im = ax.imshow(data, origin="lower", extent=extent, cmap=div,
                               norm=res_norm, aspect="equal",
                               interpolation=interp, rasterized=True)
                clabel = r"$\Delta\langle v\sin i\rangle$ [km s$^{-1}$]"
            else:
                im = ax.imshow(data, origin="lower", extent=extent, cmap=seq,
                               vmin=vsini_vmin, vmax=vsini_vmax, aspect="equal",
                               interpolation=interp, rasterized=True)
                clabel = r"$\langle v\sin i\rangle$ [km s$^{-1}$]"

            # colorbar on the rightmost column only, labelled per row
            if j == ncol - 1:
                cbar = add_colorbar(fig, ax, im, label=clabel)
                cbar.ax.tick_params(labelsize=8)
                if kind == "diverging":
                    cbar.set_ticks(np.linspace(-clim, clim, 7))

            xl, yl = axis_labels(plane)
            ax.set_xlim(-vlim, vlim)
            ax.set_ylim(-vlim, vlim)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            # x-labels only on the bottom row; y-labels only on the left column
            if i == nrow - 1:
                ax.set_xlabel(xl)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{row_labels[i]}\n{yl}")
            else:
                ax.set_yticklabels([])

            add_cavity_circle(ax, radius=r_cut, edgecolor="black",
                              linewidth=1.0, linestyle="--")
            mark_sun(ax, color="black")
            if i == 0:
                ax.set_title(plane)

    return fig, axes
