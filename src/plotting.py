import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from .style import add_cavity_circle, mark_sun, add_colorbar, set_map_axes_style
from .spatial import interpolate_map, axis_labels, plane_coords, make_grid_edges
from .statistics import vsini_error

COLORS = {
    "F": "#2166ac",
    "G": "#d68d4d",
    "ALL": "#404040",
}


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
    cmap: str = "hot",
    interpolate: bool = True,
) -> tuple:
    """
    Three-panel star-count map: XY, XZ, ZY.
    """

    planes = ["XY", "XZ", "ZY"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    edges = make_grid_edges(ul, tc, interval)
    Nc = len(edges) - 1
    extent = [-ul, ul, -ul, ul]

    for ax, plane in zip(axes, planes):
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

        smooth = interpolate_map(count_map)

        if interpolate:
            im = ax.imshow(
                smooth,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=0,
                aspect="equal",
                interpolation="bilinear",
                rasterized=True,
            )
        else:
            im = ax.imshow(
                count_map,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=0,
                aspect="equal",
                interpolation="None",
                rasterized=True,
            )

        cbar = add_colorbar(fig, ax, im, label="Number of stars", pad=0.01)

        # Density contours at the 25th, 50th, 75th, 90th percentiles
        if interpolate:
            count_map = smooth  # use the interpolated map for contours

        pos_vals = count_map[count_map > 0]
        if pos_vals.size:
            levels = np.percentile(pos_vals, [25, 50, 75, 90])
            xs = np.linspace(-ul, ul, count_map.shape[1])
            ys = np.linspace(-ul, ul, count_map.shape[0])
            ax.contour(
                xs,
                ys,
                count_map,
                levels=levels,
                colors="white",
                linewidths=0.5,
                alpha=0.5,
            )

        xl, yl = axis_labels(plane)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-ul, ul)
        ax.set_ylim(-ul, ul)

        # mark_sun(ax)

    return fig, axes


def plot_vsini_bootstrap_panel(
    original: dict,
    bootstrap: dict,
    plane: str,
    ul: float = 150.0,
    method: str = "spline36",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Spectral_r",
    figsize: tuple = (20, 8),
) -> tuple:
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

    cmap_obj = plt.get_cmap(cmap)

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
        spline = RectBivariateSpline(x_grid, y_grid, data_array.T, kx=3, ky=3)
        ima = spline(xnew, ynew)

        cax = ax.imshow(
            ima.T,
            cmap=cmap_obj,
            aspect="auto",
            origin="lower",
            interpolation=method,
            vmin=vmin,
            vmax=vmax,
        )

        cbar = add_colorbar(fig, ax, cax, label=all_titles[idx], pad=0.01, fontsize=16)
        # cbar = fig.colorbar(cax, ax=ax, pad=0)
        # cbar.set_label(all_titles[idx])
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)

        xticklabels = np.arange(-ul, ul + 1, 50)
        xticks = np.linspace(0, ima.shape[1] - 1, len(xticklabels))
        yticklabels = np.arange(-ul, ul + 1, 50)
        yticks = np.linspace(0, ima.shape[0] - 1, len(yticklabels))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=16)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=16)

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

        sm = interpolate_map(zscore_map)
        set_map_axes_style(ax, facecolor="#9faab5")
        im = ax.imshow(
            sm,
            origin="lower",
            extent=extent,
            cmap=cmap,
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
) -> tuple:
    planes = ["XY", "XZ", "ZY"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    extent = [-ul, ul, -ul, ul]

    if clim is None:
        all_res = np.concatenate(
            [
                residual_results[p]["residual"].ravel()
                for p in planes
                if p in residual_results
            ]
        )
        clim = float(np.nanpercentile(np.abs(all_res), 98))
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    for ax, plane in zip(axes, planes):
        if plane not in residual_results:
            ax.set_visible(False)
            continue

        res = residual_results[plane]["residual"]
        sm = interpolate_map(res)

        set_map_axes_style(ax, facecolor="#9faab5")
        im = ax.imshow(
            sm,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="bilinear",
            rasterized=True,
        )

        ax.contour(
            sm,
            levels=[-2, -1, 1, 2],
            extent=extent,
            colors=["0.2", "0.5", "0.5", "0.2"],
            linewidths=[0.8, 0.5, 0.5, 0.8],
            linestyles="--",
            origin="lower",
        )

        cbar = add_colorbar(
            fig, ax, im, label=r"$\Delta\langle v\sin i\rangle$ [km s$^{-1}$]"
        )
        cbar.set_ticks(np.linspace(-clim, clim, 7))
        cbar.ax.tick_params(labelsize=8)

        xl, yl = axis_labels(plane)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-ul, ul)
        ax.set_ylim(-ul, ul)

        add_cavity_circle(
            ax, radius=r_cut, edgecolor="black", linewidth=1.2, linestyle="--"
        )
        mark_sun(ax, color="black")
        ax.set_title(f"Residual ({plane})")

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
    df, cmap: str = "viridis", figsize: tuple = (10, 4.5), ax=None, label=""
) -> tuple:
    v = df["vsini"].values
    d = df["Dist"].values
    e = vsini_error(v)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc = ax.scatter(
        d, v, c=e, cmap=cmap, s=15, edgecolor="k", rasterized=True, label=label
    )

    ax.errorbar(
        d, v, yerr=e, fmt="none", ecolor="k", elinewidth=0.4, alpha=0.5, zorder=0
    )

    # colorbar per subplot
    cbar = add_colorbar(fig, ax, sc, label=r"$\sigma(v\sin i)$ [km s$^{-1}$]", pad=0.01)

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
) -> tuple:
    planes = ["XY", "XZ", "ZY"]
    row_data = [
        (df_all, "All stars (F+G)", "Grays", "#1a1a1a"),
        (df_F, "F-type", "Blues", "#f7fbff"),
        (df_G, "G-type", "Oranges", "#fff5eb"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    edges = make_grid_edges(ul, tc)
    Nc = len(edges) - 1
    extent = [-ul, ul, -ul, ul]

    for row, (df, row_label, cmap_name, nan_color) in enumerate(row_data):
        sigma_v = vsini_error(df[vsini_col].to_numpy())
        weights = df[weight_col].to_numpy()

        cmap_obj = plt.get_cmap(cmap_name).copy()
        cmap_obj.set_bad(nan_color)

        # Compute all 3 plane maps first to derive vmin/vmax from actual cell values
        err_maps = []
        for plane in planes:
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
            err_maps.append(err_map)

        cell_vals = np.concatenate([m[np.isfinite(m)] for m in err_maps])
        vmin_r = float(cell_vals.min()) if cell_vals.size else 0.0
        vmax_r = float(cell_vals.max()) if cell_vals.size else 1.0

        for col, (plane, err_map) in enumerate(zip(planes, err_maps)):
            ax = axes[row, col]

            sm = interpolate_map(err_map, factor=2)
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
                interpolation="spline16",
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
            ax.set_xlim(-ul, ul)
            ax.set_ylim(-ul, ul)
            ax.tick_params()

            add_cavity_circle(ax, radius=r_cut, edgecolor="0.3")
            mark_sun(ax, color="0.3", markeredgecolor="none", markersize=10)

            if col == 0:
                ax.set_ylabel(f"{row_label}\n{yl}")
            else:
                ax.set_ylabel(f"{yl}")

    return fig, axes
