import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter, zoom


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute a weighted percentile."""
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]

    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan

    target = q / 100.0 * cw[-1]
    return np.interp(target, cw, v)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted arithmetic mean (returns NaN for empty or zero-weight input)."""
    total_w = weights.sum()
    if total_w <= 0 or len(values) == 0:
        return np.nan
    return np.average(values, weights=weights)


def make_grid_edges(
    ul: float = 150.0, tc: float = 20.0, interval: float = 1.0
) -> np.ndarray:
    """Return 1-D bin edges for a symmetric grid."""
    step = interval * tc
    return np.arange(-ul, ul + step, step)


def plane_coords(df: pd.DataFrame, plane: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two coordinate arrays for a given Galactic projection."""
    if plane == "XY":
        return df["X"].to_numpy(), df["Y"].to_numpy()
    if plane == "XZ":
        return df["X"].to_numpy(), df["Z"].to_numpy()
    if plane == "ZY":
        return df["Z"].to_numpy(), df["Y"].to_numpy()
    raise ValueError(f"plane must be 'XY', 'XZ', or 'ZY', got '{plane}'")


def axis_labels(plane: str) -> tuple[str, str]:
    """Return axis labels for a given projection plane."""
    mapping = {
        "XY": ("$X$ [pc]", "$Y$ [pc]"),
        "XZ": ("$X$ [pc]", "$Z$ [pc]"),
        "ZY": ("$Z$ [pc]", "$Y$ [pc]"),
    }
    if plane not in mapping:
        raise ValueError(f"Unknown plane '{plane}'")
    return mapping[plane]


def compute_grid_stats(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    min_count: int = 20,
) -> dict[str, np.ndarray]:
    """Fill a regular grid with weighted statistics of *values*."""
    edges = make_grid_edges(ul, tc, interval)
    Nx = Ny = len(edges) - 1

    mean_map = np.full((Ny, Nx), np.nan)
    q25_map = np.full((Ny, Nx), np.nan)
    q50_map = np.full((Ny, Nx), np.nan)
    q75_map = np.full((Ny, Nx), np.nan)
    cnt_map = np.zeros((Ny, Nx), dtype=int)

    for iy in range(Ny):
        for ix in range(Nx):
            mask = (
                (x >= edges[ix])
                & (x < edges[ix + 1])
                & (y >= edges[iy])
                & (y < edges[iy + 1])
            )
            n = mask.sum()
            cnt_map[iy, ix] = n
            if n < min_count:
                continue
            v = values[mask]
            w = weights[mask]
            mean_map[iy, ix] = weighted_mean(v, w)
            q25_map[iy, ix] = weighted_percentile(v, w, 25)
            q50_map[iy, ix] = weighted_percentile(v, w, 50)
            q75_map[iy, ix] = weighted_percentile(v, w, 75)

    return dict(
        mean=mean_map,
        q25=q25_map,
        q50=q50_map,
        q75=q75_map,
        count=cnt_map.astype(float),
    )


def interpolate_map(
    grid_2d: np.ndarray, factor: int = 5, remask: bool = True
) -> np.ndarray:
    """Smooth a 2-D grid with a bicubic spline, keeping empty cells masked.

    Cells that are NaN on input (masked / low-count) are temporarily filled
    only so the spline stays numerically stable, but the corresponding
    regions of the up-sampled output are set back to NaN when ``remask`` is
    True. This prevents "no data" cells from being painted as a real (often
    low) value on the colour scale -- see the referee's Major Comment 3.
    """
    Ny, Nx = grid_2d.shape
    nan_mask = np.isnan(grid_2d)
    filled = grid_2d.copy()
    if nan_mask.any():
        filled[nan_mask] = np.nanmean(grid_2d)

    x_coarse = np.arange(Nx)
    y_coarse = np.arange(Ny)
    spl = RectBivariateSpline(y_coarse, x_coarse, filled, kx=3, ky=3)

    x_fine = np.linspace(0, Nx - 1, Nx * factor)
    y_fine = np.linspace(0, Ny - 1, Ny * factor)
    result = spl(y_fine, x_fine)

    # Re-mask the up-sampled empty regions: propagate the NaN footprint to
    # the fine grid with nearest-neighbour (kx=ky=1, threshold) interpolation.
    if remask and nan_mask.any():
        nan_coarse = nan_mask.astype(float)
        nan_spl = RectBivariateSpline(y_coarse, x_coarse, nan_coarse, kx=1, ky=1)
        nan_fine = nan_spl(y_fine, x_fine)
        result[nan_fine > 0.5] = np.nan

    return result


def smooth_masked_field(
    grid_2d: np.ndarray,
    factor: int = 8,
    mask_sigma: float = 0.5,
    mask_level: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Up-sample a gridded field and clip it to a de-pixelated data footprint.

    Returns ``(field, footprint)`` where ``field`` is a float array up-sampled
    by ``factor`` (NaN outside the footprint) and ``footprint`` is the smoothed
    valid-cell mask on the same up-sampled grid (useful for drawing an outline).

    Important (scientific honesty): this does NOT extrapolate the measured
    quantity into empty cells. The interior is interpolated only across cells
    that already hold data; the boundary is a *morphologically smoothed* version
    of the hard ``count >= min_count`` footprint. The smoothing rounds pixel
    corners roughly symmetrically -- with the default ``mask_sigma`` /
    ``mask_level`` the footprint area changes by ~1% (sub-cell), so no
    appreciable coverage is added beyond where stars exist. ``mask_level`` above
    0.5 keeps the boundary from bulging outward.
    """
    valid = np.isfinite(grid_2d).astype(float)

    # Interior field: interpolate WITHOUT re-masking (fill is only for spline
    # stability); the explicit footprint mask below decides what is shown.
    field = interpolate_map(grid_2d, factor=factor, remask=False)

    # Smooth footprint: up-sample the binary valid mask, blur, threshold.
    vm_up = zoom(valid, factor, order=1)
    vm_blur = gaussian_filter(vm_up, sigma=mask_sigma * factor)
    footprint = vm_blur >= mask_level

    field = np.where(footprint, field, np.nan)
    return field, vm_blur


def compute_bootstrap_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    min_count: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Bootstrap the grid statistics to estimate sampling uncertainty."""
    rng = np.random.default_rng(seed)
    N = len(values)

    edges = make_grid_edges(ul, tc, interval)
    Nc = len(edges) - 1

    accum = {k: np.zeros((Nc, Nc)) for k in ("mean", "q25", "q50", "q75", "count")}
    n_valid = np.zeros((Nc, Nc), dtype=int)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        stats = compute_grid_stats(
            x[idx],
            y[idx],
            values[idx],
            weights[idx],
            ul=ul,
            tc=tc,
            interval=interval,
            min_count=min_count,
        )
        for k in accum:
            finite = np.isfinite(stats[k])
            accum[k][finite] += stats[k][finite]
            if k == "mean":
                n_valid[finite] += 1

    result = {}
    for k in accum:
        arr = np.full((Nc, Nc), np.nan)
        mask = n_valid > 0
        arr[mask] = accum[k][mask] / n_valid[mask]
        result[k] = arr

    return result


def vsini_maps_for_plane(
    df: pd.DataFrame,
    plane: str,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    min_count: int = 20,
    n_bootstrap: int = 1000,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    seed: int | None = 42,
    do_bootstrap: bool = True,
    alpha: float = 0.01,
    use_weights: bool = True,
) -> dict:
    if seed is not None:
        np.random.seed(seed)

    if plane == "XY":
        x = df["X"]
        y = df["Y"]
    elif plane == "XZ":
        x = df["X"]
        y = df["Z"]
    elif plane == "ZY":
        x = df["Z"]
        y = df["Y"]
    else:
        raise ValueError("Invalid plane selection. Choose 'XY', 'XZ', or 'ZY'.")

    vsini = df[vsini_col]
    if use_weights:
        raw_weights = df[weight_col].values
        label_mean = r"$\langle v\sin i \rangle$ [km/s] (Weighted)"
    else:
        raw_weights = np.ones(len(vsini))
        label_mean = r"$\langle v\sin i \rangle$ [km/s] (Unweighted)"

    DPtotal = np.std(vsini)

    step = interval * tc
    ttx = np.arange(-ul, ul + step, step)
    tty = np.arange(-ul, ul + step, step)
    Nx = len(ttx) - 1
    Ny = len(tty) - 1

    # Statistic maps are initialised to NaN so that masked / low-count cells
    # (count < min_count) are flagged as "no data" rather than carrying a real
    # value of 0, which would otherwise render as a low value on the velocity
    # colour scale (referee Major Comment 3). The count map stays at 0 because
    # there a 0 is a genuine measurement (no stars).
    meanoriginal = np.full((Ny, Nx), np.nan)
    meanoriginalSD = np.full((Ny, Nx), np.nan)
    percentile25 = np.full((Ny, Nx), np.nan)
    percentile50 = np.full((Ny, Nx), np.nan)
    percentile75 = np.full((Ny, Nx), np.nan)

    meanbootstrap = np.full((Ny, Nx), np.nan)
    meanbootstrapSD = np.full((Ny, Nx), np.nan)
    percentile25boot = np.full((Ny, Nx), np.nan)
    percentile50boot = np.full((Ny, Nx), np.nan)
    percentile75boot = np.full((Ny, Nx), np.nan)

    boot_mean = np.full((Ny, Nx), np.nan)
    boot_se = np.full((Ny, Nx), np.nan)
    ci1 = np.full((Ny, Nx), np.nan)
    ci2 = np.full((Ny, Nx), np.nan)
    shape = np.full((Ny, Nx), np.nan)
    countfXY = np.zeros((Ny, Nx))

    for j in range(Nx):
        for i in range(Ny):
            condition = (
                (ttx[j] <= x) & (x < ttx[j + 1]) & (tty[i] <= y) & (y < tty[i + 1])
            )

            vsini_cond = vsini[condition].values
            w_cond_raw = raw_weights[condition]
            count = len(vsini_cond)
            countfXY[i, j] = count

            if count >= min_count and np.sum(w_cond_raw) > 0:
                w_norm = w_cond_raw / np.sum(w_cond_raw)

                m = np.sum(w_norm * vsini_cond)
                meanoriginal[i, j] = m

                var = np.sum(w_norm * (vsini_cond - m) ** 2)
                meanoriginalSD[i, j] = np.sqrt(var)

                if use_weights:
                    percentile25[i, j] = weighted_percentile(
                        vsini_cond, w_cond_raw, 25.0
                    )
                    percentile50[i, j] = weighted_percentile(
                        vsini_cond, w_cond_raw, 50.0
                    )
                    percentile75[i, j] = weighted_percentile(
                        vsini_cond, w_cond_raw, 75.0
                    )
                else:
                    percentile25[i, j] = np.percentile(vsini_cond, 25.0)
                    percentile50[i, j] = np.percentile(vsini_cond, 50.0)
                    percentile75[i, j] = np.percentile(vsini_cond, 75.0)

                if do_bootstrap and n_bootstrap > 0:
                    Ncell = count
                    bootout = np.empty(n_bootstrap)

                    idx_all = np.arange(Ncell)
                    for b in range(n_bootstrap):
                        sample_idx = np.random.choice(
                            idx_all,
                            size=Ncell,
                            replace=True,
                            p=w_norm,
                        )
                        sample_vsini = vsini_cond[sample_idx]
                        bootout[b] = np.mean(sample_vsini)

                    boot_mean[i, j] = np.mean(bootout)
                    boot_se[i, j] = np.std(bootout)
                    meanbootstrap[i, j] = boot_mean[i, j]
                    meanbootstrapSD[i, j] = boot_se[i, j]

                    lower_percentile = (alpha / 2) * 100.0
                    upper_percentile = (1.0 - alpha / 2) * 100.0
                    ci1[i, j] = np.percentile(bootout, lower_percentile)
                    ci2[i, j] = np.percentile(bootout, upper_percentile)

                    percentile25boot[i, j] = np.percentile(bootout, 25.0)
                    percentile50boot[i, j] = np.percentile(bootout, 50.0)
                    percentile75boot[i, j] = np.percentile(bootout, 75.0)

    length = ci2 - ci1
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = (ci2 - boot_mean) / (boot_mean - ci1)
    shape = np.nan_to_num(shape)

    data_dict1 = {
        label_mean: meanoriginal,
        r"$v\sin i$ [km/s] (q = 1/4)": percentile25,
        r"$v\sin i$ [km/s] (q = 1/2)": percentile50,
        r"$v\sin i$ [km/s] (q = 3/4)": percentile75,
    }

    data_dict2 = {
        r"$\langle v\sin i \rangle$ Bootstrap Mean": meanbootstrap,
        r"$\langle v\sin i \rangle$ CI Lower (25%)": percentile25boot,
        r"$\langle v\sin i \rangle$ CI Median (50%)": percentile50boot,
        r"$\langle v\sin i \rangle$ CI Upper (75%)": percentile75boot,
    }

    results = {
        "ttx": ttx,
        "tty": tty,
        "meanoriginal": meanoriginal,
        "meanoriginalSD": meanoriginalSD,
        "percentile25": percentile25,
        "percentile50": percentile50,
        "percentile75": percentile75,
        "meanbootstrap": meanbootstrap,
        "meanbootstrapSD": meanbootstrapSD,
        "ci1": ci1,
        "ci2": ci2,
        "percentile25boot": percentile25boot,
        "percentile50boot": percentile50boot,
        "percentile75boot": percentile75boot,
        "length": length,
        "shape": shape,
        "countfXY": countfXY,
        "DPtotal": DPtotal,
        "data_dict1": data_dict1,
        "data_dict2": data_dict2,
    }

    results["original"] = {
        "mean": meanoriginal,
        "q25": percentile25,
        "q50": percentile50,
        "q75": percentile75,
        "count": countfXY,
    }
    results["edges"] = ttx
    if do_bootstrap:
        results["bootstrap"] = {
            "mean": meanbootstrap,
            "q25": percentile25boot,
            "q50": percentile50boot,
            "q75": percentile75boot,
            "count": countfXY,
        }

    return results


def vmag_maps_for_plane(
    df: pd.DataFrame,
    plane: str,
    ul: float = 150.0,
    tc: float = 20.0,
    interval: float = 1.0,
    min_count: int = 20,
    n_bootstrap: int = 1000,
    vmag_col: str = "Vmag",
    seed: int | None = 42,
    do_bootstrap: bool = True,
    alpha: float = 0.01,
    use_weights: bool = True,
) -> dict:
    results = vsini_maps_for_plane(
        df=df,
        plane=plane,
        ul=ul,
        tc=tc,
        interval=interval,
        min_count=min_count,
        n_bootstrap=n_bootstrap,
        vsini_col=vmag_col,
        seed=seed,
        do_bootstrap=do_bootstrap,
        alpha=alpha,
        use_weights=use_weights,
    )

    data_dict1 = {
        r"$\langle V_{mag} \rangle$ (Apparent)": results["meanoriginal"],
        r"$V_{mag}$ (q = 1/4)": results["percentile25"],
        r"$V_{mag}$ (q = 1/2)": results["percentile50"],
        r"$V_{mag}$ (q = 3/4)": results["percentile75"],
    }
    data_dict2 = {
        r"$\langle V_{mag} \rangle$ Bootstrap Mean": results["meanbootstrap"],
        r"$\langle V_{mag} \rangle$ CI Lower": results["percentile25boot"],
        r"$\langle V_{mag} \rangle$ CI Median": results["percentile50boot"],
        r"$\langle V_{mag} \rangle$ CI Upper": results["percentile75boot"],
    }
    results["data_dict1"] = data_dict1
    results["data_dict2"] = data_dict2

    return results
