import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .spatial import make_grid_edges, plane_coords


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


def compute_vmax_weights(
    df: pd.DataFrame,
    v_lim: dict,
    type_col: str = "SpecType",
    abs_mag_col: str = "VMAG",
    out_col: str = "w_vmax",
) -> pd.Series:
    """
    Compute 1/V_max weights to correct for Malmquist bias.

    For each star the maximum accessible volume is:
        V_max = (4pi/3) r_max³
    where r_max comes from the distance modulus at the survey limit.
    The weight is w = 1 / V_max (in units of pc^(-3)).
    """
    mlim = df[type_col].map(v_lim).to_numpy()
    M_abs = df[abs_mag_col].to_numpy()
    r_max = 10.0 ** ((mlim - M_abs + 5.0) / 5.0)
    V_max = (4.0 * np.pi / 3.0) * r_max**3
    w = 1.0 / V_max
    return pd.Series(w, index=df.index, name=out_col)


def vsini_error(v: np.ndarray) -> np.ndarray:
    """
    Piecewise instrumental uncertainty in vsini.

    * v < 20 km/s  -> sigma = 1.0 km/s
    * 20 <= v < 40  ->  sigma = 3.0 km/s
    * v >= 40       ->  sigma = 0.10 x v
    """
    err = np.where(v < 20, 1.0, np.where(v < 40, 3.0, 0.10 * v))
    return err


def monte_carlo_permutation(
    df: pd.DataFrame,
    iterations: int = 5000,
    r_cut: float = 80.0,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    r_col: str = "R_xy",
    seed: int = 42,
) -> dict:
    """
    Global randomization test for the kinematic cavity.

    Shuffles vsini values across *all* stars (keeping positions and weights
    fixed) and measures how often the randomised velocity contrast
    (shell - cavity) exceeds the observed one.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if r_col not in df.columns:
        df[r_col] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)

    mask_in = df[r_col].to_numpy() <= r_cut
    mask_out = ~mask_in

    w_in = df.loc[mask_in, weight_col].to_numpy()
    w_out = df.loc[mask_out, weight_col].to_numpy()
    vsini = df[vsini_col].to_numpy()

    m_in_obs = np.average(vsini[mask_in], weights=w_in)
    m_out_obs = np.average(vsini[mask_out], weights=w_out)
    obs_diff = float(m_out_obs - m_in_obs)

    null_diffs = np.empty(iterations)
    for i in range(iterations):
        v_sh = rng.permutation(vsini)
        null_diffs[i] = np.average(v_sh[mask_out], weights=w_out) - np.average(
            v_sh[mask_in], weights=w_in
        )

    p_value = float(np.mean(null_diffs >= obs_diff))
    z_score = float((obs_diff - null_diffs.mean()) / null_diffs.std())

    return dict(
        obs_diff=obs_diff, p_value=p_value, z_score=z_score, null_dist=null_diffs
    )


def monte_carlo_stratified_teff(
    df: pd.DataFrame,
    iterations: int = 10_000,
    r_cut: float = 80.0,
    n_bins: int = 10,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    r_col: str = "R_xy",
    teff_col: str = "Teff",
    seed: int = 42,
) -> dict:
    """
    Stratified permutation test that controls for the F/G mix gradient.

    vsini is shuffled only *within* each Teff quantile bin, so the
    temperature distribution (and thus the F/G composition) is identical
    in every randomised universe.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if r_col not in df.columns:
        df[r_col] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)

    mask_in = df[r_col].to_numpy() <= r_cut
    mask_out = ~mask_in

    w_in = df.loc[mask_in, weight_col].to_numpy()
    w_out = df.loc[mask_out, weight_col].to_numpy()

    teff_bins = pd.qcut(df[teff_col], q=n_bins, labels=False, duplicates="drop")
    df["_teff_bin"] = teff_bins.astype(int)
    n_eff = int(df["_teff_bin"].max()) + 1

    vsini_orig = df[vsini_col].to_numpy()
    bin_indices = [np.where(df["_teff_bin"].to_numpy() == b)[0] for b in range(n_eff)]

    m_in_obs = np.average(vsini_orig[mask_in], weights=w_in)
    m_out_obs = np.average(vsini_orig[mask_out], weights=w_out)
    obs_diff = float(m_out_obs - m_in_obs)

    null_diffs = np.empty(iterations)
    v_shuf = vsini_orig.copy()

    for i in range(iterations):
        v_shuf[:] = vsini_orig
        for idx in bin_indices:
            v_shuf[idx] = rng.permutation(vsini_orig[idx])
        null_diffs[i] = np.average(v_shuf[mask_out], weights=w_out) - np.average(
            v_shuf[mask_in], weights=w_in
        )

    p_value = float(np.mean(null_diffs >= obs_diff))
    z_score = float((obs_diff - null_diffs.mean()) / null_diffs.std())

    return dict(
        obs_diff=obs_diff, p_value=p_value, z_score=z_score, null_dist=null_diffs
    )


def scan_cavity_significance(
    df: pd.DataFrame,
    radii: np.ndarray,
    iterations: int = 500,
    n_bins: int = 10,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    r_col: str = "R_xy",
    teff_col: str = "Teff",
    min_stars: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the stratified permutation test over a range of cavity radii."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    if r_col not in df.columns:
        df[r_col] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)

    teff_bins = pd.qcut(df[teff_col], q=n_bins, labels=False, duplicates="drop")
    df["_teff_bin"] = teff_bins.astype(int)
    n_eff = int(df["_teff_bin"].max()) + 1
    vsini_orig = df[vsini_col].to_numpy()
    bin_indices = [np.where(df["_teff_bin"].to_numpy() == b)[0] for b in range(n_eff)]

    records = []
    v_shuf = vsini_orig.copy()

    for r in radii:
        mask_in = df[r_col].to_numpy() <= r
        mask_out = ~mask_in
        n_in, n_out = mask_in.sum(), mask_out.sum()
        if n_in < min_stars or n_out < min_stars:
            continue

        w_in = df.loc[mask_in, weight_col].to_numpy()
        w_out = df.loc[mask_out, weight_col].to_numpy()

        obs = np.average(vsini_orig[mask_out], weights=w_out) - np.average(
            vsini_orig[mask_in], weights=w_in
        )

        null = np.empty(iterations)
        for i in range(iterations):
            v_shuf[:] = vsini_orig
            for idx in bin_indices:
                v_shuf[idx] = rng.permutation(vsini_orig[idx])
            null[i] = np.average(v_shuf[mask_out], weights=w_out) - np.average(
                v_shuf[mask_in], weights=w_in
            )

        records.append(
            dict(
                Radius=r,
                N_in=int(n_in),
                N_out=int(n_out),
                Observed=obs,
                Expected=null.mean(),
                Uncertainty=null.std(),
                Z_score=(obs - null.mean()) / null.std(),
                p_value=np.mean(null >= obs),
            )
        )

    return pd.DataFrame(records)


def compute_residual_maps(
    df: pd.DataFrame,
    planes: tuple = ("XY", "XZ", "ZY"),
    ul: float = 150.0,
    tc: float = 20.0,
    n_mc: int = 10_000,
    n_bins: int = 10,
    min_count: int = 20,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    teff_col: str = "Teff",
    seed: int = 42,
) -> dict:
    """
    Compute cell-by-cell residual maps: ⟨vsini⟩_obs - ⟨vsini⟩_exp.

    The expected field is the mean of N_MC spatially-gridded maps produced
    by the Teff-stratified permutation null model: stellar positions and
    1/Vmax weights are held fixed; vsini is shuffled independently within
    each of N_bins quantile bins of Teff. This preserves the empirical
    Teff-rotation relation and survey demographics while erasing any
    intrinsic spatial-kinematic coupling.
    """

    rng = np.random.default_rng(seed)

    # Teff quantile bins (fixed across all planes)
    df = df.copy()
    teff_bins = pd.qcut(df[teff_col], q=n_bins, labels=False, duplicates="drop")
    df["_teff_bin"] = teff_bins.astype(int)
    n_eff = int(df["_teff_bin"].max()) + 1
    bin_indices = [np.where(df["_teff_bin"].to_numpy() == b)[0] for b in range(n_eff)]

    vsini = df[vsini_col].to_numpy()
    weights = df[weight_col].to_numpy()
    edges = make_grid_edges(ul, tc)
    Nc = len(edges) - 1

    results = {}
    v_shuf = vsini.copy()

    for plane in planes:
        x, y = plane_coords(df, plane)

        # Cell assignment (flat index)
        ix = np.digitize(x, edges) - 1
        iy = np.digitize(y, edges) - 1
        valid = (ix >= 0) & (ix < Nc) & (iy >= 0) & (iy < Nc)

        valid_idx = np.where(valid)[0]
        valid_ids = iy[valid_idx] * Nc + ix[valid_idx]
        valid_weights = weights[valid_idx]
        valid_vsini = vsini[valid_idx]

        # Fixed quantities: total weight and count per cell
        wsum = np.bincount(valid_ids, weights=valid_weights, minlength=Nc * Nc)
        cnt = np.bincount(valid_ids, minlength=Nc * Nc)
        cell_ok = (cnt >= min_count) & (wsum > 0)

        # Observed weighted-mean map
        obs_vw = np.bincount(
            valid_ids, weights=valid_vsini * valid_weights, minlength=Nc * Nc
        )
        obs_flat = np.full(Nc * Nc, np.nan)
        obs_flat[cell_ok] = obs_vw[cell_ok] / wsum[cell_ok]

        # Monte Carlo: accumulate expected field
        accum = np.zeros(Nc * Nc)
        n_valid_mc = np.zeros(Nc * Nc, dtype=int)

        for _ in range(n_mc):
            v_shuf[:] = vsini
            for idx in bin_indices:
                v_shuf[idx] = rng.permutation(vsini[idx])

            v_sv = v_shuf[valid_idx]
            mc_vw = np.bincount(
                valid_ids, weights=v_sv * valid_weights, minlength=Nc * Nc
            )
            mc_flat = np.full(Nc * Nc, np.nan)
            mc_flat[cell_ok] = mc_vw[cell_ok] / wsum[cell_ok]

            finite = np.isfinite(mc_flat)
            accum[finite] += mc_flat[finite]
            n_valid_mc[finite] += 1

        exp_flat = np.full(Nc * Nc, np.nan)
        has_data = n_valid_mc > 0
        exp_flat[has_data] = accum[has_data] / n_valid_mc[has_data]

        obs_map = obs_flat.reshape(Nc, Nc)
        exp_map = exp_flat.reshape(Nc, Nc)
        res_map = np.where(
            np.isfinite(obs_map) & np.isfinite(exp_map), obs_map - exp_map, np.nan
        )

        results[plane] = dict(
            observed=obs_map,
            expected=exp_map,
            residual=res_map,
            count=cnt.reshape(Nc, Nc),
        )

    return results


def mc_uncertainty_propagation(
    df: pd.DataFrame,
    n_mc: int = 10_000,
    r_cut: float = 80.0,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    r_col: str = "R_xy",
    seed: int = 42,
) -> np.ndarray:
    """
    Propagate heteroscedastic vsini uncertainties via Monte Carlo.

    For each of *n_mc* realizations the observed vsini values are perturbed
    by independent Gaussian noise drawn from the piecewise uncertainty model
    (vsini_error).  Non-physical negative draws are truncated to zero.  The
    weighted shell-cavity contrast Delta v is recomputed for every realization.

    The resulting distribution characterises how much the observed contrast
    would scatter purely due to measurement noise—without any demographic
    contribution.  Comparing this distribution to the Teff-stratified null
    (see monte_carlo_stratified_teff) provides the combined significance
    Z_robust = Delta v_phys / sqrt(sigma²_obs + sigma²_null).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if r_col not in df.columns:
        df[r_col] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)

    mask_in = df[r_col].to_numpy() <= r_cut
    mask_out = ~mask_in

    vsini = df[vsini_col].to_numpy()
    errors = vsini_error(vsini)  # heteroscedastic sigma per star
    w_in = df.loc[mask_in, weight_col].to_numpy()
    w_out = df.loc[mask_out, weight_col].to_numpy()

    mc_diffs = np.empty(n_mc)
    for i in range(n_mc):
        v_pert = vsini + rng.normal(0.0, errors)
        v_pert = np.maximum(v_pert, 0.0)  # physical: vsini >= 0
        mc_diffs[i] = np.average(v_pert[mask_out], weights=w_out) - np.average(
            v_pert[mask_in], weights=w_in
        )

    return mc_diffs


def radial_density_profile(
    df: pd.DataFrame,
    radius: str = "R3D",
    bin_edges: np.ndarray | None = None,
    geometry: str = "sphere",
) -> dict:
    """Stellar number density n(R) vs. galactocentric (heliocentric) radius.

    Density is counts divided by shell volume (``geometry='sphere'``, for 3-D
    radius) or annulus area (``geometry='annulus'``, for projected radius). Used
    to show the two-population separation that motivates the contrast boundary:
    the density peaks at small R and falls smoothly outward, so the boundary
    lies in the low-density region between the nearby and distant populations.
    """
    df = df.copy()
    if radius == "R3D":
        r = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2).to_numpy()
    elif radius == "R_xy":
        r = (
            df["R_xy"].to_numpy()
            if "R_xy" in df.columns
            else np.sqrt(df["X"] ** 2 + df["Y"] ** 2).to_numpy()
        )
    else:
        raise ValueError("radius must be 'R3D' or 'R_xy'")

    if bin_edges is None:
        bin_edges = np.arange(0.0, 121.0, 20.0)
    bin_edges = np.asarray(bin_edges, dtype=float)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    count, _ = np.histogram(r, bins=bin_edges)

    if geometry == "sphere":
        shell = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    elif geometry == "annulus":
        shell = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
    else:
        raise ValueError("geometry must be 'sphere' or 'annulus'")

    density = np.divide(count, shell, out=np.full_like(shell, np.nan), where=shell > 0)
    return dict(
        radius=radius,
        bin_edges=bin_edges,
        centers=centers,
        count=count,
        density=density,
        geometry=geometry,
    )


def radial_vsini_profile(
    df: pd.DataFrame,
    radius: str = "R3D",
    bin_edges: np.ndarray | None = None,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    use_weights: bool = True,
    n_bootstrap: int = 1000,
    seed: int | None = 42,
) -> dict:
    """Weighted v sin i as a function of galactocentric radius (per star).

    This is independent of the 2-D display grid: stars are binned by their own
    radius, so the profile is not tied to the 20 pc cell size. It provides the
    quantitative anchor for the transition-scale claim in the text.

    ``radius`` selects ``"R3D"`` (sqrt(X^2+Y^2+Z^2)) or ``"R_xy"`` (projected).
    Returns bin centres, weighted mean and weighted quartiles per bin, the
    bootstrap standard error of the mean, and per-bin counts.
    """
    df = df.copy()
    if radius == "R3D":
        r = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2).to_numpy()
    elif radius == "R_xy":
        r = (
            df["R_xy"].to_numpy()
            if "R_xy" in df.columns
            else np.sqrt(df["X"] ** 2 + df["Y"] ** 2).to_numpy()
        )
    else:
        raise ValueError("radius must be 'R3D' or 'R_xy'")

    v = df[vsini_col].to_numpy()
    w = df[weight_col].to_numpy() if use_weights else np.ones_like(v, dtype=float)

    if bin_edges is None:
        bin_edges = np.arange(0.0, 121.0, 20.0)
    bin_edges = np.asarray(bin_edges, dtype=float)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nb = len(centers)

    mean = np.full(nb, np.nan)
    q25 = np.full(nb, np.nan)
    q50 = np.full(nb, np.nan)
    q75 = np.full(nb, np.nan)
    se = np.full(nb, np.nan)
    count = np.zeros(nb, dtype=int)

    rng = np.random.default_rng(seed)
    for i in range(nb):
        m = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        n = int(m.sum())
        count[i] = n
        if n == 0:
            continue
        vi, wi = v[m], w[m]
        mean[i] = weighted_mean(vi, wi)
        q25[i] = weighted_percentile(vi, wi, 25)
        q50[i] = weighted_percentile(vi, wi, 50)
        q75[i] = weighted_percentile(vi, wi, 75)
        if n > 1 and n_bootstrap > 0:
            # bootstrap the weighted mean; resample probabilities ~ weights so
            # this matches the 1/Vmax-weighted resampling used for the maps.
            p = wi / wi.sum() if use_weights else None
            boots = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True, p=p)
                boots[b] = weighted_mean(vi[idx], wi[idx])
            se[i] = float(np.std(boots, ddof=1))

    return dict(
        radius=radius,
        bin_edges=bin_edges,
        centers=centers,
        mean=mean,
        q25=q25,
        q50=q50,
        q75=q75,
        se=se,
        count=count,
    )


def weighted_median_contrast(
    df: pd.DataFrame,
    r_cut: float = 80.0,
    vsini_col: str = "vsini",
    weight_col: str = "w_vmax",
    r_col: str = "R_xy",
) -> dict:
    """
    Compute the shell-cavity contrast using the weighted median.

    The weighted mean can be sensitive to a handful of high-weight objects.
    The weighted median (50th weighted percentile) provides a robust
    alternative that is insensitive to extreme values.
    """
    df = df.copy()
    if r_col not in df.columns:
        df[r_col] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)

    mask_in = df[r_col].to_numpy() <= r_cut
    mask_out = ~mask_in

    v_in = df.loc[mask_in, vsini_col].to_numpy()
    v_out = df.loc[mask_out, vsini_col].to_numpy()
    w_in = df.loc[mask_in, weight_col].to_numpy()
    w_out = df.loc[mask_out, weight_col].to_numpy()

    med_in = weighted_percentile(v_in, w_in, 50)
    med_out = weighted_percentile(v_out, w_out, 50)
    mn_in = weighted_mean(v_in, w_in)
    mn_out = weighted_mean(v_out, w_out)

    return dict(
        median_in=med_in,
        median_out=med_out,
        delta_v_median=med_out - med_in,
        mean_in=mn_in,
        mean_out=mn_out,
        delta_v_mean=mn_out - mn_in,
    )


def morans_I(
    df: pd.DataFrame,
    value_col: str = "vsini",
    x_col: str = "X",
    y_col: str = "Y",
    k_neighbors: int = 20,
    permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Global Moran's I with k-nearest-neighbour spatial weights.
    """
    rng = np.random.default_rng(seed)
    sub = df[[value_col, x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    vals = sub[value_col].to_numpy(dtype=float)
    coords = sub[[x_col, y_col]].to_numpy(dtype=float)
    N = len(vals)

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k_neighbors + 1)
    neighbors = indices[:, 1:]

    W = np.zeros((N, N))
    rows = np.repeat(np.arange(N), k_neighbors)
    cols = neighbors.flatten()
    W[rows, cols] = 1.0
    W_sum = W.sum()

    z = vals - vals.mean()
    I_obs = float((N / W_sum) * (z @ W @ z) / (z @ z))

    E_I = -1.0 / (N - 1)

    null = np.empty(permutations)
    for i in range(permutations):
        zp = rng.permutation(z)
        null[i] = float((N / W_sum) * (zp @ W @ zp) / (zp @ zp))

    p_value = float(np.mean(np.abs(null) >= np.abs(I_obs)))
    Var_I = float(null.var())
    Z_I = float((I_obs - E_I) / np.sqrt(Var_I)) if Var_I > 0 else np.nan

    return dict(I=I_obs, E_I=E_I, Var_I=Var_I, Z_I=Z_I, p_value=p_value, null_dist=null)

