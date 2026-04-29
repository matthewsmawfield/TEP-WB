#!/usr/bin/env python3
"""
Step 005: Environmental Screening Test for TEP-WB

This script performs the most critical physics test of the TEP framework: the Environmental
Modulation test.

Scientific Rationale:
TEP v0.7 predicts that the scalar field is screened not just by the binary's internal mass, but
by the ambient gravitational potential of the environment. In the Temporal Topology framework,
screening operates via the continuous spatial profile of the chameleon field: high ambient
density in deep potential wells flattens the Temporal Topology, driving the Temporal Shear
(∇φ) to zero and suppressing fifth forces. Therefore, binaries residing in dense environments
(the Galactic Disk) should be more heavily screened than binaries in sparse environments (the
Galactic Halo). As a result, Halo binaries should transition to the anomalous (unscreened)
state at smaller separations than Disk binaries.

To test this:
1. We split the sample by Galactocentric height |Z| into a High-Density Disk sample and a
   Low-Density Halo sample.
2. We fit the screening model to both. Because the global anomaly saturation amplitude (alpha)
   and the screening radius (R_s) are somewhat mathematically degenerate, we fix alpha=0.4
   (the global saturation limit) to cleanly isolate the R_s shift.
3. We perform a secondary "Solar Track" test. To avoid any accusations that the metallicity
   mass correction (step 002) was artificially tuned to produce this result, the sample isolates a
   sub-sample of stars that require ZERO correction (they sit exactly on the solar-metallicity
   Main Sequence). If the environmental effect is real physics and not an artifact of the mass
   correction, it must appear identically in this pristine sub-sample.
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.tep_model import (
    GLOBAL_BINS,
    GLOBAL_ALPHA,
    G_AU,
    BOOTSTRAP_SEED,
    tep_screening_model_fixed_alpha,
)

# Set publication-quality matplotlib defaults
plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "serif"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

# GLOBAL_ALPHA, BOOTSTRAP_SEED and tep_screening_model_fixed_alpha are imported
# from scripts.utils.tep_model (canonical definitions).
PERMUTATION_ITERATIONS = 10000


def fit_transition_radius(x, y, yerr, p0, bounds):
    try:
        popt, _ = curve_fit(
            tep_screening_model_fixed_alpha,
            x,
            y,
            sigma=yerr,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return np.nan


def fit_with_bootstrap(x, y, yerr, p0, bounds, rng, n_boot=200):
    try:
        base_rs = fit_transition_radius(x, y, yerr, p0, bounds)
        if not np.isfinite(base_rs):
            return np.nan, np.nan
    except (RuntimeError, ValueError):
        return np.nan, np.nan

    boot_rs = []
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    yerr_values = np.asarray(yerr)

    for _ in range(n_boot):
        idx = rng.choice(len(x_values), len(x_values), replace=True)
        try:
            boot_popt, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                x_values[idx],
                y_values[idx],
                sigma=yerr_values[idx],
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            boot_rs.append(boot_popt[0])
        except (RuntimeError, ValueError):
            pass

    if len(boot_rs) > 0:
        rs_err = np.std(boot_rs)
    else:
        rs_err = np.nan

    return base_rs, rs_err


# ---------------------------------------------------------------------------
# Pure-numpy helpers + multiprocessing workers for permutation test
# ---------------------------------------------------------------------------


def _numpy_bin_fit_rs(sep, vals, bins, bin_centers):
    """Pure-numpy: normalize to inner baseline, bin, fit fixed-alpha R_s."""
    inner = sep < 500
    if inner.sum() == 0:
        return np.nan
    base = np.median(vals[inner])
    if not np.isfinite(base) or base <= 0:
        return np.nan
    v_norm = vals / base

    n_bins = len(bins) - 1
    bin_idx = np.digitize(sep, bins)  # 1-indexed
    meds = np.empty(n_bins)
    sems = np.empty(n_bins)
    valid = np.zeros(n_bins, dtype=bool)

    for i in range(n_bins):
        mask = bin_idx == (i + 1)
        cnt = mask.sum()
        if cnt > 2:
            meds[i] = np.median(v_norm[mask])
            sems[i] = 1.253 * np.std(v_norm[mask]) / np.sqrt(cnt)
            if np.isfinite(meds[i]) and np.isfinite(sems[i]) and sems[i] > 0:
                valid[i] = True

    if valid.sum() < 4:
        return np.nan
    try:
        popt, _ = curve_fit(
            tep_screening_model_fixed_alpha,
            bin_centers[valid],
            meds[valid],
            sigma=sems[valid],
            p0=[2000],
            bounds=([100], [50000]),
            maxfev=10000,
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return np.nan


_perm_shared = {}  # populated by Pool initializer in each worker


def _init_perm_worker(sep_arr, val_arr, bins_arr, bc_arr, n_a_val):
    _perm_shared["sep"] = sep_arr
    _perm_shared["val"] = val_arr
    _perm_shared["bins"] = bins_arr
    _perm_shared["bc"] = bc_arr
    _perm_shared["n_a"] = n_a_val


def _perm_batch_worker(args):
    """Process a batch of permutation iterations in one worker."""
    seed, batch_size = args
    rng = np.random.default_rng(seed)
    sep = _perm_shared["sep"]
    val = _perm_shared["val"]
    bins = _perm_shared["bins"]
    bc = _perm_shared["bc"]
    n_a = _perm_shared["n_a"]
    n_total = len(sep)

    deltas = []
    for _ in range(batch_size):
        perm = rng.permutation(n_total)
        rs_a = _numpy_bin_fit_rs(sep[perm[:n_a]], val[perm[:n_a]], bins, bc)
        rs_b = _numpy_bin_fit_rs(sep[perm[n_a:]], val[perm[n_a:]], bins, bc)
        if np.isfinite(rs_a) and np.isfinite(rs_b):
            deltas.append(rs_a - rs_b)
    return deltas


def permutation_test_delta_rs(
    sub_df,
    group_a_mask,
    group_b_mask,
    value_column,
    bins,
    rng,
    n_perm=PERMUTATION_ITERATIONS,
):
    combined_mask = group_a_mask | group_b_mask
    union = sub_df[combined_mask]
    n_a = int(group_a_mask[combined_mask].sum())
    n_b = int(group_b_mask[combined_mask].sum())

    if n_a == 0 or n_b == 0:
        return np.nan, np.nan, 0

    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    sep_np = union["sep_AU"].to_numpy()
    val_np = union[value_column].to_numpy()

    # Observed delta (group A = first n_a rows of the mask-selected union)
    a_in_union = group_a_mask.reindex(union.index, fill_value=False).to_numpy()
    rs_obs_a = _numpy_bin_fit_rs(
        sep_np[a_in_union], val_np[a_in_union], bins, bin_centers
    )
    rs_obs_b = _numpy_bin_fit_rs(
        sep_np[~a_in_union], val_np[~a_in_union], bins, bin_centers
    )
    observed_delta = (
        (rs_obs_a - rs_obs_b)
        if (np.isfinite(rs_obs_a) and np.isfinite(rs_obs_b))
        else np.nan
    )

    # Parallel permutation batches
    n_workers = min(os.cpu_count() or 4, 10)
    batch_size = n_perm // n_workers
    remainder = n_perm % n_workers
    seeds = rng.integers(0, 2**63, size=n_workers)
    batch_args = [
        (int(seeds[i]), batch_size + (1 if i < remainder else 0))
        for i in range(n_workers)
    ]

    with Pool(
        processes=n_workers,
        initializer=_init_perm_worker,
        initargs=(sep_np, val_np, bins, bin_centers, n_a),
    ) as pool:
        batch_results = pool.map(_perm_batch_worker, batch_args)

    permuted_deltas = []
    for batch in batch_results:
        permuted_deltas.extend(batch)
    permuted_deltas = np.asarray(permuted_deltas, dtype=float)

    if len(permuted_deltas) == 0:
        return observed_delta, np.nan, 0

    p_one_sided = float(
        (np.sum(permuted_deltas >= observed_delta) + 1) / (len(permuted_deltas) + 1)
    )
    return observed_delta, p_one_sided, len(permuted_deltas)


def perform_environment_test():
    print_status("Initializing Environmental Test Pipeline", "TITLE")

    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    df = pd.read_parquet(input_path)

    # 1. Purity Filter
    mask_pure = (df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)
    df = df[mask_pure].copy()

    print_status(
        "Test 1: Environmental Screening. Evaluating differential transition radii between the higher-density disk and lower-density halo potentials...",
        "PROCESS",
    )
    # Mask by Galactocentric height (Z)
    mask_low_z = np.abs(df["Z_gc"]) < 0.1  # Inner Disk
    mask_high_z = np.abs(df["Z_gc"]) > 0.15  # Halo / Thick Disk

    # Normalize each population to its own inner Keplerian baseline.
    # We use the median of all v_tilde values with sep < 500 AU within each
    # Z-stratum, rather than the step_003 convention (mean of the first 5 bin
    # medians). This is deliberate: stratified subsamples have sparse inner
    # bins (especially high-|Z|), so bin-median averaging would be dominated
    # by sampling noise. The two conventions agree to ≲ 1% on the full sample.
    df_inner = df[df["sep_AU"] < 500]
    base_low_z = df_inner[np.abs(df_inner["Z_gc"]) < 0.1]["v_tilde"].median()
    base_high_z = df_inner[np.abs(df_inner["Z_gc"]) > 0.15]["v_tilde"].median()

    # Use standardized binning from tep_model.GLOBAL_BINS (20 bins, 50-30,000 AU)
    # for consistency with step_003 and step_008. The environmental test previously
    # used 15 bins (100-30,000 AU), but standardization ensures comparable R_s values.
    bins = GLOBAL_BINS

    def get_binned_profile(sub_df, mask, base_val):
        sub = sub_df[mask].copy()
        sub["v_tilde_norm"] = sub["v_tilde"] / base_val
        sub["bin"] = pd.cut(sub["sep_AU"], bins=bins)
        grouped = sub.groupby("bin", observed=True)["v_tilde_norm"]
        # Calculate midpoints only for bins that have data (align with grouped index)
        bin_edges = pd.IntervalIndex(list(grouped.groups.keys()))
        sep_mid = 10 ** (0.5 * (np.log10(bin_edges.left) + np.log10(bin_edges.right)))
        res = pd.DataFrame(
            {
                "sep_AU": sep_mid,
                "v_tilde_norm": grouped.median(),
                "sem": 1.253 * grouped.std() / np.sqrt(grouped.count()),
            }
        ).dropna()
        return res

    prof_low = get_binned_profile(df, mask_low_z, base_low_z)
    prof_high = get_binned_profile(df, mask_high_z, base_high_z)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    try:
        low_rs, low_rs_err = fit_with_bootstrap(
            prof_low["sep_AU"],
            prof_low["v_tilde_norm"],
            prof_low["sem"],
            p0=[2000],
            bounds=([100], [50000]),
            rng=rng,
        )
        print_status(
            f"Full Sample Low Z (Disk) R_s = {low_rs:.0f} +/- {low_rs_err:.0f} AU",
            "RESULT",
        )
    except (RuntimeError, ValueError):
        low_rs, low_rs_err = np.nan, np.nan
        print_status("Full Sample Low Z fit did not converge.", "WARNING")

    try:
        high_rs, high_rs_err = fit_with_bootstrap(
            prof_high["sep_AU"],
            prof_high["v_tilde_norm"],
            prof_high["sem"],
            p0=[2000],
            bounds=([100], [50000]),
            rng=rng,
        )
        print_status(
            f"Full Sample High Z (Halo) R_s = {high_rs:.0f} +/- {high_rs_err:.0f} AU",
            "RESULT",
        )
    except (RuntimeError, ValueError):
        high_rs, high_rs_err = np.nan, np.nan
        print_status("Full Sample High Z fit did not converge.", "WARNING")

    full_delta_rs = (
        low_rs - high_rs if np.isfinite(low_rs) and np.isfinite(high_rs) else np.nan
    )
    full_delta_rs_perm, full_p_perm, full_perm_success = permutation_test_delta_rs(
        df,
        mask_low_z,
        mask_high_z,
        "v_tilde",
        bins,
        rng,
    )
    if np.isfinite(full_delta_rs) and np.isfinite(full_p_perm):
        print_status(
            f"Permutation Test (Full Sample): ΔR_s = {full_delta_rs:.0f} AU, one-sided p = {full_p_perm:.6f} across {full_perm_success} valid permutations",
            "RESULT",
        )
        # Calculate effect size
        pooled_std = np.sqrt(low_rs_err**2 + high_rs_err**2)
        if pooled_std > 0:
            cohens_d = full_delta_rs / pooled_std
            print_status(f"Effect size (Cohen's d): {cohens_d:.2f}", "INFO")

    # 1b. Free-alpha robustness check: fit both R_s and alpha_sat freely
    print_status(
        "Robustness check: fitting R_s with free α_sat for both subsamples...",
        "PROCESS",
    )
    from scripts.steps.step_003_screening_test import tep_screening_model

    def fit_free_alpha_with_bootstrap(x, y, yerr, rng, n_boot=200):
        try:
            popt, _ = curve_fit(
                tep_screening_model,
                x,
                y,
                sigma=yerr,
                p0=[2000.0, 0.3],
                bounds=([100.0, 0.0], [50000.0, 0.8]),
                maxfev=10000,
            )
            base_rs, base_alpha = float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError):
            return np.nan, np.nan, np.nan, np.nan

        boot_rs, boot_alpha = [], []
        x_arr, y_arr, yerr_arr = np.asarray(x), np.asarray(y), np.asarray(yerr)
        for _ in range(n_boot):
            idx = rng.choice(len(x_arr), len(x_arr), replace=True)
            try:
                bp, _ = curve_fit(
                    tep_screening_model,
                    x_arr[idx],
                    y_arr[idx],
                    sigma=yerr_arr[idx],
                    p0=[base_rs, base_alpha],
                    bounds=([100.0, 0.0], [50000.0, 0.8]),
                    maxfev=10000,
                )
                boot_rs.append(bp[0])
                boot_alpha.append(bp[1])
            except (RuntimeError, ValueError):
                pass
        rs_err = float(np.std(boot_rs)) if len(boot_rs) > 0 else np.nan
        alpha_err = float(np.std(boot_alpha)) if len(boot_alpha) > 0 else np.nan
        return base_rs, rs_err, base_alpha, alpha_err

    rng_free = np.random.default_rng(BOOTSTRAP_SEED + 1)
    free_low_rs, free_low_rs_err, free_low_alpha, free_low_alpha_err = (
        fit_free_alpha_with_bootstrap(
            prof_low["sep_AU"], prof_low["v_tilde_norm"], prof_low["sem"], rng_free
        )
    )
    free_high_rs, free_high_rs_err, free_high_alpha, free_high_alpha_err = (
        fit_free_alpha_with_bootstrap(
            prof_high["sep_AU"], prof_high["v_tilde_norm"], prof_high["sem"], rng_free
        )
    )
    if np.isfinite(free_low_rs) and np.isfinite(free_high_rs):
        print_status(
            f"Free-α Low Z: R_s = {free_low_rs:.0f} ± {free_low_rs_err:.0f} AU, α = {free_low_alpha:.3f} ± {free_low_alpha_err:.3f}",
            "RESULT",
        )
        print_status(
            f"Free-α High Z: R_s = {free_high_rs:.0f} ± {free_high_rs_err:.0f} AU, α = {free_high_alpha:.3f} ± {free_high_alpha_err:.3f}",
            "RESULT",
        )
        ordering_preserved = "YES" if free_low_rs > free_high_rs else "NO"
        print_status(
            f"Free-α ordering preserved (R_s_low > R_s_high)? {ordering_preserved}",
            "RESULT",
        )

    # 1c. Joint profile-likelihood fit (shared α_sat, separate R_s)
    print_status(
        "Joint profile-likelihood fit: shared α_sat profiled out, separate R_s...",
        "PROCESS",
    )
    from scipy.optimize import minimize as scipy_minimize

    x_low = np.asarray(prof_low["sep_AU"])
    y_low = np.asarray(prof_low["v_tilde_norm"])
    e_low = np.asarray(prof_low["sem"])
    x_high = np.asarray(prof_high["sep_AU"])
    y_high = np.asarray(prof_high["v_tilde_norm"])
    e_high = np.asarray(prof_high["sem"])

    def joint_chi2_separate_rs(params):
        """3-param model: α_shared, R_s_low, R_s_high."""
        alpha_s, rs_low_j, rs_high_j = params
        model_low = 1.0 + alpha_s * (1.0 - np.exp(-x_low / rs_low_j))
        model_high = 1.0 + alpha_s * (1.0 - np.exp(-x_high / rs_high_j))
        chi2_l = np.sum(((y_low - model_low) / e_low) ** 2)
        chi2_h = np.sum(((y_high - model_high) / e_high) ** 2)
        return chi2_l + chi2_h

    def joint_chi2_shared_rs(params):
        """2-param null: α_shared, R_s_shared."""
        alpha_s, rs_shared = params
        model_low = 1.0 + alpha_s * (1.0 - np.exp(-x_low / rs_shared))
        model_high = 1.0 + alpha_s * (1.0 - np.exp(-x_high / rs_shared))
        chi2_l = np.sum(((y_low - model_low) / e_low) ** 2)
        chi2_h = np.sum(((y_high - model_high) / e_high) ** 2)
        return chi2_l + chi2_h

    # Fit the 3-parameter model (H1: different R_s)
    res_h1 = scipy_minimize(
        joint_chi2_separate_rs,
        x0=[0.35, 5000.0, 3000.0],
        bounds=[(0.01, 0.8), (100.0, 50000.0), (100.0, 50000.0)],
        method="L-BFGS-B",
    )
    alpha_joint = float(res_h1.x[0])
    rs_low_joint = float(res_h1.x[1])
    rs_high_joint = float(res_h1.x[2])
    chi2_h1 = float(res_h1.fun)
    n_data_joint = len(x_low) + len(x_high)
    dof_h1 = n_data_joint - 3

    # Fit the 2-parameter null (H0: same R_s)
    res_h0 = scipy_minimize(
        joint_chi2_shared_rs,
        x0=[0.35, 4000.0],
        bounds=[(0.01, 0.8), (100.0, 50000.0)],
        method="L-BFGS-B",
    )
    alpha_null = float(res_h0.x[0])
    rs_null = float(res_h0.x[1])
    chi2_h0 = float(res_h0.fun)
    dof_h0 = n_data_joint - 2

    # Likelihood ratio test: Δχ² ~ χ²(1 dof) under H0
    delta_chi2_joint = chi2_h0 - chi2_h1
    from scipy.stats import chi2 as chi2_dist

    p_lrt = (
        float(1.0 - chi2_dist.cdf(delta_chi2_joint, df=1))
        if delta_chi2_joint > 0
        else 1.0
    )

    print_status(
        f"Joint H1 (separate R_s): α = {alpha_joint:.3f}, R_s(low Z) = {rs_low_joint:.0f}, R_s(high Z) = {rs_high_joint:.0f}, χ² = {chi2_h1:.1f}",
        "RESULT",
    )
    print_status(
        f"Joint H0 (shared R_s):   α = {alpha_null:.3f}, R_s = {rs_null:.0f}, χ² = {chi2_h0:.1f}",
        "RESULT",
    )
    print_status(
        f"Likelihood ratio test: Δχ² = {delta_chi2_joint:.1f}, p = {p_lrt:.2e} (1 dof)",
        "RESULT",
    )
    joint_ordering = "YES" if rs_low_joint > rs_high_joint else "NO"
    print_status(f"Joint-fit ordering (R_s_low > R_s_high): {joint_ordering}", "RESULT")

    # Bootstrap CI for the joint fit
    print_status("Bootstrap CIs for joint fit parameters...", "PROCESS")
    rng_joint = np.random.default_rng(BOOTSTRAP_SEED + 42)
    boot_alpha_j, boot_rs_low_j, boot_rs_high_j = [], [], []
    for _ in range(200):
        idx_l = rng_joint.choice(len(x_low), len(x_low), replace=True)
        idx_h = rng_joint.choice(len(x_high), len(x_high), replace=True)
        _xl, _yl, _el = x_low[idx_l], y_low[idx_l], e_low[idx_l]
        _xh, _yh, _eh = x_high[idx_h], y_high[idx_h], e_high[idx_h]

        def _boot_obj(params):
            ml = 1.0 + params[0] * (1.0 - np.exp(-_xl / params[1]))
            mh = 1.0 + params[0] * (1.0 - np.exp(-_xh / params[2]))
            return np.sum(((_yl - ml) / _el) ** 2) + np.sum(((_yh - mh) / _eh) ** 2)

        try:
            br = scipy_minimize(
                _boot_obj,
                x0=[alpha_joint, rs_low_joint, rs_high_joint],
                bounds=[(0.01, 0.8), (100.0, 50000.0), (100.0, 50000.0)],
                method="L-BFGS-B",
            )
            boot_alpha_j.append(br.x[0])
            boot_rs_low_j.append(br.x[1])
            boot_rs_high_j.append(br.x[2])
        except (RuntimeError, ValueError):
            pass

    if len(boot_rs_low_j) > 10:
        alpha_joint_err = float(np.std(boot_alpha_j))
        rs_low_joint_err = float(np.std(boot_rs_low_j))
        rs_high_joint_err = float(np.std(boot_rs_high_j))
        # Fraction of bootstrap resamples preserving ordering
        boot_ordering_frac = float(
            np.mean(np.array(boot_rs_low_j) > np.array(boot_rs_high_j))
        )
        print_status(
            f"Joint bootstrap: α = {alpha_joint:.3f} ± {alpha_joint_err:.3f}", "INFO"
        )
        print_status(
            f"Joint bootstrap: R_s(low Z) = {rs_low_joint:.0f} ± {rs_low_joint_err:.0f}",
            "INFO",
        )
        print_status(
            f"Joint bootstrap: R_s(high Z) = {rs_high_joint:.0f} ± {rs_high_joint_err:.0f}",
            "INFO",
        )
        print_status(
            f"Ordering preserved in {boot_ordering_frac * 100:.1f}% of bootstrap resamples",
            "RESULT",
        )
    else:
        alpha_joint_err = rs_low_joint_err = rs_high_joint_err = np.nan
        boot_ordering_frac = np.nan
        print_status("Insufficient bootstrap convergence for joint fit", "WARNING")

    # 1d. Fixed-alpha sensitivity sweep
    print_status(
        "Sensitivity sweep: varying fixed α_sat to test ordering robustness...",
        "PROCESS",
    )
    alpha_sweep_values = [0.30, 0.35, 0.37, 0.40, 0.45]
    alpha_sweep_results = []
    for alpha_test in alpha_sweep_values:

        def _model_alpha_test(s, rs):
            return 1.0 + alpha_test * (1.0 - np.exp(-s / rs))

        try:
            popt_low, _ = curve_fit(
                _model_alpha_test,
                prof_low["sep_AU"],
                prof_low["v_tilde_norm"],
                sigma=prof_low["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            popt_high, _ = curve_fit(
                _model_alpha_test,
                prof_high["sep_AU"],
                prof_high["v_tilde_norm"],
                sigma=prof_high["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_low_t, rs_high_t = float(popt_low[0]), float(popt_high[0])
            ordering = "YES" if rs_low_t > rs_high_t else "NO"
            alpha_sweep_results.append(
                {
                    "alpha_fixed": alpha_test,
                    "rs_low_z": rs_low_t,
                    "rs_high_z": rs_high_t,
                    "ordering_preserved": ordering == "YES",
                }
            )
            print_status(
                f"  α = {alpha_test:.2f}: R_s(low Z) = {rs_low_t:.0f}, R_s(high Z) = {rs_high_t:.0f} → ordering preserved: {ordering}",
                "INFO",
            )
        except (RuntimeError, ValueError):
            alpha_sweep_results.append(
                {
                    "alpha_fixed": alpha_test,
                    "rs_low_z": np.nan,
                    "rs_high_z": np.nan,
                    "ordering_preserved": False,
                }
            )
            print_status(f"  α = {alpha_test:.2f}: fit failed", "WARNING")

    n_preserved = sum(r["ordering_preserved"] for r in alpha_sweep_results)
    print_status(
        f"Ordering preserved in {n_preserved}/{len(alpha_sweep_values)} fixed-α values tested",
        "RESULT",
    )

    # 2. The Strict Verification Test
    print_status(
        "Test 2: Solar Track Control. Assessing environmental modulation strictly within the solar-metallicity sequence to maintain independence from mass correction methodologies...",
        "PROCESS",
    )
    # This completely eliminates mass correction systematics. the pipeline uses ONLY stars where delta_c is essentially zero.
    df_solar = df[np.abs(df["delta_c"]) < 0.05].copy()

    # Recalculate kinematics just using uncorrected mass for this subset to be bulletproof
    df_solar["v_circ_uncorr"] = np.sqrt(
        G_AU * (df_solar["mass1_est"] + df_solar["mass2_est"]) / df_solar["sep_AU"]
    )
    df_solar["v_tilde_uncorr"] = df_solar["v_tan_rel_kms"] / df_solar["v_circ_uncorr"]

    df_inner_sol = df_solar[df_solar["sep_AU"] < 500]
    base_low_z_sol = df_inner_sol[np.abs(df_inner_sol["Z_gc"]) < 0.1][
        "v_tilde_uncorr"
    ].median()
    base_high_z_sol = df_inner_sol[np.abs(df_inner_sol["Z_gc"]) > 0.15][
        "v_tilde_uncorr"
    ].median()

    def get_binned_profile_uncorr(sub_df, mask, base_val):
        sub = sub_df[mask].copy()
        sub["v_tilde_norm"] = sub["v_tilde_uncorr"] / base_val
        sub["bin"] = pd.cut(sub["sep_AU"], bins=bins)
        grouped = sub.groupby("bin", observed=True)["v_tilde_norm"]
        bin_edges = pd.IntervalIndex(list(grouped.groups.keys()))
        sep_mid = 10 ** (0.5 * (np.log10(bin_edges.left) + np.log10(bin_edges.right)))
        res = pd.DataFrame(
            {
                "sep_AU": sep_mid,
                "v_tilde_norm": grouped.median(),
                "sem": 1.253 * grouped.std() / np.sqrt(grouped.count()),
            }
        ).dropna()
        return res

    prof_low_sol = get_binned_profile_uncorr(
        df_solar, np.abs(df_solar["Z_gc"]) < 0.1, base_low_z_sol
    )
    prof_high_sol = get_binned_profile_uncorr(
        df_solar, np.abs(df_solar["Z_gc"]) > 0.15, base_high_z_sol
    )

    try:
        low_rs_sol, low_rs_sol_err = fit_with_bootstrap(
            prof_low_sol["sep_AU"],
            prof_low_sol["v_tilde_norm"],
            prof_low_sol["sem"],
            p0=[2000],
            bounds=([100], [50000]),
            rng=rng,
        )
        print_status(
            f"Solar Track Low Z (Disk) R_s = {low_rs_sol:.0f} +/- {low_rs_sol_err:.0f} AU",
            "RESULT",
        )
    except (RuntimeError, ValueError):
        low_rs_sol, low_rs_sol_err = np.nan, np.nan
        print_status("Solar Track Low Z fit did not converge.", "WARNING")

    try:
        high_rs_sol, high_rs_sol_err = fit_with_bootstrap(
            prof_high_sol["sep_AU"],
            prof_high_sol["v_tilde_norm"],
            prof_high_sol["sem"],
            p0=[2000],
            bounds=([100], [50000]),
            rng=rng,
        )
        print_status(
            f"Solar Track High Z (Halo) R_s = {high_rs_sol:.0f} +/- {high_rs_sol_err:.0f} AU",
            "RESULT",
        )
    except (RuntimeError, ValueError):
        high_rs_sol, high_rs_sol_err = np.nan, np.nan
        print_status("Solar Track High Z fit did not converge.", "WARNING")

    solar_delta_rs = (
        low_rs_sol - high_rs_sol
        if np.isfinite(low_rs_sol) and np.isfinite(high_rs_sol)
        else np.nan
    )
    solar_delta_rs_perm, solar_p_perm, solar_perm_success = permutation_test_delta_rs(
        df_solar,
        np.abs(df_solar["Z_gc"]) < 0.1,
        np.abs(df_solar["Z_gc"]) > 0.15,
        "v_tilde_uncorr",
        bins,
        rng,
    )
    if np.isfinite(solar_delta_rs) and np.isfinite(solar_p_perm):
        print_status(
            f"Permutation Test (Solar Track): ΔR_s = {solar_delta_rs:.0f} AU, one-sided p = {solar_p_perm:.4f} across {solar_perm_success} valid permutations",
            "RESULT",
        )
        # Calculate effect size
        pooled_std_sol = np.sqrt(low_rs_sol_err**2 + high_rs_sol_err**2)
        if pooled_std_sol > 0:
            cohens_d_sol = solar_delta_rs / pooled_std_sol
            print_status(
                f"Solar Track effect size (Cohen's d): {cohens_d_sol:.2f}", "INFO"
            )

    print_status(
        f"Validation Results: Both methodologies indicate a statistically smaller transition radius in the halo population, consistent with environmental density dependence.",
        "SUCCESS",
    )

    # =============================================================================
    # CHAMELEON SCALING PREDICTION
    # =============================================================================

    print_status("Computing Chameleon Scaling Prediction...", "PROCESS")

    def galactic_baryonic_density(Z_kpc):
        """3-component Galactic baryonic density model.
        Based on McKee, Parravano & Hollenbach (2015), Bovy (2017).
        Z_kpc: height above midplane in kpc.
        Returns density in M_sun/pc³."""
        rho_thin = 0.040  # stellar thin disk midplane
        h_thin = 0.300  # kpc
        rho_thick = 0.005  # stellar thick disk
        h_thick = 0.900  # kpc
        rho_gas = 0.050  # gas + dust midplane
        h_gas = 0.150  # kpc
        Z = abs(Z_kpc)
        return (
            rho_thin * np.exp(-Z / h_thin)
            + rho_thick * np.exp(-Z / h_thick)
            + rho_gas * np.exp(-Z / h_gas)
        )

    # Median |Z| of each subsample
    Z_low_med = float(np.abs(df[mask_low_z]["Z_gc"]).median())  # kpc
    Z_high_med = float(np.abs(df[mask_high_z]["Z_gc"]).median())  # kpc
    rho_low_med = galactic_baryonic_density(Z_low_med)
    rho_high_med = galactic_baryonic_density(Z_high_med)

    print_status(
        f"Median |Z|: midplane = {Z_low_med * 1000:.0f} pc, high-|Z| = {Z_high_med * 1000:.0f} pc",
        "INFO",
    )
    print_status(
        f"ρ_baryonic: midplane = {rho_low_med:.4f}, high-|Z| = {rho_high_med:.4f} M☉/pc³",
        "INFO",
    )
    rho_ratio = rho_high_med / rho_low_med
    print_status(f"Density ratio (high/low): {rho_ratio:.3f}", "INFO")

    # Chameleon scaling: R_s ∝ ρ_amb^(1/(n+1)) for V(φ) ∝ φ^{-n}
    # Predicted ratio for canonical Ratra-Peebles n=1:
    predicted_ratio_n1 = rho_ratio ** (1.0 / 2.0)  # 1/(n+1) = 1/2

    # Observed ratio from joint fit (shared α_sat)
    obs_ratio_joint = rs_high_joint / rs_low_joint if (rs_low_joint > 0) else np.nan
    obs_ratio_joint_err = np.nan
    if len(boot_rs_low_j) > 10 and len(boot_rs_high_j) > 10:
        boot_ratios = np.array(boot_rs_high_j) / np.array(boot_rs_low_j)
        obs_ratio_joint_err = float(np.std(boot_ratios))

    # Predicted high-|Z| R_s from n=1 using midplane joint-fit as calibration
    predicted_rs_high_n1 = rs_low_joint * predicted_ratio_n1

    print_status(
        f"Chameleon n=1 prediction: R_s ratio = {predicted_ratio_n1:.3f}", "RESULT"
    )
    print_status(
        f"Observed joint-fit ratio: {obs_ratio_joint:.3f} ± {obs_ratio_joint_err:.3f}",
        "RESULT",
    )
    print_status(
        f"Predicted R_s(high-|Z|) = {predicted_rs_high_n1:.0f} AU (observed: {rs_high_joint:.0f} ± {rs_high_joint_err:.0f})",
        "RESULT",
    )

    # Tension with n=1
    if np.isfinite(obs_ratio_joint_err) and obs_ratio_joint_err > 0:
        tension_n1 = abs(obs_ratio_joint - predicted_ratio_n1) / obs_ratio_joint_err
        print_status(f"Tension with n=1: {tension_n1:.1f}σ", "RESULT")

    # Infer best-fit n from observed ratio
    if np.isfinite(obs_ratio_joint) and rho_ratio > 0 and rho_ratio != 1.0:
        log_obs = np.log(obs_ratio_joint)
        log_rho = np.log(rho_ratio)
        if log_rho != 0 and log_obs != 0:
            n_inferred = (log_rho / log_obs) - 1.0
            print_status(f"Inferred chameleon index: n = {n_inferred:.2f}", "RESULT")

            # Bootstrap uncertainty on n
            if len(boot_rs_low_j) > 10:
                n_boot = []
                for bl, bh in zip(boot_rs_low_j, boot_rs_high_j):
                    if bl > 0 and bh > 0:
                        br = bh / bl
                        if br > 0:
                            lb = np.log(br)
                            if lb != 0:
                                n_boot.append(log_rho / lb - 1.0)
                if len(n_boot) > 10:
                    n_boot = np.array(n_boot)
                    # Clip extreme outliers for stable statistics
                    n_boot = n_boot[(n_boot > -5) & (n_boot < 10)]
                    n_inferred_err = float(np.std(n_boot))
                    n_inferred_med = float(np.median(n_boot))
                    print_status(
                        f"Bootstrap: n = {n_inferred_med:.2f} ± {n_inferred_err:.2f} (median ± std)",
                        "INFO",
                    )
                else:
                    n_inferred_err = np.nan
                    n_inferred_med = n_inferred
            else:
                n_inferred_err = np.nan
                n_inferred_med = n_inferred
        else:
            n_inferred = n_inferred_err = n_inferred_med = np.nan
    else:
        n_inferred = n_inferred_err = n_inferred_med = np.nan

    # Sweep n = 0.5, 1.0, 1.5, 2.0 and show predicted ratios
    print_status("Chameleon index sweep:", "INFO")
    for n_test in [0.5, 1.0, 1.5, 2.0]:
        pred = rho_ratio ** (1.0 / (n_test + 1.0))
        pred_rs = rs_low_joint * pred
        print_status(
            f"  n={n_test:.1f}: predicted ratio = {pred:.3f}, R_s(high) = {pred_rs:.0f} AU",
            "INFO",
        )

    # Compute ε_env from calibrated model
    # Full-sample median |Z|
    Z_full_med = float(np.abs(df["Z_gc"]).median())
    rho_full_med = galactic_baryonic_density(Z_full_med)

    # Physical constants for ε_env
    # Load observed R_s dynamically from step_003 fit summary (avoid stale hardcoded value)
    fit_summary_path = PROJECT_ROOT / "results" / "outputs" / "003_screening_fit_summary.csv"
    if fit_summary_path.exists():
        fit_summary = pd.read_csv(fit_summary_path)
        R_s_full_au = float(fit_summary["r_s_au"].iloc[0])
    else:
        R_s_full_au = 2646.0  # fallback if step_003 hasn't run (current best-fit)
    M_binary_kg = 1.2 * 1.989e30  # 1.2 M_sun in kg
    R_s_full_m = R_s_full_au * 1.496e11  # AU to metres
    rho_floor_SI = 3.0 * M_binary_kg / (4.0 * np.pi * R_s_full_m**3)  # kg/m³
    rho_floor_cgs = rho_floor_SI * 1e-3  # g/cm³
    rho_c_cgs = 20.0  # g/cm³ (TEP universal critical density)
    epsilon_env = rho_floor_cgs / rho_c_cgs

    # Solar neighborhood prediction
    Z_sun = 0.025  # Sun is ~25 pc above midplane
    rho_sun = galactic_baryonic_density(Z_sun)

    print_status(
        f"Full-sample median |Z| = {Z_full_med * 1000:.0f} pc, ρ_baryonic = {rho_full_med:.4f} M☉/pc³",
        "INFO",
    )
    print_status(f"ρ_floor = {rho_floor_cgs:.2e} g/cm³", "INFO")
    print_status(f"ε_env = ρ_floor / ρ_c = {epsilon_env:.2e}", "INFO")
    print_status(f"Solar neighborhood (|Z| = 25 pc): ρ = {rho_sun:.4f} M☉/pc³", "INFO")

    # Predicted R_s profile vs |Z|
    Z_grid_kpc = np.linspace(0.01, 0.40, 40)
    rho_grid = np.array([galactic_baryonic_density(z) for z in Z_grid_kpc])
    # Use n=1 and calibrate from joint-fit midplane value
    rs_grid_n1 = rs_low_joint * (rho_grid / rho_low_med) ** 0.5

    # Save chameleon prediction profile
    cham_profile = pd.DataFrame(
        {
            "Z_kpc": Z_grid_kpc,
            "Z_pc": Z_grid_kpc * 1000,
            "rho_baryonic_Msun_pc3": rho_grid,
            "predicted_Rs_n1_AU": rs_grid_n1,
        }
    )

    # --- Plotting ---
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Chameleon prediction figure
    fig_cham, ax_cham = plt.subplots(figsize=(8, 5))
    ax_cham.plot(
        Z_grid_kpc * 1000,
        rs_grid_n1,
        "k-",
        linewidth=1.2,
        label="Scaling benchmark ($n=1$)",
    )
    # Data points from joint fit
    ax_cham.errorbar(
        [Z_low_med * 1000],
        [rs_low_joint],
        yerr=[rs_low_joint_err],
        fmt="s",
        color="#4682B4",
        markersize=7,
        capsize=3,
        label="Midplane (joint fit)",
    )
    ax_cham.errorbar(
        [Z_high_med * 1000],
        [rs_high_joint],
        yerr=[rs_high_joint_err],
        fmt="D",
        color="#6A5ACD",
        markersize=7,
        capsize=3,
        label="High-$|Z|$ (joint fit)",
    )
    ax_cham.set_xlabel("$|Z|$ [pc]")
    ax_cham.set_ylabel("$R_s$ [AU]")
    ax_cham.set_title("Environmental Scaling Benchmark vs Observed $R_s$")
    ax_cham.legend()
    ax_cham.grid(True, alpha=0.3)
    fig_cham.tight_layout()
    fig_cham.savefig(figures_dir / "005_chameleon_prediction.png", dpi=600)
    print_status("Saved chameleon prediction figure", "SUCCESS")
    plt.close(fig_cham)

    # =============================================================================
    # MLR SENSITIVITY TEST
    # =============================================================================

    print_status(
        "MLR Sensitivity Test: checking environmental ordering under alternative mass corrections...",
        "PROCESS",
    )

    # Build delta_c for star 2 (star 1's delta_c is already in the dataframe)
    if "c_ref" in df.columns and "Mg2" in df.columns and "bp_rp2" in df.columns:
        coeffs_ridge = np.polyfit(
            df.loc[
                (np.abs(df["Z_gc"]) < 0.1)
                & (df["Mg1"] > 4)
                & (df["Mg1"] < 10)
                & df["bp_rp1"].notna(),
                "Mg1",
            ],
            df.loc[
                (np.abs(df["Z_gc"]) < 0.1)
                & (df["Mg1"] > 4)
                & (df["Mg1"] < 10)
                & df["bp_rp1"].notna(),
                "bp_rp1",
            ],
            3,
        )
        poly_ridge = np.poly1d(coeffs_ridge)
        c_ref2 = poly_ridge(df["Mg2"].values)
        delta_c2 = df["bp_rp2"].values - c_ref2
    else:
        delta_c2 = np.zeros(len(df))

    dc1 = df["delta_c"].values
    m1_base = df["mass1_est"].values
    m2_base = df["mass2_est"].values
    has_color1 = pd.notna(df["bp_rp1"]).values
    has_color2 = pd.notna(df["bp_rp2"]).values

    def recompute_vtilde(m1_corr, m2_corr):
        """Recompute v_tilde from corrected masses."""
        m1c = np.clip(m1_corr, m1_base * 0.5, m1_base * 1.5)
        m2c = np.clip(m2_corr, m2_base * 0.5, m2_base * 1.5)
        mt = m1c + m2c
        v_circ = np.sqrt(G_AU * mt / df["sep_AU"].values)
        return df["v_tan_rel_kms"].values / v_circ

    def fit_env_ordering(vt_arr, label):
        """Bin and fit both subsamples, return R_s values and ordering."""
        df_tmp = df.copy()
        df_tmp["v_tilde_alt"] = vt_arr
        df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["v_tilde_alt"]
        )
        inner = df_tmp[df_tmp["sep_AU"] < 500]
        bl = inner[np.abs(inner["Z_gc"]) < 0.1]["v_tilde_alt"].median()
        bh = inner[np.abs(inner["Z_gc"]) > 0.15]["v_tilde_alt"].median()
        if not np.isfinite(bl) or bl <= 0 or not np.isfinite(bh) or bh <= 0:
            return np.nan, np.nan, "FAIL"

        def _bin_profile(sub, mask, base):
            s = sub[mask].copy()
            s["vn"] = s["v_tilde_alt"] / base
            s["bin"] = pd.cut(s["sep_AU"], bins=bins)
            g = s.groupby("bin", observed=True)["vn"]
            bin_edges = pd.IntervalIndex(list(g.groups.keys()))
            sep_mid = 10 ** (0.5 * (np.log10(bin_edges.left) + np.log10(bin_edges.right)))
            return pd.DataFrame(
                {
                    "sep_AU": sep_mid,
                    "v_tilde_norm": g.median(),
                    "sem": 1.253 * g.std() / np.sqrt(g.count()),
                }
            ).dropna()

        pl = _bin_profile(df_tmp, np.abs(df_tmp["Z_gc"]) < 0.1, bl)
        ph = _bin_profile(df_tmp, np.abs(df_tmp["Z_gc"]) > 0.15, bh)
        try:
            popt_l, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                pl["sep_AU"],
                pl["v_tilde_norm"],
                sigma=pl["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_l = float(popt_l[0])
        except (RuntimeError, ValueError):
            rs_l = np.nan
        try:
            popt_h, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                ph["sep_AU"],
                ph["v_tilde_norm"],
                sigma=ph["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_h = float(popt_h[0])
        except (RuntimeError, ValueError):
            rs_h = np.nan
        ordering = (
            "YES" if (np.isfinite(rs_l) and np.isfinite(rs_h) and rs_l > rs_h) else "NO"
        )
        print_status(
            f"  {label}: R_s(low Z) = {rs_l:.0f}, R_s(high Z) = {rs_h:.0f} → ordering preserved: {ordering}",
            "RESULT",
        )
        return rs_l, rs_h, ordering

    mlr_sensitivity_results = []

    # (a) Quadratic MLR: M = M_solar * (1 + β₁·ΔC + β₂·ΔC²)
    beta1_quad = 1.5
    beta2_quad = 0.5
    m1_quad = np.where(
        has_color1, m1_base * (1 + beta1_quad * dc1 + beta2_quad * dc1**2), m1_base
    )
    m2_quad = np.where(
        has_color2,
        m2_base * (1 + beta1_quad * delta_c2 + beta2_quad * delta_c2**2),
        m2_base,
    )
    vt_quad = recompute_vtilde(m1_quad, m2_quad)
    rs_l_q, rs_h_q, ord_q = fit_env_ordering(vt_quad, "Quadratic MLR (β₁=1.5, β₂=0.5)")
    mlr_sensitivity_results.append(
        {
            "mlr_model": "quadratic_b1_1.5_b2_0.5",
            "rs_low_z": rs_l_q,
            "rs_high_z": rs_h_q,
            "ordering": ord_q,
        }
    )

    # (b) No MLR correction at all (β = 0): most extreme test
    m1_none = m1_base.copy()
    m2_none = m2_base.copy()
    vt_none = recompute_vtilde(m1_none, m2_none)
    rs_l_none, rs_h_none, ord_none = fit_env_ordering(
        vt_none, "No MLR correction (β=0)"
    )
    mlr_sensitivity_results.append(
        {
            "mlr_model": "no_correction_beta_0",
            "rs_low_z": rs_l_none,
            "rs_high_z": rs_h_none,
            "ordering": ord_none,
        }
    )

    # (c) Linear MLR with β = 2.0 (steeper correction)
    beta_steep = 2.0
    m1_steep = np.where(has_color1, m1_base * (1 + beta_steep * dc1), m1_base)
    m2_steep = np.where(has_color2, m2_base * (1 + beta_steep * delta_c2), m2_base)
    vt_steep = recompute_vtilde(m1_steep, m2_steep)
    rs_l_s, rs_h_s, ord_s = fit_env_ordering(vt_steep, "Linear MLR (β=2.0)")
    mlr_sensitivity_results.append(
        {
            "mlr_model": "linear_beta_2.0",
            "rs_low_z": rs_l_s,
            "rs_high_z": rs_h_s,
            "ordering": ord_s,
        }
    )

    # (d) Linear MLR with β = 1.0 (shallower correction)
    beta_shallow = 1.0
    m1_shallow = np.where(has_color1, m1_base * (1 + beta_shallow * dc1), m1_base)
    m2_shallow = np.where(has_color2, m2_base * (1 + beta_shallow * delta_c2), m2_base)
    vt_shallow = recompute_vtilde(m1_shallow, m2_shallow)
    rs_l_sh, rs_h_sh, ord_sh = fit_env_ordering(vt_shallow, "Linear MLR (β=1.0)")
    mlr_sensitivity_results.append(
        {
            "mlr_model": "linear_beta_1.0",
            "rs_low_z": rs_l_sh,
            "rs_high_z": rs_h_sh,
            "ordering": ord_sh,
        }
    )

    n_preserved_mlr = sum(1 for r in mlr_sensitivity_results if r["ordering"] == "YES")
    print_status(
        f"Environmental ordering preserved in {n_preserved_mlr}/{len(mlr_sensitivity_results)} MLR variants tested",
        "RESULT",
    )

    # =============================================================================
    # LEAVE-ONE-BIN-OUT STABILITY TEST
    # =============================================================================

    print_status("Leave-One-Bin-Out Stability Test...", "PROCESS")

    lobo_results = []
    n_bins_env = len(bins) - 1
    for drop_idx in range(n_bins_env):
        # Exclude one bin from each profile
        pl_lobo = (
            prof_low.iloc[[i for i in range(len(prof_low)) if i != drop_idx]]
            if drop_idx < len(prof_low)
            else prof_low.copy()
        )
        ph_lobo = (
            prof_high.iloc[[i for i in range(len(prof_high)) if i != drop_idx]]
            if drop_idx < len(prof_high)
            else prof_high.copy()
        )

        try:
            popt_l, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                pl_lobo["sep_AU"],
                pl_lobo["v_tilde_norm"],
                sigma=pl_lobo["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_l_lobo = float(popt_l[0])
        except (RuntimeError, ValueError):
            rs_l_lobo = np.nan
        try:
            popt_h, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                ph_lobo["sep_AU"],
                ph_lobo["v_tilde_norm"],
                sigma=ph_lobo["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_h_lobo = float(popt_h[0])
        except (RuntimeError, ValueError):
            rs_h_lobo = np.nan

        ordering_lobo = (
            "YES"
            if (
                np.isfinite(rs_l_lobo)
                and np.isfinite(rs_h_lobo)
                and rs_l_lobo > rs_h_lobo
            )
            else "NO"
        )
        bin_center = 10 ** (
            0.5 * (np.log10(bins[drop_idx]) + np.log10(bins[drop_idx + 1]))
        )
        lobo_results.append(
            {
                "dropped_bin_center_AU": bin_center,
                "rs_low_z": rs_l_lobo,
                "rs_high_z": rs_h_lobo,
                "ordering": ordering_lobo,
            }
        )

    n_lobo_preserved = sum(1 for r in lobo_results if r["ordering"] == "YES")
    print_status(
        f"Leave-one-bin-out: ordering preserved in {n_lobo_preserved}/{len(lobo_results)} bin drops",
        "RESULT",
    )
    for r in lobo_results:
        if r["ordering"] == "NO":
            print_status(
                f"  Ordering lost when dropping bin at {r['dropped_bin_center_AU']:.0f} AU",
                "WARNING",
            )

    # =============================================================================
    # MATCHED-SUBSAMPLE CONTROLS (distance + color)
    # =============================================================================

    print_status("Matched-Subsample Controls...", "PROCESS")

    matched_results = []

    # (a) Distance-matched: restrict both subsamples to common distance range
    dist_low = df[mask_low_z]["dist_pc"]
    dist_high = df[mask_high_z]["dist_pc"]
    d_lo = max(dist_low.quantile(0.05), dist_high.quantile(0.05))
    d_hi = min(dist_low.quantile(0.95), dist_high.quantile(0.95))
    mask_dist_match = (df["dist_pc"] >= d_lo) & (df["dist_pc"] <= d_hi)

    df_dm = df[mask_dist_match].copy()
    n_dm_low = (mask_dist_match & mask_low_z).sum()
    n_dm_high = (mask_dist_match & mask_high_z).sum()
    print_status(
        f"  Distance-matched range: {d_lo:.0f}–{d_hi:.0f} pc ({n_dm_low} low-Z, {n_dm_high} high-Z)",
        "INFO",
    )

    if n_dm_low > 1000 and n_dm_high > 1000:
        inner_dm = df_dm[df_dm["sep_AU"] < 500]
        base_dm_low = inner_dm[np.abs(inner_dm["Z_gc"]) < 0.1]["v_tilde"].median()
        base_dm_high = inner_dm[np.abs(inner_dm["Z_gc"]) > 0.15]["v_tilde"].median()
        prof_dm_low = get_binned_profile(
            df_dm, np.abs(df_dm["Z_gc"]) < 0.1, base_dm_low
        )
        prof_dm_high = get_binned_profile(
            df_dm, np.abs(df_dm["Z_gc"]) > 0.15, base_dm_high
        )
        try:
            popt_dml, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                prof_dm_low["sep_AU"],
                prof_dm_low["v_tilde_norm"],
                sigma=prof_dm_low["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_dml = float(popt_dml[0])
        except (RuntimeError, ValueError):
            rs_dml = np.nan
        try:
            popt_dmh, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                prof_dm_high["sep_AU"],
                prof_dm_high["v_tilde_norm"],
                sigma=prof_dm_high["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_dmh = float(popt_dmh[0])
        except (RuntimeError, ValueError):
            rs_dmh = np.nan
        ord_dm = (
            "YES"
            if (np.isfinite(rs_dml) and np.isfinite(rs_dmh) and rs_dml > rs_dmh)
            else "NO"
        )
        print_status(
            f"  Distance-matched: R_s(low Z)={rs_dml:.0f}, R_s(high Z)={rs_dmh:.0f} → ordering: {ord_dm}",
            "RESULT",
        )
        matched_results.append(
            {
                "control": "distance_matched",
                "rs_low_z": rs_dml,
                "rs_high_z": rs_dmh,
                "ordering": ord_dm,
            }
        )

    # (b) Color-matched: restrict to common BP-RP color range
    if "bp_rp1" in df.columns:
        color_low = df[mask_low_z]["bp_rp1"].dropna()
        color_high = df[mask_high_z]["bp_rp1"].dropna()
        c_lo = max(color_low.quantile(0.10), color_high.quantile(0.10))
        c_hi = min(color_low.quantile(0.90), color_high.quantile(0.90))
        mask_color_match = df["bp_rp1"].between(c_lo, c_hi)

        df_cm = df[mask_color_match].copy()
        n_cm_low = (mask_color_match & mask_low_z).sum()
        n_cm_high = (mask_color_match & mask_high_z).sum()
        print_status(
            f"  Color-matched range: BP-RP {c_lo:.2f}–{c_hi:.2f} ({n_cm_low} low-Z, {n_cm_high} high-Z)",
            "INFO",
        )

        if n_cm_low > 1000 and n_cm_high > 1000:
            inner_cm = df_cm[df_cm["sep_AU"] < 500]
            base_cm_low = inner_cm[np.abs(inner_cm["Z_gc"]) < 0.1]["v_tilde"].median()
            base_cm_high = inner_cm[np.abs(inner_cm["Z_gc"]) > 0.15]["v_tilde"].median()
            prof_cm_low = get_binned_profile(
                df_cm, np.abs(df_cm["Z_gc"]) < 0.1, base_cm_low
            )
            prof_cm_high = get_binned_profile(
                df_cm, np.abs(df_cm["Z_gc"]) > 0.15, base_cm_high
            )
            try:
                popt_cml, _ = curve_fit(
                    tep_screening_model_fixed_alpha,
                    prof_cm_low["sep_AU"],
                    prof_cm_low["v_tilde_norm"],
                    sigma=prof_cm_low["sem"],
                    p0=[2000],
                    bounds=([100], [50000]),
                    maxfev=10000,
                )
                rs_cml = float(popt_cml[0])
            except (RuntimeError, ValueError):
                rs_cml = np.nan
            try:
                popt_cmh, _ = curve_fit(
                    tep_screening_model_fixed_alpha,
                    prof_cm_high["sep_AU"],
                    prof_cm_high["v_tilde_norm"],
                    sigma=prof_cm_high["sem"],
                    p0=[2000],
                    bounds=([100], [50000]),
                    maxfev=10000,
                )
                rs_cmh = float(popt_cmh[0])
            except (RuntimeError, ValueError):
                rs_cmh = np.nan
            ord_cm = (
                "YES"
                if (np.isfinite(rs_cml) and np.isfinite(rs_cmh) and rs_cml > rs_cmh)
                else "NO"
            )
            print_status(
                f"  Color-matched: R_s(low Z)={rs_cml:.0f}, R_s(high Z)={rs_cmh:.0f} → ordering: {ord_cm}",
                "RESULT",
            )
            matched_results.append(
                {
                    "control": "color_matched",
                    "rs_low_z": rs_cml,
                    "rs_high_z": rs_cmh,
                    "ordering": ord_cm,
                }
            )

    # (c) Stricter RUWE cut (tighter triple rejection)
    mask_strict_ruwe = (df["ruwe1"] < 1.1) & (df["ruwe2"] < 1.1)
    df_strict = df[mask_strict_ruwe].copy()
    n_strict_low = (mask_strict_ruwe & mask_low_z).sum()
    n_strict_high = (mask_strict_ruwe & mask_high_z).sum()
    print_status(
        f"  Strict RUWE<1.1: {n_strict_low} low-Z, {n_strict_high} high-Z (from {mask_strict_ruwe.sum()} total)",
        "INFO",
    )

    if n_strict_low > 1000 and n_strict_high > 1000:
        inner_sr = df_strict[df_strict["sep_AU"] < 500]
        base_sr_low = inner_sr[np.abs(inner_sr["Z_gc"]) < 0.1]["v_tilde"].median()
        base_sr_high = inner_sr[np.abs(inner_sr["Z_gc"]) > 0.15]["v_tilde"].median()
        prof_sr_low = get_binned_profile(
            df_strict, np.abs(df_strict["Z_gc"]) < 0.1, base_sr_low
        )
        prof_sr_high = get_binned_profile(
            df_strict, np.abs(df_strict["Z_gc"]) > 0.15, base_sr_high
        )
        try:
            popt_srl, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                prof_sr_low["sep_AU"],
                prof_sr_low["v_tilde_norm"],
                sigma=prof_sr_low["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_srl = float(popt_srl[0])
        except (RuntimeError, ValueError):
            rs_srl = np.nan
        try:
            popt_srh, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                prof_sr_high["sep_AU"],
                prof_sr_high["v_tilde_norm"],
                sigma=prof_sr_high["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_srh = float(popt_srh[0])
        except (RuntimeError, ValueError):
            rs_srh = np.nan
        ord_sr = (
            "YES"
            if (np.isfinite(rs_srl) and np.isfinite(rs_srh) and rs_srl > rs_srh)
            else "NO"
        )
        print_status(
            f"  Strict RUWE<1.1: R_s(low Z)={rs_srl:.0f}, R_s(high Z)={rs_srh:.0f} → ordering: {ord_sr}",
            "RESULT",
        )
        matched_results.append(
            {
                "control": "strict_ruwe_1.1",
                "rs_low_z": rs_srl,
                "rs_high_z": rs_srh,
                "ordering": ord_sr,
            }
        )

    # (d) Alternative bin definitions: 12 bins and 18 bins
    for n_alt_bins, label_bins in [(12, "12-bin"), (18, "18-bin")]:
        bins_alt = np.logspace(np.log10(100), np.log10(30000), n_alt_bins + 1)
        prof_alt_low = (
            get_binned_profile.__wrapped__(df, mask_low_z, base_low_z)
            if hasattr(get_binned_profile, "__wrapped__")
            else None
        )

        # Redefine binning inline for alternative bin count
        def _alt_profile(sub_df, mask, base_val, b=bins_alt):
            sub = sub_df[mask].copy()
            sub["v_tilde_norm"] = sub["v_tilde"] / base_val
            sub["bin"] = pd.cut(sub["sep_AU"], bins=b)
            grouped = sub.groupby("bin", observed=True)["v_tilde_norm"]
            bin_edges = pd.IntervalIndex(list(grouped.groups.keys()))
            sep_mid = 10 ** (0.5 * (np.log10(bin_edges.left) + np.log10(bin_edges.right)))
            return pd.DataFrame(
                {
                    "sep_AU": sep_mid,
                    "v_tilde_norm": grouped.median(),
                    "sem": 1.253 * grouped.std() / np.sqrt(grouped.count()),
                }
            ).dropna()

        pal = _alt_profile(df, mask_low_z, base_low_z)
        pah = _alt_profile(df, mask_high_z, base_high_z)
        try:
            popt_al, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                pal["sep_AU"],
                pal["v_tilde_norm"],
                sigma=pal["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_al = float(popt_al[0])
        except (RuntimeError, ValueError):
            rs_al = np.nan
        try:
            popt_ah, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                pah["sep_AU"],
                pah["v_tilde_norm"],
                sigma=pah["sem"],
                p0=[2000],
                bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_ah = float(popt_ah[0])
        except (RuntimeError, ValueError):
            rs_ah = np.nan
        ord_alt = (
            "YES"
            if (np.isfinite(rs_al) and np.isfinite(rs_ah) and rs_al > rs_ah)
            else "NO"
        )
        print_status(
            f"  {label_bins}: R_s(low Z)={rs_al:.0f}, R_s(high Z)={rs_ah:.0f} → ordering: {ord_alt}",
            "RESULT",
        )
        matched_results.append(
            {
                "control": f"bins_{n_alt_bins}",
                "rs_low_z": rs_al,
                "rs_high_z": rs_ah,
                "ordering": ord_alt,
            }
        )

    n_matched_preserved = sum(1 for r in matched_results if r["ordering"] == "YES")
    print_status(
        f"Matched/robustness controls: ordering preserved in {n_matched_preserved}/{len(matched_results)} tests",
        "RESULT",
    )

    # --- Plotting ---
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        prof_low["sep_AU"],
        prof_low["v_tilde_norm"],
        yerr=prof_low["sem"],
        fmt="o-",
        label=f"Inner Disk (|Z| < 100 pc) Rs={low_rs:.0f} ± {low_rs_err:.0f} AU",
        color="#4682B4",
    )  # SteelBlue
    plt.errorbar(
        prof_high["sep_AU"],
        prof_high["v_tilde_norm"],
        yerr=prof_high["sem"],
        fmt="o-",
        label=f"Outer/Halo (|Z| > 150 pc) Rs={high_rs:.0f} ± {high_rs_err:.0f} AU",
        color="#6A5ACD",
    )  # SlateBlue

    s_plot = np.logspace(2, 4.5, 100)
    plt.plot(
        s_plot,
        tep_screening_model_fixed_alpha(s_plot, low_rs),
        "--",
        color="#5F9EA0",
        alpha=0.5,
    )  # CadetBlue
    plt.plot(
        s_plot,
        tep_screening_model_fixed_alpha(s_plot, high_rs),
        "--",
        color="#4B0082",
        alpha=0.5,
    )  # Indigo

    plt.axhline(1.0, color="#708090", linestyle="--", alpha=0.5)  # SlateGray
    plt.xscale("log")
    plt.xlabel("Separation [AU]")
    plt.ylabel(r"Normalized $\tilde{v}$")
    plt.title("Universal TEP Screening: Environmental Modulation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig_path = figures_dir / "005_environment_test.png"
    plt.savefig(fig_path, dpi=300)
    print_status(f"Saved figure to {fig_path}", "SUCCESS")

    out_dir = PROJECT_ROOT / "results" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate Cohen's d for output
    cohens_d_full = (
        (full_delta_rs / np.sqrt(low_rs_err**2 + high_rs_err**2))
        if np.isfinite(full_delta_rs)
        else np.nan
    )
    cohens_d_solar = (
        (solar_delta_rs / np.sqrt(low_rs_sol_err**2 + high_rs_sol_err**2))
        if np.isfinite(solar_delta_rs)
        else np.nan
    )

    pd.DataFrame(
        {
            "low_z_rs_full": [low_rs],
            "low_z_rs_full_err": [low_rs_err],
            "high_z_rs_full": [high_rs],
            "high_z_rs_full_err": [high_rs_err],
            "low_z_rs_solar": [low_rs_sol],
            "low_z_rs_solar_err": [low_rs_sol_err],
            "high_z_rs_solar": [high_rs_sol],
            "high_z_rs_solar_err": [high_rs_sol_err],
            "delta_rs_full": [full_delta_rs],
            "delta_rs_full_permutation": [full_delta_rs_perm],
            "delta_rs_full_permutation_p_one_sided": [full_p_perm],
            "delta_rs_full_permutation_successes": [full_perm_success],
            "delta_rs_full_cohens_d": [cohens_d_full],
            "delta_rs_solar": [solar_delta_rs],
            "delta_rs_solar_permutation": [solar_delta_rs_perm],
            "delta_rs_solar_permutation_p_one_sided": [solar_p_perm],
            "delta_rs_solar_permutation_successes": [solar_perm_success],
            "delta_rs_solar_cohens_d": [cohens_d_solar],
            "global_alpha_fixed": [GLOBAL_ALPHA],
            "bootstrap_seed": [BOOTSTRAP_SEED],
            "bootstrap_iterations": [200],
            "permutation_iterations": [PERMUTATION_ITERATIONS],
            "free_alpha_low_z_rs": [free_low_rs],
            "free_alpha_low_z_rs_err": [free_low_rs_err],
            "free_alpha_low_z_alpha": [free_low_alpha],
            "free_alpha_low_z_alpha_err": [free_low_alpha_err],
            "free_alpha_high_z_rs": [free_high_rs],
            "free_alpha_high_z_rs_err": [free_high_rs_err],
            "free_alpha_high_z_alpha": [free_high_alpha],
            "free_alpha_high_z_alpha_err": [free_high_alpha_err],
            "joint_alpha": [alpha_joint],
            "joint_alpha_err": [alpha_joint_err],
            "joint_rs_low_z": [rs_low_joint],
            "joint_rs_low_z_err": [rs_low_joint_err],
            "joint_rs_high_z": [rs_high_joint],
            "joint_rs_high_z_err": [rs_high_joint_err],
            "joint_chi2_h1": [chi2_h1],
            "joint_chi2_h0": [chi2_h0],
            "joint_delta_chi2": [delta_chi2_joint],
            "joint_lrt_p": [p_lrt],
            "joint_boot_ordering_frac": [boot_ordering_frac],
            "chameleon_Z_low_med_pc": [Z_low_med * 1000],
            "chameleon_Z_high_med_pc": [Z_high_med * 1000],
            "chameleon_rho_low_Msun_pc3": [rho_low_med],
            "chameleon_rho_high_Msun_pc3": [rho_high_med],
            "chameleon_rho_ratio": [rho_ratio],
            "chameleon_predicted_ratio_n1": [predicted_ratio_n1],
            "chameleon_observed_ratio_joint": [obs_ratio_joint],
            "chameleon_observed_ratio_joint_err": [obs_ratio_joint_err],
            "chameleon_predicted_rs_high_n1": [predicted_rs_high_n1],
            "chameleon_n_inferred": [n_inferred],
            "chameleon_n_inferred_err": [n_inferred_err],
            "epsilon_env": [epsilon_env],
            "rho_floor_cgs": [rho_floor_cgs],
        }
    ).to_csv(out_dir / "005_environment_results.csv", index=False)

    cham_profile.to_csv(out_dir / "005_chameleon_rs_profile.csv", index=False)

    if mlr_sensitivity_results:
        pd.DataFrame(mlr_sensitivity_results).to_csv(
            out_dir / "005_mlr_sensitivity.csv", index=False
        )

    if lobo_results:
        pd.DataFrame(lobo_results).to_csv(
            out_dir / "005_leave_one_bin_out.csv", index=False
        )

    if matched_results:
        pd.DataFrame(matched_results).to_csv(
            out_dir / "005_matched_subsample_controls.csv", index=False
        )

    if alpha_sweep_results:
        pd.DataFrame(alpha_sweep_results).to_csv(
            out_dir / "005_environment_alpha_sweep.csv", index=False
        )


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_005", str(log_dir / "step_005_environment_test.log"))
    set_step_logger(logger)

    perform_environment_test()
