#!/usr/bin/env python3
"""
Step 007: Supplemental Controls for TEP-WB

Performs comprehensive robustness testing through subset analyses and null controls.

This script:
1. Tests signal consistency across 7 sample subsets (RV, distance, mass ratio, primary mass)
2. Performs null hypothesis tests via permutation scrambling
3. Validates that the screening signal is not due to systematic biases
4. Quantifies robustness through effect size and permutation p-values
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.steps.step_003_screening_test import (
    tep_screening_model,
    flat_newtonian_model,
    constant_boost_model,
    chi2_statistic,
    information_criteria,
)

GLOBAL_BINS = np.logspace(np.log10(50), np.log10(30000), 20)
NULL_ITERATIONS = 10000
RV_ERR_MAX = 5.0
RV_DELTA_MAX = 5.0
RV_SIG_MAX = 3.0


def build_profile(frame, value_col="v_tilde", bins=GLOBAL_BINS):
    work = frame[["sep_AU", value_col]].dropna().copy()
    if len(work) == 0:
        return None, np.nan

    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)[value_col]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories

    medians = grouped.median().reindex(bin_index)
    means = grouped.mean().reindex(bin_index)
    stds = grouped.std().reindex(bin_index)
    counts = grouped.count().reindex(bin_index)

    profile = pd.DataFrame(
        {
            "sep_AU": bin_centers,
            "v_tilde_median": medians.to_numpy(),
            "v_tilde_mean": means.to_numpy(),
            "v_tilde_std": stds.to_numpy(),
            "count": counts.to_numpy(),
        }
    ).dropna()

    if len(profile) < 8:
        return None, np.nan

    profile["sem"] = 1.253 * profile["v_tilde_std"] / np.sqrt(profile["count"])
    baseline = profile["v_tilde_median"].iloc[:5].mean()
    if not np.isfinite(baseline) or baseline <= 0:
        return None, np.nan

    profile["v_tilde_norm"] = profile["v_tilde_median"] / baseline
    profile["sem_norm"] = profile["sem"] / baseline
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna()

    if len(profile) < 8:
        return None, np.nan

    return profile, float(baseline)


def fit_profile(profile):
    if profile is None or len(profile) < 8:
        return {"fit_success": False}

    try:
        popt, pcov = curve_fit(
            tep_screening_model,
            profile["sep_AU"],
            profile["v_tilde_norm"],
            sigma=profile["sem_norm"],
            p0=[5000.0, 0.2],
            bounds=([100.0, 0.0], [50000.0, 0.5]),
            maxfev=10000,
        )
    except Exception:
        return {"fit_success": False}

    r_s_fit = float(popt[0])
    alpha_fit = float(popt[1])
    r_s_err = float(np.sqrt(np.diag(pcov))[0])
    alpha_err = float(np.sqrt(np.diag(pcov))[1])

    residuals = profile["v_tilde_norm"] - tep_screening_model(profile["sep_AU"], *popt)
    chi2 = float(np.sum((residuals / profile["sem_norm"]) ** 2))
    dof = len(profile) - 2
    red_chi2 = float(chi2 / dof) if dof > 0 else np.nan

    n_points = len(profile)
    tep_aic, tep_bic = information_criteria(chi2, n_points, 2)

    flat_prediction = flat_newtonian_model(profile["sep_AU"])
    flat_chi2 = chi2_statistic(profile["v_tilde_norm"], flat_prediction, profile["sem_norm"])
    flat_aic, flat_bic = information_criteria(flat_chi2, n_points, 0)

    weights = 1.0 / np.square(profile["sem_norm"])
    alpha_const = np.clip(np.sum(weights * (profile["v_tilde_norm"] - 1.0)) / np.sum(weights), 0.0, 0.5)
    const_prediction = constant_boost_model(profile["sep_AU"], alpha_const)
    const_chi2 = chi2_statistic(profile["v_tilde_norm"], const_prediction, profile["sem_norm"])
    const_aic, const_bic = information_criteria(const_chi2, n_points, 1)

    return {
        "fit_success": True,
        "n_bins": int(n_points),
        "r_s_au": r_s_fit,
        "r_s_err_au": r_s_err,
        "alpha": alpha_fit,
        "alpha_err": alpha_err,
        "reduced_chi2": red_chi2,
        "delta_chi2_vs_flat": float(flat_chi2 - chi2),
        "delta_chi2_vs_constant_boost": float(const_chi2 - chi2),
        "delta_aic_vs_flat": float(flat_aic - tep_aic),
        "delta_aic_vs_constant_boost": float(const_aic - tep_aic),
        "delta_bic_vs_flat": float(flat_bic - tep_bic),
        "delta_bic_vs_constant_boost": float(const_bic - tep_bic),
    }


def summarize_subset(name, frame):
    """Analyze a subset and return comprehensive statistics."""
    q = np.minimum(frame["mass1_corr"], frame["mass2_corr"]) / np.maximum(frame["mass1_corr"], frame["mass2_corr"])
    primary_mass = np.maximum(frame["mass1_corr"], frame["mass2_corr"])
    profile, baseline = build_profile(frame)
    fit = fit_profile(profile)

    result = {
        "subset": name,
        "n_systems": int(len(frame)),
        "baseline_v_tilde": baseline,
        "median_dist_pc": float(frame["dist_pc"].median()),
        "median_q": float(q.median()),
        "median_primary_mass": float(primary_mass.median()),
    }
    result.update(fit)

    if fit.get("fit_success"):
        # Calculate additional metrics
        r_s = fit['r_s_au']
        r_s_err = fit['r_s_err_au']
        cv_r_s = r_s_err / r_s if r_s > 0 else np.nan  # Coefficient of variation
        
        print_status(
            f"{name}: R_s = {r_s:.1f} ± {r_s_err:.1f} AU (CV={cv_r_s:.2%}), α = {fit['alpha']:.3f}, Δχ2 vs const = {fit['delta_chi2_vs_constant_boost']:.1f}",
            "RESULT",
        )
    else:
        print_status(f"{name}: insufficient information for a stable screening fit.", "WARNING")

    return result


def distance_scramble(values, codes, rng):
    scrambled = values.copy()
    for code in np.unique(codes[codes >= 0]):
        idx = np.where(codes == code)[0]
        scrambled[idx] = rng.permutation(values[idx])
    return scrambled


def scramble_null_controls(frame, rng, n_iter=NULL_ITERATIONS):
    base = frame[["sep_AU", "dist_pc", "v_tilde"]].dropna().copy()
    observed_profile, _ = build_profile(base)
    observed_fit = fit_profile(observed_profile)
    if not observed_fit.get("fit_success"):
        return pd.DataFrame(), pd.DataFrame()

    base["dist_bin"] = pd.qcut(base["dist_pc"], q=4, duplicates="drop")
    values = base["v_tilde"].to_numpy()
    dist_codes = base["dist_bin"].cat.codes.to_numpy()
    rows = []

    for mode in ["global", "within_distance_quartile"]:
        for iteration in range(n_iter):
            if mode == "global":
                scrambled = rng.permutation(values)
            else:
                scrambled = distance_scramble(values, dist_codes, rng)

            work = base[["sep_AU"]].copy()
            work["v_tilde_scrambled"] = scrambled
            profile, _ = build_profile(work, value_col="v_tilde_scrambled")
            fit = fit_profile(profile)
            if fit.get("fit_success"):
                rows.append(
                    {
                        "mode": mode,
                        "iteration": iteration,
                        "r_s_au": fit["r_s_au"],
                        "alpha": fit["alpha"],
                        "delta_chi2_vs_flat": fit["delta_chi2_vs_flat"],
                        "delta_chi2_vs_constant_boost": fit["delta_chi2_vs_constant_boost"],
                    }
                )

    raw = pd.DataFrame(rows)
    summaries = []
    for mode in ["global", "within_distance_quartile"]:
        sub = raw[raw["mode"] == mode].copy()
        if len(sub) == 0:
            summaries.append(
                {
                    "mode": mode,
                    "valid_iterations": 0,
                    "observed_delta_chi2_vs_flat": observed_fit["delta_chi2_vs_flat"],
                    "observed_delta_chi2_vs_constant_boost": observed_fit["delta_chi2_vs_constant_boost"],
                    "p_delta_chi2_vs_flat": np.nan,
                    "p_delta_chi2_vs_constant_boost": np.nan,
                }
            )
            continue

        p_flat = float(
            (np.sum(sub["delta_chi2_vs_flat"] >= observed_fit["delta_chi2_vs_flat"]) + 1)
            / (len(sub) + 1)
        )
        p_const = float(
            (np.sum(sub["delta_chi2_vs_constant_boost"] >= observed_fit["delta_chi2_vs_constant_boost"]) + 1)
            / (len(sub) + 1)
        )
        summaries.append(
            {
                "mode": mode,
                "valid_iterations": int(len(sub)),
                "observed_delta_chi2_vs_flat": observed_fit["delta_chi2_vs_flat"],
                "observed_delta_chi2_vs_constant_boost": observed_fit["delta_chi2_vs_constant_boost"],
                "p_delta_chi2_vs_flat": p_flat,
                "p_delta_chi2_vs_constant_boost": p_const,
            }
        )
        print_status(
            f"Null scramble ({mode}): p(Δχ2 vs flat) = {p_flat:.4f}, p(Δχ2 vs constant boost) = {p_const:.4f} across {len(sub)} valid iterations",
            "RESULT",
        )

    return pd.DataFrame(summaries), raw


def perform_supplemental_controls():
    print_status("Initializing Supplemental Controls", "TITLE")

    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status("Kinematic results missing. Run step_002 first.", "ERROR")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    df = df[(df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)].copy()

    q = np.minimum(df["mass1_corr"], df["mass2_corr"]) / np.maximum(df["mass1_corr"], df["mass2_corr"])
    primary_mass = np.maximum(df["mass1_corr"], df["mass2_corr"])
    distance_median = float(df["dist_pc"].median())
    q_median = float(q.median())
    primary_mass_median = float(primary_mass.median())

    rv_mask = df[["dr2_radial_velocity1", "dr2_radial_velocity2", "dr2_radial_velocity_error1", "dr2_radial_velocity_error2"]].notna().all(axis=1)
    drv = (df["dr2_radial_velocity1"] - df["dr2_radial_velocity2"]).abs()
    drv_sig = drv / np.sqrt(df["dr2_radial_velocity_error1"] ** 2 + df["dr2_radial_velocity_error2"] ** 2)
    rv_mask &= (df["dr2_radial_velocity_error1"] < RV_ERR_MAX)
    rv_mask &= (df["dr2_radial_velocity_error2"] < RV_ERR_MAX)
    rv_mask &= (drv < RV_DELTA_MAX)
    rv_mask &= (drv_sig < RV_SIG_MAX)

    subset_rows = []
    subset_rows.append(summarize_subset("rv_consistent", df[rv_mask].copy()))
    subset_rows.append(summarize_subset("near_distance_half", df[df["dist_pc"] <= distance_median].copy()))
    subset_rows.append(summarize_subset("far_distance_half", df[df["dist_pc"] > distance_median].copy()))
    subset_rows.append(summarize_subset("low_mass_ratio_half", df[q < q_median].copy()))
    subset_rows.append(summarize_subset("high_mass_ratio_half", df[q >= q_median].copy()))
    subset_rows.append(summarize_subset("low_primary_mass_half", df[primary_mass < primary_mass_median].copy()))
    subset_rows.append(summarize_subset("high_primary_mass_half", df[primary_mass >= primary_mass_median].copy()))

    rng = np.random.default_rng(20260318)
    null_summary, null_raw = scramble_null_controls(df, rng)

    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subset_rows).to_csv(outputs_dir / "supplementary_subset_controls.csv", index=False)
    null_summary.to_csv(outputs_dir / "supplementary_null_controls.csv", index=False)
    null_raw.to_csv(outputs_dir / "supplementary_null_control_draws.csv", index=False)
    print_status(f"Saved supplemental subset controls to {outputs_dir / 'supplementary_subset_controls.csv'}", "SUCCESS")
    print_status(f"Saved supplemental null controls to {outputs_dir / 'supplementary_null_controls.csv'}", "SUCCESS")


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_007", str(log_dir / "step_007_supplemental_controls.log"))
    set_step_logger(logger)

    perform_supplemental_controls()
