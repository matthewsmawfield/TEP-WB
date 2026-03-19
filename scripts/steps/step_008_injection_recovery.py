#!/usr/bin/env python3
"""
Step 008: Injection-Recovery Test for TEP-WB

Validates that the analysis pipeline can faithfully recover a known TEP screening
signal injected into a realistic mock catalog, and that it does NOT falsely recover
a signal when none is injected.

This script:
1. Constructs a mock catalog by shuffling real v_tilde values to destroy any true signal
   while preserving the marginal distributions of separation, distance, mass, and errors.
2. Injects a known TEP screening signal (with specified R_s and alpha_sat) into the
   shuffled catalog.
3. Runs the standard profile-building and fitting pipeline on the injected catalog.
4. Measures recovery fidelity: how well R_s and alpha_sat are recovered.
5. Repeats with a pure-Newtonian injection (alpha=0) to confirm the pipeline does not
   hallucinate a transition.
6. Repeats at multiple injected R_s values to map recovery accuracy vs. signal strength.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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
N_REALIZATIONS = 100
SEED = 271828


def build_profile(frame, value_col="v_tilde_injected", bins=GLOBAL_BINS):
    """Build a binned velocity profile from a dataframe."""
    work = frame[["sep_AU", value_col]].dropna().copy()
    if len(work) == 0:
        return None, np.nan

    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)[value_col]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories

    medians = grouped.median().reindex(bin_index)
    stds = grouped.std().reindex(bin_index)
    counts = grouped.count().reindex(bin_index)

    profile = pd.DataFrame(
        {
            "sep_AU": bin_centers,
            "v_tilde_median": medians.to_numpy(),
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
    """Fit TEP screening model to a binned profile."""
    if profile is None or len(profile) < 8:
        return {"fit_success": False}

    try:
        popt, pcov = curve_fit(
            tep_screening_model,
            profile["sep_AU"],
            profile["v_tilde_norm"],
            sigma=profile["sem_norm"],
            p0=[5000.0, 0.2],
            bounds=([100.0, 0.0], [50000.0, 0.8]),
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

    # Compare to constant boost
    weights = 1.0 / np.square(profile["sem_norm"])
    alpha_const = np.clip(
        np.sum(weights * (profile["v_tilde_norm"] - 1.0)) / np.sum(weights), 0.0, 0.5
    )
    const_prediction = constant_boost_model(profile["sep_AU"], alpha_const)
    const_chi2 = chi2_statistic(
        profile["v_tilde_norm"], const_prediction, profile["sem_norm"]
    )

    return {
        "fit_success": True,
        "r_s_au": r_s_fit,
        "r_s_err_au": r_s_err,
        "alpha": alpha_fit,
        "alpha_err": alpha_err,
        "chi2": chi2,
        "dof": int(dof),
        "delta_chi2_vs_constant_boost": float(const_chi2 - chi2),
    }


def inject_signal(df, r_s_inject, alpha_inject, rng):
    """
    Create a mock catalog by:
    1. Shuffling v_tilde GLOBALLY across all separations to destroy any
       separation-dependent structure (i.e., the real signal).
    2. Normalizing the shuffled values so the global median is unity.
    3. Multiplying by the TEP screening enhancement at each system's separation.

    Global shuffling ensures that the per-bin medians are flat (~1.0) before
    injection, so any recovered signal comes exclusively from the injected
    parameters.  The trade-off is that the per-bin scatter is now the global
    scatter rather than the local scatter, but because the pipeline uses
    medians (robust to outliers) the recovery test remains valid.
    """
    work = df[["sep_AU", "v_tilde"]].dropna().copy()

    # Global shuffle: destroy all separation-dependent structure
    shuffled = rng.permutation(work["v_tilde"].values).copy()

    # Normalize so global median ≈ 1
    global_median = np.median(shuffled)
    if np.isfinite(global_median) and global_median > 0:
        shuffled = shuffled / global_median

    # Inject the TEP signal
    enhancement = 1.0 + alpha_inject * (1.0 - np.exp(-work["sep_AU"].values / r_s_inject))
    work["v_tilde_injected"] = shuffled * enhancement

    return work


def run_injection_recovery():
    print_status("Initializing Injection-Recovery Test", "TITLE")

    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status("Kinematic results missing. Run step_002 first.", "ERROR")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    df = df[(df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)].copy()
    print_status(f"Loaded {len(df):,} systems after RUWE cut", "INFO")

    rng = np.random.default_rng(SEED)

    # =========================================================================
    # TEST 1: Recover known TEP signal at the observed parameters
    # =========================================================================
    print_status("Test 1: Recovering known TEP signal at observed parameters", "PROCESS")

    true_r_s = 2646.0
    true_alpha = 0.366

    recovered_rs = []
    recovered_alpha = []
    recovered_dchi2 = []

    for i in range(N_REALIZATIONS):
        mock = inject_signal(df, true_r_s, true_alpha, rng)
        profile, _ = build_profile(mock)
        fit = fit_profile(profile)
        if fit.get("fit_success"):
            recovered_rs.append(fit["r_s_au"])
            recovered_alpha.append(fit["alpha"])
            recovered_dchi2.append(fit["delta_chi2_vs_constant_boost"])

    recovered_rs = np.array(recovered_rs)
    recovered_alpha = np.array(recovered_alpha)
    recovered_dchi2 = np.array(recovered_dchi2)

    rs_median = float(np.median(recovered_rs))
    rs_bias = float(rs_median - true_r_s)
    rs_scatter = float(np.std(recovered_rs))
    alpha_median = float(np.median(recovered_alpha))
    alpha_bias = float(alpha_median - true_alpha)
    alpha_scatter = float(np.std(recovered_alpha))
    dchi2_median = float(np.median(recovered_dchi2))
    recovery_rate = float(np.mean(recovered_dchi2 > 10))

    print_status(
        f"R_s recovery: median = {rs_median:.0f} AU (true = {true_r_s:.0f}), "
        f"bias = {rs_bias:+.0f} AU, scatter = {rs_scatter:.0f} AU",
        "RESULT",
    )
    print_status(
        f"alpha recovery: median = {alpha_median:.4f} (true = {true_alpha:.3f}), "
        f"bias = {alpha_bias:+.4f}, scatter = {alpha_scatter:.4f}",
        "RESULT",
    )
    print_status(
        f"Detection rate (Δχ² > 10 vs const boost): {recovery_rate:.0%} of {len(recovered_rs)} realizations",
        "RESULT",
    )

    # =========================================================================
    # TEST 2: Null injection — confirm no false positive
    # =========================================================================
    print_status("Test 2: Null injection (alpha=0) — confirming no false positive", "PROCESS")

    null_rs = []
    null_alpha = []
    null_dchi2 = []

    for i in range(N_REALIZATIONS):
        mock = inject_signal(df, 2646.0, 0.0, rng)
        profile, _ = build_profile(mock)
        fit = fit_profile(profile)
        if fit.get("fit_success"):
            null_rs.append(fit["r_s_au"])
            null_alpha.append(fit["alpha"])
            null_dchi2.append(fit["delta_chi2_vs_constant_boost"])

    null_alpha = np.array(null_alpha)
    null_dchi2 = np.array(null_dchi2)
    false_positive_rate = float(np.mean(null_dchi2 > 10))
    null_alpha_median = float(np.median(null_alpha))

    print_status(
        f"Null alpha recovery: median = {null_alpha_median:.4f} (should be ~0)",
        "RESULT",
    )
    print_status(
        f"False positive rate (Δχ² > 10 vs const boost): {false_positive_rate:.0%} of {len(null_dchi2)} realizations",
        "RESULT",
    )

    # =========================================================================
    # TEST 3: Sweep across injected R_s values
    # =========================================================================
    print_status("Test 3: Recovery fidelity across injected R_s values", "PROCESS")

    sweep_rs_values = [1000, 2000, 2646, 3500, 5000, 8000]
    sweep_results = []

    for inject_rs in sweep_rs_values:
        sweep_recovered = []
        sweep_alpha_recovered = []
        sweep_dchi2_recovered = []

        for i in range(N_REALIZATIONS):
            mock = inject_signal(df, inject_rs, true_alpha, rng)
            profile, _ = build_profile(mock)
            fit = fit_profile(profile)
            if fit.get("fit_success"):
                sweep_recovered.append(fit["r_s_au"])
                sweep_alpha_recovered.append(fit["alpha"])
                sweep_dchi2_recovered.append(fit["delta_chi2_vs_constant_boost"])

        sweep_recovered = np.array(sweep_recovered)
        sweep_alpha_recovered = np.array(sweep_alpha_recovered)
        sweep_dchi2_recovered = np.array(sweep_dchi2_recovered)

        row = {
            "injected_r_s": inject_rs,
            "injected_alpha": true_alpha,
            "n_successful": len(sweep_recovered),
            "recovered_r_s_median": float(np.median(sweep_recovered)) if len(sweep_recovered) > 0 else np.nan,
            "recovered_r_s_std": float(np.std(sweep_recovered)) if len(sweep_recovered) > 0 else np.nan,
            "recovered_r_s_bias": float(np.median(sweep_recovered) - inject_rs) if len(sweep_recovered) > 0 else np.nan,
            "recovered_alpha_median": float(np.median(sweep_alpha_recovered)) if len(sweep_alpha_recovered) > 0 else np.nan,
            "recovered_alpha_std": float(np.std(sweep_alpha_recovered)) if len(sweep_alpha_recovered) > 0 else np.nan,
            "median_delta_chi2": float(np.median(sweep_dchi2_recovered)) if len(sweep_dchi2_recovered) > 0 else np.nan,
            "detection_rate": float(np.mean(sweep_dchi2_recovered > 10)) if len(sweep_dchi2_recovered) > 0 else np.nan,
        }
        sweep_results.append(row)
        print_status(
            f"R_s={inject_rs}: recovered {row['recovered_r_s_median']:.0f} AU "
            f"(bias={row['recovered_r_s_bias']:+.0f}), "
            f"detection={row['detection_rate']:.0%}",
            "RESULT",
        )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "test": "signal_recovery",
                "injected_r_s": true_r_s,
                "injected_alpha": true_alpha,
                "n_realizations": N_REALIZATIONS,
                "n_successful": len(recovered_rs),
                "recovered_r_s_median": rs_median,
                "recovered_r_s_std": rs_scatter,
                "recovered_r_s_bias": rs_bias,
                "recovered_r_s_bias_pct": float(rs_bias / true_r_s * 100),
                "recovered_alpha_median": alpha_median,
                "recovered_alpha_std": alpha_scatter,
                "recovered_alpha_bias": alpha_bias,
                "median_delta_chi2_vs_const": dchi2_median,
                "detection_rate": recovery_rate,
            },
            {
                "test": "null_injection",
                "injected_r_s": 2646.0,
                "injected_alpha": 0.0,
                "n_realizations": N_REALIZATIONS,
                "n_successful": len(null_dchi2),
                "recovered_r_s_median": float(np.median(null_rs)) if len(null_rs) > 0 else np.nan,
                "recovered_r_s_std": float(np.std(null_rs)) if len(null_rs) > 0 else np.nan,
                "recovered_r_s_bias": np.nan,
                "recovered_r_s_bias_pct": np.nan,
                "recovered_alpha_median": null_alpha_median,
                "recovered_alpha_std": float(np.std(null_alpha)),
                "recovered_alpha_bias": null_alpha_median,
                "median_delta_chi2_vs_const": float(np.median(null_dchi2)),
                "detection_rate": false_positive_rate,
            },
        ]
    )
    summary.to_csv(outputs_dir / "injection_recovery_summary.csv", index=False)

    pd.DataFrame(sweep_results).to_csv(
        outputs_dir / "injection_recovery_sweep.csv", index=False
    )

    print_status(
        f"Saved injection-recovery results to {outputs_dir / 'injection_recovery_summary.csv'}",
        "SUCCESS",
    )
    print_status(
        f"Saved sweep results to {outputs_dir / 'injection_recovery_sweep.csv'}",
        "SUCCESS",
    )


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger(
        "step_008", str(log_dir / "step_008_injection_recovery.log")
    )
    set_step_logger(logger)

    run_injection_recovery()
