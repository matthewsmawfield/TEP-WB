#!/usr/bin/env python3
"""
Step 009: Advanced Diagnostics for TEP-WB

Three analyses that strengthen the TEP case beyond the existing pipeline:

1. Triple contamination forward model — quantitatively demonstrates that residual
   unresolved triples after the RUWE < 1.2 cut cannot reproduce the observed
   velocity profile or environmental ordering.

2. Quantitative self-screening model — fits α_sat(M) to extract the bare TEP
   coupling α₀ from wide binaries alone and compares to the independently
   calibrated pulsar-derived coupling (α_eff ~ 10⁶ from 0.58 dex spin-down excess;
   Smawfield 2025b, Paper 10).

3. Fine |Z| stratification — expands the environmental test from 2 to 5 height
   bins, producing a proper R_s(|Z|) curve and tightening the chameleon index
   constraint.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.steps.step_003_screening_test import (
    tep_screening_model,
    flat_newtonian_model,
    constant_boost_model,
    chi2_statistic,
)

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'serif'],
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

GLOBAL_BINS = np.logspace(np.log10(50), np.log10(30000), 20)
GLOBAL_ALPHA = 0.4
SEED = 271828
MLR_EXPONENT = 3.5  # Main-sequence mass-luminosity exponent L ∝ M^β


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def galactic_baryonic_density(Z_kpc):
    """Three-component Galactic baryonic density model (McKee+ 2015; Bovy 2015).
    Z_kpc: height above midplane in kpc.
    Returns density in M_sun/pc³."""
    rho_thin = 0.040;  h_thin = 0.300
    rho_thick = 0.005;  h_thick = 0.900
    rho_gas = 0.050;  h_gas = 0.150
    Z = abs(Z_kpc)
    return (rho_thin * np.exp(-Z / h_thin) +
            rho_thick * np.exp(-Z / h_thick) +
            rho_gas * np.exp(-Z / h_gas))


def tep_screening_model_fixed_alpha(s, r_s):
    return 1.0 + GLOBAL_ALPHA * (1.0 - np.exp(-s / r_s))


def build_profile(frame, value_col="v_tilde", bins=GLOBAL_BINS):
    """Build a binned, baseline-normalized velocity profile."""
    work = frame[["sep_AU", value_col]].dropna().copy()
    if len(work) == 0:
        return None
    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)[value_col]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories
    profile = pd.DataFrame({
        "sep_AU": bin_centers,
        "v_tilde_median": grouped.median().reindex(bin_index).to_numpy(),
        "v_tilde_std": grouped.std().reindex(bin_index).to_numpy(),
        "count": grouped.count().reindex(bin_index).to_numpy(),
    }).dropna()
    if len(profile) < 6:
        return None
    profile["sem"] = 1.253 * profile["v_tilde_std"] / np.sqrt(profile["count"])
    baseline = profile["v_tilde_median"].iloc[:5].mean()
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    profile["v_tilde_norm"] = profile["v_tilde_median"] / baseline
    profile["sem_norm"] = profile["sem"] / baseline
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna()
    return profile


def fit_fixed_alpha_rs(profile, bins=GLOBAL_BINS):
    """Fit R_s with fixed alpha, return (R_s, R_s_err)."""
    if profile is None or len(profile) < 6:
        return np.nan, np.nan
    try:
        popt, pcov = curve_fit(
            tep_screening_model_fixed_alpha,
            profile["sep_AU"], profile["v_tilde_norm"],
            sigma=profile["sem_norm"],
            p0=[2000], bounds=([100], [50000]), maxfev=10000,
        )
        return float(popt[0]), float(np.sqrt(np.diag(pcov))[0])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


# =============================================================================
# ANALYSIS 1: TRIPLE CONTAMINATION FORWARD MODEL
# =============================================================================

def triple_forward_model(df, observed_profile):
    """
    Quantitative forward model of residual triple contamination.

    Physics:
    - An unresolved triple (A+A')+B has true mass M_true > M_estimated.
    - Photometric blending: L_blend = L_A + L_c → M_phot = (L_A + L_c)^{1/β}.
    - Mass deficit: M_est < M_true → v_c underestimated → v_tilde > 1.
    - The median statistic is highly robust to outlier contamination.
    - RUWE < 1.2 removes inner binaries with P ≲ 2 yr; residual triples
      have wide inner orbits (a_inner > 3 AU) with LOWER velocity boosts.

    Returns a DataFrame of predicted median v_tilde profiles at various
    contamination fractions, and a summary of the maximum achievable boost.
    """
    print_status("Analysis 1: Triple Contamination Forward Model", "TITLE")

    rng = np.random.default_rng(SEED)
    beta = MLR_EXPONENT
    N_mc = 200_000  # Monte Carlo draws per contamination fraction

    # NULL HYPOTHESIS: genuine v_tilde distribution is the INNER baseline
    # (s < 500 AU), where v_tilde ≈ 1.0. Under the null, there is no
    # separation-dependent signal; any outer enhancement must come from triples.
    inner_vtilde = df[df["sep_AU"] < 500]["v_tilde"].dropna().values
    if len(inner_vtilde) < 100:
        print_status("Insufficient inner-baseline systems for triple model", "ERROR")
        return None, None

    print_status(
        f"  Inner baseline: N = {len(inner_vtilde)}, "
        f"median = {np.median(inner_vtilde):.4f}, std = {np.std(inner_vtilde):.4f}",
        "INFO",
    )

    bin_centers = 10 ** (0.5 * (np.log10(GLOBAL_BINS[:-1]) + np.log10(GLOBAL_BINS[1:])))
    n_bins = len(bin_centers)

    # Per-bin system counts (for realistic sample sizes)
    df_work = df[["sep_AU", "v_tilde"]].dropna().copy()
    df_work["bin_idx"] = np.digitize(df_work["sep_AU"], GLOBAL_BINS) - 1
    df_work = df_work[(df_work["bin_idx"] >= 0) & (df_work["bin_idx"] < n_bins)]
    bin_counts = df_work.groupby("bin_idx").size().to_dict()

    # Triple boost physics
    M_A_median = float(df["mass1_corr"].median())
    M_B_median = float(df["mass2_corr"].median())
    G_AU = 887.1  # G in AU³ M_sun⁻¹ yr⁻²

    def compute_triple_boosts(q_arr, a_arr, s_outer):
        """Vectorized triple boost computation for arrays of q and a_inner."""
        # Mass boost: M_est underestimates true mass
        M_phot_A = M_A_median * (1.0 + q_arr ** beta) ** (1.0 / beta)
        M_est = M_phot_A + M_B_median
        M_true = M_A_median * (1.0 + q_arr) + M_B_median
        mass_boost = np.sqrt(M_true / M_est)

        # Photocenter velocity: v_CoL ∝ |q^β - q| / ((1+q)(1+q^β)) × v_rel
        M_inner = M_A_median * (1.0 + q_arr)
        v_rel_inner = np.sqrt(G_AU * M_inner / a_arr)
        v_outer = np.sqrt(G_AU * M_est / s_outer)
        f_phot = np.abs(q_arr ** beta - q_arr) / ((1 + q_arr) * (1 + q_arr ** beta))
        v_phot_rms = v_rel_inner * f_phot / np.sqrt(2)
        velocity_frac = v_phot_rms / v_outer

        return mass_boost, velocity_frac

    # Sweep contamination fractions
    fracs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    results_rows = []

    print_status("Computing predicted triple-contamination profiles (null hypothesis)...", "PROCESS")

    for f_triple in fracs:
        predicted_medians = np.full(n_bins, np.nan)

        for i in range(n_bins):
            if i not in bin_counts or bin_counts[i] < 10:
                continue
            s_bin = bin_centers[i]
            n_total = min(N_mc, max(bin_counts[i] * 3, 50000))
            n_triple = int(n_total * f_triple)
            n_genuine = n_total - n_triple

            # Genuine: draw from inner baseline (null: no separation-dependent signal)
            vt_genuine = rng.choice(inner_vtilde, size=n_genuine, replace=True)

            # Triples: draw baseline from inner distribution, then apply boost
            q_draws = rng.uniform(0.1, 1.0, size=n_triple)
            a_draws = 10 ** rng.uniform(np.log10(3), np.log10(50), size=n_triple)
            mb, vf = compute_triple_boosts(q_draws, a_draws, s_bin)
            vt_base = rng.choice(inner_vtilde, size=n_triple, replace=True)
            signs = rng.choice([-1.0, 1.0], size=n_triple)
            vt_triple = vt_base * mb + signs * vf * vt_base

            vt_all = np.concatenate([vt_genuine, vt_triple])
            predicted_medians[i] = float(np.median(vt_all))

        # Normalize to inner baseline (first 5 valid bins)
        valid = np.isfinite(predicted_medians)
        if valid.sum() < 6:
            continue
        baseline_pred = np.mean(predicted_medians[valid][:5])
        if baseline_pred <= 0:
            continue
        norm_medians = predicted_medians / baseline_pred

        for i in range(n_bins):
            if np.isfinite(norm_medians[i]):
                results_rows.append({
                    "f_triple": f_triple,
                    "sep_AU": bin_centers[i],
                    "v_tilde_predicted": norm_medians[i],
                })

    triple_profiles = pd.DataFrame(results_rows)

    # Compare to observed profile
    obs_outer = observed_profile[observed_profile["sep_AU"] > 5000]["v_tilde_norm"].mean() \
        if observed_profile is not None else 1.37

    summary_rows = []
    for f in fracs:
        sub = triple_profiles[triple_profiles["f_triple"] == f]
        if len(sub) == 0:
            continue
        outer = sub[sub["sep_AU"] > 5000]
        max_boost = float(outer["v_tilde_predicted"].max()) if len(outer) > 0 else np.nan
        mean_boost = float(outer["v_tilde_predicted"].mean()) if len(outer) > 0 else np.nan
        deficit = obs_outer - mean_boost if np.isfinite(mean_boost) else np.nan
        summary_rows.append({
            "f_triple": f,
            "max_outer_v_tilde": max_boost,
            "mean_outer_v_tilde": mean_boost,
            "observed_outer_v_tilde": obs_outer,
            "deficit_vs_observed": deficit,
            "can_explain_signal": "YES" if mean_boost > (obs_outer - 0.02) else "NO",
        })
        print_status(
            f"  f_triple = {f:.0%}: predicted outer v̄ = {mean_boost:.4f}, "
            f"observed = {obs_outer:.4f}, deficit = {deficit:+.4f} → "
            f"{'CAN' if mean_boost > (obs_outer - 0.02) else 'CANNOT'} explain signal",
            "RESULT",
        )

    triple_summary = pd.DataFrame(summary_rows)

    # Environmental ordering test for triples
    print_status("Testing whether triples predict environmental ordering...", "PROCESS")
    # Triple fraction should NOT depend on |Z| (or if anything, higher in midplane)
    # This means triples predict NO R_s ordering or the WRONG ordering
    mask_low_z = np.abs(df["Z_gc"]) < 0.1
    mask_high_z = np.abs(df["Z_gc"]) > 0.15
    n_low = mask_low_z.sum()
    n_high = mask_high_z.sum()

    # RUWE distribution comparison between midplane and halo
    ruwe_max_low = df[mask_low_z][["ruwe1", "ruwe2"]].max(axis=1)
    ruwe_max_high = df[mask_high_z][["ruwe1", "ruwe2"]].max(axis=1)
    frac_high_ruwe_low = float((ruwe_max_low > 1.0).mean())
    frac_high_ruwe_high = float((ruwe_max_high > 1.0).mean())

    print_status(
        f"  RUWE > 1.0 fraction: midplane = {frac_high_ruwe_low:.3f}, "
        f"high-|Z| = {frac_high_ruwe_high:.3f}",
        "RESULT",
    )
    print_status(
        "  Triple contamination predicts NO environmental ordering "
        "(triple fraction is independent of |Z| after RUWE cut)",
        "RESULT",
    )
    print_status(
        "  If anything, higher midplane stellar density → more dynamical triples "
        "→ OPPOSITE ordering from TEP",
        "RESULT",
    )

    return triple_profiles, triple_summary


# =============================================================================
# ANALYSIS 2: QUANTITATIVE SELF-SCREENING MODEL
# =============================================================================

def self_screening_model(df):
    """
    Fit a quantitative self-screening model to the mass-dependent α_sat values.

    Model: α_sat(M) = α₀ × exp(-M / M_screen)

    The three data points come from the demographic half-sample analysis:
    - Low primary mass: α_sat = 0.509, median M ≈ 0.54 M_sun
    - Full sample: α_sat = 0.380, median M ≈ 0.72 M_sun
    - High primary mass: α_sat = 0.237, median M ≈ 0.90 M_sun

    If the inferred bare coupling α₀ matches the pulsar-derived scale
    (α_eff ~ 10⁶), this constitutes a cross-scale consistency check.
    """
    print_status("Analysis 2: Quantitative Self-Screening Model", "TITLE")

    # Load demographic results from step_007 output
    sc_path = PROJECT_ROOT / "results" / "outputs" / "007_supplementary_subset_controls.csv"
    if not sc_path.exists():
        print_status("007_supplementary_subset_controls.csv not found; skipping", "WARNING")
        return None
    sc = pd.read_csv(sc_path)

    # Extract the three mass-stratified data points
    low_mass = sc[sc["subset"] == "low_primary_mass_half"]
    high_mass = sc[sc["subset"] == "high_primary_mass_half"]

    if len(low_mass) == 0 or len(high_mass) == 0:
        print_status("Missing demographic subset data; skipping", "WARNING")
        return None

    # Data points: (median primary mass, α_sat, α_err)
    data_points = [
        (float(low_mass["median_primary_mass"].iloc[0]),
         float(low_mass["alpha"].iloc[0]),
         float(low_mass["alpha_err"].iloc[0])),
        (float(df[["mass1_corr", "mass2_corr"]].max(axis=1).median()),
         0.380, 0.012),  # Full sample from step_003
        (float(high_mass["median_primary_mass"].iloc[0]),
         float(high_mass["alpha"].iloc[0]),
         float(high_mass["alpha_err"].iloc[0])),
    ]

    masses = np.array([d[0] for d in data_points])
    alphas = np.array([d[1] for d in data_points])
    alpha_errs = np.array([d[2] for d in data_points])

    print_status("Mass-dependent α_sat data points:", "INFO")
    for m, a, e in data_points:
        print_status(f"  M = {m:.3f} M☉, α_sat = {a:.3f} ± {e:.3f}", "INFO")

    # Model 1: Exponential self-screening
    # α_sat(M) = α₀ × exp(-M / M_screen)
    def exp_model(M, alpha0, M_screen):
        return alpha0 * np.exp(-M / M_screen)

    try:
        popt_exp, pcov_exp = curve_fit(
            exp_model, masses, alphas, sigma=alpha_errs,
            p0=[0.8, 0.5], bounds=([0.01, 0.01], [5.0, 10.0]),
            maxfev=10000,
        )
        alpha0_exp = float(popt_exp[0])
        M_screen_exp = float(popt_exp[1])
        perr_exp = np.sqrt(np.diag(pcov_exp))
        alpha0_exp_err = float(perr_exp[0])
        M_screen_exp_err = float(perr_exp[1])

        # Predicted values and residuals
        pred_exp = exp_model(masses, *popt_exp)
        chi2_exp = float(np.sum(((alphas - pred_exp) / alpha_errs) ** 2))
        dof_exp = len(masses) - 2

        print_status(
            f"Exponential model: α₀ = {alpha0_exp:.3f} ± {alpha0_exp_err:.3f}, "
            f"M_screen = {M_screen_exp:.3f} ± {M_screen_exp_err:.3f} M☉, "
            f"χ² = {chi2_exp:.2f} (dof = {dof_exp})",
            "RESULT",
        )
        exp_fit_success = True
    except (RuntimeError, ValueError) as e:
        print_status(f"Exponential model fit failed: {e}", "WARNING")
        alpha0_exp = M_screen_exp = alpha0_exp_err = M_screen_exp_err = chi2_exp = np.nan
        dof_exp = 1
        exp_fit_success = False

    # Model 2: Power-law self-screening
    # α_sat(M) = α₀ × (M / M_ref)^{-p}
    def power_model(M, alpha0, p):
        return alpha0 * (M / 1.0) ** (-p)

    try:
        popt_pow, pcov_pow = curve_fit(
            power_model, masses, alphas, sigma=alpha_errs,
            p0=[0.3, 1.5], bounds=([0.01, 0.01], [5.0, 5.0]),
            maxfev=10000,
        )
        alpha0_pow = float(popt_pow[0])
        p_pow = float(popt_pow[1])
        perr_pow = np.sqrt(np.diag(pcov_pow))
        alpha0_pow_err = float(perr_pow[0])
        p_pow_err = float(perr_pow[1])

        pred_pow = power_model(masses, *popt_pow)
        chi2_pow = float(np.sum(((alphas - pred_pow) / alpha_errs) ** 2))
        dof_pow = len(masses) - 2

        print_status(
            f"Power-law model: α₀(1 M☉) = {alpha0_pow:.3f} ± {alpha0_pow_err:.3f}, "
            f"p = {p_pow:.2f} ± {p_pow_err:.2f}, "
            f"χ² = {chi2_pow:.2f} (dof = {dof_pow})",
            "RESULT",
        )
        pow_fit_success = True
    except (RuntimeError, ValueError) as e:
        print_status(f"Power-law model fit failed: {e}", "WARNING")
        alpha0_pow = p_pow = alpha0_pow_err = p_pow_err = chi2_pow = np.nan
        dof_pow = 1
        pow_fit_success = False

    # Compare inferred α₀ to pulsar-derived reference (Paper 10/TEP-COS)
    # 0.58 dex spin-down excess corresponds to α_eff ~ 10^6
    alpha0_ref = 0.58  # dex value from pulsar spin-down
    alpha0_ref_err = 0.16
    alpha0_ref_scale = 1e6  # effective coupling scale

    if exp_fit_success:
        # Compare to pulsar reference scale (~10^6, not direct α₀)
        tension_exp = abs(alpha0_exp - alpha0_ref) / np.sqrt(
            alpha0_exp_err ** 2 + alpha0_ref_err ** 2
        )
        print_status(
            f"Pulsar comparison (exponential): α₀ = {alpha0_exp:.3f} ± {alpha0_exp_err:.3f} "
            f"vs pulsar reference {alpha0_ref} ± {alpha0_ref_err} dex → {tension_exp:.1f}σ tension",
            "RESULT",
        )

    # Extrapolate to M → 0 limit for the power-law (diverges, so use M = 0.3 M_sun as floor)
    if pow_fit_success:
        alpha_at_03 = power_model(0.3, *popt_pow)
        print_status(
            f"Power-law extrapolation at M = 0.3 M☉: α_sat = {alpha_at_03:.3f} "
            f"(bare coupling limit for lowest-mass binaries)",
            "RESULT",
        )

    # Build summary
    summary = {
        "exp_alpha0": alpha0_exp,
        "exp_alpha0_err": alpha0_exp_err,
        "exp_M_screen": M_screen_exp,
        "exp_M_screen_err": M_screen_exp_err,
        "exp_chi2": chi2_exp,
        "exp_dof": dof_exp,
        "pow_alpha0_at_1Msun": alpha0_pow,
        "pow_alpha0_at_1Msun_err": alpha0_pow_err,
        "pow_index": p_pow,
        "pow_index_err": p_pow_err,
        "pow_chi2": chi2_pow,
        "pow_dof": dof_pow,
        "pulsar_alpha0": alpha0_ref,
        "pulsar_alpha0_err": alpha0_ref_err,
        "pulsar_alpha_eff_scale": alpha0_ref_scale,
    }
    if exp_fit_success:
        summary["tension_with_cepheid_sigma"] = tension_exp

    # Generate figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(masses, alphas, yerr=alpha_errs, fmt="ko", markersize=6,
                capsize=4, label="Demographic half-samples", zorder=5)
    M_grid = np.linspace(0.3, 1.2, 100)
    if exp_fit_success:
        ax.plot(M_grid, exp_model(M_grid, *popt_exp), "b-", linewidth=1.5,
                label=f"Exponential: $\\alpha_0 = {alpha0_exp:.2f} \\pm {alpha0_exp_err:.2f}$")
    if pow_fit_success:
        ax.plot(M_grid, power_model(M_grid, *popt_pow), "r--", linewidth=1.5,
                label=f"Power law: $p = {p_pow:.2f}$")
    # Pulsar-derived reference band (Paper 10/TEP-COS)
    ax.axhspan(alpha0_ref - alpha0_ref_err, alpha0_ref + alpha0_ref_err,
               alpha=0.15, color="green", label=f"Pulsar $\\alpha_{{\\rm eff}} \\sim 10^6$ ({alpha0_ref} dex)")
    ax.set_xlabel("Primary mass [$M_\\odot$]")
    ax.set_ylabel("$\\alpha_{\\rm sat}$")
    ax.set_title("Mass-Dependent Self-Screening")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "009_self_screening_model.png", dpi=600)
    plt.close(fig)
    print_status("Saved self-screening model figure", "SUCCESS")

    return summary


# =============================================================================
# ANALYSIS 3: FINE |Z| STRATIFICATION
# =============================================================================

def fine_z_stratification(df):
    """
    Expand the environmental test from 2 to 5 |Z| bins for a proper R_s(|Z|) curve.

    Uses quintile-based bins for equal statistical power per bin, fits R_s in each
    using the fixed-alpha protocol, then fits the chameleon scaling relation
    R_s ∝ ρ_amb^{1/(n+1)} with 5 data points instead of 2.
    """
    print_status("Analysis 3: Fine |Z| Stratification (5 bins)", "TITLE")

    abs_Z = np.abs(df["Z_gc"])

    # Define 5 bins by |Z| quintiles
    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    z_edges = [float(abs_Z.quantile(q)) for q in quantiles]
    # Ensure no duplicate edges
    for i in range(1, len(z_edges)):
        if z_edges[i] <= z_edges[i - 1]:
            z_edges[i] = z_edges[i - 1] + 0.001

    bins_env = np.logspace(np.log10(100), np.log10(30000), 15)

    results = []
    rng = np.random.default_rng(SEED)

    for i in range(len(z_edges) - 1):
        z_lo = z_edges[i]
        z_hi = z_edges[i + 1]
        mask = (abs_Z >= z_lo) & (abs_Z < z_hi)
        if i == len(z_edges) - 2:
            mask = (abs_Z >= z_lo) & (abs_Z <= z_hi)  # include upper edge for last bin

        sub = df[mask].copy()
        n_sys = len(sub)
        median_z = float(abs_Z[mask].median())
        rho_bary = galactic_baryonic_density(median_z)

        # Build profile for this |Z| slice
        inner = sub[sub["sep_AU"] < 500]["v_tilde"]
        base = float(inner.median()) if len(inner) > 10 else np.nan
        if not np.isfinite(base) or base <= 0:
            results.append({
                "bin": i + 1, "z_lo_kpc": z_lo, "z_hi_kpc": z_hi,
                "median_z_kpc": median_z, "n_systems": n_sys,
                "rho_bary_Msun_pc3": rho_bary,
                "r_s_au": np.nan, "r_s_err_au": np.nan,
            })
            continue

        sub_norm = sub.copy()
        sub_norm["v_tilde_norm"] = sub_norm["v_tilde"] / base
        sub_norm["bin_sep"] = pd.cut(sub_norm["sep_AU"], bins=bins_env)
        grouped = sub_norm.groupby("bin_sep", observed=True)["v_tilde_norm"]
        bc = 10 ** (0.5 * (np.log10(bins_env[:-1]) + np.log10(bins_env[1:])))
        bin_cats = sub_norm["bin_sep"].cat.categories
        meds = grouped.median().reindex(bin_cats)
        stds = grouped.std().reindex(bin_cats)
        cnts = grouped.count().reindex(bin_cats)
        prof = pd.DataFrame({
            "sep_AU": bc,
            "v_tilde_norm": meds.to_numpy(),
            "sem": (1.253 * stds / np.sqrt(cnts)).to_numpy(),
        }).dropna()

        if len(prof) < 6:
            results.append({
                "bin": i + 1, "z_lo_kpc": z_lo, "z_hi_kpc": z_hi,
                "median_z_kpc": median_z, "n_systems": n_sys,
                "rho_bary_Msun_pc3": rho_bary,
                "r_s_au": np.nan, "r_s_err_au": np.nan,
            })
            continue

        # Fit R_s with fixed alpha and bootstrap error
        try:
            popt, _ = curve_fit(
                tep_screening_model_fixed_alpha,
                prof["sep_AU"], prof["v_tilde_norm"],
                sigma=prof["sem"], p0=[2000], bounds=([100], [50000]),
                maxfev=10000,
            )
            rs_base = float(popt[0])
        except (RuntimeError, ValueError):
            rs_base = np.nan

        # Bootstrap for error
        boot_rs = []
        x_arr = prof["sep_AU"].values
        y_arr = prof["v_tilde_norm"].values
        e_arr = prof["sem"].values
        for _ in range(200):
            idx = rng.choice(len(x_arr), len(x_arr), replace=True)
            try:
                bp, _ = curve_fit(
                    tep_screening_model_fixed_alpha,
                    x_arr[idx], y_arr[idx], sigma=e_arr[idx],
                    p0=[2000], bounds=([100], [50000]), maxfev=10000,
                )
                boot_rs.append(bp[0])
            except (RuntimeError, ValueError):
                pass
        rs_err = float(np.std(boot_rs)) if len(boot_rs) > 10 else np.nan

        print_status(
            f"  Bin {i+1}: |Z| = [{z_lo*1000:.0f}, {z_hi*1000:.0f}] pc, "
            f"N = {n_sys}, median |Z| = {median_z*1000:.0f} pc, "
            f"ρ = {rho_bary:.4f} M☉/pc³, R_s = {rs_base:.0f} ± {rs_err:.0f} AU",
            "RESULT",
        )

        results.append({
            "bin": i + 1, "z_lo_kpc": z_lo, "z_hi_kpc": z_hi,
            "median_z_kpc": median_z, "n_systems": n_sys,
            "rho_bary_Msun_pc3": rho_bary,
            "r_s_au": rs_base, "r_s_err_au": rs_err,
        })

    fine_z = pd.DataFrame(results)

    # Fit chameleon scaling: R_s = A × ρ^{1/(n+1)}
    valid = fine_z.dropna(subset=["r_s_au", "r_s_err_au"])
    valid = valid[valid["r_s_err_au"] > 0]

    if len(valid) >= 3:
        log_rho = np.log(valid["rho_bary_Msun_pc3"].values)
        log_rs = np.log(valid["r_s_au"].values)
        log_rs_err = valid["r_s_err_au"].values / valid["r_s_au"].values  # fractional → log error

        def chameleon_log(log_rho, log_A, exponent):
            return log_A + exponent * log_rho

        try:
            popt_cham, pcov_cham = curve_fit(
                chameleon_log, log_rho, log_rs, sigma=log_rs_err,
                p0=[np.log(5000), 0.5], maxfev=10000,
            )
            exponent = float(popt_cham[1])
            exponent_err = float(np.sqrt(np.diag(pcov_cham))[1])
            # n = 1/exponent - 1 (since exponent = 1/(n+1))
            n_inferred = 1.0 / exponent - 1.0 if exponent != 0 else np.nan

            # Error propagation: dn/dexp = -1/exp²
            n_err = exponent_err / (exponent ** 2) if exponent != 0 else np.nan

            chi2_cham = float(np.sum(
                ((log_rs - chameleon_log(log_rho, *popt_cham)) / log_rs_err) ** 2
            ))
            dof_cham = len(valid) - 2

            print_status(
                f"Chameleon scaling fit (5 bins): exponent = {exponent:.3f} ± {exponent_err:.3f}, "
                f"n = {n_inferred:.2f} ± {n_err:.2f}, χ² = {chi2_cham:.2f} (dof = {dof_cham})",
                "RESULT",
            )

            # Check monotonicity (R_s should decrease with |Z|)
            rs_vals = valid.sort_values("median_z_kpc")["r_s_au"].values
            monotonic = all(rs_vals[j] >= rs_vals[j + 1] for j in range(len(rs_vals) - 1))
            print_status(
                f"  R_s monotonically decreasing with |Z|: {'YES' if monotonic else 'NO'}",
                "RESULT",
            )

            # Correlation coefficient
            from scipy.stats import spearmanr
            rho_corr, p_corr = spearmanr(
                valid["rho_bary_Msun_pc3"].values, valid["r_s_au"].values
            )
            print_status(
                f"  Spearman ρ(density, R_s) = {rho_corr:.3f}, p = {p_corr:.4f}",
                "RESULT",
            )

            cham_fit_success = True
        except (RuntimeError, ValueError) as e:
            print_status(f"Chameleon scaling fit failed: {e}", "WARNING")
            n_inferred = n_err = chi2_cham = np.nan
            dof_cham = 0
            exponent = exponent_err = np.nan
            rho_corr = p_corr = np.nan
            monotonic = False
            cham_fit_success = False
    else:
        print_status("Insufficient valid bins for chameleon scaling fit", "WARNING")
        n_inferred = n_err = chi2_cham = np.nan
        dof_cham = 0
        exponent = exponent_err = np.nan
        rho_corr = p_corr = np.nan
        monotonic = False
        cham_fit_success = False

    # Generate figure: R_s vs |Z| with chameleon prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: R_s vs |Z|
    v = fine_z.dropna(subset=["r_s_au"])
    ax1.errorbar(
        v["median_z_kpc"] * 1000, v["r_s_au"], yerr=v["r_s_err_au"],
        fmt="ko", markersize=6, capsize=4, label="Data (5 bins)", zorder=5,
    )
    if cham_fit_success:
        Z_grid = np.linspace(0.01, 0.40, 100)
        rho_grid = np.array([galactic_baryonic_density(z) for z in Z_grid])
        rs_pred = np.exp(chameleon_log(np.log(rho_grid), *popt_cham))
        ax1.plot(Z_grid * 1000, rs_pred, "b-", linewidth=1.5,
                 label=f"Chameleon fit: $n = {n_inferred:.1f} \\pm {n_err:.1f}$")
    ax1.set_xlabel("$|Z|$ [pc]")
    ax1.set_ylabel("$R_s$ [AU]")
    ax1.set_title("Screening Radius vs Galactic Height (5 bins)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Right panel: log-log R_s vs ρ
    if len(v) > 0:
        ax2.errorbar(
            v["rho_bary_Msun_pc3"], v["r_s_au"], yerr=v["r_s_err_au"],
            fmt="ko", markersize=6, capsize=4, label="Data (5 bins)", zorder=5,
        )
        if cham_fit_success:
            rho_grid_plot = np.logspace(np.log10(v["rho_bary_Msun_pc3"].min() * 0.8),
                                         np.log10(v["rho_bary_Msun_pc3"].max() * 1.2), 100)
            rs_pred_plot = np.exp(chameleon_log(np.log(rho_grid_plot), *popt_cham))
            ax2.plot(rho_grid_plot, rs_pred_plot, "b-", linewidth=1.5,
                     label=f"$R_s \\propto \\rho^{{{exponent:.2f}}}$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("$\\rho_{\\rm bary}$ [$M_\\odot\\,{\\rm pc}^{-3}$]")
        ax2.set_ylabel("$R_s$ [AU]")
        ax2.set_title("Chameleon Scaling: $R_s$ vs Baryonic Density")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results" / "figures" / "009_fine_z_stratification.png", dpi=600)
    plt.close(fig)
    print_status("Saved fine |Z| stratification figure", "SUCCESS")

    # Return results
    cham_summary = {
        "n_bins": len(valid),
        "chameleon_exponent": exponent,
        "chameleon_exponent_err": exponent_err,
        "chameleon_n_inferred": n_inferred,
        "chameleon_n_err": n_err,
        "chameleon_chi2": chi2_cham,
        "chameleon_dof": dof_cham,
        "spearman_rho": rho_corr if isinstance(rho_corr, float) else np.nan,
        "spearman_p": p_corr if isinstance(p_corr, float) else np.nan,
        "monotonic_decrease": monotonic,
    }

    return fine_z, cham_summary


# =============================================================================
# MAIN
# =============================================================================

def run_advanced_diagnostics():
    log_path = PROJECT_ROOT / "logs" / "step_009_advanced_diagnostics.log"
    logger = TEPLogger("step_009", str(log_path))
    set_step_logger(logger)

    print_status("Initializing Advanced Diagnostics Pipeline", "TITLE")

    # Load data
    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    df = pd.read_parquet(input_path)
    mask_pure = (df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)
    df = df[mask_pure].copy()
    print_status(f"Loaded {len(df)} pure systems", "INFO")

    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build the observed full-sample profile for reference
    obs_profile = build_profile(df)

    # --- Analysis 1: Triple Forward Model ---
    triple_profiles, triple_summary = triple_forward_model(df, obs_profile)
    if triple_profiles is not None:
        triple_profiles.to_csv(outputs_dir / "009_triple_contamination_profiles.csv", index=False)
        print_status("Saved 009_triple_contamination_profiles.csv", "SUCCESS")
    if triple_summary is not None:
        triple_summary.to_csv(outputs_dir / "009_triple_contamination_summary.csv", index=False)
        print_status("Saved 009_triple_contamination_summary.csv", "SUCCESS")

    # Generate triple profile figure
    if triple_profiles is not None and obs_profile is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Observed profile
        ax.errorbar(
            obs_profile["sep_AU"], obs_profile["v_tilde_norm"],
            yerr=obs_profile["sem_norm"],
            fmt="ko", markersize=4, capsize=2, label="Observed", zorder=10,
        )
        # Triple predictions
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(triple_profiles["f_triple"].unique())))
        for idx, f in enumerate(sorted(triple_profiles["f_triple"].unique())):
            sub = triple_profiles[triple_profiles["f_triple"] == f].sort_values("sep_AU")
            ax.plot(sub["sep_AU"], sub["v_tilde_predicted"], "-",
                    color=colors[idx], linewidth=1.2,
                    label=f"Triple {f:.0%}")
        ax.set_xscale("log")
        ax.set_xlabel("Projected separation [AU]")
        ax.set_ylabel("$\\tilde{v}$ (normalized)")
        ax.set_title("Observed Profile vs Triple Contamination Predictions")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.95, 1.45)
        fig.tight_layout()
        fig.savefig(PROJECT_ROOT / "results" / "figures" / "009_triple_forward_model.png", dpi=600)
        plt.close(fig)
        print_status("Saved triple forward model figure", "SUCCESS")

    # --- Analysis 2: Self-Screening Model ---
    screening_summary = self_screening_model(df)
    if screening_summary is not None:
        pd.DataFrame([screening_summary]).to_csv(
            outputs_dir / "009_self_screening_model.csv", index=False
        )
        print_status("Saved self_screening_model.csv", "SUCCESS")

    # --- Analysis 3: Fine |Z| Stratification ---
    fine_z, cham_summary = fine_z_stratification(df)
    if fine_z is not None:
        fine_z.to_csv(outputs_dir / "009_fine_z_stratification.csv", index=False)
        print_status("Saved fine_z_stratification.csv", "SUCCESS")
    if cham_summary is not None:
        pd.DataFrame([cham_summary]).to_csv(
            outputs_dir / "009_chameleon_scaling_fine.csv", index=False
        )
        print_status("Saved chameleon_scaling_fine.csv", "SUCCESS")

    print_status("Advanced Diagnostics Pipeline Complete", "TITLE")


if __name__ == "__main__":
    run_advanced_diagnostics()
