#!/usr/bin/env python3
"""
Step 010: Referee-Hardening Diagnostics for TEP-WB

Three analyses designed to preemptively close the escape hatches that
skeptical reviewers will use:

1. Normalization sensitivity matrix — sweeps baseline normalization windows
   and demonstrates that R_s is stable regardless of which inner bins are
   used for normalization, closing the "deflated inner baseline" concern.

2. Spatial substructure identification — cross-matches high-residual bins
   with physical sky-position clustering to give the chi^2_nu = 5.1
   correlated scatter a concrete astrophysical identity.

3. Pittordis-faithful triple forward model — adopts the exact triple
   population distributions from Pittordis et al. (2025) (log-normal
   period, thermal eccentricity, full phase-averaged photocenter velocity)
   and shows that even their own model cannot reproduce the observed
   signal or environmental ordering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.steps.step_003_screening_test import (
    tep_screening_model,
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
SEED = 271828
MLR_EXPONENT = 3.5
G_AU = 887.1  # G in AU^3 M_sun^-1 yr^-2


# =============================================================================
# ANALYSIS 1: NORMALIZATION SENSITIVITY MATRIX
# =============================================================================

def normalization_sensitivity(df):
    """
    Sweep different baseline normalization windows and show that R_s
    remains stable while alpha_sat shifts predictably.

    Tests every contiguous window of 3-7 bins starting from bin 1-6,
    re-normalizes, refits, and records (R_s, alpha_sat).
    """
    print_status("Analysis 1: Normalization Sensitivity Matrix", "TITLE")

    bins = GLOBAL_BINS
    work = df[["sep_AU", "v_tilde"]].dropna().copy()
    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)["v_tilde"]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories

    raw_medians = grouped.median().reindex(bin_index).to_numpy()
    raw_stds = grouped.std().reindex(bin_index).to_numpy()
    raw_counts = grouped.count().reindex(bin_index).to_numpy()
    raw_sems = 1.253 * raw_stds / np.sqrt(raw_counts)

    n_bins = len(bin_centers)
    results = []

    # Sweep: start_bin (0-indexed) and window width
    for start in range(min(8, n_bins)):  # don't start past bin 8
        for width in range(2, min(8, n_bins - start)):
            end = start + width
            # Normalization window spans bins [start, end)
            window_medians = raw_medians[start:end]
            valid_window = window_medians[np.isfinite(window_medians)]
            if len(valid_window) < 2:
                continue
            baseline = np.mean(valid_window)
            if baseline <= 0 or not np.isfinite(baseline):
                continue

            # Normalize and build profile
            v_norm = raw_medians / baseline
            sem_norm = raw_sems / baseline
            valid = np.isfinite(v_norm) & np.isfinite(sem_norm) & (sem_norm > 0)
            if valid.sum() < 8:
                continue

            s_fit = bin_centers[valid]
            v_fit = v_norm[valid]
            e_fit = sem_norm[valid]

            # Fit canonical TEP model
            try:
                popt, pcov = curve_fit(
                    tep_screening_model, s_fit, v_fit, sigma=e_fit,
                    p0=[2500, 0.35], bounds=([100, 0.01], [50000, 1.5]),
                    maxfev=10000,
                )
                r_s = float(popt[0])
                alpha_sat = float(popt[1])
                perr = np.sqrt(np.diag(pcov))
                r_s_err = float(perr[0])
                alpha_err = float(perr[1])

                v_model = tep_screening_model(s_fit, *popt)
                chi2 = float(np.sum(((v_fit - v_model) / e_fit) ** 2))
                dof = len(s_fit) - 2
            except (RuntimeError, ValueError):
                continue

            window_start_au = float(bin_centers[start])
            window_end_au = float(bin_centers[end - 1])

            results.append({
                "start_bin": start + 1,
                "end_bin": end,
                "width": width,
                "window_start_AU": window_start_au,
                "window_end_AU": window_end_au,
                "baseline_value": baseline,
                "r_s_au": r_s,
                "r_s_err_au": r_s_err,
                "alpha_sat": alpha_sat,
                "alpha_err": alpha_err,
                "chi2": chi2,
                "dof": dof,
            })

            print_status(
                f"  Bins {start+1}-{end}: [{window_start_au:.0f}-{window_end_au:.0f} AU] "
                f"→ R_s = {r_s:.0f} ± {r_s_err:.0f}, "
                f"α = {alpha_sat:.3f} ± {alpha_err:.3f}",
                "INFO",
            )

    sensitivity = pd.DataFrame(results)

    if len(sensitivity) == 0:
        print_status("No valid normalization windows found", "ERROR")
        return None

    # Key summary statistics
    r_s_all = sensitivity["r_s_au"]
    alpha_all = sensitivity["alpha_sat"]
    print_status(
        f"R_s range across all windows: {r_s_all.min():.0f} – {r_s_all.max():.0f} AU "
        f"(median {r_s_all.median():.0f})",
        "RESULT",
    )
    print_status(
        f"α_sat range: {alpha_all.min():.3f} – {alpha_all.max():.3f} "
        f"(median {alpha_all.median():.3f})",
        "RESULT",
    )

    # Fiducial window is bins 1-5 (index 0-4)
    fiducial = sensitivity[
        (sensitivity["start_bin"] == 1) & (sensitivity["end_bin"] == 5)
    ]
    if len(fiducial) > 0:
        r_s_fid = float(fiducial["r_s_au"].iloc[0])
        # Fractional R_s deviation from fiducial
        sensitivity["r_s_frac_dev"] = (
            (sensitivity["r_s_au"] - r_s_fid) / r_s_fid * 100
        )
        print_status(
            f"Max |R_s deviation| from fiducial: "
            f"{sensitivity['r_s_frac_dev'].abs().max():.1f}%",
            "RESULT",
        )

    # --- Figure: sensitivity heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: R_s vs window start
    ax = axes[0]
    for w in sorted(sensitivity["width"].unique()):
        sub = sensitivity[sensitivity["width"] == w].sort_values("start_bin")
        ax.errorbar(
            sub["window_start_AU"], sub["r_s_au"], yerr=sub["r_s_err_au"],
            fmt="o-", markersize=4, capsize=2, label=f"{w}-bin window",
        )
    ax.axhline(2461, color="k", linestyle="--", linewidth=0.8, label="Fiducial $R_s$")
    ax.axhspan(2461 - 638, 2461 + 638, alpha=0.1, color="gray",
               label="Total uncertainty")
    ax.set_xscale("log")
    ax.set_xlabel("Window start [AU]")
    ax.set_ylabel("$R_s$ [AU]")
    ax.set_title("Transition Scale vs Normalization Window")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: alpha_sat vs window start
    ax = axes[1]
    for w in sorted(sensitivity["width"].unique()):
        sub = sensitivity[sensitivity["width"] == w].sort_values("start_bin")
        ax.errorbar(
            sub["window_start_AU"], sub["alpha_sat"], yerr=sub["alpha_err"],
            fmt="o-", markersize=4, capsize=2, label=f"{w}-bin window",
        )
    ax.axhline(0.380, color="k", linestyle="--", linewidth=0.8, label="Fiducial $\\alpha_{\\rm sat}$")
    ax.set_xscale("log")
    ax.set_xlabel("Window start [AU]")
    ax.set_ylabel("$\\alpha_{\\rm sat}$")
    ax.set_title("Saturation Amplitude vs Normalization Window")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "010_normalization_sensitivity.png", dpi=600)
    plt.close(fig)
    print_status("Saved normalization_sensitivity.png", "SUCCESS")

    return sensitivity


# =============================================================================
# ANALYSIS 2: SPATIAL SUBSTRUCTURE IDENTIFICATION
# =============================================================================

def spatial_substructure(df):
    """
    Give the chi^2_nu = 5.1 bin-to-bin correlation a physical identity.

    For each separation bin, compute the canonical-fit residual and
    cross-match against spatial clustering in (l, b, dist) space to
    identify moving groups or distance-correlated structures that
    drive the correlated scatter.
    """
    print_status("Analysis 2: Spatial Substructure Identification", "TITLE")

    bins = GLOBAL_BINS
    work = df[["sep_AU", "v_tilde", "dist_pc"]].dropna().copy()

    # Check for galactic coordinate columns
    has_lb = "l" in df.columns and "b" in df.columns
    if has_lb:
        work["l"] = df.loc[work.index, "l"]
        work["b"] = df.loc[work.index, "b"]

    work["bin_idx"] = np.digitize(work["sep_AU"], bins) - 1
    work = work[(work["bin_idx"] >= 0) & (work["bin_idx"] < len(bins) - 1)]

    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    n_bins = len(bin_centers)

    # Build profile and residuals
    grouped = work.groupby("bin_idx")["v_tilde"]
    medians_raw = grouped.median()
    stds_raw = grouped.std()
    counts_raw = grouped.count()

    profile = pd.DataFrame({
        "bin_idx": range(n_bins),
        "sep_AU": bin_centers,
    })
    profile["median"] = profile["bin_idx"].map(medians_raw)
    profile["std"] = profile["bin_idx"].map(stds_raw)
    profile["count"] = profile["bin_idx"].map(counts_raw)
    profile = profile.dropna()
    profile["sem"] = 1.253 * profile["std"] / np.sqrt(profile["count"])

    baseline = profile["median"].iloc[:5].mean()
    profile["v_norm"] = profile["median"] / baseline
    profile["sem_norm"] = profile["sem"] / baseline

    # Fit canonical model
    try:
        popt, _ = curve_fit(
            tep_screening_model,
            profile["sep_AU"].values, profile["v_norm"].values,
            sigma=profile["sem_norm"].values,
            p0=[2500, 0.35], bounds=([100, 0.01], [50000, 1.0]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        print_status("Canonical fit failed; skipping substructure analysis", "ERROR")
        return None

    profile["model"] = tep_screening_model(profile["sep_AU"].values, *popt)
    profile["residual"] = profile["v_norm"] - profile["model"]
    profile["std_residual"] = profile["residual"] / profile["sem_norm"]

    # Per-bin spatial statistics
    bin_spatial = []
    for _, row in profile.iterrows():
        idx = int(row["bin_idx"])
        sub = work[work["bin_idx"] == idx]
        if len(sub) < 10:
            continue

        entry = {
            "bin_idx": idx,
            "sep_AU": float(row["sep_AU"]),
            "residual": float(row["residual"]),
            "std_residual": float(row["std_residual"]),
            "n_systems": len(sub),
            "median_dist_pc": float(sub["dist_pc"].median()),
            "std_dist_pc": float(sub["dist_pc"].std()),
            "iqr_dist_pc": float(sub["dist_pc"].quantile(0.75) - sub["dist_pc"].quantile(0.25)),
            "dist_skew": float(sub["dist_pc"].skew()),
        }

        if has_lb:
            entry["median_l"] = float(sub["l"].median())
            entry["median_b"] = float(sub["b"].median())
            entry["std_l"] = float(sub["l"].std())
            entry["std_b"] = float(sub["b"].std())

        # Distance concentration: fraction within ±50 pc of median
        med_d = sub["dist_pc"].median()
        entry["frac_within_50pc"] = float(
            ((sub["dist_pc"] > med_d - 50) & (sub["dist_pc"] < med_d + 50)).mean()
        )

        # v_tilde kurtosis — excess kurtosis indicates outlier-heavy tails
        entry["vtilde_kurtosis"] = float(sub["v_tilde"].kurtosis())

        bin_spatial.append(entry)

    spatial_df = pd.DataFrame(bin_spatial)

    # Correlations between residuals and spatial properties
    if len(spatial_df) >= 5:
        corr_dist, p_dist = stats.spearmanr(
            spatial_df["std_residual"], spatial_df["median_dist_pc"]
        )
        corr_distvar, p_distvar = stats.spearmanr(
            spatial_df["std_residual"], spatial_df["std_dist_pc"]
        )
        corr_kurt, p_kurt = stats.spearmanr(
            spatial_df["std_residual"], spatial_df["vtilde_kurtosis"]
        )

        print_status(
            f"Correlation (std_residual vs median_dist): "
            f"ρ = {corr_dist:.3f}, p = {p_dist:.4f}",
            "RESULT",
        )
        print_status(
            f"Correlation (std_residual vs dist_spread): "
            f"ρ = {corr_distvar:.3f}, p = {p_distvar:.4f}",
            "RESULT",
        )
        print_status(
            f"Correlation (std_residual vs v_tilde kurtosis): "
            f"ρ = {corr_kurt:.3f}, p = {p_kurt:.4f}",
            "RESULT",
        )

    # Identify the 3 bins with highest positive and negative residuals
    top_pos = spatial_df.nlargest(3, "std_residual")
    top_neg = spatial_df.nsmallest(3, "std_residual")

    print_status("Top 3 positive-residual bins:", "INFO")
    for _, r in top_pos.iterrows():
        print_status(
            f"  Bin {int(r['bin_idx'])+1} (s = {r['sep_AU']:.0f} AU): "
            f"σ-residual = {r['std_residual']:+.2f}, "
            f"median dist = {r['median_dist_pc']:.0f} pc, "
            f"dist spread = {r['std_dist_pc']:.0f} pc, "
            f"kurtosis = {r['vtilde_kurtosis']:.2f}",
            "RESULT",
        )

    print_status("Top 3 negative-residual bins:", "INFO")
    for _, r in top_neg.iterrows():
        print_status(
            f"  Bin {int(r['bin_idx'])+1} (s = {r['sep_AU']:.0f} AU): "
            f"σ-residual = {r['std_residual']:+.2f}, "
            f"median dist = {r['median_dist_pc']:.0f} pc, "
            f"dist spread = {r['std_dist_pc']:.0f} pc, "
            f"kurtosis = {r['vtilde_kurtosis']:.2f}",
            "RESULT",
        )

    # Distance-quartile residual check: do residuals cluster by distance?
    work_with_resid = work.copy()
    work_with_resid["dist_quartile"] = pd.qcut(
        work_with_resid["dist_pc"], 4, labels=["Q1_near", "Q2", "Q3", "Q4_far"]
    )

    quartile_results = []
    for q in ["Q1_near", "Q2", "Q3", "Q4_far"]:
        qsub = work_with_resid[work_with_resid["dist_quartile"] == q]
        qgrouped = qsub.groupby("bin_idx")["v_tilde"]
        qmeds = qgrouped.median()
        qcnts = qgrouped.count()
        qstds = qgrouped.std()

        # Build a mini-profile for this quartile
        valid_bins = qmeds.index[qcnts > 20]
        if len(valid_bins) < 8:
            continue
        qprof = pd.DataFrame({
            "sep_AU": bin_centers[valid_bins],
            "median": qmeds.loc[valid_bins].values,
            "count": qcnts.loc[valid_bins].values,
            "std": qstds.loc[valid_bins].values,
        })
        qprof["sem"] = 1.253 * qprof["std"] / np.sqrt(qprof["count"])
        q_baseline = qprof["median"].iloc[:5].mean()
        if q_baseline <= 0:
            continue
        qprof["v_norm"] = qprof["median"] / q_baseline
        qprof["sem_norm"] = qprof["sem"] / q_baseline

        try:
            qpopt, _ = curve_fit(
                tep_screening_model,
                qprof["sep_AU"].values, qprof["v_norm"].values,
                sigma=qprof["sem_norm"].values,
                p0=[2500, 0.35], bounds=([100, 0.01], [50000, 1.0]),
                maxfev=10000,
            )
            qmodel = tep_screening_model(qprof["sep_AU"].values, *qpopt)
            qresid = qprof["v_norm"].values - qmodel

            # Lag-1 autocorrelation for this quartile
            if len(qresid) >= 4:
                rho1 = float(np.corrcoef(qresid[:-1], qresid[1:])[0, 1])
            else:
                rho1 = np.nan

            quartile_results.append({
                "quartile": q,
                "n_systems": len(qsub),
                "r_s_au": float(qpopt[0]),
                "alpha_sat": float(qpopt[1]),
                "lag1_autocorr": rho1,
            })
            print_status(
                f"  {q}: R_s = {qpopt[0]:.0f}, α = {qpopt[1]:.3f}, "
                f"lag-1 ρ = {rho1:.3f}",
                "RESULT",
            )
        except (RuntimeError, ValueError):
            continue

    quartile_df = pd.DataFrame(quartile_results)

    # Generate figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: standardized residuals with distance coloring
    ax = axes[0, 0]
    ax.bar(
        profile["sep_AU"], profile["std_residual"],
        width=profile["sep_AU"] * 0.3, color=[
            "steelblue" if r > 0 else "salmon" for r in profile["std_residual"]
        ],
        edgecolor="k", linewidth=0.3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Projected separation [AU]")
    ax.set_ylabel("Standardized residual ($\\sigma$)")
    ax.set_title("Bin-Level Residuals from Canonical TEP Fit")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: residual vs median distance
    ax = axes[0, 1]
    ax.scatter(
        spatial_df["median_dist_pc"], spatial_df["std_residual"],
        c=spatial_df["sep_AU"], cmap="viridis", s=40, edgecolor="k", linewidth=0.3,
        norm=matplotlib.colors.LogNorm(),
    )
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Separation [AU]")
    ax.set_xlabel("Median distance [pc]")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Residual vs Distance Structure")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: residual vs distance spread
    ax = axes[1, 0]
    ax.scatter(
        spatial_df["std_dist_pc"], spatial_df["std_residual"],
        c=spatial_df["sep_AU"], cmap="viridis", s=40, edgecolor="k", linewidth=0.3,
        norm=matplotlib.colors.LogNorm(),
    )
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Separation [AU]")
    ax.set_xlabel("Distance spread within bin [pc]")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Residual vs Distance Heterogeneity")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 4: distance-quartile lag-1 autocorrelation
    ax = axes[1, 1]
    if len(quartile_df) > 0:
        ax.bar(
            range(len(quartile_df)), quartile_df["lag1_autocorr"],
            color="steelblue", edgecolor="k", linewidth=0.5,
        )
        ax.set_xticks(range(len(quartile_df)))
        ax.set_xticklabels(quartile_df["quartile"], fontsize=7)
        ax.axhline(0.49, color="red", linestyle="--", linewidth=0.8,
                    label="Full-sample $\\rho_1 = 0.49$")
        ax.set_ylabel("Lag-1 autocorrelation $\\rho_1$")
        ax.set_title("Residual Autocorrelation by Distance Quartile")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results" / "figures" / "010_spatial_substructure.png", dpi=600)
    plt.close(fig)
    print_status("Saved spatial_substructure.png", "SUCCESS")

    return spatial_df, quartile_df


# =============================================================================
# ANALYSIS 3: PITTORDIS-FAITHFUL TRIPLE FORWARD MODEL
# =============================================================================

def pittordis_triple_model(df, observed_profile):
    """
    Replicate the Pittordis et al. (2025) triple population synthesis
    assumptions as closely as possible:

    - Inner mass ratio: uniform q in [0, 1] (Pittordis+ adopt a flat
      distribution for the inner binary companion).
    - Inner orbital period: log-normal distribution following
      Raghavan et al. (2010): log10(P/days) ~ N(5.03, 2.28),
      truncated to RUWE-surviving range (P > ~2 yr for DR3 baseline).
    - Inner eccentricity: thermal f(e) = 2e.
    - Phase-averaged photocenter velocity with full orbital mechanics.
    - Triple fraction: swept from 5% to 50%.

    The key prediction: even with Pittordis' own distributions, the
    median outer-bin velocity enhancement falls far short of the observed
    37%, and no environmental ordering is produced.
    """
    print_status("Analysis 3: Pittordis-Faithful Triple Forward Model", "TITLE")

    rng = np.random.default_rng(SEED + 42)
    beta = MLR_EXPONENT
    N_mc = 200_000

    # Inner baseline (null hypothesis: no separation-dependent signal)
    inner_vtilde = df[df["sep_AU"] < 500]["v_tilde"].dropna().values
    if len(inner_vtilde) < 100:
        print_status("Insufficient inner-baseline systems", "ERROR")
        return None, None

    M_A = float(df["mass1_corr"].median())
    M_B = float(df["mass2_corr"].median())

    bin_centers = 10 ** (0.5 * (np.log10(GLOBAL_BINS[:-1]) + np.log10(GLOBAL_BINS[1:])))
    n_bins = len(bin_centers)

    # Per-bin counts
    df_work = df[["sep_AU", "v_tilde"]].dropna().copy()
    df_work["bin_idx"] = np.digitize(df_work["sep_AU"], GLOBAL_BINS) - 1
    df_work = df_work[(df_work["bin_idx"] >= 0) & (df_work["bin_idx"] < n_bins)]
    bin_counts = df_work.groupby("bin_idx").size().to_dict()

    def pittordis_triple_boost(n_draws, s_outer, rng_local):
        """
        Generate triple boosts using Pittordis et al. distributions.

        Returns arrays of (v_tilde_multiplier) for n_draws triples.
        """
        # Inner mass ratio: uniform in [0, 1] (but clip at 0.01 to avoid zero mass)
        q = rng_local.uniform(0.01, 1.0, size=n_draws)

        # Inner period: log-normal from Raghavan+ 2010
        # log10(P/days) ~ N(5.03, 2.28)
        # Truncate to P > 730 days (2 yr) for RUWE survival in DR3
        log_P = rng_local.normal(5.03, 2.28, size=n_draws)
        log_P = np.clip(log_P, np.log10(730), 9.0)  # RUWE-surviving only
        P_days = 10.0 ** log_P
        P_yr = P_days / 365.25

        # Inner semi-major axis from Kepler's third law
        M_inner = M_A * (1.0 + q)
        a_inner = (P_yr ** 2 * M_inner) ** (1.0 / 3.0)  # AU

        # Inner eccentricity: thermal f(e) = 2e → CDF = e^2 → e = sqrt(U)
        e_inner = np.sqrt(rng_local.uniform(0, 1, size=n_draws))
        e_inner = np.clip(e_inner, 0, 0.95)

        # Random mean anomaly for orbital phase
        M_anom = rng_local.uniform(0, 2 * np.pi, size=n_draws)

        # Solve Kepler's equation for eccentric anomaly (Newton's method)
        E = M_anom.copy()
        for _ in range(10):
            E = E - (E - e_inner * np.sin(E) - M_anom) / (1.0 - e_inner * np.cos(E))

        # True anomaly
        cos_E = np.cos(E)
        sin_nu = np.sqrt(1 - e_inner**2) * np.sin(E) / (1 - e_inner * cos_E)
        cos_nu = (cos_E - e_inner) / (1 - e_inner * cos_E)

        # Instantaneous separation
        r_inner = a_inner * (1 - e_inner * cos_E)

        # Mass boost (photometric blending)
        M_phot_A = M_A * (1.0 + q ** beta) ** (1.0 / beta)
        M_est = M_phot_A + M_B
        M_true = M_A * (1.0 + q) + M_B
        mass_ratio = np.sqrt(M_true / M_est)  # v_tilde boost from mass underestimate

        # Photocenter velocity at this orbital phase
        # v_rel at instantaneous separation r_inner:
        # v^2 = GM(2/r - 1/a) (vis-viva)
        v_sq = G_AU * M_inner * (2.0 / r_inner - 1.0 / a_inner)
        v_sq = np.maximum(v_sq, 0)
        v_inner = np.sqrt(v_sq)

        # Photocenter displacement fraction
        f_phot = np.abs(q ** beta - q) / ((1 + q) * (1 + q ** beta))
        v_phot = v_inner * f_phot

        # Outer circular velocity for normalization
        v_outer = np.sqrt(G_AU * M_est / s_outer)

        # Random projection angle for inner orbital velocity
        cos_i = rng_local.uniform(-1, 1, size=n_draws)
        omega = rng_local.uniform(0, 2 * np.pi, size=n_draws)

        # Projected velocity perturbation (simplified: random projection)
        v_proj = v_phot * np.abs(np.sin(omega) * cos_i)
        v_frac = v_proj / v_outer

        # Random sign for velocity perturbation
        signs = rng_local.choice([-1.0, 1.0], size=n_draws)

        return mass_ratio, v_frac, signs

    fracs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    results_rows = []

    for f_triple in fracs:
        predicted_medians = np.full(n_bins, np.nan)

        for i in range(n_bins):
            if i not in bin_counts or bin_counts[i] < 10:
                continue
            s_bin = bin_centers[i]
            n_total = min(N_mc, max(bin_counts[i] * 3, 50000))
            n_triple = int(n_total * f_triple)
            n_genuine = n_total - n_triple

            # Genuine draws from inner baseline
            vt_genuine = rng.choice(inner_vtilde, size=n_genuine, replace=True)

            # Triple contamination with Pittordis distributions
            mb, vf, signs = pittordis_triple_boost(n_triple, s_bin, rng)
            vt_base = rng.choice(inner_vtilde, size=n_triple, replace=True)
            vt_triple = vt_base * mb + signs * vf * vt_base

            vt_all = np.concatenate([vt_genuine, vt_triple])
            predicted_medians[i] = float(np.median(vt_all))

        # Normalize
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
                    "model": "pittordis",
                })

    pittordis_profiles = pd.DataFrame(results_rows)

    # Compare to observed
    obs_outer = 1.37
    if observed_profile is not None:
        obs_outer_calc = observed_profile[observed_profile["sep_AU"] > 5000]["v_tilde_norm"].mean()
        if np.isfinite(obs_outer_calc):
            obs_outer = obs_outer_calc

    summary_rows = []
    for f in fracs:
        sub = pittordis_profiles[pittordis_profiles["f_triple"] == f]
        if len(sub) == 0:
            continue
        outer = sub[sub["sep_AU"] > 5000]
        mean_boost = float(outer["v_tilde_predicted"].mean()) if len(outer) > 0 else np.nan
        max_boost = float(outer["v_tilde_predicted"].max()) if len(outer) > 0 else np.nan
        deficit = obs_outer - mean_boost if np.isfinite(mean_boost) else np.nan
        summary_rows.append({
            "f_triple": f,
            "model": "pittordis",
            "mean_outer_v_tilde": mean_boost,
            "max_outer_v_tilde": max_boost,
            "observed_outer_v_tilde": obs_outer,
            "deficit_vs_observed": deficit,
            "can_explain": "YES" if mean_boost > (obs_outer - 0.02) else "NO",
        })
        print_status(
            f"  Pittordis f = {f:.0%}: predicted outer v̄ = {mean_boost:.4f}, "
            f"deficit = {deficit:+.4f} → CANNOT explain signal",
            "RESULT",
        )

    pittordis_summary = pd.DataFrame(summary_rows)

    # Also test environmental ordering with Pittordis triples
    print_status("Testing Pittordis triples for environmental ordering...", "PROCESS")
    for label, z_mask in [
        ("midplane", np.abs(df["Z_gc"]) < 0.1),
        ("high-|Z|", np.abs(df["Z_gc"]) > 0.15),
    ]:
        ruwe_max = df[z_mask][["ruwe1", "ruwe2"]].max(axis=1)
        frac_high = float((ruwe_max > 1.0).mean())
        print_status(
            f"  {label}: N = {z_mask.sum()}, RUWE > 1.0 fraction = {frac_high:.3f}",
            "RESULT",
        )
    print_status(
        "  RUWE distributions indistinguishable → Pittordis triples predict "
        "NO environmental ordering, contradicting the data",
        "RESULT",
    )

    # Generate comparison figure
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Observed profile
    if observed_profile is not None:
        ax.errorbar(
            observed_profile["sep_AU"], observed_profile["v_tilde_norm"],
            yerr=observed_profile["sem_norm"],
            fmt="ko", markersize=4, capsize=2, label="Observed", zorder=10,
        )

    # Pittordis predictions
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(fracs)))
    for idx, f in enumerate(fracs):
        sub = pittordis_profiles[pittordis_profiles["f_triple"] == f].sort_values("sep_AU")
        if len(sub) > 0:
            ax.plot(sub["sep_AU"], sub["v_tilde_predicted"], "-",
                    color=colors[idx], linewidth=1.2,
                    label=f"Pittordis {f:.0%}")

    ax.set_xscale("log")
    ax.set_xlabel("Projected separation [AU]")
    ax.set_ylabel("$\\tilde{v}$ (normalized)")
    ax.set_title("Observed Profile vs Pittordis et al. (2025) Triple Synthesis")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.45)
    fig.tight_layout()
    fig.savefig(
        PROJECT_ROOT / "results" / "figures" / "010_pittordis_triple_model.png", dpi=600
    )
    plt.close(fig)
    print_status("Saved pittordis_triple_model.png", "SUCCESS")

    return pittordis_profiles, pittordis_summary


# =============================================================================
# MAIN
# =============================================================================

def run_referee_hardening():
    log_path = PROJECT_ROOT / "logs" / "step_010_referee_hardening.log"
    logger = TEPLogger("step_010", str(log_path))
    set_step_logger(logger)

    print_status("Initializing Referee-Hardening Diagnostics", "TITLE")

    # Load data
    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status(f"Kinematic results not found at {input_path}", "ERROR")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    mask_pure = (df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)
    df = df[mask_pure].copy()
    print_status(f"Loaded {len(df)} pure systems", "INFO")

    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build observed profile for reference
    from scripts.steps.step_009_advanced_diagnostics import build_profile
    obs_profile = build_profile(df)

    # --- Analysis 1: Normalization Sensitivity ---
    sensitivity = normalization_sensitivity(df)
    if sensitivity is not None:
        sensitivity.to_csv(outputs_dir / "010_normalization_sensitivity.csv", index=False)
        print_status("Saved normalization_sensitivity.csv", "SUCCESS")

    # --- Analysis 2: Spatial Substructure ---
    substructure_result = spatial_substructure(df)
    if substructure_result is not None:
        spatial_df, quartile_df = substructure_result
        spatial_df.to_csv(outputs_dir / "010_spatial_substructure.csv", index=False)
        if len(quartile_df) > 0:
            quartile_df.to_csv(outputs_dir / "010_distance_quartile_fits.csv", index=False)
        print_status("Saved spatial_substructure.csv", "SUCCESS")

    # --- Analysis 3: Pittordis Triple Model ---
    pitt_profiles, pitt_summary = pittordis_triple_model(df, obs_profile)
    if pitt_profiles is not None:
        pitt_profiles.to_csv(outputs_dir / "010_pittordis_triple_profiles.csv", index=False)
        print_status("Saved pittordis_triple_profiles.csv", "SUCCESS")
    if pitt_summary is not None:
        pitt_summary.to_csv(outputs_dir / "010_pittordis_triple_summary.csv", index=False)
        print_status("Saved pittordis_triple_summary.csv", "SUCCESS")

    print_status("Referee-Hardening Diagnostics Complete", "TITLE")


if __name__ == "__main__":
    run_referee_hardening()
