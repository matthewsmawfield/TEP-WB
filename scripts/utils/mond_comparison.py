#!/usr/bin/env python3
"""
MOND ν-function comparison for the wide-binary velocity profile.

Fits two standard MOND interpolating functions to the same binned
velocity profile used for the TEP screening analysis, enabling a
direct χ²/AIC comparison.

The MOND prediction for the velocity ratio is:
    ṽ(s) = √(ν(g_N/a_0))
where g_N = GM/s² and ν is the interpolating function.

Two variants are tested:
    1. Simple ν: ν(x) = (1 + √(1 + 4/x)) / 2  [Famaey & Binney 2005]
    2. Standard ν: ν(x) = 1 / √(1 − exp(−√x))  [Milgrom 1983]

The fit uses s_M = √(GM/a_0) as the single free parameter,
avoiding any assumption about the median binary mass.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── MOND interpolating functions ──────────────────────────────────────

def nu_simple(x):
    """Simple (n=1) ν-function (Famaey & Binney 2005, Eq. 8). x = g_N/a_0.

    ν(x) = (1 + √(1 + 4/x)) / 2,  deep-MOND limit ν → x^(-1/2).
    """
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / np.maximum(x, 1e-30)))

def nu_standard(x):
    """Standard (n=2) ν-function — inverse of Milgrom (1983) μ(y)=y/√(1+y²).

    Famaey & Binney (2005, Eq. 9):
        ν(x) = √[ (1 + √(1 + 4/x²)) / 2 ],  deep-MOND limit ν → x^(-1/2).
    """
    x_safe = np.maximum(x, 1e-30)
    return np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / x_safe**2)))

# ── MOND model for the normalised velocity profile ───────────────────

def mond_profile(s, s_mond, nu_func, s_norm):
    """
    MOND velocity-ratio prediction, self-consistently normalised
    the same way as the data (divide by mean of first 5 bins).

    Parameters
    ----------
    s : array  – projected separations (AU)
    s_mond : float – √(GM/a_0) in AU (single free parameter)
    nu_func : callable – ν(x) interpolating function
    s_norm : array – separation values of the 5 normalisation bins
    """
    x = (s_mond / s) ** 2          # g_N / a_0
    v_raw = np.sqrt(nu_func(x))

    x_norm = (s_mond / s_norm) ** 2
    v_norm_factor = np.mean(np.sqrt(nu_func(x_norm)))

    return v_raw / v_norm_factor

def _make_mond_model(nu_func, s_norm):
    """Return a curve_fit-compatible function with one free parameter."""
    def model(s, s_mond):
        return mond_profile(s, s_mond, nu_func, s_norm)
    return model

# ── TEP model (for reference chi2) ───────────────────────────────────

from scripts.utils.tep_model import tep_screening_model  # noqa: E402

# ── Fit helpers ───────────────────────────────────────────────────────

def chi2_stat(obs, exp, err):
    return float(np.sum(((obs - exp) / err) ** 2))

def aic(chi2, k):
    return chi2 + 2 * k

# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Load binned profile
    data_path = PROJECT_ROOT / "results" / "outputs" / "003_screening_test_results.csv"
    df = pd.read_csv(data_path)

    s   = df["sep_AU"].values
    v   = df["v_tilde_norm"].values
    err = df["sem_norm"].values
    n   = len(s)

    s_norm = s[:5]  # first 5 bins used for normalisation

    print(f"Loaded {n} bins, separations {s[0]:.0f} – {s[-1]:.0f} AU\n")

    # ── Fit TEP (for baseline comparison) ─────────────────────────────
    tep_popt, _ = curve_fit(tep_screening_model, s, v, sigma=err,
                            p0=[3000, 0.4], bounds=([100, 0], [50000, 0.8]))
    v_tep = tep_screening_model(s, *tep_popt)
    chi2_tep = chi2_stat(v, v_tep, err)
    aic_tep  = aic(chi2_tep, 2)

    print(f"TEP exponential  : R_s = {tep_popt[0]:.0f} AU, α = {tep_popt[1]:.4f}")
    print(f"                   χ² = {chi2_tep:.1f},  dof = {n-2},  χ²_ν = {chi2_tep/(n-2):.2f},  AIC = {aic_tep:.1f}")
    print()

    # ── Fit MOND simple ν ─────────────────────────────────────────────
    model_simple = _make_mond_model(nu_simple, s_norm)
    try:
        popt_s, pcov_s = curve_fit(model_simple, s, v, sigma=err,
                                   p0=[7000], bounds=([500], [100000]))
        s_mond_simple = popt_s[0]
        s_mond_simple_err = float(np.sqrt(pcov_s[0, 0]))
        v_mond_s = model_simple(s, s_mond_simple)
        chi2_ms = chi2_stat(v, v_mond_s, err)
        aic_ms  = aic(chi2_ms, 1)
        print(f"MOND simple ν    : s_M = {s_mond_simple:.0f} ± {s_mond_simple_err:.0f} AU")
        print(f"                   χ² = {chi2_ms:.1f},  dof = {n-1},  χ²_ν = {chi2_ms/(n-1):.2f},  AIC = {aic_ms:.1f}")
        print(f"                   Δχ² vs TEP = {chi2_ms - chi2_tep:+.1f},  ΔAIC vs TEP = {aic_ms - aic_tep:+.1f}")
    except Exception as e:
        print(f"MOND simple ν fit failed: {e}")
        chi2_ms = np.inf
        aic_ms = np.inf
        s_mond_simple = np.nan
        s_mond_simple_err = np.nan
        v_mond_s = np.full_like(s, np.nan)

    print()

    # ── Fit MOND standard ν ───────────────────────────────────────────
    model_standard = _make_mond_model(nu_standard, s_norm)
    try:
        popt_st, pcov_st = curve_fit(model_standard, s, v, sigma=err,
                                     p0=[7000], bounds=([500], [100000]))
        s_mond_standard = popt_st[0]
        s_mond_standard_err = float(np.sqrt(pcov_st[0, 0]))
        v_mond_st = model_standard(s, s_mond_standard)
        chi2_mst = chi2_stat(v, v_mond_st, err)
        aic_mst  = aic(chi2_mst, 1)
        print(f"MOND standard ν  : s_M = {s_mond_standard:.0f} ± {s_mond_standard_err:.0f} AU")
        print(f"                   χ² = {chi2_mst:.1f},  dof = {n-1},  χ²_ν = {chi2_mst/(n-1):.2f},  AIC = {aic_mst:.1f}")
        print(f"                   Δχ² vs TEP = {chi2_mst - chi2_tep:+.1f},  ΔAIC vs TEP = {aic_mst - aic_tep:+.1f}")
    except Exception as e:
        print(f"MOND standard ν fit failed: {e}")
        chi2_mst = np.inf
        aic_mst = np.inf
        s_mond_standard = np.nan
        s_mond_standard_err = np.nan
        v_mond_st = np.full_like(s, np.nan)

    print()

    # ── Physical interpretation ───────────────────────────────────────
    # Estimate implied a_0 from median binary mass
    # Typical wide-binary total mass ~ 1.3–1.8 M_sun
    M_sun = 1.989e30   # kg
    G     = 6.674e-11  # m³ kg⁻¹ s⁻²
    AU_m  = 1.496e11   # m
    M_median = 1.5 * M_sun  # representative

    for label, s_m in [("simple", s_mond_simple), ("standard", s_mond_standard)]:
        if np.isnan(s_m):
            continue
        s_m_m = s_m * AU_m
        a0_implied = G * M_median / s_m_m**2
        a0_ratio = a0_implied / 1.2e-10
        print(f"MOND {label:8s}: implied a_0 = {a0_implied:.2e} m/s²  "
              f"(= {a0_ratio:.2f} × canonical a_0, assuming M = 1.5 M_sun)")

    print()

    # ── Residual shape diagnostic ─────────────────────────────────────
    print("Bin-by-bin residual comparison (data − model):")
    print(f"{'s (AU)':>10s}  {'data':>8s}  {'TEP':>8s}  {'MOND_s':>8s}  {'res_TEP':>8s}  {'res_MOND':>8s}")
    for i in range(n):
        res_t = v[i] - v_tep[i]
        res_m = v[i] - v_mond_s[i] if not np.isnan(v_mond_s[i]) else np.nan
        print(f"{s[i]:10.0f}  {v[i]:8.4f}  {v_tep[i]:8.4f}  "
              f"{v_mond_s[i]:8.4f}  {res_t:+8.4f}  {res_m:+8.4f}")

    # ── Save results ──────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "results" / "outputs" / "011_mond_comparison.csv"
    out_df = pd.DataFrame({
        "model": ["TEP_exponential", "MOND_simple_nu", "MOND_standard_nu"],
        "k": [2, 1, 1],
        "transition_param_AU": [tep_popt[0], s_mond_simple, s_mond_standard],
        "transition_err_AU": [np.nan, s_mond_simple_err, s_mond_standard_err],
        "chi2": [chi2_tep, chi2_ms, chi2_mst],
        "dof": [n - 2, n - 1, n - 1],
        "reduced_chi2": [chi2_tep / (n-2), chi2_ms / (n-1), chi2_mst / (n-1)],
        "AIC": [aic_tep, aic_ms, aic_mst],
        "delta_chi2_vs_TEP": [0, chi2_ms - chi2_tep, chi2_mst - chi2_tep],
        "delta_AIC_vs_TEP": [0, aic_ms - aic_tep, aic_mst - aic_tep],
    })
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MOND COMPARISON SUMMARY")
    print("=" * 60)
    best_mond = "simple" if chi2_ms <= chi2_mst else "standard"
    best_mond_chi2 = min(chi2_ms, chi2_mst)
    print(f"Best MOND fit ({best_mond} ν):  χ² = {best_mond_chi2:.1f}  (AIC = {min(aic_ms, aic_mst):.1f})")
    print(f"TEP exponential:              χ² = {chi2_tep:.1f}  (AIC = {aic_tep:.1f})")
    print(f"Δχ²(MOND − TEP) = {best_mond_chi2 - chi2_tep:+.1f}")
    print(f"ΔAIC(MOND − TEP) = {min(aic_ms, aic_mst) - aic_tep:+.1f}")
    if best_mond_chi2 > chi2_tep:
        print("→ TEP exponential is preferred over MOND on this dataset.")
    else:
        print("→ MOND is preferred (or comparable) — investigate shape.")
    print("=" * 60)

if __name__ == "__main__":
    main()
