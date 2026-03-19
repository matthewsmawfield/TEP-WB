#!/usr/bin/env python3
"""
Two analyses that strengthen the TEP case for the wide-binary manuscript:

  A) A priori R_s prediction from TEP's independently measured acceleration
     scale (g_TEP from SPARC rotation curves, Paper 7).

  B) MOND ν-function comparison with:
     (i)   the External Field Effect (EFE) included,
     (ii)  per-bin median masses instead of a single representative mass,
     (iii) both fixed-g_e (1-parameter) and free-g_e (2-parameter) variants.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import quad

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Physical constants ────────────────────────────────────────────────
G_SI   = 6.67430e-11   # m^3 kg^-1 s^-2
M_SUN  = 1.98847e30    # kg
AU_M   = 1.49598e11    # m
A0_CANONICAL = 1.2e-10 # m/s^2 (Milgrom 1983)

# ── MOND ν-functions ─────────────────────────────────────────────────

def nu_simple(x):
    """Simple ν-function (Famaey & Binney 2005). x = g/a_0."""
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / np.maximum(x, 1e-30)))

def nu_standard(x):
    """Standard ν-function (Milgrom 1983)."""
    return 1.0 / np.sqrt(np.maximum(1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-30))), 1e-30))

# ── EFE-modified MOND (angle-averaged QUMOND) ────────────────────────

def mond_efe_vtilde_single(g_N, a0, g_e, nu_func, n_theta=50):
    """
    Angle-averaged EFE velocity ratio for a single (g_N, a0, g_e).
    Uses the QUMOND prescription (Milgrom 2010):
      g = ν(|g_N+g_e|/a0)(g_N+g_e) - ν(g_e/a0) g_e
    The radial component along the binary axis gives v_tilde^2.
    """
    if g_N < 1e-30:
        return 1.0

    e = g_e / a0
    x = g_N / a0

    # Integrate over angle θ between g_N and g_e
    theta = np.linspace(0, np.pi, n_theta)
    sin_theta = np.sin(theta)

    # |g_N + g_e| for each angle
    g_tot = np.sqrt(g_N**2 + g_e**2 + 2*g_N*g_e*np.cos(theta))
    y = g_tot / a0

    nu_y = nu_func(y)
    nu_e = nu_func(e)

    # Radial (along binary axis) component of the MONDian acceleration
    # g_r = ν(y)(g_N + g_e cosθ) - ν(e) g_e cosθ
    g_r = nu_y * (g_N + g_e * np.cos(theta)) - nu_e * g_e * np.cos(theta)

    # v_tilde^2 = g_r / g_N
    v2 = g_r / g_N

    # Angle-average with sin(θ) weighting
    # <v^2> = (1/2) ∫_0^π v^2(θ) sinθ dθ
    v2_avg = 0.5 * np.trapz(v2 * sin_theta, theta)

    return np.sqrt(max(v2_avg, 0.0))


def mond_efe_profile(s_au, M_msun_per_bin, a0, g_e, nu_func, s_norm_au, M_norm_msun):
    """
    Compute the EFE-modified MOND velocity ratio profile,
    self-consistently normalised like the data.

    Parameters
    ----------
    s_au : array – bin centre separations (AU)
    M_msun_per_bin : array – median total mass per bin (M_sun)
    a0 : float – MOND acceleration scale (m/s^2)
    g_e : float – external field strength (m/s^2)
    nu_func : callable
    s_norm_au : array – separations of normalisation bins
    M_norm_msun : array – masses of normalisation bins
    """
    v_raw = np.zeros(len(s_au))
    for i, (s, M) in enumerate(zip(s_au, M_msun_per_bin)):
        g_N = G_SI * M * M_SUN / (s * AU_M)**2
        v_raw[i] = mond_efe_vtilde_single(g_N, a0, g_e, nu_func)

    # Normalisation: same first-5-bin mean as the data
    v_norm = np.zeros(len(s_norm_au))
    for i, (s, M) in enumerate(zip(s_norm_au, M_norm_msun)):
        g_N = G_SI * M * M_SUN / (s * AU_M)**2
        v_norm[i] = mond_efe_vtilde_single(g_N, a0, g_e, nu_func)

    norm_factor = np.mean(v_norm)
    return v_raw / norm_factor


# ── No-EFE MOND with per-bin masses (for fair comparison) ────────────

def mond_no_efe_profile(s_au, M_msun_per_bin, a0, nu_func, s_norm_au, M_norm_msun):
    """MOND without EFE but with per-bin masses."""
    v_raw = np.zeros(len(s_au))
    for i, (s, M) in enumerate(zip(s_au, M_msun_per_bin)):
        g_N = G_SI * M * M_SUN / (s * AU_M)**2
        x = g_N / a0
        v_raw[i] = np.sqrt(nu_func(x))

    v_norm = np.zeros(len(s_norm_au))
    for i, (s, M) in enumerate(zip(s_norm_au, M_norm_msun)):
        g_N = G_SI * M * M_SUN / (s * AU_M)**2
        x = g_N / a0
        v_norm[i] = np.sqrt(nu_func(x))

    return v_raw / np.mean(v_norm)


# ── TEP model ────────────────────────────────────────────────────────

def tep_model(s, r_s, alpha):
    return 1.0 + alpha * (1.0 - np.exp(-s / r_s))


# ── Fit helpers ───────────────────────────────────────────────────────

def chi2_stat(obs, exp, err):
    return float(np.sum(((obs - exp) / err)**2))

def aic(chi2, k):
    return chi2 + 2*k


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # ── Load binned profile ───────────────────────────────────────────
    data_path = PROJECT_ROOT / "results" / "outputs" / "screening_test_results.csv"
    df = pd.read_csv(data_path)
    s   = df["sep_AU"].values
    v   = df["v_tilde_norm"].values
    err = df["sem_norm"].values
    n   = len(s)

    # ── Load per-bin masses from the parquet ──────────────────────────
    kin_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    kin = pd.read_parquet(kin_path)
    mask = (kin["ruwe1"] < 1.2) & (kin["ruwe2"] < 1.2)
    kin = kin[mask]

    bins = np.logspace(np.log10(50), np.log10(30000), 20)
    kin["bin_idx"] = pd.cut(kin["sep_AU"], bins=bins, labels=False)

    M_per_bin = np.zeros(n)
    for i in range(n):
        sub = kin[kin["bin_idx"] == i]
        M_per_bin[i] = sub["mass_total"].median() if len(sub) > 0 else 1.2

    s_norm = s[:5]
    M_norm = M_per_bin[:5]

    print(f"Loaded {n} bins, per-bin masses range {M_per_bin.min():.3f}–{M_per_bin.max():.3f} M_sun")
    print()

    # ==================================================================
    # PART A: TEP R_s PREDICTION FROM SPARC-DERIVED g_TEP
    # ==================================================================
    print("=" * 60)
    print("PART A: A PRIORI R_s PREDICTION FROM TEP")
    print("=" * 60)

    g_TEP = 5.0e-10  # m/s^2, from Smawfield 2025b (Paper 7, SPARC fits)
    R_s_obs = 2646.0  # AU

    # Mass at the transition bins (bins 10-12, separations 1715-3363 AU)
    M_transition = np.mean(M_per_bin[10:13])
    print(f"Mean mass at transition bins (1700–3400 AU): {M_transition:.3f} M_sun")

    # Prediction: R_s = sqrt(GM / g_screen)
    # The screening transition occurs at g_N ≈ η * g_TEP, η ~ O(1)
    R_s_pred_eta1 = np.sqrt(G_SI * M_transition * M_SUN / g_TEP) / AU_M
    R_s_pred_eta2 = np.sqrt(G_SI * M_transition * M_SUN / (2 * g_TEP)) / AU_M

    print(f"\nPrediction from g_TEP = 5.0e-10 m/s^2 (Paper 7, SPARC):")
    print(f"  η=1: R_s^pred = {R_s_pred_eta1:.0f} AU  (ratio to obs: {R_s_pred_eta1/R_s_obs:.2f})")
    print(f"  η=2: R_s^pred = {R_s_pred_eta2:.0f} AU  (ratio to obs: {R_s_pred_eta2/R_s_obs:.2f})")

    # What η gives the observed R_s?
    g_N_at_Rs = G_SI * M_transition * M_SUN / (R_s_obs * AU_M)**2
    eta_obs = g_N_at_Rs / g_TEP
    print(f"\nObserved g_N at R_s = {g_N_at_Rs:.2e} m/s^2")
    print(f"Implied η = g_N(R_s) / g_TEP = {eta_obs:.2f}")
    print(f"  (η > 1 expected: Galactic pre-screening raises the threshold)")

    # Cross-check: ratio to a_0
    print(f"  g_N(R_s) / a_0 = {g_N_at_Rs / A0_CANONICAL:.1f}")
    print(f"  g_TEP / a_0 = {g_TEP / A0_CANONICAL:.1f}")

    # Full-sample median mass check
    M_med = kin["mass_total"].median()
    R_s_pred_med = np.sqrt(G_SI * M_med * M_SUN / g_TEP) / AU_M
    print(f"\nUsing full-sample median mass ({M_med:.3f} M_sun):")
    print(f"  η=1: R_s^pred = {R_s_pred_med:.0f} AU  (ratio: {R_s_pred_med/R_s_obs:.2f})")

    print()

    # ==================================================================
    # PART B: MOND WITH EFE AND PER-BIN MASSES
    # ==================================================================
    print("=" * 60)
    print("PART B: MOND COMPARISON WITH EFE AND PER-BIN MASSES")
    print("=" * 60)

    # External field at solar position
    # g_e = v_c^2 / R_0, v_c = 220 km/s, R_0 = 8.2 kpc
    v_c = 220e3     # m/s
    R_0 = 8.2e3 * 3.086e16  # m
    g_e_solar = v_c**2 / R_0
    print(f"External field: g_e = v_c^2/R_0 = {g_e_solar:.2e} m/s^2  (= {g_e_solar/A0_CANONICAL:.2f} a_0)")
    print()

    # ── Fit TEP (baseline) ────────────────────────────────────────────
    tep_popt, _ = curve_fit(tep_model, s, v, sigma=err,
                            p0=[3000, 0.4], bounds=([100, 0], [50000, 0.8]))
    v_tep = tep_model(s, *tep_popt)
    chi2_tep = chi2_stat(v, v_tep, err)
    aic_tep = aic(chi2_tep, 2)
    print(f"TEP exponential : R_s={tep_popt[0]:.0f} AU, α={tep_popt[1]:.4f}, χ²={chi2_tep:.1f}, AIC={aic_tep:.1f}")
    print()

    results = []
    results.append({
        "model": "TEP_exponential", "k": 2,
        "chi2": chi2_tep, "aic": aic_tep,
        "dof": n-2, "red_chi2": chi2_tep/(n-2),
        "note": f"R_s={tep_popt[0]:.0f}, alpha={tep_popt[1]:.4f}"
    })

    # ── MOND without EFE, per-bin masses ──────────────────────────────
    print("--- MOND without EFE (per-bin masses) ---")
    for label, nu_func in [("simple", nu_simple), ("standard", nu_standard)]:
        def cost_no_efe(log_a0):
            a0 = 10**log_a0
            vpred = mond_no_efe_profile(s, M_per_bin, a0, nu_func, s_norm, M_norm)
            return chi2_stat(v, vpred, err)

        res = minimize_scalar(cost_no_efe, bounds=(-11, -8), method='bounded')
        a0_best = 10**res.x
        vpred = mond_no_efe_profile(s, M_per_bin, a0_best, nu_func, s_norm, M_norm)
        c2 = chi2_stat(v, vpred, err)
        a_aic = aic(c2, 1)

        print(f"  {label:10s}: a_0 = {a0_best:.2e} m/s^2 ({a0_best/A0_CANONICAL:.2f}× canonical)")
        print(f"             χ² = {c2:.1f},  Δχ² vs TEP = {c2 - chi2_tep:+.1f},  AIC = {a_aic:.1f}")

        results.append({
            "model": f"MOND_{label}_no_EFE_perbin", "k": 1,
            "chi2": c2, "aic": a_aic,
            "dof": n-1, "red_chi2": c2/(n-1),
            "note": f"a0={a0_best:.2e}"
        })
    print()

    # ── MOND with EFE, fixed g_e (1 free parameter: a_0) ─────────────
    print("--- MOND with EFE, fixed g_e (1 free param: a_0) ---")
    for label, nu_func in [("simple", nu_simple), ("standard", nu_standard)]:
        def cost_efe_fixed(log_a0):
            a0 = 10**log_a0
            vpred = mond_efe_profile(s, M_per_bin, a0, g_e_solar, nu_func,
                                     s_norm, M_norm)
            return chi2_stat(v, vpred, err)

        res = minimize_scalar(cost_efe_fixed, bounds=(-11, -8), method='bounded')
        a0_best = 10**res.x
        vpred = mond_efe_profile(s, M_per_bin, a0_best, g_e_solar, nu_func,
                                  s_norm, M_norm)
        c2 = chi2_stat(v, vpred, err)
        a_aic = aic(c2, 1)

        # Saturation level
        v_sat = vpred[-1]

        print(f"  {label:10s}: a_0 = {a0_best:.2e} m/s^2 ({a0_best/A0_CANONICAL:.2f}× canonical)")
        print(f"             χ² = {c2:.1f},  Δχ² vs TEP = {c2 - chi2_tep:+.1f},  AIC = {a_aic:.1f}")
        print(f"             saturation at 25kAU: ṽ = {v_sat:.3f} (data: {v[-1]:.3f})")

        results.append({
            "model": f"MOND_{label}_EFE_fixed_ge", "k": 1,
            "chi2": c2, "aic": a_aic,
            "dof": n-1, "red_chi2": c2/(n-1),
            "note": f"a0={a0_best:.2e}, ge={g_e_solar:.2e} (fixed)"
        })
    print()

    # ── MOND with EFE, free g_e (2 free parameters: a_0, g_e) ────────
    print("--- MOND with EFE, free g_e (2 free params: a_0, g_e) ---")
    for label, nu_func in [("simple", nu_simple), ("standard", nu_standard)]:
        def cost_efe_free(params):
            log_a0, log_ge = params
            a0 = 10**log_a0
            ge = 10**log_ge
            vpred = mond_efe_profile(s, M_per_bin, a0, ge, nu_func,
                                     s_norm, M_norm)
            return chi2_stat(v, vpred, err)

        from scipy.optimize import minimize
        best = None
        best_c2 = np.inf
        # Grid search for initial guess
        for la0 in np.linspace(-11, -8, 10):
            for lge in np.linspace(-12, -8, 10):
                try:
                    c2 = cost_efe_free([la0, lge])
                    if c2 < best_c2:
                        best_c2 = c2
                        best = [la0, lge]
                except (ValueError, RuntimeError):
                    pass

        res = minimize(cost_efe_free, best, method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 0.1, 'maxiter': 5000})
        a0_best = 10**res.x[0]
        ge_best = 10**res.x[1]
        vpred = mond_efe_profile(s, M_per_bin, a0_best, ge_best, nu_func,
                                  s_norm, M_norm)
        c2 = chi2_stat(v, vpred, err)
        a_aic = aic(c2, 2)
        v_sat = vpred[-1]

        print(f"  {label:10s}: a_0 = {a0_best:.2e} ({a0_best/A0_CANONICAL:.2f}× canonical), g_e = {ge_best:.2e} ({ge_best/A0_CANONICAL:.2f}× a_0)")
        print(f"             χ² = {c2:.1f},  Δχ² vs TEP = {c2 - chi2_tep:+.1f},  AIC = {a_aic:.1f}")
        print(f"             saturation at 25kAU: ṽ = {v_sat:.3f}")

        results.append({
            "model": f"MOND_{label}_EFE_free_ge", "k": 2,
            "chi2": c2, "aic": a_aic,
            "dof": n-2, "red_chi2": c2/(n-2),
            "note": f"a0={a0_best:.2e}, ge={ge_best:.2e}"
        })
    print()

    # ── Residual comparison for best models ───────────────────────────
    print("=" * 60)
    print("BIN-BY-BIN COMPARISON (best models)")
    print("=" * 60)

    # Recompute best MOND-EFE with simple ν, free ge
    # (Already done above, but let's get the vectors explicitly)
    best_efe = [r for r in results if "simple_EFE_free" in r["model"]][0]
    # Parse a0 and ge from note
    parts = best_efe["note"].split(", ")
    a0_refit = float(parts[0].split("=")[1])
    ge_refit = float(parts[1].split("=")[1])
    v_mond_efe = mond_efe_profile(s, M_per_bin, a0_refit, ge_refit, nu_simple,
                                   s_norm, M_norm)

    print(f"{'s (AU)':>10s}  {'M (Msun)':>8s}  {'data':>7s}  {'TEP':>7s}  {'MOND+EFE':>9s}  {'res_TEP':>8s}  {'res_MOND':>9s}")
    for i in range(n):
        rt = v[i] - v_tep[i]
        rm = v[i] - v_mond_efe[i]
        print(f"{s[i]:10.0f}  {M_per_bin[i]:8.3f}  {v[i]:7.4f}  {v_tep[i]:7.4f}  {v_mond_efe[i]:9.4f}  {rt:+8.4f}  {rm:+9.4f}")

    # ── Save results ──────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results" / "outputs" / "mond_efe_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPart A: R_s prediction")
    print(f"  g_TEP from SPARC (Paper 7): 5.0e-10 m/s^2")
    print(f"  R_s^pred (η=1) = {R_s_pred_eta1:.0f} AU")
    print(f"  R_s^obs        = {R_s_obs:.0f} AU")
    print(f"  Ratio           = {R_s_pred_eta1/R_s_obs:.2f}")
    print(f"  Implied η       = {eta_obs:.2f} (Galactic pre-screening enhancement)")

    print(f"\nPart B: MOND comparison")
    print(f"  {'Model':<35s}  {'k':>2s}  {'χ²':>8s}  {'Δχ² vs TEP':>12s}  {'AIC':>8s}")
    for r in sorted(results, key=lambda x: x["chi2"]):
        dc = r["chi2"] - chi2_tep
        print(f"  {r['model']:<35s}  {r['k']:>2d}  {r['chi2']:>8.1f}  {dc:>+12.1f}  {r['aic']:>8.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
