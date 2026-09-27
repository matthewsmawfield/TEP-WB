#!/usr/bin/env python3
"""
Step 014 — Screening-Shell Consistency Map

Evaluates the corpus's resolved nested two-body screening operator across
the separation ladder, verifying that the wide-binary saturation amplitude
and the Solar-System bounds are the interior and exterior readings of a
single continuous screening shell around each source.

Resolved operator (Paper 0 master-action closure; nested two-body
evaluation in TEP step_30):
    R(pair) = S_Sigma(X_env)^2 * y(s)
    y (1 + y^2 (r*/r)^4) = 1   (exact flux-conserving profile of
                                P = X - V + X|X|/Lambda^4)
    r*(M)   = sqrt(G M / g_t),   g_t = c H_0 / (2 beta_A^2)
    S_Sigma(X) = [1 + 2X/Lambda^4]^-1

X_env is the embedding ambient of the pair (hierarchy level above),
excluding the pair's own co-generated mutual shear field. Two regimes:

  * hierarchical pair (test member inside a dominant member's nonlinear
    shell — all Solar-System channels): X_env = the dominant member's
    field, and the identity 2X_dom/Lambda^4 = (1-y)/y makes the vertex
    factors equal to y(s), so R ~ y^3 ~ s^4 — the quartic recovered
    exactly as the deep-interior nested response.
  * comparable-mass pair (field wide binaries): the mutual field is the
    measured response itself and cancels at the pair's bridge region, so
    the vertex ambient saturates at the Galactic floor S_env ~ 0.5 and
    the transition profile is the flux-conserving s^(4/3) law selected
    by the data (step_015).

The shell is continuous, not a thin-shell step.

Consistency checks:
  - Cassini conjunction (s = 1.6 R_sun) and the planetary region sit deep
    inside the solar shell -> S_eff ~ 1e-23, satisfying S_Sigma^(sun) <~ 5.8e-6.
  - The Earth-Moon pair sits deep inside the terrestrial shell -> ~1e-14 (LLR).
  - Earth-orbiting GNSS clocks sit inside the solar shell at 1 AU and inside
    the terrestrial well; the measured clock correlation is the terrestrial
    well's own shear (S_A ~ 1, S_Sigma ~ 1e-21 at the surface).
  - Oort-cloud separations (1e4-1e5 AU) lie outside the solar shell ->
    pairwise factor ~> 0.95, the recovered-shear regime (outer-boundary
    phenomenology, this series Paper 36).
  - The observed saturation alpha_sat = 0.366 corresponds to an extra
    centripetal acceleration ~0.87 g_N, i.e. ~43% of the unscreened
    conformal response 2 beta_A^2 g_N = 2 g_N: a partial recovery bounded
    by the ambient Galactic floor S_body ~ 0.66 (corpus ambient value
    S_Sigma(MW) ~ 0.74), not a fully unsuppressed charge.

Output: results/outputs/014_shell_consistency_map.json
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import brentq

# Physical constants
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_SI = 299792.458e3         # m/s
AU_M = 1.495978707e11       # m
M_SUN = 1.98892e30          # kg
M_EARTH = 5.9722e24         # kg
R_SUN_M = 6.957e8           # m
PC_M = 3.08567758e16        # m

# Corpus parameters
BETA_A = -1.0
H0_SI = 70e3 / (1e6 * PC_M)          # 70 km/s/Mpc -> s^-1
G_T = C_SI * H0_SI / (2.0 * BETA_A**2)  # m/s^2, canonical corpus transition acceleration

# Wide-binary datum (this work, canonical fit)
RS_FIT_AU = 2646.0
RS_FIT_ERR_AU = 182.0
ALPHA_SAT = 0.366
M_SAMPLE_MSUN = 1.24     # sample mean total mass
M_MEDIAN_MSUN = 1.2      # sample median total mass


def shell_radius_au(mass_kg):
    """Derived shell radius R_s = sqrt(GM/g_t), in AU."""
    return np.sqrt(G_SI * mass_kg / G_T) / AU_M


def y_profile(x):
    """Exact flux-conserving profile y = phi'/phi'_lin at x = r/r*,
    solving y(1 + y^2/x^4) = 1. Interior asymptote y ~ x^(4/3)."""
    if x <= 0:
        return 0.0
    return float(brentq(lambda y: y * (1.0 + y * y / x**4) - 1.0,
                        1e-30, 1.0, xtol=1e-14))


def pair_response_hierarchical(s_au, rs_au):
    """Pair response of a test member embedded in a dominant member's
    nonlinear shell: R = S_Sigma(X_env)^2 * y with vertex factors equal
    to y(s), i.e. R = y^3 (~ s^4 deep interior)."""
    return y_profile(s_au / rs_au) ** 3


def pair_response_comparable(s_au, rs_au, s_env=0.5):
    """Pair response of a comparable-mass binary: vertex ambient saturates
    at the external (Galactic) floor, R = S_env^2 * y(s) (~ s^(4/3)
    transition profile)."""
    return s_env**2 * y_profile(s_au / rs_au)


def env_operator(g_si):
    """Environmental operator S_Sigma(E) = [1 + (g/g_t)^2]^-1."""
    return 1.0 / (1.0 + (g_si / G_T) ** 2)


def main():
    results = {
        "constants": {
            "g_t_m_s2": G_T,
            "beta_A": BETA_A,
            "formula": "R(pair) = S_Sigma(X_env)^2 * y(s), "
                       "y(1+y^2(r*/r)^4)=1; hierarchical pairs (X_env = "
                       "dominant member's field) -> R = y^3 ~ s^4; "
                       "comparable-mass pairs (X_env = Galactic floor) -> "
                       "R = S_env^2 y ~ s^(4/3); R_s(M) = sqrt(GM/g_t); "
                       "S_Sigma(E) = [1+(g/g_t)^2]^-1",
        },
        "shell_radii_AU": {
            "derived_1Msun": float(shell_radius_au(M_SUN)),
            "derived_1p24Msun": float(shell_radius_au(M_SAMPLE_MSUN * M_SUN)),
            "derived_Earth": float(shell_radius_au(M_EARTH)),
            "fitted_wb_1p24Msun": RS_FIT_AU,
            "derived_over_fitted": float(
                shell_radius_au(M_SAMPLE_MSUN * M_SUN) / RS_FIT_AU
            ),
        },
        "shell_map": {},
        "amplitude_ledger": {},
    }

    # --- Pairwise response across the solar shell (1 M_sun source) --------
    # Hierarchical channels (photon/planet test member inside the solar
    # shell): R = y(s)^3. The wide-binary transition is a comparable-mass
    # pair: R = S_env^2 * y(s) with S_env ~ 0.5.
    rs_sun_derived = shell_radius_au(M_SUN)
    rs_sun_fitted = RS_FIT_AU * (1.0 / M_MEDIAN_MSUN) ** 0.5  # M^1/2 law
    separations = {
        "Cassini conjunction 1.6 Rsun": (1.6 * R_SUN_M / AU_M, "hier"),
        "1 AU": (1.0, "hier"),
        "Saturn 9.5 AU": (9.5, "hier"),
        "Neptune 30 AU": (30.0, "hier"),
        "WB transition 2646 AU": (RS_FIT_AU, "comp"),
        "Oort inner 1e4 AU": (1.0e4, "hier"),
        "Oort outer 1e5 AU": (1.0e5, "hier"),
    }
    for name, (s, kind) in separations.items():
        fn = (pair_response_hierarchical if kind == "hier"
              else pair_response_comparable)
        results["shell_map"][name] = {
            "s_AU": float(s),
            "regime": kind,
            "S_eff_derived_shell": float(fn(s, rs_sun_derived)),
            "S_eff_measured_shell": float(fn(s, rs_sun_fitted)),
        }

    # Earth-Moon pair: embedded in Earth's nonlinear field AND the solar
    # field at 1 AU (both pre-existing ambients for the pair).
    rs_earth = shell_radius_au(M_EARTH)
    s_llr_au = 3.844e8 / AU_M
    y_em = y_profile(s_llr_au / rs_earth)
    y_1au = y_profile(1.0 / rs_sun_derived)
    x_env_em = (1.0 - y_em) / y_em + (1.0 - y_1au) / y_1au
    q_env_em = 1.0 / (1.0 + x_env_em)
    results["shell_map"]["Earth-Moon 384400 km"] = {
        "s_AU": float(s_llr_au),
        "regime": "hier",
        "S_eff_derived_shell": float(q_env_em**2 * y_em),
    }

    # --- Environmental operator at benchmark field strengths ----------------
    results["environmental_S_Sigma"] = {
        "Earth surface g=9.82": float(env_operator(9.82)),
        "GNSS orbit 26560 km g=0.565": float(env_operator(0.565)),
        "Sun field at 1 AU": float(env_operator(G_SI * M_SUN / AU_M**2)),
        "MW solar-circle ambient g=2e-10": float(env_operator(2e-10)),
        "Void g=1e-12": float(env_operator(1e-12)),
    }

    # --- Amplitude ledger ----------------------------------------------------
    # v_tilde^2 -> 1 + 2 beta_A^2 S_eff(infty) at saturation; alpha_sat = 0.366
    # gives extra acceleration (1+alpha)^2 - 1 = 0.867 of g_N.
    extra_over_gN = (1.0 + ALPHA_SAT) ** 2 - 1.0
    s_eff_infty = extra_over_gN / (2.0 * BETA_A**2)
    results["amplitude_ledger"] = {
        "alpha_sat": ALPHA_SAT,
        "extra_acceleration_over_gN": float(extra_over_gN),
        "unscreened_response_over_gN": 2.0 * BETA_A**2,
        "S_eff_infty_implied": float(s_eff_infty),
        "partial_recovery_fraction": float(extra_over_gN / (2.0 * BETA_A**2)),
        "S_body_per_star_implied": float(np.sqrt(s_eff_infty)),
        "corpus_MW_ambient_S_Sigma": float(env_operator(2e-10)),
    }

    out = Path("results/outputs/014_shell_consistency_map.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("=== Screening-Shell Consistency Map ===")
    print(f"g_t = {G_T:.3e} m/s^2")
    print(f"R_shell(1 Msun)  derived = {rs_sun_derived:.0f} AU")
    print(f"R_shell(1.24 Msun) derived = {results['shell_radii_AU']['derived_1p24Msun']:.0f} AU"
          f"  vs fitted {RS_FIT_AU:.0f} AU"
          f"  (ratio {results['shell_radii_AU']['derived_over_fitted']:.2f})")
    print(f"R_shell(Earth) derived = {rs_earth:.0f} AU")
    for name, row in results["shell_map"].items():
        print(f"  {name:32s} S_eff derived={row['S_eff_derived_shell']:.2e}"
              f"  measured={row.get('S_eff_measured_shell', float('nan')):.2e}")
    for name, v in results["environmental_S_Sigma"].items():
        print(f"  {name:32s} S_Sigma = {v:.2e}")
    al = results["amplitude_ledger"]
    print(f"  plateau: +{al['extra_acceleration_over_gN']:.3f} g_N"
          f" = {al['partial_recovery_fraction']*100:.1f}% of unscreened 2*beta_A^2*g_N"
          f"  (S_body ~ {al['S_body_per_star_implied']:.2f})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
