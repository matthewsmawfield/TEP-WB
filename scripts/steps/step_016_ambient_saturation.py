#!/usr/bin/env python3
"""step_016_ambient_saturation.py — ambient-floor closure of the wide-binary
saturation amplitude (corpus issue 0-28).

The pair response of a comparable-mass binary is

    R = S_env^2 * y(s)      (ambient vertex product x profile)

so at large separations (y -> 1) the residual acceleration enhancement is
set entirely by the ambient Galactic suppression:

    (1 + alpha_sat)^2 - 1 = 2 beta_A^2 S_eff(infty),  S_eff = S_env^2.

This step performs the honest bidirectional check the audit asked for:

  FORWARD:  corpus ambient benchmark g_amb -> S_env -> predicted alpha_sat
            (prediction, not a fit of alpha_sat)
  INVERSE:  observed alpha_sat -> implied S_env -> implied g_amb
            (read-out of the ambient field from the plateau)

A genuine closure requires the corpus's *independent* Galactic ambient
field (temporal-landscape shear + Newtonian baryons at the solar circle)
to land on the implied g_amb.  The step therefore scans a grid of ambient
field estimates and reports where each convention lands, so the plateau
amplitude becomes a *probe* of the local ambient rather than a tuned
parameter.

Inputs:  none external — uses the canonical corpus constants and the
         fitted alpha_sat values already produced by steps 003/014/015.

Output:  results/outputs/016_ambient_saturation.json
"""
import json
import numpy as np
from pathlib import Path

G_SI = 6.67430e-11
C_SI = 299792.458e3
PC_M = 3.08567758e16
BETA_A = -1.0
H0_SI = 70e3 / (1e6 * PC_M)
G_T = C_SI * H0_SI / (2.0 * BETA_A**2)   # 3.40e-10 m/s^2


def env_operator(g_si):
    """S_Sigma(E) = [1 + (g/g_t)^2]^-1."""
    return 1.0 / (1.0 + (g_si / G_T) ** 2)


def alpha_from_s_eff(s_eff):
    """Velocity-boost saturation from a vertex-squared effective suppression."""
    return np.sqrt(1.0 + 2.0 * BETA_A**2 * s_eff) - 1.0


def s_eff_from_alpha(alpha):
    return ((1.0 + alpha) ** 2 - 1.0) / (2.0 * BETA_A**2)


def g_from_s_env(s_env):
    """Invert S_env = [1+(g/g_t)^2]^-1."""
    return G_T * np.sqrt(1.0 / s_env - 1.0)


def run():
    # Observed plateau amplitudes already in the pipeline
    alpha_canonical = 0.366   # step_003 TEP-exponential fit
    alpha_canonical_err = 0.012
    alpha_free = 0.44         # step_015 free-exponent fit

    results = {"g_t_m_s2": float(G_T), "beta_A": BETA_A}

    # ---------- INVERSE: observed plateau -> implied ambient ----------
    inverse = {}
    for name, alpha in [("canonical_fit", alpha_canonical),
                        ("free_exponent_fit", alpha_free)]:
        s_eff = s_eff_from_alpha(alpha)
        # Convention A (corpus vertex product): S_eff = S_env^2
        s_env_A = np.sqrt(s_eff)
        g_A = g_from_s_env(s_env_A)
        # Convention B (single-vertex): S_eff = S_env
        s_env_B = s_eff
        g_B = g_from_s_env(s_env_B)
        inverse[name] = {
            "alpha_sat": alpha,
            "S_eff_inf": float(s_eff),
            "vertex_product_convention": {
                "S_env_implied": float(s_env_A),
                "g_amb_implied_m_s2": float(g_A),
            },
            "single_vertex_convention": {
                "S_env_implied": float(s_env_B),
                "g_amb_implied_m_s2": float(g_B),
            },
        }
    results["inverse_implied_ambient"] = inverse

    # ---------- FORWARD: independent ambient benchmarks -> predicted plateau
    # Ambient-field benchmarks spanning plausible solar-circle values.
    # Newtonian baryonic ~1.5-2.2e-10 m/s^2 (McMillan 2017, de Salas+ 2019);
    # TEP ambient adds the temporal-landscape shear, so the effective
    # ambient seen by the pair can exceed the Newtonian floor.
    bench = {}
    for label, g in [
        ("Newtonian floor g=1.5e-10", 1.5e-10),
        ("corpus MW solar-circle benchmark g=2e-10", 2.0e-10),
        ("mid g=2.45e-10", 2.45e-10),
        ("high g=3.16e-10", 3.16e-10),
        ("transition scale g=g_t=3.40e-10", G_T),
    ]:
        s_env = env_operator(g)
        s_eff = s_env ** 2          # corpus vertex-product convention
        alpha_pred = alpha_from_s_eff(s_eff)
        bench[label] = {
            "g_amb_m_s2": float(g),
            "S_env": float(s_env),
            "S_eff_inf_vertex_sq": float(s_eff),
            "alpha_sat_predicted": float(alpha_pred),
        }
    results["forward_predicted"] = bench

    # ---------- Direct comparison ----------
    corpus_pred = bench["corpus MW solar-circle benchmark g=2e-10"]["alpha_sat_predicted"]
    resid = {
        "observed_alpha_sat": alpha_canonical,
        "observed_alpha_sat_err": alpha_canonical_err,
        "corpus_benchmark_predicted": float(corpus_pred),
        "overshoot_ratio": float(corpus_pred / alpha_canonical),
        "implied_g_amb_vertex_sq": inverse["canonical_fit"]["vertex_product_convention"]["g_amb_implied_m_s2"],
        "implied_g_amb_single_vertex": inverse["canonical_fit"]["single_vertex_convention"]["g_amb_implied_m_s2"],
    }
    resid["interpretation"] = (
        "Under the corpus's own R = S_env^2 y vertex-product structure, the "
        "benchmark ambient g_amb = 2e-10 m/s^2 predicts alpha_sat = %.3f "
        "vs observed 0.366+-0.012: a ~%.0f%% overshoot — the plateau is "
        "a partial recovery, not the fully unsuppressed response.  Inverting "
        "the observed plateau gives g_amb ~ %.2e-%.2e m/s^2 (convention-"
        "dependent), i.e. the solar-circle ambient must lie somewhat above "
        "the bare Newtonian floor — consistent in order with a Newtonian "
        "baryonic background plus a temporal-landscape shear contribution. "
        "This is a consistency *read-out* of the ambient field, not an "
        "independent verification; closing it requires the ambient field "
        "to be predicted a priori by the Galactic temporal-landscape model."
        % (corpus_pred, 100.0 * (corpus_pred / alpha_canonical - 1.0),
           resid["implied_g_amb_vertex_sq"],
           resid["implied_g_amb_single_vertex"]))
    results["residual_check"] = resid

    out = Path("results/outputs/016_ambient_saturation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("=== ambient saturation closure ===")
    print(json.dumps(resid, indent=2))
    return results


if __name__ == "__main__":
    run()
