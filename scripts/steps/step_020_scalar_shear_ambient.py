#!/usr/bin/env python3
"""Step 020: scalar-shear ambient + projection-corrected WB estimator.

Corrects three analysis errors in the step_017/019 estimator chain
(D2 suspects 1, 2 and the missing deprojection):

(1) AMBIENT SEMANTICS (suspect 1 -- reversion).  The embedded solve's
    boundary condition u0 is the scalar field's gradient variable u_bar
    (the Temporal Shear), whose flux obeys P_X(xi) u_bar = g_N/g_t.
    The corpus convention u_sun = 0.57 set u0 to the *total* solar-circle
    acceleration a_sun/g_t -- treating the g-sector Newtonian part as
    part of grad phi.  The self-consistent ambient is derived, not
    assumed: solve g_obs = g_N + 2 beta_A^2 g_t u_bar together with
    J = P_X u_bar = g_N/g_t, i.e.

        u_bar (P_X(u_bar^2/2) + 2 beta_A^2) = g_obs,sun / g_t

    with g_obs,sun = v_c^2/R_sun.  Each sector thereby predicts its own
    implied baryonic fraction g_N/g_obs at the solar circle -- a
    falsifiable side output.

(2) VERTEX DOUBLE-COUNTING (suspect 2 -- pre-derivation ansatz).
    y_emb is the output of a *resolved* embedded solve: the ambient
    medium's stiffness already suppresses the propagator once.  The
    corpus's extra q^2 = S_Sigma(u0)^2 vertex is a point-source/
    algebraic convention which, applied on top of the resolved
    propagator, applies the ambient suppression twice.  The pure
    propagator reading is the honest output of the resolved machinery;
    a per-star self-screening vertex would be set by each star's own
    well depth, not by the ambient stiffness.  All three readings are
    still reported for comparison.

(3) DEPROJECTION.  The observed v_tilde(s) bins by *projected*
    separation while the propagator lives on the true separation
    d >= s (mean d ~ 1.57 s for isotropic orbits).  The forward model
    here convolves y(d/r*) over the isotropic projection kernel
    d = s/sin(theta), as step_015's Monte Carlo already does for the
    ansatz family.

Inputs:  results/outputs/015_derived_operator_profiles.csv (observed)
         TEP step_56 machinery (embedded solves at the derived u0)
Outputs: results/outputs/020_scalar_shear_ambient.json
"""
import csv
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq, curve_fit

AU = 1.496e11
M_SUN = 1.989e30
G = 6.674e-11
C = 299792458.0
H0 = 70e3 / 3.085677581e22
BETA = -1.0
G_T = C * H0 / (2 * BETA ** 2)
K = 16.03

# solar-circle inputs (state provenance in output)
R_SUN_KPC = 8.2
V_C_KMS = 233.0           # local circular speed
KPC_M = 3.085677581e19
G_OBS_SUN = (V_C_KMS * 1e3) ** 2 / (R_SUN_KPC * KPC_M)   # ~2.15e-10
Y_OBS = G_OBS_SUN / G_T

R_S_FIT = 2646.0
R_S_ERR_TOTAL = 609.0
ALPHA_OBS = 0.366
ALPHA_ERR = 0.012


def px_sector(u, sector):
    """P_X(xi) with xi = u^2/2 for the candidate sectors."""
    if sector == "baseline":
        return 1.0 + u * u
    if sector == "exp_interp":
        return (1.0 - np.exp(-K * abs(u) / math.sqrt(2.0))) + u * u
    return K * abs(u) / math.sqrt(2.0) + u * u   # two-branch


def ambient_self_consistent(sector):
    """Solve u_bar (P_X + 2 beta^2) = g_obs,sun/g_t for the field's
    own gradient variable at the solar circle.  Returns u0 and the
    implied baryonic share g_N/g_obs."""
    y_obs = G_OBS_SUN / G_T

    def eq(ub):
        return ub * (px_sector(ub, sector) + 2.0 * BETA ** 2) - y_obs

    u0 = brentq(eq, 1e-6, 50.0)
    gN_over_gt = px_sector(u0, sector) * u0
    gN = gN_over_gt * G_T
    gshear = 2.0 * BETA ** 2 * u0 * G_T
    return {"u0": float(u0),
            "g_N_m_s2": float(gN),
            "g_shear_m_s2": float(gshear),
            "baryonic_share_of_vc2": float(gN / G_OBS_SUN),
            "vc_baryonic_kms": float(V_C_KMS * math.sqrt(gN / G_OBS_SUN))}


# ---------------------------------------------------------------------------
# embedded-solve machinery imported from the TEP repo (same solver state
# hygiene as step_59: restore every mutable module global on exit)
# ---------------------------------------------------------------------------
def load_step56():
    tep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "TEP")
    spec = importlib.util.spec_from_file_location(
        "step56", os.path.join(tep, "scripts", "steps",
                               "step_56_embedded_two_branch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def embedded_profile_at(u0, sector):
    """Run the resolved embedded solve at the derived ambient and return
    the orientation-mean y(r/r*) interpolant inputs."""
    m = load_step56()
    saved = {k: getattr(m, k) for k in
             ("STARVE", "STARVE_LAM", "KPHI_C", "KPHI_PERT",
              "PSI_FROZEN", "PHI0")}
    try:
        psi = m.solve_embedded(u0, sector)
        radii, ym = m.orient_mean_profile(psi)
        f0 = m.f_flux(abs(u0), sector)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    return radii, ym, float(f0)


def mk_interp(xr, yr):
    xr = np.asarray(xr)
    yr = np.asarray(yr)

    def f(x):
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.interp(x, xr, yr)
        lo = x < xr[0]
        out[lo] = yr[0] * (x[lo] / xr[0]) ** (4.0 / 3.0)
        out[x > xr[-1]] = yr[-1]
        return float(out[0]) if scalar else out
    return f


def r_star_au(m_msun):
    return math.sqrt(G * m_msun * M_SUN / G_T) / AU


def canonical(x, a, rs):
    return 1.0 + a * (1.0 - np.exp(-x / rs))


def forward_profile(s_obs, rs_au_arr, f, q2, ntheta=24, deproject=True):
    """v_tilde(s) = sqrt(1 + 2 beta^2 q2 <y(d/r*)>_M,theta).

    deproject=True convolves the pair propagator over the isotropic
    projection kernel d = s / sin(theta), theta isotropic on the sky
    (mu = cos theta uniform in [0,1)); deproject=False reproduces the
    step_019 convention y(s) for comparison."""
    mu = (np.arange(ntheta) + 0.5) / ntheta       # mu = cos(theta)
    inv_sin = 1.0 / np.sqrt(np.clip(1.0 - mu ** 2, 1e-3, None))
    out = np.zeros(s_obs.size)
    rs_col = rs_au_arr[:, None]                   # (nmass, 1)
    for i, si in enumerate(s_obs):
        if deproject:
            # true separations d = s/sin(theta): (nmass, ntheta) grid
            x = si * inv_sin[None, :] / rs_col
        else:
            x = np.full((rs_au_arr.size, 1), si) / rs_col
        ybar = float(np.mean(f(x)))
        out[i] = math.sqrt(1.0 + 2.0 * BETA ** 2 * q2 * ybar)
    return out


def run():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "..", "results", "outputs")
    src = os.path.join(outdir, "015_derived_operator_profiles.csv")

    s, obs, sem = [], [], []
    with open(src) as f:
        for r in csv.DictReader(f):
            s.append(float(r["sep_AU"]))
            obs.append(float(r["observed"]))
            sem.append(float(r["sem"]))
    s = np.array(s)
    obs = np.array(obs)
    sem = np.array(sem)

    obs_n = obs / obs[:5].mean()
    p_obs_wt, _ = curve_fit(canonical, s, obs_n, p0=[0.36, 2646],
                            sigma=sem)
    obs_a, obs_r = float(p_obs_wt[0]), float(p_obs_wt[1])

    # identical sample mass draw as step_017/019 for comparability
    rng = np.random.default_rng(314159)
    sig = math.sqrt(math.log(1 + 0.42 ** 2))
    mu_l = math.log(1.24) - 0.5 * sig ** 2
    M = rng.lognormal(mu_l, sig, 4000)
    M = M[(M >= 0.1) & (M <= 5.7)]
    rs_arr = np.array([r_star_au(m) for m in M])

    OUT = {
        "inputs": {
            "g_t": G_T,
            "R_sun_kpc": R_SUN_KPC,
            "v_c_kms": V_C_KMS,
            "g_obs_sun_m_s2": G_OBS_SUN,
            "observed_refit_alpha": obs_a,
            "observed_refit_Rs": obs_r,
            "corrections": [
                "u0 is the scalar-shear gradient u_bar solving "
                "u_bar(P_X+2) = g_obs,g_t-normalised -- not the total "
                "acceleration (corpus u_sun=0.57 was the wrong variable)",
                "resolved-solve propagator carries ambient suppression "
                "once; pure-propagator reading is the honest output",
                "isotropic deprojection convolution added (d = s/sin th)"],
        },
        "ambients": {},
        "sectors": {},
    }

    t0 = time.time()
    for sector in ("two_branch", "baseline", "exp_interp"):
        amb = ambient_self_consistent(sector)
        OUT["ambients"][sector] = amb
        u0 = amb["u0"]
        print(f"[{sector}] derived ambient u0 = {u0:.4f} "
              f"(baryonic share {100*amb['baryonic_share_of_vc2']:.0f}%)",
              flush=True)

        radii, ym, f0 = embedded_profile_at(u0, sector)
        f = mk_interp(radii, ym)

        sec = {"u0_derived": u0, "ambient_PX": f0,
               "y_far_derived": float(ym[-6:].mean()),
               "profiles_r_over_rstar": radii.tolist(),
               "y_emb": ym.tolist(), "readings": {}}
        for label, q2 in (("pure_propagator", 1.0),
                          ("single_vertex_S", 1.0 / f0),
                          ("corpus_vertex_S2", (1.0 / f0) ** 2)):
            for dp, tag in ((True, "deprojected"), (False, "raw_s")):
                pred = forward_profile(s, rs_arr, f, q2, deproject=dp)
                pn = pred / pred[:5].mean()
                p, _ = curve_fit(canonical, s, pn, p0=[0.36, 2646],
                                 sigma=sem)
                sec["readings"][f"{label}_{tag}"] = {
                    "q2": float(q2), "deprojected": dp,
                    "alpha": float(p[0]), "Rs_AU": float(p[1]),
                    "alpha_sigma": float(abs(p[0] - obs_a) / ALPHA_ERR),
                    "Rs_sigma": float(abs(p[1] - obs_r) / R_S_ERR_TOTAL)}
        OUT["sectors"][sector] = sec
        print(f"[{sector}] done in {time.time()-t0:.0f}s", flush=True)

    # heterogeneity: lognormal jitter on the derived ambient (the disk's
    # local field varies across the sample's z and R range)
    sec = OUT["sectors"]["two_branch"]
    f_der = mk_interp(sec["profiles_r_over_rstar"], sec["y_emb"])
    jitter = {}
    for sig_u in (0.15, 0.30):
        us = sec["u0_derived"] * np.exp(
            rng.normal(0.0, sig_u, 4000))
        # profile depends on u0 through y_emb; approximate the spread by
        # weighting the derived-u0 profile and the bracketing published
        # grid points is deferred -- record the ambient spread only
        jitter[f"ambient_lognormal_sigma_{sig_u}"] = {
            "u0_p16_p84": [float(np.percentile(us, 16)),
                           float(np.percentile(us, 84))]}

    # verdict: joint-closure test at the *derived* ambient per sector
    verdict = {}
    for sector, sec in OUT["sectors"].items():
        r = sec["readings"]["pure_propagator_deprojected"]
        chi2 = ((r["alpha"] - obs_a) / ALPHA_ERR) ** 2 + \
               ((r["Rs_AU"] - obs_r) / R_S_ERR_TOTAL) ** 2
        verdict[sector] = {
            "u0": sec["u0_derived"],
            "alpha_pred": r["alpha"], "Rs_pred": r["Rs_AU"],
            "alpha_sigma": r["alpha_sigma"], "Rs_sigma": r["Rs_sigma"],
            "joint_chi2": float(chi2),
            "baryonic_share": OUT["ambients"][sector]
                ["baryonic_share_of_vc2"]}
    OUT["verdict"] = {
        "observed": {"alpha": obs_a, "alpha_err": ALPHA_ERR,
                     "Rs_AU": obs_r, "Rs_err": R_S_ERR_TOTAL},
        "ambient_jitter_scales": jitter,
        "per_sector": verdict,
        "note": ("alpha and R_s are evaluated at each sector's own "
                 "self-consistent solar-circle ambient, derived from the "
                 "same flux law -- no per-channel ambient tuning.")}

    dest = os.path.join(outdir, "020_scalar_shear_ambient.json")
    with open(dest, "w") as fh:
        json.dump(OUT, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    run()
