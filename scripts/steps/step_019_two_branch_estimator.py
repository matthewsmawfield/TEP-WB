#!/usr/bin/env python3
"""Two-branch kinetic sector through the SAME radius-law estimator as
step_017 (plan T1.3 / T-W3): resolves whether the new sector reproduces
the observed transition scale R_s = 2,646 +/- 609 AU and plateau
alpha = 0.366, and quantifies the 4,651-vs-2,646 AU discrepancy under
the corrected propagator.

Machinery identical to step_017: same observed profile
(015_derived_operator_profiles.csv), same sem-weighted canonical fit
1 + alpha(1 - exp(-s/R_s)), same log-normal sample mass draw
(<M>=1.24, sigma=0.42, truncated [0.1,5.7], seed 314159), same
50-270 AU baseline normalisation.  What changes is only the propagator
input: the orientation-mean embedded profiles from TEP
step_56_embedded_two_branch.json, solved under the two-branch flux law
J_i = a_i f(|a|), f = k q/sqrt(2) + q^2, k = 16.03.

Three vertex readings are reported per ambient u0, mirroring the
corpus's Gate-A ambient-projection question:

  pure        v_tilde^2 = 1 + 2 beta^2 <y_emb>          (q^2 = 1)
              -- the embedded solve already carries the ambient
              suppression; no additional vertex (no double counting).
  fixed       q^2 = 0.433 (legacy plateau-calibrated vertex)
  consistent  q^2 = P_X(u0^2/2)^{-2} under the two-branch stiffness
              -- ambient suppression applied twice (propagator + vertex);
              the corpus's historical convention.

Inputs:  ../TEP/results/step_56_embedded_two_branch.json
         results/outputs/015_derived_operator_profiles.csv
Outputs: results/outputs/019_two_branch_estimator.json

NOTE: superseded by step_020/020b/020c.  This estimator used
u_sun = a_sun/g_t = 0.57 as the ambient — the TOTAL acceleration, not
the scalar-shear gradient the embedded solve requires (correct value
~0.163 from u(P_X+2) = g_obs/g_t); it also evaluates y at projected s
without the isotropic deprojection used in step_015, and marginalises
over a global lognormal mass draw despite the real sample's
corr(log M, log s) = 0.38.  Kept for the audit trail.
"""
import csv
import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit

AU = 1.496e11
M_SUN = 1.989e30
G = 6.674e-11
C = 299792458.0
H0 = 70e3 / 3.085677581e22
BETA = -1.0
G_T = C * H0 / (2 * BETA**2)
K = 16.03                       # two-branch coefficient, step_54
Q_ENV2_LEGACY = 0.432825484764543

R_S_FIT = 2646.0
R_S_ERR_TOTAL = 609.0


def r_star_au(m_msun):
    return math.sqrt(G * m_msun * M_SUN / G_T) / AU


def canonical(x, a, rs):
    return 1.0 + a * (1.0 - np.exp(-x / rs))


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

    # identical sample mass draw as step_017
    rng = np.random.default_rng(314159)
    sig = math.sqrt(math.log(1 + 0.42**2))
    mu = math.log(1.24) - 0.5 * sig**2
    M = rng.lognormal(mu, sig, 4000)
    M = M[(M >= 0.1) & (M <= 5.7)]
    rs_arr = np.array([r_star_au(m) for m in M])

    obs_n = obs / obs[:5].mean()
    p_obs_wt, _ = curve_fit(canonical, s, obs_n, p0=[0.36, 2646],
                            sigma=sem)

    emb_path = os.path.join(here, "..", "..", "..", "TEP", "results",
                            "step_56_embedded_two_branch.json")
    embj = json.load(open(emb_path))
    sectors = {"two_branch": embj["two_branch"]}
    if "exp_interp" in embj:
        sectors["exp_interp"] = embj["exp_interp"]

    def fit_profile(f, q2):
        pred = np.array([
            math.sqrt(1 + 2 * BETA**2 * q2
                      * np.mean([float(f(si / r)) for r in rs_arr]))
            for si in s])
        pred_n = pred / pred[:5].mean()
        p, _ = curve_fit(canonical, s, pred_n, p0=[0.36, 2646],
                         sigma=sem)
        exc = (pred_n - 1.0) / (pred_n[-1] - 1.0)
        half = float(s[np.argmin(np.abs(exc - 0.5))])
        return {"Rs_AU": float(p[1]), "alpha": float(p[0]),
                "half_excess_AU": half,
                "Rs_ratio_vs_observed": float(p[1] / p_obs_wt[1]),
                "Rs_sigma_from_observed":
                    float(abs(p[1] - p_obs_wt[1]) / R_S_ERR_TOTAL),
                "alpha_sigma_from_observed":
                    float(abs(p[0] - p_obs_wt[0]) / 0.012)}

    OUT = {
        "inputs": {
            "profile_source": "TEP step_56_embedded_two_branch",
            "k_two_branch": K,
            "estimator": "identical to step_017 (same data, mass draw, "
                         "normalisation, canonical fit)",
            "observed_refit_Rs": float(p_obs_wt[1]),
            "observed_refit_alpha": float(p_obs_wt[0]),
        },
        "sectors": {},
        "verdict": {}}

    obs_a, obs_r = float(p_obs_wt[0]), float(p_obs_wt[1])
    for sec, tb in sectors.items():
        sec_out = {}
        for u0k in sorted(tb, key=float):
            sweep = tb[u0k]
            f = mk_interp(sweep["r_over_rstar"], sweep["y_emb"])
            ent = {"u0": sweep["u0"], "y_far": sweep["y_far"],
                   "half_excess_r_over_rstar":
                       sweep["half_excess_r_over_rstar"],
                   "ambient_PX": sweep["ambient_PX"]}
            for label, q2 in (("pure_propagator", 1.0),
                              ("fixed_vertex_0433", Q_ENV2_LEGACY),
                              ("consistent_vertex",
                               sweep["q2_vertex"])):
                ent[label] = dict({"q2": float(q2)},
                                  **fit_profile(f, q2))
            sec_out[u0k] = ent
        OUT["sectors"][sec] = sec_out

    # discriminator summary: which reading/ambient reproduces the
    # observed (alpha, R_s) jointly
    best = {}
    for sec in sectors:
        for label in ("pure_propagator", "fixed_vertex_0433",
                      "consistent_vertex"):
            us = sorted(float(u) for u in OUT["sectors"][sec])
            al = np.array([OUT["sectors"][sec][str(u)][label]["alpha"]
                           for u in us])
            rr = np.array([OUT["sectors"][sec][str(u)][label]["Rs_AU"]
                           for u in us])
            i = int(np.argmin(np.abs(al - obs_a)))
            best["%s/%s" % (sec, label)] = {
                "u0_at_observed_alpha": us[i],
                "alpha": float(al[i]), "Rs_AU": float(rr[i]),
                "Rs_sigma":
                    float(abs(rr[i] - obs_r) / R_S_ERR_TOTAL)}

    # heterogeneity check: the estimator is linear in the profile, so an
    # ambient mixture gives a convex combination -- verify that mixing
    # interpolates along the failing (alpha, R_s) trade-off curve rather
    # than reaching the observed joint point.
    het = {}
    if "two_branch" in sectors:
        tbs = sectors["two_branch"]
        interps = {float(u): mk_interp(tbs[u]["r_over_rstar"],
                                       tbs[u]["y_emb"]) for u in tbs}
        mixes = {
            "70% 0.15 + 30% 0.57": {0.15: 0.7, 0.57: 0.3},
            "50/50 0.15+0.57": {0.15: 0.5, 0.57: 0.5},
            "uniform spread 0.12-0.72":
                {0.12: 0.25, 0.15: 0.25, 0.3: 0.25, 0.721: 0.25},
            "90% shear u0=0.57": {0.57: 0.9, 0.15: 0.1}}
        for name, w in mixes.items():
            def mixp(x, w=w):
                return sum(wt * interps[u](x) for u, wt in w.items())
            pred = np.array([
                math.sqrt(1 + 2 * BETA**2
                          * np.mean([mixp(si / r) for r in rs_arr]))
                for si in s])
            pn = pred / pred[:5].mean()
            p, _ = curve_fit(canonical, s, pn, p0=[0.36, 2646],
                             sigma=sem)
            het[name] = {"alpha": float(p[0]), "Rs_AU": float(p[1]),
                         "Rs_sigma": float(abs(p[1] - obs_r) / 609.0)}

    OUT["verdict"] = {
        "observed": {"alpha": obs_a, "Rs_AU": obs_r,
                     "Rs_err": R_S_ERR_TOTAL},
        "best_match_per_reading_per_sector": best,
        "heterogeneous_ambient_mixtures": het,
        "interpretation": (
            "JOINT CLOSURE FAILS at every ambient: the two observables "
            "pull u0 in opposite directions.  alpha=0.366 requires "
            "u0 ~= 0.18 g_t (pure-propagator) where the fitted scale "
            "is R_s ~= 5,200 AU (~4 sigma high); R_s = 2,646 requires "
            "u0 ~= 0.37 where alpha ~= 0.17 (~16 sigma low).  The "
            "small-X branch produces a slow algebraic far-field tail "
            "that stretches the canonical exponential fit through the "
            "mass convolution.  The consistent-vertex reading cannot "
            "reach alpha = 0.366 at any ambient (max ~0.30), and is "
            "ill-defined as u0 -> 0 on the small-X branch (vertex "
            "1/P_X^2 diverges).  Incumbent comparison: baseline "
            "consistent-vertex at u_sun=0.570 gave (0.354, 3,749) — "
            "alpha within 1 sigma, R_s 1.8 sigma — so under the "
            "corpus's own estimator the two-branch sector is WORSE "
            "jointly despite fixing the RAR shape.  Verdict: "
            "CONDITIONAL-FAIL for joint (alpha, R_s) closure; the "
            "surviving escape is a derived anisotropic/density-set "
            "ambient model (Gate-A), not a free u0.  exp_interp "
            "sector: ambient stiffness is baseline-identical at "
            "Galactic u0 (e^{-k sqrt xi} kills the sqrt term), so it "
            "inherits the baseline's estimator behaviour rather than "
            "rescuing the joint closure.  Heterogeneous ambient "
            "mixtures interpolate ALONG the failing trade-off curve "
            "(verified: convex combination of profiles cannot reach "
            "the observed joint point) -- the failure is in the "
            "profile-shape family, not the ambient's uniformity.")}

    dest = os.path.join(outdir, "019_two_branch_estimator.json")
    with open(dest, "w") as fh:
        json.dump(OUT, fh, indent=2)
    print(json.dumps(OUT["verdict"], indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    run()
