#!/usr/bin/env python3
"""Step 020c: refined two-branch WB estimator — per-bin mass
marginalisation, ambient heterogeneity, and the two-centre profile.

Refinements over step_020 (same corrected ambient semantics):

(a) PER-BIN MASS MARGINALISATION.  The sample's system mass correlates
    with separation (selection: wider pairs are detected at larger
    distances and are more massive; corr(log M, log s) = 0.38).  Each
    observed bin is therefore marginalised over its OWN mass
    distribution drawn from the real catalogue rather than the global
    log-normal draw.

(b) AMBIENT HETEROGENEITY.  The solar-neighbourhood sample spans disk
    heights z ~ +-250 pc, so the local scalar shear varies about the
    solar-circle value.  Profiles are interpolated (log-linear in u0)
    across the step_56 ambient grid and the response is marginalised
    over a lognormal u0 distribution centred on the derived 0.163.

(c) TWO-CENTRE PAIR PROFILE, ORIENTATION AVERAGED.  The observable is
    the pair mutual force.  The parallel two-centre solve (020b, stars
    nested inside each other's axial wake) is the pessimistic channel;
    the perpendicular channel sits beside the compressed wake and
    follows the monopole equatorial response y_perp.  The isotropic
    estimate is <y> = (2/3) y_perp + (1/3) y_par.

(d) PER-STAR RESPONSE SCALE.  Each star's well scales with its own
    mass, r*_i = sqrt(G m_i/g_t); the relative scalar acceleration is
    (m2/m_tot) y(d/r*_2) + (m1/m_tot) y(d/r*_1).  For equal pairs the
    transition appears sqrt(2) earlier than the total-mass convention
    used in step_019/020.

(e) BASELINE-WINDOW SENSITIVITY.  The innermost observed bin carries
    documented unresolved-companion dilution; the observed canonical
    fit is refit under alternative normalisation windows.

Outputs: results/outputs/020c_refined_estimator.json
"""
import csv
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

AU = 1.496e11
M_SUN = 1.989e30
G = 6.674e-11
C = 299792458.0
H0 = 70e3 / 3.085677581e22
BETA = -1.0
G_T = C * H0 / (2 * BETA**2)
K = 16.03
U0_DERIVED = 0.16291036984356919

R_S_FIT = 2646.0
R_S_ERR_TOTAL = 609.0
ALPHA_OBS = 0.366
ALPHA_ERR = 0.012

BINS = np.logspace(np.log10(50), np.log10(30000), 20)


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


def profile_stack():
    """Load the step_56 two-branch profiles keyed by u0 and build a
    log-linear-in-u0 interpolant of y(x; u0) on a common x grid."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "..", "..", "..", "TEP", "results",
                     "step_56_embedded_two_branch.json")
    tb = json.load(open(p))["two_branch"]
    grids = {}
    us = sorted(float(u) for u in tb)
    xr = np.asarray(tb[str(us[0])]["r_over_rstar"])
    Y = np.zeros((len(us), xr.size))
    for i, u in enumerate(us):
        e = tb[str(u)]
        f = mk_interp(e["r_over_rstar"], e["y_emb"])
        Y[i] = f(xr)
    us = np.array(us)

    def y_at(x, u0):
        lx = np.log(np.asarray(x))
        iu = np.clip(np.searchsorted(us, u0), 1, len(us) - 1)
        t = (np.log(u0) - np.log(us[iu - 1])) / \
            (np.log(us[iu]) - np.log(us[iu - 1]))
        y0 = np.interp(lx, np.log(xr), Y[iu - 1])
        y1 = np.interp(lx, np.log(xr), Y[iu])
        return np.exp((1 - t) * np.log(np.clip(y0, 1e-30, None))
                      + t * np.log(np.clip(y1, 1e-30, None)))
    return y_at, xr, us


def canonical(x, a, rs):
    return 1.0 + a * (1.0 - np.exp(-x / rs))


def r_star_au(m_msun):
    return math.sqrt(G * m_msun * M_SUN / G_T) / AU


def forward_per_bin(df, s_obs, y_at, q2=1.0, ntheta=24,
                    het_sigma=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(314159)
    """Predicted median v_tilde per observed bin, marginalising over
    each bin's own mass distribution and (optionally) a lognormal
    ambient spread."""
    mu = (np.arange(ntheta) + 0.5) / ntheta
    inv_sin = 1.0 / np.sqrt(np.clip(1.0 - mu**2, 1e-3, None))
    out = np.zeros(s_obs.size)
    for i, si in enumerate(s_obs):
        m_bin = df.loc[(df["sep_AU"] >= BINS[i])
                       & (df["sep_AU"] < BINS[i + 1]),
                       "mass_total"].dropna().values
        if m_bin.size > 4000:
            m_bin = rng.choice(m_bin, 4000, replace=False)
        rs = np.array([r_star_au(mm) for mm in m_bin])
        u0s = (U0_DERIVED * np.exp(rng.normal(0.0, het_sigma, rs.size))
               if het_sigma else np.full(rs.size, U0_DERIVED))
        # (nmass, ntheta) true-separation/r* grid
        x = si * inv_sin[None, :] / rs[:, None]
        # ambient interpolation is scalar in u0 -- vectorise by sorting
        ybar = 0.0
        if het_sigma:
            # evaluate at each mass's own u0 (chunked for speed)
            for uu in np.unique(np.round(u0s, 3)):
                sel = np.abs(u0s - uu) < 5e-4
                ybar += float(np.mean(y_at(x[sel], uu))) * sel.mean()
        else:
            ybar = float(np.mean(y_at(x, U0_DERIVED)))
        out[i] = math.sqrt(1.0 + 2.0 * BETA**2 * q2 * ybar)
    return out


def forward_per_bin_component(df, s_obs, y_at, ntheta=24,
                              rng=None):
    """Honest per-star mass treatment.

    Each star's scalar well scales with its OWN mass: r*_i =
    sqrt(G m_i / g_t).  The relative scalar acceleration is

        a_phi / g_N = (m2/m_tot) y(d/r*_2) + (m1/m_tot) y(d/r*_1),

    evaluated at true separation d = s_proj/sin(theta).  For equal
    masses this evaluates the profile at r*_tot/sqrt(2), i.e. the
    transition appears sqrt(2) earlier in projected s than the
    total-mass convention used in step_019/020.
    """
    if rng is None:
        rng = np.random.default_rng(314159)
    mu = (np.arange(ntheta) + 0.5) / ntheta
    inv_sin = 1.0 / np.sqrt(np.clip(1.0 - mu**2, 1e-3, None))
    out = np.zeros(s_obs.size)
    for i, si in enumerate(s_obs):
        sel = ((df["sep_AU"] >= BINS[i]) & (df["sep_AU"] < BINS[i + 1]))
        sub = df.loc[sel, ["mass1_corr", "mass2_corr"]].dropna()
        if sub.empty:
            out[i] = 1.0
            continue
        if sub.shape[0] > 4000:
            sub = sub.iloc[rng.choice(sub.shape[0], 4000, replace=False)]
        m1 = sub["mass1_corr"].values
        m2 = sub["mass2_corr"].values
        mt = m1 + m2  # = mass_total, the canonical v_circ mass
        rs1 = np.array([r_star_au(mm) for mm in m1])
        rs2 = np.array([r_star_au(mm) for mm in m2])
        # (nmass, ntheta) true separation
        d = si * inv_sin[None, :]
        ybar = float(np.mean((m2 / mt)[:, None] * y_at(d / rs2[:, None],
                                                      U0_DERIVED)
                             + (m1 / mt)[:, None] * y_at(d / rs1[:, None],
                                                        U0_DERIVED)))
        out[i] = math.sqrt(1.0 + 2.0 * BETA**2 * ybar)
    return out


def pair_profile_interp():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "..", "..", "results", "outputs",
                     "020b_pair_profile_derived_u0.json")
    d = json.load(open(p))
    xs = np.array([r["d_over_rstar"] for r in d["rows"]])
    ys = np.array([r["y_pair_direct"] for r in d["rows"]])
    # extend: interior x^{4/3}-like approach, exterior -> monopole far
    # field plateau at the derived ambient
    return mk_interp(xs, ys)


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
    s, obs, sem = np.array(s), np.array(obs), np.array(sem)
    obs_n = obs / obs[:5].mean()
    p_obs, _ = curve_fit(canonical, s, obs_n, p0=[0.36, 2646], sigma=sem)

    df = pd.read_parquet(os.path.join(
        here, "..", "..", "data", "processed",
        "kinematic_results.parquet"))
    rng = np.random.default_rng(314159)

    y_at, xr, us = profile_stack()
    OUT = {"u0_derived": U0_DERIVED,
           "observed_refit": {"alpha": float(p_obs[0]),
                              "Rs": float(p_obs[1])},
           "runs": {}}

    def score(tag, pred):
        pn = pred / pred[:5].mean()
        p, _ = curve_fit(canonical, s, pn, p0=[0.36, 2646], sigma=sem)
        OUT["runs"][tag] = {
            "alpha": float(p[0]), "Rs_AU": float(p[1]),
            "alpha_sigma": float(abs(p[0] - p_obs[0]) / ALPHA_ERR),
            "Rs_sigma": float(abs(p[1] - p_obs[1]) / R_S_ERR_TOTAL),
            "profile": pn.tolist()}
        print("%-46s a=%.3f (%.2f) Rs=%.0f (%.2f)" %
              (tag, p[0], abs(p[0] - p_obs[0]) / ALPHA_ERR,
               p[1], abs(p[1] - p_obs[1]) / R_S_ERR_TOTAL), flush=True)

    # A. per-bin masses, single ambient, monopole profile
    pred = forward_per_bin(df, s, y_at)
    score("perbin_mass_monopole", pred)

    # B. per-bin masses + ambient heterogeneity sigma=0.2
    pred = forward_per_bin(df, s, y_at, het_sigma=0.2, rng=rng)
    score("perbin_mass_monopole_het0.2", pred)

    # C. pair profile (derived u0) with per-bin masses
    try:
        fp = pair_profile_interp()

        def y_pair_at(x, u0=U0_DERIVED):
            return fp(x)
        pred = forward_per_bin(df, s, y_pair_at)
        score("perbin_mass_pair_u0=%.3f" % U0_DERIVED, pred)
    except Exception as e:
        OUT["runs"]["pair_error"] = str(e)

    # E. per-STAR r* convention: monopole profile marginalised over
    #    real component masses per bin
    pred = forward_per_bin_component(df, s, y_at)
    score("component_mass_monopole", pred)

    # F. per-star r* + pair profile
    try:
        fp = pair_profile_interp()

        def y_pair_at2(x, u0=U0_DERIVED):
            return fp(x)
        pred = forward_per_bin_component(df, s, y_pair_at2)
        score("component_mass_pair", pred)
    except Exception as e:
        OUT["runs"]["component_pair_error"] = str(e)

    # G. HEADLINE: orientation-averaged pair at derived ambient.
    #    Mutual force for pair axis at angle theta to the ambient:
    #    perpendicular channel ~ monopole equatorial response y_perp
    #    (star sits beside the other's compressed wake), parallel
    #    channel = two-centre solve (stars nested inside each other's
    #    deep axial wake).  Isotropic pair axes give
    #    <y> = (2/3) y_perp + (1/3) y_pair_parallel.
    tb = json.load(open(os.path.join(
        here, "..", "..", "..", "TEP", "results",
        "step_56_embedded_two_branch.json")))["two_branch"]["0.15"]
    fperp = mk_interp(np.asarray(tb["s_over_rstar"]),
                      np.asarray(tb["y_emb_perp"]))
    fp = pair_profile_interp()

    def y_orient(x, u0=U0_DERIVED):
        return (2.0 / 3.0) * fperp(x) + (1.0 / 3.0) * fp(x)

    # corr component masses are the canonical mass_total components —
    # used for both the response weights and per-star r*
    pred = forward_per_bin_component(df, s, y_orient)
    score("component_mass_orient_pair", pred)

    # I. baseline-window sensitivity of the OBSERVED fit.  Bin 1
    #    (59 AU, N=128, residual -0.10) carries documented unresolved-
    #    companion dilution; excluding it from the normalisation is a
    #    data-quality choice the manuscript flags.
    sens = {}
    for lo, hi, tag in ((0, 5, "bins1-5"), (1, 5, "bins2-5"),
                        (2, 6, "bins3-6")):
        on = obs / obs[lo:hi].mean()
        pb, _ = curve_fit(canonical, s, on, p0=[0.36, 2646], sigma=sem)
        sens[tag] = {"alpha": float(pb[0]), "Rs_AU": float(pb[1]),
                     "plateau": float(on[-4:].mean())}
    OUT["baseline_window_sensitivity"] = sens

    # D. pair profile + heterogeneity handled implicitly (pair profile
    #    already at derived ambient only) -- approximate spread with the
    #    monopole u0-interpolated map around the pair's shape ratio is
    #    deferred; report the pair number as the headline.

    dest = os.path.join(outdir, "020c_refined_estimator.json")
    with open(dest, "w") as fh:
        json.dump(OUT, fh, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    run()
