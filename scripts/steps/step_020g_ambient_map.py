#!/usr/bin/env python3
"""Empirical ambient map u_eff(|Z|) — inverts the fine |Z|-stratified
transition radii (step_009, fixed-alpha = 0.4 convention) through the
corrected forward model into a per-stratum ambient shear u0.

Inputs
- pair mutual-force profiles: 020b (u0=0.163), 020e (0.10, 0.23),
  020f (0.14, 0.17, 0.20) -- parallel channel, interpolated in u0.
- perpendicular channel: monopole equatorial response y_emb_perp from
  step_56 embedded solves (u0 = 0.08..1.1 incl. low-u0 aux file).
- forward model: component-mass per-star r* marginalisation +
  isotropic deprojection, as in 020c.

The map reports u_eff at the five step_009 stratum median heights and
compares the measured z-dependence against candidate projections
(u_eff ~ rho^n over the disk density model, and the |grad-phi|(z)
magnitude from the 1D vertical solve).

Outputs results/outputs/020g_ambient_map.json
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "results", "outputs")
TEP_RES = os.path.join(HERE, "..", "..", "..", "TEP", "results")

G = 6.674e-11
M_SUN = 1.989e30
G_T = 3.4e-10  # canonical corpus value: c H_0 / (2 beta_A^2), H_0 = 70 km/s/Mpc
AU = 1.496e11
BETA = -1.0
BINS = np.logspace(np.log10(50), np.log10(30000), 20)   # GLOBAL_BINS
ALPHA_FIXED = 0.4          # step_009 convention
BOOT_SEED = 314159


def mk_interp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x)
    lx, ly = np.log(x[o]), np.log(np.clip(y[o], 1e-30, None))
    return lambda xx: np.exp(np.interp(np.log(np.asarray(xx)), lx, ly,
                                       left=ly[0], right=ly[-1]))


def load_pair_stack():
    """Pair (parallel-channel) profiles keyed by u0 -> interp."""
    files = ["020b_pair_profile_derived_u0.json",
             "020e_pair_profile_strata.json",
             "020f_pair_profile_grid.json"]
    stack = {}
    for f in files:
        p = os.path.join(OUTDIR, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["rows"]
        if isinstance(rows, dict):
            for u0, rr in rows.items():
                xs = [r["d_over_rstar"] for r in rr]
                ys = [r.get("y_pair_direct", r["y_pair"]) for r in rr]
                stack[float(u0)] = (np.asarray(xs), np.asarray(ys))
        else:
            u0 = float(d.get("u0", 0.163))
            xs = [r["d_over_rstar"] for r in rows]
            ys = [r.get("y_pair_direct", r["y_pair"]) for r in rows]
            stack[u0] = (np.asarray(xs), np.asarray(ys))
    return stack


def load_perp_stack():
    """Monopole equatorial (perpendicular-channel) profiles per u0."""
    stack = {}
    main = json.load(open(os.path.join(TEP_RES,
                          "step_56_embedded_two_branch.json")))["two_branch"]
    low_path = os.path.join(TEP_RES, "step_56_emb_low_u0.json")
    low = json.load(open(low_path)) if os.path.exists(low_path) else {}
    for key, e in list(main.items()) + list(low.items()):
        if "y_emb_perp" in e and "s_over_rstar" in e:
            stack[float(e["u0"])] = (np.asarray(e["s_over_rstar"]),
                                     np.asarray(e["y_emb_perp"]))
    return stack


def stack_interp(stack):
    """u0-interpolated profile: log-linear in u0 over the grid."""
    us = np.array(sorted(stack.keys()))
    interps = {u: mk_interp(*v) for u, v in stack.items()}

    def f(x, u0):
        iu = int(np.clip(np.searchsorted(us, u0), 1, len(us) - 1))
        t = (np.log(u0) - np.log(us[iu - 1])) / \
            (np.log(us[iu]) - np.log(us[iu - 1]))
        y0 = np.clip(interps[us[iu - 1]](x), 1e-30, None)
        y1 = np.clip(interps[us[iu]](x), 1e-30, None)
        return np.exp((1 - t) * np.log(y0) + t * np.log(y1))
    return f, us


def r_star_au(m):
    return math.sqrt(G * m * M_SUN / G_T) / AU


def forward_component(df, s_obs, y_orient, u0, ntheta=24, rng=None):
    if rng is None:
        rng = np.random.default_rng(BOOT_SEED)
    mu = (np.arange(ntheta) + 0.5) / ntheta
    inv_sin = 1.0 / np.sqrt(np.clip(1.0 - mu**2, 1e-3, None))
    out = np.zeros(s_obs.size)
    for i, si in enumerate(s_obs):
        sel = (df["sep_AU"] >= BINS[i]) & (df["sep_AU"] < BINS[i + 1])
        sub = df.loc[sel, ["mass1_corr", "mass2_corr"]].dropna()
        if sub.empty:
            out[i] = 1.0
            continue
        if sub.shape[0] > 4000:
            sub = sub.iloc[rng.choice(sub.shape[0], 4000, replace=False)]
        m1 = sub["mass1_corr"].values
        m2 = sub["mass2_corr"].values
        mt = m1 + m2
        rs1 = np.array([r_star_au(mm) for mm in m1])
        rs2 = np.array([r_star_au(mm) for mm in m2])
        d = si * inv_sin[None, :]
        ybar = float(np.mean((m2 / mt)[:, None]
                     * y_orient(d / rs2[:, None], u0)
                     + (m1 / mt)[:, None]
                     * y_orient(d / rs1[:, None], u0)))
        out[i] = math.sqrt(1.0 + 2.0 * BETA**2 * ybar)
    return out


def canonical(x, a, rs):
    return 1.0 + a * (1.0 - np.exp(-x / rs))


def canonical_fixed(x, rs):
    return canonical(x, ALPHA_FIXED, rs)


def main():
    pair = load_pair_stack()
    perp = load_perp_stack()
    f_pair, u_pair = stack_interp(pair)
    f_perp, u_perp = stack_interp(perp)
    print("pair u0 grid:", u_pair.tolist())
    print("perp u0 grid:", u_perp.tolist())

    def y_orient(x, u0):
        return (2.0 / 3.0) * f_perp(x, u0) + (1.0 / 3.0) * f_pair(x, u0)

    df = pd.read_parquet(os.path.join(
        HERE, "..", "..", "data", "processed",
        "kinematic_results.parquet"))

    # observed 19-bin separation grid (as 020c reads it from the CSV)
    import csv as _csv
    with open(os.path.join(OUTDIR, "015_derived_operator_profiles.csv")) \
            as fh:
        s_obs = np.array([float(r["sep_AU"])
                          for r in _csv.DictReader(fh)])

    u0_grid = np.array([0.08, 0.10, 0.11, 0.13, 0.15, 0.163,
                        0.18, 0.20, 0.22, 0.24, 0.27])
    maps = {}
    for u0 in u0_grid:
        pred = forward_component(df, s_obs, y_orient, float(u0))
        pn = pred / pred[:5].mean()
        try:
            pf, _ = curve_fit(canonical, s_obs, pn, p0=[0.36, 2646])
            px, _ = curve_fit(canonical_fixed, s_obs, pn,
                              p0=[2646], bounds=([100], [100000]))
        except RuntimeError:
            continue
        maps["%.3f" % u0] = {"alpha_free": float(pf[0]),
                             "Rs_free": float(pf[1]),
                             "Rs_fixed_alpha04": float(px[0])}
        print("u0=%.3f  a=%.3f Rs_free=%.0f  Rs_fix=%.0f"
              % (u0, pf[0], pf[1], px[0]), flush=True)

    # invert the fine-|Z| R_s values (step_009, fixed-alpha=0.4)
    fz = pd.read_csv(os.path.join(OUTDIR,
                     "009_fine_z_stratification.csv"))
    us_fit = np.array(sorted(float(k) for k in maps))
    rs_fit = np.array([maps["%.3f" % u]["Rs_fixed_alpha04"]
                       for u in us_fit])
    o = np.argsort(rs_fit)

    def invert_rs(rs):
        # monotone-in-u0 section of the fit map
        return float(np.interp(rs, rs_fit[o], us_fit[o]))

    strata_out = []
    for _, r in fz.iterrows():
        ue = invert_rs(r["r_s_au"])
        err = r["r_s_err_au"]
        dudr = np.gradient(us_fit[o], rs_fit[o])
        ue_lo = invert_rs(r["r_s_au"] + err)
        ue_hi = invert_rs(r["r_s_au"] - err)
        strata_out.append({
            "z_med_pc": float(r["median_z_kpc"] * 1000),
            "rho": float(r["rho_bary_Msun_pc3"]),
            "rs_au": float(r["r_s_au"]), "rs_err": float(err),
            "u_eff": ue, "u_eff_lo": ue_lo, "u_eff_hi": ue_hi})
        print("z=%.0fpc rho=%.4f Rs=%.0f+-%0.f -> u_eff=%.3f (%.3f-%.3f)"
              % (r["median_z_kpc"] * 1000, r["rho_bary_Msun_pc3"],
                 r["r_s_au"], err, ue, ue_lo, ue_hi))

    # empirical exponent: u_eff vs rho
    rhos = np.array([s["rho"] for s in strata_out])
    ues = np.array([s["u_eff"] for s in strata_out])
    n_fit = np.polyfit(np.log(rhos), np.log(ues), 1)[0]

    # free-(alpha, Rs) fits per fine-|Z| stratum (same 5 quintile
    # bins as step_009, own inner-<500AU normalization).  Each
    # stratum's map is marginalised over the STRATUM'S OWN mass
    # distribution -- median system mass runs 1.11 -> 1.38 Msun
    # across quintiles, so a global mass kernel misassigns the
    # mass-driven R_s shift to ambient.
    free_strata = []
    zgc = np.abs(df["Z_gc"])
    qs = np.quantile(zgc, [0, .2, .4, .6, .8, 1.0])
    rng_f = np.random.default_rng(7)
    s_mid = np.sqrt(BINS[:-1] * BINS[1:])
    for i in range(5):
        sub = df[(zgc >= qs[i]) & (zgc <= qs[i + 1])].copy()
        base = sub.loc[sub["sep_AU"] < 500, "v_tilde"].median()
        sub["vn"] = sub["v_tilde"] / base
        g = sub.groupby(pd.cut(sub["sep_AU"], BINS), observed=True)["vn"]
        rows = [(np.sqrt(iv.left * iv.right), grp.median(),
                 1.253 * grp.std() / np.sqrt(len(grp)), len(grp))
                for iv, grp in g if len(grp) > 30]
        prof = pd.DataFrame(rows, columns=["sep", "v", "sem", "n"])
        if len(prof) < 6:
            continue
        try:
            p, _ = curve_fit(canonical, prof["sep"], prof["v"],
                             sigma=prof["sem"], p0=[0.36, 2646],
                             bounds=([0.0, 10.0], [1.0, 1e6]))
        except RuntimeError:
            continue
        xa, ya, ea = (prof["sep"].values, prof["v"].values,
                      prof["sem"].values)
        bs = []
        for _ in range(150):
            idx = rng_f.choice(len(xa), len(xa), replace=True)
            try:
                pb, _ = curve_fit(canonical, xa[idx], ya[idx],
                                  sigma=ea[idx], p0=list(p),
                                  bounds=([0.0, 10.0], [1.0, 1e6]))
                bs.append(pb)
            except RuntimeError:
                pass
        bs = np.array(bs)
        # per-stratum forward maps: this stratum's own mass kernel
        amap, rmap = [], []
        nbase = max(3, int((prof["sep"] < 500).sum()))
        for u0 in u0_grid:
            pred = forward_component(sub, prof["sep"].values,
                                     y_orient, float(u0))
            pn = pred / pred[:nbase].mean()
            pf, _ = curve_fit(canonical, prof["sep"], pn,
                              p0=[0.36, 2646], bounds=([0, 10], [1, 1e6]))
            amap.append(pf[0]); rmap.append(pf[1])
        free_strata.append({
            "z_med_pc": float(sub["Z_gc"].abs().median() * 1000),
            "mass_med": float(sub["mass_total"].median()),
            "alpha_obs": float(p[0]),
            "alpha_err": float(bs[:, 0].std()) if len(bs) > 10 else None,
            "rs_obs": float(p[1]),
            "rs_err": float(bs[:, 1].std()) if len(bs) > 10 else None,
            "stratum_alpha_map": [float(x) for x in amap],
            "stratum_Rs_map": [float(x) for x in rmap]})

    for s in free_strata:
        am = np.array(s["stratum_alpha_map"])
        rm = np.array(s["stratum_Rs_map"])
        s["u_eff_from_alpha"] = float(np.interp(
            s["alpha_obs"], am[::-1], u0_grid[::-1]))
        s["u_eff_from_Rs"] = float(np.interp(
            s["rs_obs"], rm[np.argsort(rm)], u0_grid[np.argsort(rm)]))
        print("z=%4.0fpc M=%.2f a=%.3f Rs=%5.0f -> u_a=%.3f u_Rs=%.3f"
              % (s["z_med_pc"], s["mass_med"], s["alpha_obs"],
                 s["rs_obs"], s["u_eff_from_alpha"],
                 s["u_eff_from_Rs"]))
    n_alpha = None
    if free_strata:
        # density-weighted rho per stratum over the actual pairs
        def rho_d(zpc):
            return (0.040 * np.exp(-zpc / 300.)
                    + 0.005 * np.exp(-zpc / 900.)
                    + 0.050 * np.exp(-zpc / 150.))
        rh = np.array([float(rho_d(np.abs(
            df.loc[(zgc >= qs[i]) & (zgc <= qs[i + 1]), "Z_gc"])
            * 1000).mean()) for i in range(len(free_strata))])
        ua = np.array([s["u_eff_from_alpha"] for s in free_strata])
        n_alpha = float(np.polyfit(np.log(rh), np.log(ua), 1)[0])
        for s, r_ in zip(free_strata, rh):
            s["rho_density_weighted"] = float(r_)

    out = {"alpha_fixed": ALPHA_FIXED,
           "u0_grid_forward_maps": maps,
           "fine_z_strata_fixed_alpha_convention": strata_out,
           "empirical_exponent_fixed_alpha": float(n_fit),
           "free_fit_strata": free_strata,
           "empirical_exponent_alpha_channel": n_alpha,
           "note": "CORRECTED (per-stratum mass kernel + density-weighted "
                   "rho): u_eff ~ rho^%.2f, contrast %.2f. Alpha and Rs "
                   "inversions agree at endpoints (~0.21/~0.20 midplane, "
                   "~0.12/~0.08 halo) but DIVERGE mid-strata (z=68pc: "
                   "0.177 vs 0.241) -- a single ambient per stratum does "
                   "not reproduce both profile parameters there. "
                   "Mechanism class: rho^{1/3} packing vs rho^{1/2} "
                   "density-coupled both inside ~1.5 sigma; the "
                   "constraint-slice (AUD-3) projection remains the "
                   "derivation."
                   % (n_alpha or 0.0,
                      free_strata[0]["u_eff_from_alpha"]
                      / free_strata[-1]["u_eff_from_alpha"]
                      if free_strata else float("nan"))}
    dest = os.path.join(OUTDIR, "020g_ambient_map.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
