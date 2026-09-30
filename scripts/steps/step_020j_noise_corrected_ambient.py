#!/usr/bin/env python3
"""Step 020j: Noise-corrected ambient inversion.

The definitive ambient inversion: per |Z| stratum, the two-branch
embedded response y(d/r*, u0) is applied to per-pair Newtonian
velocities (conditioned on observed projected separation, component
masses), the measured per-pair PM errors are added to the tangential
velocity vector, and the resulting noisy mock is binned and
inner-baseline-normalised exactly like the data.  chi2 of the
noise-aware TEP prediction against the observed profile is evaluated
on a u0 grid -> the noise-corrected effective ambient per stratum.

This supersedes the 020g raw-profile inversion, which fitted the
noiseless forward model to profiles that carry a Rice-rectified
noise floor (step_020i).  Thermal eccentricity law; the orbital
draw is fixed across u0 (common random numbers).

Outputs results/outputs/020j_noise_corrected_ambient.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "results", "outputs")
TEP_RES = os.path.join(HERE, "..", "..", "..", "TEP", "results")

G = 6.674e-11
M_SUN = 1.989e30
AU = 1.496e11
G_T = 3.4e-10  # canonical corpus value: c H_0 / (2 beta_A^2), H_0 = 70 km/s/Mpc
BINS = np.logspace(np.log10(50), np.log10(30000), 20)
U0_GRID = np.array([0.163, 0.18, 0.20, 0.24, 0.27,
                    0.35, 0.45, 0.60, 0.90, 1.5])
SEED = 7


def mk_interp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x)
    lx, ly = np.log(x[o]), np.log(np.clip(y[o], 1e-30, None))
    return lambda xx: np.exp(np.interp(np.log(np.asarray(xx)), lx, ly,
                                       left=ly[0], right=ly[-1]))


def load_pair_stack():
    st = {}
    for f in ["020b_pair_profile_derived_u0.json",
              "020e_pair_profile_strata.json",
              "020f_pair_profile_grid.json"]:
        p = os.path.join(OUTDIR, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["rows"]
        if isinstance(rows, dict):
            for u0, rr in rows.items():
                st[float(u0)] = (
                    np.asarray([r["d_over_rstar"] for r in rr]),
                    np.asarray([r.get("y_pair_direct", r["y_pair"])
                                for r in rr]))
        else:
            u0 = float(d.get("u0", 0.163))
            st[u0] = (
                np.asarray([r["d_over_rstar"] for r in rows]),
                np.asarray([r.get("y_pair_direct", r["y_pair"])
                            for r in rows]))
    return st


def load_perp_stack():
    st = {}
    main = json.load(open(os.path.join(
        TEP_RES, "step_56_embedded_two_branch.json")))["two_branch"]
    lp = os.path.join(TEP_RES, "step_56_emb_low_u0.json")
    low = json.load(open(lp)) if os.path.exists(lp) else {}
    for k, e in list(main.items()) + list(low.items()):
        if "y_emb_perp" in e and "s_over_rstar" in e:
            st[float(e["u0"])] = (np.asarray(e["s_over_rstar"]),
                                  np.asarray(e["y_emb_perp"]))
    return st


def lerp(usv, ipv, x, u0):
    if u0 >= usv[-1]:
        return np.clip(ipv[usv[-1]](x), 1e-30, None)
    if u0 <= usv[0]:
        return np.clip(ipv[usv[0]](x), 1e-30, None)
    iu = int(np.clip(np.searchsorted(usv, u0), 1, len(usv) - 1))
    t = (np.log(u0) - np.log(usv[iu - 1])) / \
        (np.log(usv[iu]) - np.log(usv[iu - 1]))
    return np.exp((1 - t) * np.log(np.clip(ipv[usv[iu - 1]](x), 1e-30, None))
                  + t * np.log(np.clip(ipv[usv[iu]](x), 1e-30, None)))


def rstar_au(m):
    return np.sqrt(G * m * M_SUN / G_T) / AU


def solve_kepler(ma, e):
    E = ma.copy()
    E[e > 0.8] = np.pi
    for _ in range(40):
        E -= (E - e * np.sin(E) - ma) / (1 - e * np.cos(E))
    return E


def main():
    pair = load_pair_stack()
    perp = load_perp_stack()
    us_p = np.array(sorted(pair.keys()))
    ip_p = {u: mk_interp(*v) for u, v in pair.items()}
    us_q = np.array(sorted(perp.keys()))
    ip_q = {u: mk_interp(*v) for u, v in perp.items()}

    def y_orient(x, u0):
        return ((2.0 / 3.0) * lerp(us_q, ip_q, x, u0)
                + (1.0 / 3.0) * lerp(us_p, ip_p, x, u0))

    df = pd.read_parquet(os.path.join(
        HERE, "..", "..", "data", "processed",
        "kinematic_results.parquet"))
    df = df[(df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)]
    df = df[(df["mass_total"] > 0) & (df["sep_AU"] > BINS[0])
            & (df["sep_AU"] < BINS[-1])].reset_index(drop=True)
    zpc = np.abs(df["Z_gc"]) * 1000.0
    qs = np.quantile(zpc, [0, .2, .4, .6, .8, 1.0])

    s = df["sep_AU"].to_numpy()
    M = df["mass_total"].to_numpy()
    d = df["dist_pc"].to_numpy()
    n = len(s)
    sig_ra = 4.74047e-3 * np.sqrt(df["pmra_error1"] ** 2
                                  + df["pmra_error2"] ** 2) * d
    sig_dec = 4.74047e-3 * np.sqrt(df["pmdec_error1"] ** 2
                                   + df["pmdec_error2"] ** 2) * d
    v_circ2d = np.sqrt(G * M_SUN * M / (s * AU)) / 1000.0

    rng = np.random.default_rng(SEED)
    cpsi = rng.uniform(-0.98, 0.98, n)
    spsi = np.maximum(np.sqrt(1 - cpsi ** 2), 1e-6)
    r3 = s / spsi                                   # true 3D separation (AU)
    vc3 = np.sqrt(G * M_SUN * M / (r3 * AU)) / 1000.0
    e_ = 0.95 * np.sqrt(rng.uniform(0, 1, n))       # thermal
    ma = rng.uniform(0, 2 * np.pi, n)
    E = solve_kepler(ma, e_)
    cnu = (np.cos(E) - e_) / (1 - e_ * np.cos(E))
    sfac = np.sqrt(np.maximum(2 - (1 - e_ * e_) /
                              np.maximum(1 + e_ * cnu, 1e-6), 0))
    ph = rng.uniform(0, 2 * np.pi, n)
    vpf = np.sqrt(np.cos(ph) ** 2 * cpsi ** 2 + np.sin(ph) ** 2)
    vt = vc3 * sfac * vpf
    nra = rng.standard_normal(n)
    ndec = rng.standard_normal(n)
    ang0 = rng.uniform(0, 2 * np.pi, n)
    ca, sa_ = np.cos(ang0), np.sin(ang0)

    m1 = df["mass1_corr"].to_numpy()
    m2 = df["mass2_corr"].to_numpy()
    rs1 = rstar_au(m1); rs2 = rstar_au(m2); mt = m1 + m2

    def vtm_at_u0(u0):
        yb = ((m2 / mt) * y_orient(r3 / rs2, u0)
              + (m1 / mt) * y_orient(r3 / rs1, u0))
        fac = np.sqrt(1.0 + 2.0 * yb)
        vx = vt * fac * ca + sig_ra * nra
        vy = vt * fac * sa_ + sig_dec * ndec
        return np.sqrt(vx * vx + vy * vy) / v_circ2d

    VTM = {u0: vtm_at_u0(u0) for u0 in U0_GRID}

    out = {"method": "TEP y(d/r*,u0) through per-pair PM-noise "
                     "kernel; thermal e-law; per-stratum chi2 on u0 grid",
           "u0_grid": U0_GRID.tolist(), "strata": []}
    for i in range(5):
        idx = np.where((zpc >= qs[i]) & (zpc <= qs[i + 1]))[0]
        sub = pd.DataFrame({"sep": s[idx],
                            "vt": df["v_tilde"].to_numpy()[idx]})
        sub["bin"] = pd.cut(sub["sep"], BINS)
        g = sub.groupby("bin", observed=False)["vt"]
        med = g.median().to_numpy()
        sd = g.std().to_numpy()
        cnt = g.count().to_numpy()
        so = np.isfinite(med) & (cnt > 50)
        base = np.nanmean(med[so][:5])
        yo = med / base
        eo = 1.253 * sd / np.sqrt(np.maximum(cnt, 1)) / base

        chis = []
        for u0 in U0_GRID:
            w = pd.DataFrame({"sep": s[idx], "vt": VTM[u0][idx]})
            w["bin"] = pd.cut(w["sep"], BINS)
            pm_ = w.groupby("bin", observed=False)["vt"].median().to_numpy()
            pm_ = pm_ / np.nanmean(pm_[np.isfinite(pm_)][:5])
            msk = so & np.isfinite(pm_) & (eo > 0)
            chis.append(float(np.sum(((yo[msk] - pm_[msk])
                                      / eo[msk]) ** 2)))
        chis = np.array(chis)
        ib = int(np.argmin(chis))
        # 1-sigma band: chi2 <= min + 1
        ok = np.where(chis <= chis[ib] + 1.0)[0]
        rec = {"stratum": i + 1,
               "median_z_pc": float(np.median(zpc[idx])),
               "n_pairs": int(len(idx)),
               "best_u0": float(U0_GRID[ib]),
               "u0_lo_1sig": float(U0_GRID[ok[0]]),
               "u0_hi_1sig": float(U0_GRID[ok[-1]]),
               "chi2_min": float(chis[ib]),
               "chi2_per_u0": dict(zip([f"{u:.3f}" for u in U0_GRID],
                                       chis.tolist())),
               "at_grid_edge": bool(ib == 0 or ib == len(U0_GRID) - 1)}
        out["strata"].append(rec)
        print(f"z{i+1} z~{rec['median_z_pc']:.0f}pc "
              f"n={len(idx)}: best u0={U0_GRID[ib]:.2f} "
              f"[{U0_GRID[ok[0]]:.2f},{U0_GRID[ok[-1]]:.2f}] "
              f"chi2={chis[ib]:.0f} edge={rec['at_grid_edge']}")

    outpath = os.path.join(OUTDIR, "020j_noise_corrected_ambient.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", outpath)


if __name__ == "__main__":
    main()
