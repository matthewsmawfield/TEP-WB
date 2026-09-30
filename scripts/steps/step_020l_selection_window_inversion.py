#!/usr/bin/env python3
"""Step 020l: Ambient inversion on the selection-insensitive window.

Step_020k showed the R_chance_align < 0.01 cut manufactures the widest-bin
downturn: per-stratum profiles are identical across R_chance thresholds for
s <~ 9 kAU (the cut removes <1.3% of pairs) but diverge above it.  Fits and
ambient inversions that use the outer bins are therefore contaminated by the
selection function, not by the physical response.

This step repeats the step_020j noise-aware ambient inversion -- the
two-branch embedded response y(d/r*, u0) applied to per-pair Newtonian
velocities conditioned on observed projected separation and component
masses, with measured per-pair PM errors added to the tangential velocity
vector -- but evaluates chi2 ONLY over bins with bin centre < 9 kAU, the
selection-insensitive window.  A pure-Newtonian reference (response factor
unity through the same noise kernel) supplies the window-restricted
detection delta-chi2 per stratum.

Two profile conventions are evaluated.  The pipeline's per-stratum
self-normalisation (each profile divided by the mean of its first five
populated bins) anchors different strata on different separation ranges --
the high-|Z| anchor is itself noise-inflated -- which manufactures a
spurious mid-strata "Newtonian preference".  The absolute (unnormalised)
median-v_tilde comparison removes the artifact; mass-calibration bias
cancels because the forward model uses the same mass_total column as the
data.  The absolute-profile inversion is the headline result; the
normalised comparison is retained as a diagnostic.  Eccentricity-law
sensitivity is scanned over circular/uniform/thermal/superthermal.

Outputs results/outputs/020l_selection_window.json
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
S_MAX_WINDOW = 9000.0          # selection-insensitive window (step_020k)
U0_GRID = np.concatenate([
    np.linspace(0.05, 0.30, 26),
    np.linspace(0.35, 1.5, 24),
])
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

    m1 = df["mass1_corr"].to_numpy()
    m2 = df["mass2_corr"].to_numpy()
    rs1 = rstar_au(m1); rs2 = rstar_au(m2); mt = m1 + m2
    obs_vt = df["v_tilde"].to_numpy()

    bc = np.sqrt(BINS[:-1] * BINS[1:])
    win = bc < S_MAX_WINDOW

    def draw_ecc(kind, rng):
        if kind == "circular":
            return np.zeros(n)
        if kind == "uniform":
            return rng.uniform(0.0, 0.95, n)
        if kind == "thermal":
            return 0.95 * np.sqrt(rng.uniform(0.0, 1.0, n))
        return 0.95 * rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)

    # Observed per-stratum bin statistics (computed once, shared across
    # eccentricity laws and both profile conventions)
    strata_obs = []
    for i in range(5):
        idx = np.where((zpc >= qs[i]) & (zpc <= qs[i + 1]))[0]
        sub = pd.DataFrame({"sep": s[idx], "vt": obs_vt[idx]})
        sub["bin"] = pd.cut(sub["sep"], BINS)
        g = sub.groupby("bin", observed=False)["vt"]
        med = g.median().to_numpy()
        sd = g.std().to_numpy()
        cnt = g.count().to_numpy()
        so = np.isfinite(med) & (cnt > 50) & win
        strata_obs.append({
            "idx": idx, "med": med, "cnt": cnt, "so": so,
            "sem": 1.253 * sd / np.sqrt(np.maximum(cnt, 1)),
            "median_z_pc": float(np.median(zpc[idx])),
        })

    def median_profile(vtm, idx):
        w = pd.DataFrame({"sep": s[idx], "vt": vtm[idx]})
        w["bin"] = pd.cut(w["sep"], BINS)
        return w.groupby("bin", observed=False)["vt"].median().to_numpy()

    out = {"method": "TEP y(d/r*,u0) through per-pair PM-noise kernel; "
                     "chi2 restricted to bins < 9 kAU "
                     "(selection-insensitive window, step_020k); "
                     "absolute (unnormalised) and self-normalised "
                     "conventions; four eccentricity laws",
           "s_max_window_au": S_MAX_WINDOW,
           "u0_grid": U0_GRID.tolist(),
           "eccentricity_laws": {}}

    for ecc in ["circular", "uniform", "thermal", "superthermal"]:
        rng = np.random.default_rng(SEED)
        cpsi = rng.uniform(-0.98, 0.98, n)
        spsi = np.maximum(np.sqrt(1 - cpsi ** 2), 1e-6)
        r3 = s / spsi
        vc3 = np.sqrt(G * M_SUN * M / (r3 * AU)) / 1000.0
        e_ = draw_ecc(ecc, rng)
        ma = rng.uniform(0, 2 * np.pi, n)
        if ecc == "circular":
            sfac = np.ones(n)
        else:
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

        def vtm_at_u0(u0):
            yb = ((m2 / mt) * y_orient(r3 / rs2, u0)
                  + (m1 / mt) * y_orient(r3 / rs1, u0))
            fac = np.sqrt(1.0 + 2.0 * yb)
            vx = vt * fac * ca + sig_ra * nra
            vy = vt * fac * sa_ + sig_dec * ndec
            return np.sqrt(vx * vx + vy * vy) / v_circ2d

        vtm_newton = np.sqrt((vt * ca + sig_ra * nra) ** 2
                             + (vt * sa_ + sig_dec * ndec) ** 2) / v_circ2d
        VTM = {u0: vtm_at_u0(u0) for u0 in U0_GRID}

        ecc_rec = {"strata": []}
        for i, st in enumerate(strata_obs):
            idx, med, so = st["idx"], st["med"], st["so"]
            sem = st["sem"]
            # convention 1: absolute medians (no self-normalisation)
            eo_abs = sem
            # convention 2: pipeline self-normalisation on first 5
            base = np.nanmean(med[so][:5])
            yo_n = med / base
            eo_n = sem / base

            def chi2(vtm, normalise):
                pm_ = median_profile(vtm, idx)
                if normalise:
                    pm_ = pm_ / np.nanmean(pm_[np.isfinite(pm_) & so][:5])
                    yv, ev = yo_n, eo_n
                else:
                    yv, ev = med, eo_abs
                msk = so & np.isfinite(pm_) & np.isfinite(ev) & (ev > 0)
                return float(np.sum(((yv[msk] - pm_[msk]) / ev[msk]) ** 2))

            rec = {"stratum": i + 1,
                   "median_z_pc": st["median_z_pc"],
                   "n_pairs": int(len(idx)),
                   "n_bins_window": int(so.sum())}
            for tag, norm in [("absolute", False), ("normalised", True)]:
                cN = chi2(vtm_newton, norm)
                chis = np.array([chi2(VTM[u0], norm) for u0 in U0_GRID])
                ib = int(np.argmin(chis))
                ok1 = np.where(chis <= chis[ib] + 1.0)[0]
                rec[tag] = {
                    "chi2_newton": cN,
                    "best_u0": float(U0_GRID[ib]),
                    "u0_lo_1sig": float(U0_GRID[ok1[0]]),
                    "u0_hi_1sig": float(U0_GRID[ok1[-1]]),
                    "chi2_min": float(chis[ib]),
                    "delta_chi2_vs_newton": float(cN - chis[ib]),
                    "at_grid_edge": bool(ib == 0
                                         or ib == len(U0_GRID) - 1)}
            ecc_rec["strata"].append(rec)
            a = rec["absolute"]
            print(f"{ecc} z{i+1} z~{st['median_z_pc']:.0f}pc "
                  f"nbins={rec['n_bins_window']}: abs u0={a['best_u0']:.3f} "
                  f"[{a['u0_lo_1sig']:.3f},{a['u0_hi_1sig']:.3f}] "
                  f"chi2={a['chi2_min']:.0f} "
                  f"dchi2_vs_N={a['delta_chi2_vs_newton']:.0f} "
                  f"edge={a['at_grid_edge']} | "
                  f"norm u0={rec['normalised']['best_u0']:.3f} "
                  f"dchi2={rec['normalised']['delta_chi2_vs_newton']:.0f}")
        out["eccentricity_laws"][ecc] = ecc_rec

    outpath = os.path.join(OUTDIR, "020l_selection_window.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", outpath)


if __name__ == "__main__":
    main()
