#!/usr/bin/env python3
"""Step 020i: Noise-aware Newtonian null for the v_tilde profile.

The v_tilde observable is a *magnitude* of a noisy 2D vector:
  v_tilde = |mu_rel| * d / v_circ.
Gaussian proper-motion errors rectify the magnitude upward (Rice bias),
and the per-pair noise floor sigma_vtilde = sigma_mu * d / v_circ grows
with distance d (hence with |Z| stratum) and with projected separation s
(v_circ ~ s^{-1/2}).  The step_012 Newtonian forward model is noiseless,
so the published null underestimates the expected profile at wide bins --
most severely in the distant high-|Z| strata.

This step rebuilds the Newtonian null with the measured per-pair
astrometric errors added to the tangential velocity vector, and reports:

  * the noise-rectified Newtonian profile per eccentricity law,
  * the residual profile obs / null per |Z| stratum,
  * corrected (alpha, R_s) fits on the residual profiles,
  * the environmental ordering of the corrected residuals.

Outputs results/outputs/020i_noise_null.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "results", "outputs")

G = 6.674e-11
M_SUN = 1.989e30
AU = 1.496e11
BINS = np.logspace(np.log10(50), np.log10(30000), 20)
SEED = 271828
N_ECC_LAWS = ["circular", "uniform", "thermal", "superthermal"]


def solve_kepler(ma, e, tol=1e-10, max_iter=40):
    E = np.asarray(ma, dtype=float).copy()
    E[e > 0.8] = np.pi
    for _ in range(max_iter):
        step = (E - e * np.sin(E) - ma) / (1.0 - e * np.cos(E))
        E -= step
        if np.nanmax(np.abs(step)) < tol:
            break
    return E


def draw_ecc(kind, n, rng):
    if kind == "circular":
        return np.zeros(n)
    if kind == "uniform":
        return rng.uniform(0.0, 0.95, n)
    if kind == "thermal":
        return 0.95 * np.sqrt(rng.uniform(0.0, 1.0, n))
    if kind == "superthermal":
        return 0.95 * rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)
    raise ValueError(kind)


def draw_newtonian_vtan(s_proj, mass, ecc_kind, rng):
    """Projected tangential Newtonian relative speed (km/s) per pair,
    conditioned on the observed projected separation -- same construction
    as step_012."""
    n = len(s_proj)
    mu = G * M_SUN * mass
    cpsi = rng.uniform(-0.98, 0.98, n)
    spsi = np.maximum(np.sqrt(1.0 - cpsi * cpsi), 1e-6)
    r3 = s_proj / spsi                                   # AU
    vc3 = np.sqrt(mu / (r3 * AU)) / 1000.0               # km/s
    if ecc_kind == "circular":
        sf = np.ones(n)
    else:
        e = draw_ecc(ecc_kind, n, rng)
        ma = rng.uniform(0.0, 2.0 * np.pi, n)
        E = solve_kepler(ma, e)
        cnu = (np.cos(E) - e) / (1.0 - e * np.cos(E))
        sf = np.sqrt(np.maximum(2.0 - (1.0 - e * e) /
                                np.maximum(1.0 + e * cnu, 1e-6), 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    vpf = np.sqrt(np.cos(ph) ** 2 * cpsi ** 2 + np.sin(ph) ** 2)
    return vc3 * sf * vpf


def bin_profile(sep, vt, bins=BINS):
    w = pd.DataFrame({"sep": sep, "vt": vt})
    w["bin"] = pd.cut(w["sep"], bins)
    g = w.groupby("bin", observed=False)["vt"]
    med = g.median().to_numpy()   # length len(bins)-1, NaN where empty
    cnt = g.count().to_numpy()
    return med, cnt


def canonical(x, a, rs):
    return 1.0 + a * (1.0 - np.exp(-x / rs))


def fit_alpha_rs(x, y):
    try:
        p, _ = curve_fit(canonical, x, y, p0=[0.3, 3000.0],
                         bounds=([-0.5, 100.0], [3.0, 1e6]), maxfev=20000)
        return float(p[0]), float(p[1])
    except Exception:
        return float("nan"), float("nan")


def main():
    df = pd.read_parquet(
        os.path.join(HERE, "..", "..", "data", "processed",
                     "kinematic_results.parquet"))
    df = df[(df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)]
    df = df[(df["mass_total"] > 0) & (df["sep_AU"] > BINS[0])
            & (df["sep_AU"] < BINS[-1])].reset_index(drop=True)

    zpc = np.abs(df["Z_gc"]) * 1000.0
    qs = np.quantile(zpc, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    s = df["sep_AU"].to_numpy()
    M = df["mass_total"].to_numpy()
    d = df["dist_pc"].to_numpy()
    n = len(s)

    # per-component noise on the relative tangential velocity (km/s)
    sig_ra = 4.74047e-3 * np.sqrt(df["pmra_error1"] ** 2
                                  + df["pmra_error2"] ** 2) * d
    sig_dec = 4.74047e-3 * np.sqrt(df["pmdec_error1"] ** 2
                                   + df["pmdec_error2"] ** 2) * d
    v_circ2d = np.sqrt(G * M_SUN * M / (s * AU)) / 1000.0
    vt_noise_mag = np.sqrt(sig_ra ** 2 + sig_dec ** 2) / v_circ2d

    bc = np.sqrt(BINS[:-1] * BINS[1:])
    obs_vt = df["v_tilde"].to_numpy()

    # full-sample noise-floor diagnostics
    wide = (s >= 8000) & (s < 25000)
    noise_floor_wide = [
        float(np.median(vt_noise_mag[(zpc >= qs[i]) & (zpc <= qs[i + 1])
                                     & wide]))
        for i in range(5)]

    rng = np.random.default_rng(SEED)
    results = {"noise_floor_wide_bin_per_stratum": noise_floor_wide,
               "n_pairs": int(n), "bin_centers_au": bc.tolist(),
               "eccentricity_laws": {}}

    obs_med, obs_cnt = bin_profile(s, obs_vt)
    obs_base = np.nanmean(obs_med[np.isfinite(obs_med)][:5])
    obs_norm = obs_med / obs_base
    a_obs, r_obs = fit_alpha_rs(bc, obs_norm)
    results["observed_fit_all"] = {"alpha": a_obs, "r_s_au": r_obs}

    for ecc in N_ECC_LAWS:
        vt_true = draw_newtonian_vtan(s, M, ecc, rng)
        ang = rng.uniform(0.0, 2.0 * np.pi, n)
        vx = vt_true * np.cos(ang) + sig_ra * rng.standard_normal(n)
        vy = vt_true * np.sin(ang) + sig_dec * rng.standard_normal(n)
        vt_noisy = np.sqrt(vx * vx + vy * vy) / v_circ2d

        null_med, _ = bin_profile(s, vt_noisy)
        null_norm = null_med / np.nanmean(null_med[np.isfinite(null_med)][:5])
        a_null, r_null = fit_alpha_rs(bc, null_norm)

        # residual profile: obs_norm / null_norm
        fin = np.isfinite(obs_norm) & np.isfinite(null_norm)
        resid = obs_norm / null_norm
        a_res, r_res = fit_alpha_rs(bc[fin], resid[fin])

        # per-stratum residuals (each stratum normalized to its own
        # inner baseline, matching the stratified analysis)
        strata = []
        for i in range(5):
            m = (zpc >= qs[i]) & (zpc <= qs[i + 1])
            mo, co = bin_profile(s[m], obs_vt[m])
            mn_, cn = bin_profile(s[m], vt_noisy[m])
            fo = np.isfinite(mo) & (co > 30)
            fn = np.isfinite(mn_) & (cn > 30)
            io = np.where(fo)[0][:5]
            in_ = np.where(fn)[0][:5]
            ro = mo / np.nanmean(mo[io])
            rn = mn_ / np.nanmean(mn_[in_])
            good = fo & fn
            rr = ro[good] / rn[good]
            a_i, r_i = fit_alpha_rs(bc[good], rr)
            strata.append({
                "stratum": i + 1,
                "median_z_pc": float(np.median(zpc[m])),
                "median_dist_pc": float(np.median(d[m])),
                "residual_alpha": a_i,
                "residual_r_s_au": r_i,
                "residual_profile": [float(x) for x in rr],
                "residual_bins_au": [float(x) for x in bc[good]],
            })

        # unbiased low-noise subsample: absolute noise-floor cut, which does
        # not condition on the observed v_tilde (no selection bias)
        low_noise = {}
        for cut in (0.10, 0.15, 0.20):
            keep = vt_noise_mag < cut
            per = []
            for i in range(5):
                m = ((zpc >= qs[i]) & (zpc <= qs[i + 1])
                     & np.asarray(keep))
                mo, co = bin_profile(s[m], obs_vt[m])
                fo = np.isfinite(mo) & (co > 50)
                io = np.where(fo)[0][:5]
                rn_ = mo / np.nanmean(mo[io])
                per.append({
                    "stratum": i + 1,
                    "median_z_pc": float(np.median(zpc[(zpc >= qs[i])
                                                & (zpc <= qs[i + 1])])),
                    "n_kept": int(m.sum()),
                    "profile": [float(x) for x in rn_[fo]],
                    "bins_au": [float(x) for x in bc[fo]],
                })
            low_noise[f"sigma_vt_lt_{cut}"] = {
                "frac_kept": float(np.mean(keep)), "strata": per}
        results["low_noise_subsample_profiles"] = low_noise

        # ---- joint likelihood: TEP-like boost THROUGH the noise kernel ----
        # fixed noise draws (common random numbers) for a smooth chi2 surface
        nra = rng.standard_normal(n)
        ndec = rng.standard_normal(n)
        ang0 = rng.uniform(0.0, 2.0 * np.pi, n)
        udir = np.stack([np.cos(ang0), np.sin(ang0)])      # (2,n)

        def model_profile(boost_a, boost_rs, idx):
            """Boosted-Newtonian true speed -> +noise -> |.|/v_circ ->
            binned median profile normalized to its own inner baseline,
            evaluated on subsample index mask idx."""
            fac = 1.0 + boost_a * (1.0 - np.exp(-s / boost_rs))
            vx = vt_true * fac * udir[0] + sig_ra * nra
            vy = vt_true * fac * udir[1] + sig_dec * ndec
            vtm = np.sqrt(vx * vx + vy * vy) / v_circ2d
            med, cnt = bin_profile(s[idx], vtm[idx])
            ok = np.isfinite(med) & (cnt > 50)
            base = np.nanmean(med[np.isfinite(med)][:5])
            return med / base, ok

        joint = []
        for i in range(5):
            idx = np.where((zpc >= qs[i]) & (zpc <= qs[i + 1]))[0]
            mo, co = bin_profile(s[idx], obs_vt[idx])
            so = np.isfinite(mo) & (co > 50)
            yo = mo / np.nanmean(mo[so][:5])
            # SEM of the median ~ 1.253 sigma/sqrt(n); use bin MAD proxy
            sub = pd.DataFrame({"sep": s[idx], "vt": obs_vt[idx]})
            sub["bin"] = pd.cut(sub["sep"], BINS)
            sd = sub.groupby("bin", observed=False)["vt"].std().to_numpy()
            sem = 1.253 * sd / np.sqrt(np.maximum(co, 1))
            eo = sem / np.nanmean(mo[so][:5])
            bo = np.nanmean(mo[so][:5])

            def chi2_obs(pred):
                m = so & np.isfinite(pred) & np.isfinite(eo) & (eo > 0)
                if m.sum() < 6:
                    return np.inf
                return float(np.sum(((yo[m] - pred[m]) / eo[m]) ** 2))

            # null chi2
            pn, _ = model_profile(0.0, 1e9, idx)
            chi2_null = chi2_obs(pn)

            # grid over boost (alpha, Rs)
            best = None
            for ba in np.linspace(0.0, 0.8, 17):
                for br in np.logspace(np.log10(500), np.log10(30000), 20):
                    pm_, _ = model_profile(ba, br, idx)
                    c2 = chi2_obs(pm_)
                    if best is None or c2 < best[0]:
                        best = (c2, ba, br)
            joint.append({
                "stratum": i + 1,
                "median_z_pc": float(np.median(zpc[idx])),
                "chi2_null": chi2_null,
                "chi2_tep": best[0],
                "delta_chi2": chi2_null - best[0],
                "alpha_fit": best[1],
                "r_s_fit_au": best[2],
            })
            print(f"   {ecc} z{i+1}: chi2_null={chi2_null:.0f} "
                  f"chi2_tep={best[0]:.0f} dchi2={chi2_null-best[0]:.0f} "
                  f"(a={best[1]:.2f}, Rs={best[2]:.0f})")

        results["eccentricity_laws"][ecc] = {
            "null_fit": {"alpha": a_null, "r_s_au": r_null},
            "residual_fit_all": {"alpha": a_res, "r_s_au": r_res},
            "null_outer_median": float(np.nanmean(null_norm[bc > 5000])),
            "strata": strata,
            "joint_fit_per_stratum": joint,
        }
        print(f"{ecc}: null outer={np.nanmean(null_norm[bc>5000]):.3f} "
              f"null (a,Rs)=({a_null:.3f},{r_null:.0f}) "
              f"resid (a,Rs)=({a_res:.3f},{r_res:.0f})")
        for st in strata:
            print(f"   z{st['stratum']} (z~{st['median_z_pc']:.0f} pc): "
                  f"resid a={st['residual_alpha']:.3f} "
                  f"Rs={st['residual_r_s_au']:.0f}")

    out = os.path.join(OUTDIR, "020i_noise_null.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
