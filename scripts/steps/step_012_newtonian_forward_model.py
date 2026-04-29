#!/usr/bin/env python3
"""
Step 012: Newtonian Orbital Forward Model for TEP-WB

This control replaces the flat-profile straw null with a projected Newtonian
binary population. It draws Keplerian orbits, projects them onto the sky, applies
the same separation binning and inner-baseline normalization used by step_003,
and asks whether ordinary orbital phase, eccentricity, projection, and the
observed mass distribution can mimic the TEP-like transition.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import (
    GLOBAL_BINS,
    INJECTION_SEED,
    G_AU,
    tep_screening_model,
    constant_boost_model,
    chi2_statistic,
)

N_SIM = 900_000
N_KEEP_TARGET = 341_315


def solve_kepler(mean_anomaly, eccentricity, tol=1e-10, max_iter=40):
    """Vectorized Newton-Raphson solve of E - e sin E = M."""
    eccentricity = np.asarray(eccentricity)
    eccentric_anomaly = np.asarray(mean_anomaly).copy()
    high_e = eccentricity > 0.8
    eccentric_anomaly[high_e] = np.pi
    for _ in range(max_iter):
        step = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= step
        if np.nanmax(np.abs(step)) < tol:
            break
    return eccentric_anomaly


def build_profile(sep_au, v_tilde, bins=GLOBAL_BINS):
    work = pd.DataFrame({"sep_AU": sep_au, "v_tilde": v_tilde}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    work = work[(work["sep_AU"] > bins[0]) & (work["sep_AU"] < bins[-1])]
    if len(work) < 1000:
        return None

    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)["v_tilde"]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories
    profile = pd.DataFrame({
        "sep_AU": bin_centers,
        "v_tilde_median": grouped.median().reindex(bin_index).to_numpy(),
        "v_tilde_std": grouped.std().reindex(bin_index).to_numpy(),
        "count": grouped.count().reindex(bin_index).to_numpy(),
    }).dropna()
    if len(profile) < 8:
        return None

    profile["sem"] = 1.253 * profile["v_tilde_std"] / np.sqrt(profile["count"])
    baseline = profile["v_tilde_median"].iloc[:5].mean()
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    profile["v_tilde_norm"] = profile["v_tilde_median"] / baseline
    profile["sem_norm"] = profile["sem"] / baseline
    return profile.replace([np.inf, -np.inf], np.nan).dropna()


def fit_tep_and_constant(profile):
    if profile is None or len(profile) < 8:
        return None
    x = profile["sep_AU"].to_numpy()
    y = profile["v_tilde_norm"].to_numpy()
    e = profile["sem_norm"].to_numpy()
    try:
        popt, pcov = curve_fit(
            tep_screening_model,
            x,
            y,
            sigma=e,
            p0=[3000.0, 0.15],
            bounds=([100.0, 0.0], [50000.0, 0.8]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None

    model = tep_screening_model(x, *popt)
    chi2_tep = chi2_statistic(y, model, e)

    weights = 1.0 / np.square(e)
    alpha_const = np.clip(np.sum(weights * (y - 1.0)) / np.sum(weights), 0.0, 0.8)
    const = constant_boost_model(x, alpha_const)
    chi2_const = chi2_statistic(y, const, e)

    perr = np.sqrt(np.diag(pcov))
    return {
        "r_s_au": float(popt[0]),
        "r_s_err_au": float(perr[0]),
        "alpha": float(popt[1]),
        "alpha_err": float(perr[1]),
        "chi2_tep": float(chi2_tep),
        "chi2_const": float(chi2_const),
        "delta_chi2_vs_const": float(chi2_const - chi2_tep),
    }


def draw_eccentricities(kind, n, rng):
    if kind == "circular":
        return np.zeros(n)
    if kind == "uniform":
        return rng.uniform(0.0, 0.95, n)
    if kind == "thermal":
        return 0.95 * np.sqrt(rng.uniform(0.0, 1.0, n))
    if kind == "superthermal":
        return 0.95 * rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)
    raise ValueError(f"Unknown eccentricity law: {kind}")


def simulate_newtonian_catalog(masses, observed_sep, ecc_kind, rng, n_sim=N_SIM):
    """Return projected separation and velocity ratio for a Newtonian null.

    This null conditions on the observed projected-separation distribution. It
    therefore tests whether projection/orbital phase/eccentricity alone create
    a rising normalized profile at fixed observed s.
    """
    draw_idx = rng.integers(0, len(observed_sep), size=min(n_sim, N_KEEP_TARGET))
    sep_proj = observed_sep[draw_idx]
    mass = masses[draw_idx]
    mu = G_AU * mass

    # Random line-of-sight depth: r_hat = (sin psi, 0, cos psi), so the observed
    # sky-plane separation is s = r sin psi.
    cos_psi = rng.uniform(-0.98, 0.98, len(sep_proj))
    sin_psi = np.sqrt(1.0 - cos_psi * cos_psi)
    r_3d = sep_proj / np.maximum(sin_psi, 1e-6)

    # Circular speed at the true 3D separation, with eccentric-orbit speed
    # multipliers drawn from the requested phase-mixed eccentricity law.
    v_circ_3d = np.sqrt(mu / np.maximum(r_3d, 1e-9))
    e = draw_eccentricities(ecc_kind, len(sep_proj), rng)
    if ecc_kind == "circular":
        speed_factor = np.ones(len(sep_proj))
    else:
        mean_anomaly = rng.uniform(0.0, 2.0 * np.pi, len(sep_proj))
        eccentric_anomaly = solve_kepler(mean_anomaly, e)
        cos_e = np.cos(eccentric_anomaly)
        cos_nu = (cos_e - e) / (1.0 - e * cos_e)
        speed_factor = np.sqrt(
            np.maximum(2.0 - (1.0 - e * e) / np.maximum(1.0 + e * cos_nu, 1e-6), 0.0)
        )

    # Velocity direction is randomized in the plane perpendicular to r_hat.
    phase = rng.uniform(0.0, 2.0 * np.pi, len(sep_proj))
    cph = np.cos(phase)
    sph = np.sin(phase)
    # Basis e1=(cos psi,0,-sin psi), e2=(0,1,0)
    vx_hat = cph * cos_psi
    vy_hat = sph
    v_proj_factor = np.sqrt(vx_hat * vx_hat + vy_hat * vy_hat)

    v_proj = v_circ_3d * speed_factor * v_proj_factor
    v_circ_projected = np.sqrt(mu / np.maximum(sep_proj, 1e-9))
    v_tilde = v_proj / v_circ_projected

    mask = np.isfinite(v_tilde) & (v_tilde > 0)
    return sep_proj[mask], v_tilde[mask]


def run_newtonian_forward_model():
    print_status("Initializing Newtonian Orbital Forward Model", "TITLE")

    kin_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    profile_path = PROJECT_ROOT / "results" / "outputs" / "003_screening_test_results.csv"
    if not kin_path.exists() or not profile_path.exists():
        print_status("Required kinematic/profile outputs missing. Run steps 002 and 003 first.", "ERROR")
        sys.exit(1)

    kin = pd.read_parquet(kin_path)
    kin = kin[(kin["ruwe1"] < 1.2) & (kin["ruwe2"] < 1.2)].copy()
    kin = kin.replace([np.inf, -np.inf], np.nan).dropna(subset=["mass_total", "sep_AU"])
    kin = kin[(kin["mass_total"] > 0) & (kin["sep_AU"] > GLOBAL_BINS[0]) & (kin["sep_AU"] < GLOBAL_BINS[-1])]

    obs = pd.read_csv(profile_path)
    x_obs = obs["sep_AU"].to_numpy()
    y_obs = obs["v_tilde_norm"].to_numpy()
    e_obs = obs["sem_norm"].to_numpy()

    rng = np.random.default_rng(INJECTION_SEED + 12)
    masses = kin["mass_total"].to_numpy()
    observed_sep = kin["sep_AU"].to_numpy()

    rows = []
    profile_rows = []
    for ecc_kind in ["circular", "uniform", "thermal", "superthermal"]:
        print_status(f"Simulating Newtonian catalog: eccentricity law = {ecc_kind}", "PROCESS")
        sep_sim, vt_sim = simulate_newtonian_catalog(masses, observed_sep, ecc_kind, rng)
        profile = build_profile(sep_sim, vt_sim)
        fit = fit_tep_and_constant(profile)
        if profile is None or fit is None:
            print_status(f"Newtonian simulation failed for {ecc_kind}", "WARNING")
            continue

        sim_interp = np.interp(x_obs, profile["sep_AU"], profile["v_tilde_norm"])
        chi2_vs_observed = chi2_statistic(y_obs, sim_interp, e_obs)
        outer_sim = float(profile.loc[profile["sep_AU"] > 5000, "v_tilde_norm"].mean())
        outer_obs = float(obs.loc[obs["sep_AU"] > 5000, "v_tilde_norm"].mean())
        deficit = outer_obs - outer_sim

        rows.append({
            "eccentricity_law": ecc_kind,
            "n_simulated_kept": int(len(sep_sim)),
            "newtonian_outer_v_tilde": outer_sim,
            "observed_outer_v_tilde": outer_obs,
            "outer_deficit_vs_observed": deficit,
            "chi2_newtonian_profile_vs_observed": chi2_vs_observed,
            **fit,
        })

        tmp = profile.copy()
        tmp["eccentricity_law"] = ecc_kind
        profile_rows.append(tmp)

        print_status(
            f"{ecc_kind}: outer v_tilde = {outer_sim:.3f} vs observed {outer_obs:.3f}; "
            f"chi2(data - Newtonian forward profile) = {chi2_vs_observed:.1f}; "
            f"spurious TEP alpha = {fit['alpha']:.3f}",
            "RESULT",
        )

    out_dir = PROJECT_ROOT / "results" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "012_newtonian_forward_model_summary.csv", index=False)
    if profile_rows:
        pd.concat(profile_rows, ignore_index=True).to_csv(
            out_dir / "012_newtonian_forward_model_profiles.csv",
            index=False,
        )

    if len(summary) > 0:
        best = summary.sort_values("chi2_newtonian_profile_vs_observed").iloc[0]
        print_status(
            f"Best Newtonian orbital null ({best['eccentricity_law']}) differs by "
            f"{best['outer_deficit_vs_observed']:.3f} in outer normalized velocity and has "
            f"chi2 = {best['chi2_newtonian_profile_vs_observed']:.1f} against the observed profile.",
            "SUCCESS",
        )
    else:
        print_status("No Newtonian forward-model summaries were produced.", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_012", str(log_dir / "step_012_newtonian_forward_model.log"))
    set_step_logger(logger)
    run_newtonian_forward_model()
