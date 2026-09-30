#!/usr/bin/env python3
"""
Step 020n: Purity-weighted profiles -- soft bound-probability weights replace
the hard R_chance_align < 0.01 cut.

The hard purity cut truncates the high-v_tilde tail at wide separations
(step_020k) because R_chance rises with s, manufacturing the widest-bin
downturn and corrupting every full-range (alpha, R_s) fit.  Simply relaxing
the cut is not the fix: the excluded population is independently unbound
(median v_tilde ~ 2.1).

R_chance_align is, by construction, the probability that a pair is a chance
alignment given the local sky density and proper-motion space.  The
corresponding bound probability is w = 1 - R_chance_align.  Weighting each
pair by w implements the selection smoothly: interlopers contribute ~0
weight at any separation, and no separation-dependent truncation is imposed.

This step rebuilds the kinematic pool with ALL step_001 cuts except the
R_chance threshold, computes w-weighted median v_tilde profiles per |Z|
stratum over the full separation range, and refits the canonical saturating
profile  f(s) = 1 + alpha * (1 - exp(-s/R_s))  on the weighted profiles.
If the purity-weighted profiles saturate rather than downturn, the outer
bins carry recoverable signal and the (alpha, R_s) ordering can be quoted
over the full range.

Outputs: results/outputs/020n_purity_weighted_profiles.json
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import GLOBAL_BINS, G_AU
from scripts.steps.step_020k_selection_downturn import build_pool, N_STRATA

OUT_PATH = PROJECT_ROOT / "results" / "outputs" / "020n_purity_weighted_profiles.json"
RESTRICT_MAX_AU = 9000.0


def wmedian(v, w, q=0.5):
    """Weighted quantile via the empirical CDF."""
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if len(v) < 5:
        return np.nan
    i = np.argsort(v)
    cw = np.cumsum(w[i])
    cw /= cw[-1]
    return float(np.interp(q, cw, v[i]))


def model(s, alpha, rs):
    return 1.0 + alpha * (1.0 - np.exp(-s / rs))


def fit_profile(bc, prof, p0=(0.3, 4000.0)):
    m = np.isfinite(prof)
    if m.sum() < 5:
        return None
    try:
        p, c = curve_fit(model, bc[m], prof[m], p0=p0,
                         bounds=([0.0, 100.0], [3.0, 60000.0]), maxfev=20000)
        e = np.sqrt(np.diag(c))
        return {"alpha": float(p[0]), "alpha_err": float(e[0]),
                "R_s_AU": float(p[1]), "R_s_err": float(e[1])}
    except Exception:
        return None


def bound_component_median(v, f_unbound, v_unb):
    """CDF-level deconvolution of a two-component mixture.

    Observed C_obs = f_bound*C_bound + f_unbound*C_unbound.  The interloper
    shape C_unbound is estimated from the clearly-unbounded population
    (R_chance > 0.2) in the same bin.  Returns the bound-component median.
    """
    if f_unbound >= 0.8 or len(v) < 30 or len(v_unb) < 30:
        return np.nan
    f_bound = 1.0 - f_unbound
    grid = np.quantile(v, np.linspace(0, 1, 200))
    c_obs = np.searchsorted(np.sort(v), grid, side='right') / len(v)
    c_unb = np.searchsorted(np.sort(v_unb), grid, side='right') / len(v_unb)
    c_bnd = np.clip((c_obs - f_unbound * c_unb) / f_bound, 0.0, 1.0)
    c_bnd = np.maximum.accumulate(c_bnd)
    return float(np.interp(0.5, c_bnd, grid))


def run():
    print_status("Step 020n: purity-weighted v_tilde profiles", "TITLE")
    d = build_pool()
    d = d[np.isfinite(d['R_chance_align'])].copy()
    d['w_bound'] = np.clip(1.0 - d['R_chance_align'], 0.0, 1.0)

    bins = GLOBAL_BINS
    bc = np.sqrt(bins[:-1] * bins[1:])
    z_abs = np.abs(d['Z_gc']) * 1000.0
    edges = np.quantile(z_abs[d['R_chance_align'] < 0.01],
                        np.linspace(0, 1, N_STRATA + 1))

    # Effective sample size of the weighting: (sum w)^2 / sum w^2
    n_eff = float(d['w_bound'].sum() ** 2 / (d['w_bound'] ** 2).sum())
    frac_w = float(d['w_bound'].sum() / len(d))
    print_status(f"pool={len(d)}  <w>={frac_w:.3f}  N_eff(w)={n_eff:.0f}", "INFO")

    strata = []
    for i in range(N_STRATA):
        m = (z_abs >= edges[i]) & (z_abs <= edges[i + 1])
        # stratum-pooled interloper reference (clearly unbound pairs)
        mp = m & (d['R_chance_align'] > 0.2)
        v_unb_stratum = d['v_tilde'].values[mp]
        s_unb_stratum = d['sep_AU'].values[mp]
        prof_w, prof_w16, prof_w84, prof_hard = [], [], [], []
        prof_bnd, f_unb = [], []
        for j in range(len(bins) - 1):
            mb = m & (d['sep_AU'] >= bins[j]) & (d['sep_AU'] < bins[j + 1])
            v = d['v_tilde'].values[mb]
            w = d['w_bound'].values[mb]
            if (w > 0).sum() > 15:
                prof_w.append(wmedian(v, w, 0.5))
                prof_w16.append(wmedian(v, w, 0.16))
                prof_w84.append(wmedian(v, w, 0.84))
            else:
                prof_w.append(np.nan)
                prof_w16.append(np.nan)
                prof_w84.append(np.nan)
            mh = mb & (d['R_chance_align'] < 0.01)
            prof_hard.append(float(np.nanmedian(d['v_tilde'][mh]))
                             if mh.sum() > 15 else np.nan)
            # mixture-corrected bound-component median.  The interloper
            # v_tilde shape scales as sqrt(s) (v_tan is s-independent while
            # v_circ ~ 1/sqrt(s)), so a stratum-pooled sample of clearly
            # unbound pairs is rescaled to each bin centre; a purely
            # per-bin reference is only populated in the widest bins.
            fu = float(np.nanmean(np.clip(d['R_chance_align'].values[mb],
                                          0.0, 1.0))) if mb.sum() else np.nan
            v_unb = d['v_tilde'].values[mb & (d['R_chance_align'] > 0.2)]
            med = bound_component_median(v, fu, v_unb)
            if not np.isfinite(med) and np.isfinite(fu) and len(v_unb_stratum) >= 30:
                # rescale pooled interlopers to this bin centre
                v_unb_scaled = v_unb_stratum * np.sqrt(bc[j] / s_unb_stratum)
                med = bound_component_median(v, fu, v_unb_scaled)
            prof_bnd.append(med if np.isfinite(med) else np.nan)
            f_unb.append(fu)

        prof_w = np.array(prof_w); prof_hard = np.array(prof_hard)
        base_w = np.nanmean(prof_w[:5]); base_h = np.nanmean(prof_hard[:5])
        pw, ph = prof_w / base_w, prof_hard / base_h

        fit_full = fit_profile(bc, pw)
        mr = bc < RESTRICT_MAX_AU
        fit_rest = fit_profile(bc[mr], pw[mr])

        strata.append({
            'z_med_pc': float(np.nanmedian(z_abs[m])),
            'n_pairs': int(m.sum()),
            'w_mean': float(d['w_bound'].values[m].mean()),
            'profile_w_norm': pw.tolist(),
            'profile_hard_norm': ph.tolist(),
            'profile_w_q16': (np.array(prof_w16) / base_w).tolist(),
            'profile_w_q84': (np.array(prof_w84) / base_w).tolist(),
            'profile_bound_component': prof_bnd,
            'frac_unbound_est': f_unb,
            'fit_full_range': fit_full,
            'fit_restricted_lt9kAU': fit_rest,
        })

        tail_w = ' '.join(f'{x:.3f}' for x in pw[12:])
        tail_h = ' '.join(f'{x:.3f}' for x in ph[12:])
        print_status(f"z{i+1} ~{strata[-1]['z_med_pc']:.0f} pc "
                     f"<w>={strata[-1]['w_mean']:.3f}", "INFO")
        print_status(f"   outer weighted: {tail_w}", "INFO")
        print_status(f"   outer hard-cut: {tail_h}", "INFO")

    # Absolute (un-normalized) weighted medians per stratum x separation
    # window -- the environmental ordering read off the data with no
    # hard cut and no per-stratum baseline normalization.
    sep_windows = [(500, 2000), (2000, 5000), (5000, 9000),
                   (9000, 15000), (15000, 30000)]
    abs_medians = {}
    for i in range(N_STRATA):
        mz = (z_abs >= edges[i]) & (z_abs <= edges[i + 1])
        row = {}
        for slo, shi in sep_windows:
            m = mz & (d['sep_AU'] >= slo) & (d['sep_AU'] < shi)
            row[f'{slo}-{shi}'] = wmedian(d['v_tilde'].values[m],
                                          d['w_bound'].values[m])
        abs_medians[f'z{i+1}'] = row

    # Bootstrap significance of the z5 - z1 absolute contrast per window.
    rng = np.random.default_rng(7)
    contrast = {}
    for slo, shi in sep_windows:
        meds, sems = [], []
        for i in (0, N_STRATA - 1):
            m = (z_abs >= edges[i]) & (z_abs <= edges[i + 1]) \
                & (d['sep_AU'] >= slo) & (d['sep_AU'] < shi)
            idx = np.where(m.values if hasattr(m, 'values') else m)[0]
            n = len(idx)
            bs = np.array([
                wmedian(d['v_tilde'].values[r], d['w_bound'].values[r])
                for r in (idx[rng.integers(0, n, n)] for _ in range(300))
            ])
            meds.append(wmedian(d['v_tilde'].values[idx],
                                d['w_bound'].values[idx]))
            sems.append(float(np.std(bs)))
        c = meds[1] - meds[0]
        e = float(np.hypot(sems[0], sems[1]))
        contrast[f'{slo}-{shi}'] = {
            'z1': meds[0], 'z5': meds[1],
            'contrast': float(c), 'contrast_err': e,
            'sigma': float(c / e),
        }
        print_status(f"z5-z1 contrast {slo}-{shi} AU: {c:+.3f} +- {e:.4f} "
                     f"({c / e:.1f} sigma)", "INFO")

    # Pessimistic-interloper sweep on the 20-50 kAU bin: R_chance is a prior
    # that can underestimate the true unbound fraction at wide s.  The
    # v_tilde>1.5 tail suggests f_unbound up to ~0.3 there; the bound
    # component is recomputed under 0.11 (R_chance prior), 0.20, 0.30.
    sweep = {}
    for i in range(N_STRATA):
        m = (z_abs >= edges[i]) & (z_abs <= edges[i + 1])
        mb = m & (d['sep_AU'] >= 20000) & (d['sep_AU'] < 50000)
        vv = d['v_tilde'].values[mb]
        mu = mb & (d['R_chance_align'] > 0.2)
        vu = d['v_tilde'].values[mu]
        su = d['sep_AU'].values[mu]
        vu_s = vu * np.sqrt(30000.0 / su) if len(vu) else vu
        sweep[f'z{i+1}'] = {
            'obs_median': float(np.median(vv)) if len(vv) else None,
            'baseline_median_lt2kAU': float(np.median(
                d['v_tilde'].values[m & (d['sep_AU'] < 2000)])),
            'bound_median_f0.11': bound_component_median(vv, 0.11, vu_s),
            'bound_median_f0.20': bound_component_median(vv, 0.20, vu_s),
            'bound_median_f0.30': bound_component_median(vv, 0.30, vu_s),
        }

    # Does the downturn persist under purity weighting?
    downturn = {}
    for i, s in enumerate(strata):
        pw = np.array(s['profile_w_norm'])
        ph = np.array(s['profile_hard_norm'])
        pk = np.nanargmax(pw[:20]) if np.any(np.isfinite(pw[:20])) else -1
        downturn[f'z{i+1}'] = {
            'weighted_peak_bin': int(pk),
            'weighted_outer_decline': float(np.nanmax(pw[:20]) - pw[-1])
                if np.isfinite(pw[-1]) else None,
            'hard_outer_decline': float(np.nanmax(ph[:20]) - ph[-1])
                if np.isfinite(ph[-1]) else None,
        }

    out = {
        'description': 'bound-probability (1 - R_chance) weighted profiles; '
                       'canonical saturating-exponential refits',
        'n_pool': int(len(d)),
        'w_mean': frac_w,
        'n_eff_weights': n_eff,
        'z_edges_pc': edges.tolist(),
        'bin_centers_AU': bc.tolist(),
        'strata': strata,
        'absolute_weighted_medians': abs_medians,
        'z5_minus_z1_contrast_bootstrap': contrast,
        'pessimistic_unbound_sweep_20_50kAU': sweep,
        'downturn_audit': downturn,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print_status(f"Saved {OUT_PATH}", "SUCCESS")


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_020n", str(log_dir / "step_020n_purity_weighted_profiles.log"))
    set_step_logger(logger)
    run()
