#!/usr/bin/env python3
"""
Step 020k: Selection-cut origin of the widest-bin downturn.

Observed per-stratum v_tilde profiles are bump-shaped: they rise to s ~ 15-18 kAU
then decline at the widest bins. No model in the canonical saturating family
produces a decline, so every (alpha, R_s) fit and every u_eff inversion was a
forced compromise on the outer bins.

This step tests whether the downturn is manufactured by the purity cut
R_chance_align < 0.01 (step_001). The chance-alignment probability rises with
projected separation; at s >~ 8 kAU the 1% threshold removes a substantial,
velocity-biased fraction of pairs, truncating the high-v_tilde tail and
depressing the outer-bin medians.

Method:
  1. Rebuild the kinematic pool from the raw catalog applying ALL step_001 cuts
     except the R_chance_align threshold, and recompute v_tilde with the
     step_002 conventions (Mamajek mass estimate; metallicity correction not
     applied -- it is a second-order mass rescaling that cannot affect the
     within-stratum shape comparison performed here).
  2. Form per-|Z|-stratum normalized profiles at several R_chance thresholds
     {0.01, 0.05, 0.20, no cut}, with identical stratum edges (baseline-sample
     quintiles) so all variants are compared on the same |Z| ranges.
  3. Report the separation at which profiles diverge across thresholds, and the
     per-stratum ordering in the cut-insensitive region.

Outputs: results/outputs/020k_selection_downturn.json
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import GLOBAL_BINS, G_AU

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "all_columns_catalog.fits.gz"
MAMAJEK_PATH = PROJECT_ROOT / "data" / "processed" / "mamajek_clean.csv"
OUT_PATH = PROJECT_ROOT / "results" / "outputs" / "020k_selection_downturn.json"

NEED_COLS = [
    'sep_AU', 'R_chance_align', 'ra1', 'dec1',
    'parallax1', 'parallax2', 'parallax_error1', 'parallax_error2',
    'parallax_over_error1', 'parallax_over_error2',
    'pmra1', 'pmdec1', 'pmra2', 'pmdec2',
    'pmra_error1', 'pmdec_error1', 'pmra_error2', 'pmdec_error2',
    'ruwe1', 'ruwe2', 'phot_g_mean_mag1', 'phot_g_mean_mag2',
]

R_CHANCE_VARIANTS = [0.01, 0.05, 0.20, 1.01]  # 1.01 ~ no cut
N_STRATA = 5


def build_pool():
    """Rebuild the kinematic pool with all non-R_chance cuts applied once."""
    print_status("Loading raw catalog for pool reconstruction...", "PROCESS")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        t = Table.read(RAW_PATH)
    df = pd.DataFrame({c: np.asarray(t[c]) for c in NEED_COLS})
    for c in df.columns:
        df.loc[df[c] > 1e19, c] = np.nan
        df.loc[df[c] == 99.99999, c] = np.nan

    pm_snr1 = np.hypot(df['pmra1'], df['pmdec1']) / np.hypot(df['pmra_error1'], df['pmdec_error1'])
    pm_snr2 = np.hypot(df['pmra2'], df['pmdec2']) / np.hypot(df['pmra_error2'], df['pmdec_error2'])
    m0 = (
        (df['parallax_over_error1'] > 20) & (df['parallax_over_error2'] > 20)
        & (pm_snr1 > 10) & (pm_snr2 > 10)
        & (df['ruwe1'] < 1.2) & (df['ruwe2'] < 1.2)
        & (df['sep_AU'] > 50) & (df['sep_AU'] < 50000)
    )
    d = df[m0].copy()
    print_status(f"Pool after all non-chance cuts: {len(d)}", "INFO")

    w1 = 1 / d['parallax_error1'] ** 2
    w2 = 1 / d['parallax_error2'] ** 2
    d['dist_pc'] = 1000.0 / ((d['parallax1'] * w1 + d['parallax2'] * w2) / (w1 + w2))

    mg = pd.read_csv(MAMAJEK_PATH).sort_values('M_G')
    mi = interp1d(mg['M_G'], mg['M_sun'], bounds_error=False, fill_value='extrapolate')
    Mg1 = d['phot_g_mean_mag1'] - 5 * np.log10(d['dist_pc']) + 5
    Mg2 = d['phot_g_mean_mag2'] - 5 * np.log10(d['dist_pc']) + 5
    m1 = mi(Mg1)
    m2 = mi(Mg2)
    mm = (m1 > 0.05) & (m1 < 5.0) & (m2 > 0.05) & (m2 < 5.0)
    d = d[mm]
    m1 = m1[mm]
    m2 = m2[mm]

    coords = SkyCoord(
        ra=d['ra1'].values * u.deg,
        dec=d['dec1'].values * u.deg,
        distance=d['dist_pc'].values * u.pc,
        frame='icrs',
    )
    gc = coords.transform_to('galactocentric')
    d['Z_gc'] = gc.z.to(u.kpc).value

    mu_rel = np.hypot(d['pmra1'] - d['pmra2'], d['pmdec1'] - d['pmdec2'])
    v_tan = 4.74047e-3 * mu_rel * d['dist_pc']
    v_circ = np.sqrt(G_AU * (m1 + m2) / d['sep_AU'])
    d['v_tilde'] = v_tan / v_circ
    d = d[np.isfinite(d['v_tilde']) & (d['v_tilde'] > 0)]
    print_status(f"Final pool with v_tilde: {len(d)}", "INFO")
    return d


def stratum_profiles(d, m_rc, edges, bins):
    """Per-stratum normalized median profiles under an R_chance mask."""
    z = np.abs(d['Z_gc']) * 1000.0
    bc = np.sqrt(bins[:-1] * bins[1:])
    out = []
    for i in range(N_STRATA):
        m = m_rc & (z >= edges[i]) & (z <= edges[i + 1])
        prof = []
        for j in range(len(bins) - 1):
            mb = m & (d['sep_AU'] >= bins[j]) & (d['sep_AU'] < bins[j + 1])
            prof.append(float(np.nanmedian(d['v_tilde'][mb])) if mb.sum() > 15 else np.nan)
        prof = np.array(prof)
        base = np.nanmean(prof[:5])
        out.append({
            'z_med_pc': float(np.nanmedian(z[m])),
            'n_pairs': int(m.sum()),
            'baseline': float(base),
            'profile_norm': (prof / base).tolist(),
            'profile_raw': prof.tolist(),
        })
    return out


def run():
    print_status("Step 020k: Selection-cut origin of the widest-bin downturn", "TITLE")
    d = build_pool()

    bins = GLOBAL_BINS
    bc = np.sqrt(bins[:-1] * bins[1:])
    z_abs = np.abs(d['Z_gc']) * 1000.0
    # Stratum edges from the baseline (R_chance < 0.01) sample so every variant
    # is evaluated on identical |Z| ranges.
    edges = np.quantile(z_abs[d['R_chance_align'] < 0.01], np.linspace(0, 1, N_STRATA + 1))

    variants = {}
    for rc in R_CHANCE_VARIANTS:
        m_rc = d['R_chance_align'] < rc
        variants[str(rc)] = stratum_profiles(d, m_rc, edges, bins)

    # Fraction of the otherwise-qualified pool removed by the baseline cut,
    # per separation decade.
    frac_removed = {}
    for slo, shi, lab in [(50, 500, '50-500'), (500, 2000, '500-2k'),
                          (2000, 8000, '2k-8k'), (8000, 30000, '8k-30k')]:
        m = (d['sep_AU'] >= slo) & (d['sep_AU'] < shi)
        frac_removed[lab] = float((d['R_chance_align'][m] >= 0.01).mean())

    # First bin index where baseline vs relaxed profiles differ by > 0.02
    div_idx = {}
    for i in range(N_STRATA):
        p0 = np.array(variants['0.01'][i]['profile_norm'])
        p1 = np.array(variants['1.01'][i]['profile_norm'])
        diff = np.abs(p0 - p1)
        idx = int(np.argmax(diff > 0.02)) if np.any(diff > 0.02) else -1
        div_idx[f'z{i+1}'] = {
            'bin_index': idx,
            'sep_AU': float(bc[idx]) if idx >= 0 else None,
        }

    result = {
        'description': 'R_chance_align selection-cut audit of the widest-bin downturn',
        'n_pool': int(len(d)),
        'z_edges_pc': edges.tolist(),
        'bin_centers_AU': bc.tolist(),
        'frac_removed_by_baseline_cut_per_sep': frac_removed,
        'divergence_onset': div_idx,
        'variants': variants,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print_status(f"Saved {OUT_PATH}", "SUCCESS")

    for rc in R_CHANCE_VARIANTS:
        print_status(f"R_chance < {rc}", "INFO")
        for i, s in enumerate(variants[str(rc)]):
            pn = np.array(s['profile_norm'])
            outer = ' '.join(f'{x:.3f}' for x in pn[12:])
            print_status(f"  z{i+1} (~{s['z_med_pc']:.0f} pc): outer bins = {outer}", "INFO")


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_020k", str(log_dir / "step_020k_selection_downturn.log"))
    set_step_logger(logger)
    run()
