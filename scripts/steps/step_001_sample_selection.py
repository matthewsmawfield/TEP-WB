#!/usr/bin/env python3
"""
Step 001: Sample Selection for TEP-WB

This script filters the raw Gaia DR3 wide binary catalog (El-Badry et al. 2021) to produce a 
high-purity sample suitable for testing gravitational anomalies.

Scientific Rationale:
The primary systematic error in wide binary tests of gravity is contamination from unresolved 
hierarchical triples. If one of the stars is actually a tight, unresolved binary, its orbital 
motion will perturb the measured proper motion, falsely inflating the observed relative velocity
and mimicking a breakdown of General Relativity.

To mitigate this, the methodology applies extremely strict astrometric purity cuts, notably a RUWE < 1.2 cut.
"""

import sys
from pathlib import Path
import numpy as np
from astropy.table import Table
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

import warnings
from astropy.utils.exceptions import AstropyWarning

def select_sample():
    print_status("Initializing Sample Selection Pipeline", "TITLE")
    
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "all_columns_catalog.fits.gz"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "clean_binary_sample.parquet"

    if not raw_data_path.exists():
        print_status(f"Raw data not found at {raw_data_path}. Run step_000 first.", "ERROR")
        sys.exit(1)

    print_status("Loading catalog (this may take a moment)...", "PROCESS")
    
    # Suppress harmless FITS unit parsing warnings from the El-Badry catalog
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        t = Table.read(raw_data_path)

    df = t.to_pandas()
    print_status(f"Loaded {len(df)} candidate pairs.", "INFO")

    print_status("Cleaning extreme fill values (removing Gaia placeholders > 1e19)...", "PROCESS")
    # Clean FITS placeholder values. Gaia often uses extreme numbers (e.g. 1e19) for missing data.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.where(df[col] > 1e19, np.nan, df[col])
        df[col] = np.where(df[col] == 99.99999, np.nan, df[col])
        
    initial_count = len(df)
    
    # 1. Chance Alignment Cut
    # Removes visual binaries (stars that appear close on the sky but are at different distances).
    if 'R_chance_align' in df.columns:
        df_clean = df[df['R_chance_align'] < 0.01].copy()
        print_status(f"Applied Chance Alignment < 1% filter (removes unbound visual interlopers): {len(df_clean)} / {initial_count} remaining", "INFO")
    else:
        df_clean = df.copy()

    # 2. Astrometric Signal-to-Noise Cut
    # Ensure the distance measurement is highly precise. Velocity errors scale linearly with distance errors.
    df_clean = df_clean[
        (df_clean['parallax_over_error1'] > 20) & 
        (df_clean['parallax_over_error2'] > 20)
    ]
    print_status(f"Applied Parallax S/N > 20 filter (ensures sub-5% distance/velocity error): {len(df_clean)} remaining", "INFO")


    # 2.5 Proper Motion Signal-to-Noise Cut
    # Removes systems where the relative proper motion is dominated by noise,
    # which could artificially inflate the high-velocity tail and mimic anomalous gravity.
    # We require the total proper motion to be at least 10x its uncertainty for both stars.
    pm_snr1 = np.sqrt(df_clean['pmra1']**2 + df_clean['pmdec1']**2) / np.sqrt(df_clean['pmra_error1']**2 + df_clean['pmdec_error1']**2)
    pm_snr2 = np.sqrt(df_clean['pmra2']**2 + df_clean['pmdec2']**2) / np.sqrt(df_clean['pmra_error2']**2 + df_clean['pmdec_error2']**2)
    
    mask_pm_snr = (pm_snr1 > 10) & (pm_snr2 > 10)
    df_clean = df_clean[mask_pm_snr]
    print_status(f"Applied Proper Motion S/N > 10 filter (prevents high-velocity noise artifacts): {len(df_clean)} remaining", "INFO")

    # 3. Triple Suppression Cut (RUWE)

    # The Renormalised Unit Weight Error (RUWE) measures how well the single-star astrometric model fits.
    # RUWE ~ 1.0 means a perfect single-star fit. RUWE > 1.4 strongly indicates an unresolved binary.
    # the pipeline uses a hyper-conservative RUWE < 1.2 to eliminate any doubt that the anomaly is just triple contamination.
    if 'ruwe1' in df_clean.columns and 'ruwe2' in df_clean.columns:
        df_clean = df_clean[
            (df_clean['ruwe1'] < 1.2) & 
            (df_clean['ruwe2'] < 1.2)
        ]
        print_status(f"Applied rigorous RUWE < 1.2 filter (suppresses unresolved hierarchical triple systems to minimize kinematic contamination): {len(df_clean)} remaining", "INFO")

    # 4. Separation Cut
    # s > 50 AU: Avoids contamination from tight binaries where astrometry becomes confused.
    # s < 50,000 AU: Avoids unbound pairs and the extreme limit of the Galactic tidal field.
    sep_col = 'sep_AU'
    if sep_col in df_clean.columns:
        df_clean = df_clean[
            (df_clean[sep_col] > 50) & 
            (df_clean[sep_col] < 50000)
        ]
        print_status(f"Applied separation cuts (50 < s < 50,000 AU) to consistently sample the theoretical screening regime: {len(df_clean)} remaining", "INFO")

    try:
        df_clean.to_parquet(output_path)
        print_status(f"Saved clean sample to {output_path}", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to save output: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_001", str(log_dir / "step_001_sample_selection.log"))
    set_step_logger(logger)

    select_sample()
