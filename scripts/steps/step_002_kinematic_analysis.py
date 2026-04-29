#!/usr/bin/env python3
"""
Step 002: Kinematic Analysis for TEP-WB

Calculates the dimensionless velocity parameter (v_tilde) for the clean wide binary sample.
v_tilde = Delta_v_2D / v_circular(s)

Scientific Rationale & Math:
To test gravity, a comparison is made the observed 1D projected velocity difference on the sky (Delta_v_2D)
to the velocity expected from purely Newtonian/Keplerian physics given the masses and separation.

1. Distance: Calculated as 1000 / parallax_avg (in parsecs).
2. Proper Motion to Velocity: v_tan = 4.74 * mu * d. 
   Since Gaia proper motions are in mas/yr, the script scales by 1e-3 to get km/s.
3. Mass Estimation: Uses the empirical Mamajek (2022) Main Sequence relations (M_G -> M_sun).
4. Metallicity Mass Bias Correction:
   - Halo stars are older and metal-poor compared to Disk stars.
   - Lower metallicity means lower opacity, so the star is hotter and bluer for a given mass.
   - If the pipeline uses a single solar-metallicity relation (Mamajek), we will systematically over-estimate 
     the masses of bluer Halo stars. 
   - Overestimating the mass inflates the expected Newtonian velocity (v_circ), artificially 
     depressing the observed v_tilde anomaly.
   - We fit a color-magnitude ridge line to the local Disk (solar metallicity) and correct masses
     using the color residual Delta_C.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.download_mamajek import download_mamajek
from scripts.utils.parse_mamajek import clean_mamajek
from scripts.utils.gaia_metallicity import add_metallicity_column
from scripts.utils.tep_model import G_AU, MLR_BETA_PRIOR

def analyze_kinematics():
    print_status("Initializing Kinematic Analysis Pipeline", "TITLE")
    
    input_path = PROJECT_ROOT / "data" / "processed" / "clean_binary_sample.parquet"
    output_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    mamajek_path = PROJECT_ROOT / "data" / "processed" / "mamajek_clean.csv"
    mamajek_raw_path = PROJECT_ROOT / "data" / "raw" / "EEM_dwarf_UBVIJHK_colors_Teff.txt"

    if not input_path.exists():
        print_status(f"Clean sample not found at {input_path}. Run step_001 first.", "ERROR")
        sys.exit(1)

    if not mamajek_path.exists():
        if not mamajek_raw_path.exists():
            print_status("Mamajek table missing. Downloading source table...", "PROCESS")
            if not download_mamajek(mamajek_raw_path):
                print_status("Failed to download Mamajek source table.", "ERROR")
                sys.exit(1)
        print_status("Preparing cleaned Mamajek table...", "PROCESS")
        clean_mamajek(mamajek_raw_path, mamajek_path)

    print_status("Loading clean sample...", "PROCESS")
    df = pd.read_parquet(input_path)
    
    print_status("Loading empirical mass relations (Mamajek 2022)...", "PROCESS")
    mg_df = pd.read_csv(mamajek_path)
    mg_df = mg_df.sort_values('M_G')
    
    # Create interpolation function: Absolute Magnitude (M_G) -> Solar Masses (M_sun)
    mass_interp = interp1d(mg_df['M_G'], mg_df['M_sun'], bounds_error=False, fill_value="extrapolate")

    print_status("Calculating absolute magnitudes using distance from inverse variance-weighted parallax...", "PROCESS")
    # Inverse variance weighted mean of the two parallaxes
    w1 = 1 / df['parallax_error1']**2
    w2 = 1 / df['parallax_error2']**2
    mean_parallax = (df['parallax1'] * w1 + df['parallax2'] * w2) / (w1 + w2)
    
    # Distance in parsecs
    df['dist_pc'] = 1000.0 / mean_parallax
    
    # Absolute magnitude: M = m - 5*log10(d) + 5
    df['Mg1'] = df['phot_g_mean_mag1'] - 5 * np.log10(df['dist_pc']) + 5
    df['Mg2'] = df['phot_g_mean_mag2'] - 5 * np.log10(df['dist_pc']) + 5
    
    print_status("Estimating initial solar-metallicity baseline masses via Mamajek (2022) empirical relations...", "PROCESS")
    df['mass1_est'] = mass_interp(df['Mg1'])
    df['mass2_est'] = mass_interp(df['Mg2'])
    
    # Filter to main-sequence stars within reliable Mamajek bounds
    df = df[(df['mass1_est'] > 0.05) & (df['mass1_est'] < 5.0)]
    df = df[(df['mass2_est'] > 0.05) & (df['mass2_est'] < 5.0)]
    
    print_status("Transforming ICRS to Galactocentric coordinates to enable Disk/Halo environmental stratification...", "PROCESS")
    # Transform from ICRS (RA/Dec) to Galactocentric coordinates (R, Z) to separate Disk and Halo
    coords = SkyCoord(
        ra=df['ra1'].values * u.deg,
        dec=df['dec1'].values * u.deg,
        distance=df['dist_pc'].values * u.pc,
        frame='icrs'
    )
    galcen = coords.transform_to('galactocentric')
    df['R_gc'] = np.sqrt(galcen.x.to(u.kpc).value**2 + galcen.y.to(u.kpc).value**2)
    df['Z_gc'] = galcen.z.to(u.kpc).value
    
    print_status("Deriving photometric metallicities for β_MLR calibration...", "PROCESS")
    df = add_metallicity_column(df, source_id_col='source_id1')
    
    print_status("Applying metallicity mass correction: Adjusting inferred masses for the lower-metallicity halo population to ensure an accurate Newtonian baseline...", "PROCESS")
    # Define empirical ridge line for the local disk (|Z| < 100 pc)
    mask_disk_ref = (np.abs(df['Z_gc']) < 0.1) & (df['Mg1'] > 4) & (df['Mg1'] < 10) & df['bp_rp1'].notna()
    df_disk = df[mask_disk_ref]
    
    # Guard against insufficient data for polynomial fit
    if len(df_disk) < 10:
        print_status(f"Insufficient disk stars for metallicity ridge fit ({len(df_disk)} stars). Skipping mass correction.", "WARNING")
        df['mass1_corr'] = df['mass1_est']
        df['mass2_corr'] = df['mass2_est']
    else:
        # Fit 3rd order polynomial to the main sequence color-magnitude track
        coeffs_ridge = np.polyfit(df_disk['Mg1'], df_disk['bp_rp1'], 3)
        poly_ridge = np.poly1d(coeffs_ridge)
        
        # Calculate color residual from the solar-metallicity track
        # This is computed for ALL stars in the dataframe, not just disk stars
        df['c_ref'] = poly_ridge(df['Mg1'])
        df['delta_c'] = df['bp_rp1'] - df['c_ref']
        
        # Re-create df_disk with delta_c now available (for beta calibration)
        df_disk = df[mask_disk_ref]
        
        # Metal poor (Halo) stars are bluer (delta_c < 0).
        # Therefore, we must decrease their mass relative to the Mamajek solar estimate.
        # Beta is an empirical calibration factor relating color residual to mass offset.
        #
        # CALIBRATION GUARDRAIL:
        # Use only independent spectroscopic [Fe/H] values to calibrate beta_MLR.
        # Photometric [Fe/H] proxies are derived from delta_c and would be circular
        # if regressed back onto delta_c. If no spectroscopic cache is present, use
        # the conservative literature prior from tep_model.MLR_BETA_PRIOR; step_005
        # then stress-tests the environmental ordering across beta=0, 1, 2 and a
        # quadratic correction.

        if 'feh' not in df_disk.columns:
            print_status("ERROR: Metallicity column 'feh' not found. Cannot perform mass-bias calibration.", "ERROR")
            sys.exit(1)

        source = df_disk.get('feh_source', pd.Series('', index=df_disk.index)).fillna('')
        mask_valid_feh = (
            df_disk['feh'].notna() &
            np.isfinite(df_disk['feh']) &
            source.eq('spectroscopic')
        )
        n_valid_feh = mask_valid_feh.sum()

        if n_valid_feh >= 100:
            # Calculate mass residual from independent spectroscopic metallicity.
            # Metal-poor stars (feh < 0) are more luminous -> lower mass at fixed Mg.
            # M_true / M_solar_est = 1 + 0.8 * [Fe/H] (stellar-physics prior).
            df_disk_cal = df_disk[mask_valid_feh].copy()
            delta_feh = df_disk_cal['feh'].values
            mass_ratio_true = 1.0 + 0.8 * delta_feh
            fractional_mass_residual = mass_ratio_true - 1.0
            delta_c_cal = df_disk_cal['delta_c'].values

            valid_mask = (
                (np.abs(delta_c_cal) < 0.5) &
                np.isfinite(fractional_mass_residual) &
                (np.abs(fractional_mass_residual) < 0.5)
            )
            n_for_regression = valid_mask.sum()

            if n_for_regression < 50:
                print_status(f"ERROR: Insufficient spectroscopic stars after outlier rejection (N={n_for_regression}, need ≥50)", "ERROR")
                sys.exit(1)

            from scipy import stats as scipy_stats
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                delta_c_cal[valid_mask], fractional_mass_residual[valid_mask]
            )

            if p_value > 0.001:
                print_status(f"ERROR: β calibration statistically insignificant (p={p_value:.4f}, need p<0.001)", "ERROR")
                sys.exit(1)

            if not (0.3 < slope < 4.0):
                print_status(f"ERROR: β = {slope:.3f} outside physically reasonable range (0.3–4.0)", "ERROR")
                sys.exit(1)

            beta = float(slope)
            beta_err = float(std_err)
            calibration_mode = "spectroscopic"
            p_value_out = max(float(p_value), 1e-300)

            print_status("=" * 60, "INFO")
            print_status("INDEPENDENT SPECTROSCOPIC β_MLR CALIBRATION COMPLETE", "SUCCESS")
            print_status(f"  β = {beta:.4f} ± {beta_err:.4f}", "RESULT")
            print_status(f"  Pearson r = {r_value:.4f}", "RESULT")
            print_status(f"  p-value = {p_value_out:.2e}", "RESULT")
            print_status(f"  N = {n_for_regression} spectroscopic stars", "RESULT")
            print_status(f"  intercept = {intercept:.4f}", "RESULT")
            print_status("=" * 60, "INFO")
        else:
            beta = float(MLR_BETA_PRIOR)
            beta_err = 0.5
            calibration_mode = "literature_prior"
            r_value = np.nan
            p_value_out = np.nan
            n_for_regression = int(n_valid_feh)
            intercept = np.nan
            print_status(
                f"Only {n_valid_feh} independent spectroscopic metallicities available; "
                f"using conservative literature β_MLR prior = {beta:.2f} ± {beta_err:.2f}. "
                "Photometric [Fe/H] proxies are retained for diagnostics but are not used to self-calibrate β.",
                "WARNING",
            )

        # Save calibration results for downstream analysis
        calib_results = {
            'beta': float(beta),
            'beta_err': float(beta_err),
            'r_value': None if not np.isfinite(r_value) else float(r_value),
            'p_value': None if not np.isfinite(p_value_out) else float(p_value_out),
            'n_stars': int(n_for_regression),
            'n_spectroscopic_available': int(n_valid_feh),
            'intercept': None if not np.isfinite(intercept) else float(intercept),
            'calibration_mode': calibration_mode,
            'reference': 'Spectroscopic-only calibration when available; otherwise conservative Covey/Ivezic color-MLR prior with explicit sensitivity tests'
        }

        calib_path = PROJECT_ROOT / "results" / "outputs" / "002_beta_mlr_calibration.json"
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(calib_path, 'w') as f:
            json.dump(calib_results, f, indent=2)
        print_status(f"Saved calibration results to {calib_path}", "INFO")
        df['mass1_corr'] = np.where(pd.notna(df['bp_rp1']), df['mass1_est'] * (1 + beta * df['delta_c']), df['mass1_est'])
        
        c_ref2 = poly_ridge(df['Mg2'])
        delta_c2 = df['bp_rp2'] - c_ref2
        df['mass2_corr'] = np.where(pd.notna(df['bp_rp2']), df['mass2_est'] * (1 + beta * delta_c2), df['mass2_est'])
    
    # Safety clip to prevent extreme unphysical corrections due to bad photometry
    df['mass1_corr'] = np.clip(df['mass1_corr'], df['mass1_est']*0.5, df['mass1_est']*1.5)
    df['mass2_corr'] = np.clip(df['mass2_corr'], df['mass2_est']*0.5, df['mass2_est']*1.5)
    
    df['mass_total'] = df['mass1_corr'] + df['mass2_corr']
    
    print_status("Calculating exact 1D transverse velocities (km/s) and Newtonian expected circular velocities...", "PROCESS")
    # Proper motion difference in mas/yr
    d_pmra = df['pmra1'] - df['pmra2']
    d_pmdec = df['pmdec1'] - df['pmdec2']
    mu_rel = np.sqrt(d_pmra**2 + d_pmdec**2)
    
    # Convert to km/s. The 1e-3 factor accounts for the milli-arcsecond unit.
    df['v_tan_rel_kms'] = 4.74047 * 1e-3 * mu_rel * df['dist_pc']
    
    # Keplerian circular velocity expectation
    s_au = df['sep_AU']
    df['v_circ_newton'] = np.sqrt(G_AU * df['mass_total'] / s_au)
    
    # The key observable: Dimensionless velocity ratio
    df['v_tilde'] = df['v_tan_rel_kms'] / df['v_circ_newton']
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['v_tilde', 'sep_AU'])
    
    try:
        df.to_parquet(output_path)
        print_status(f"Saved kinematic results to {output_path}", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to save output: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_002", str(log_dir / "step_002_kinematic_analysis.log"))
    set_step_logger(logger)

    analyze_kinematics()
