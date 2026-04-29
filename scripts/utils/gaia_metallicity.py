#!/usr/bin/env python3
"""
Gaia Metallicity Utility for β_MLR Calibration

Uses Gaia DR3 XP spectrophotometry-derived metallicities when available,
or photometric metallicity proxies from the color-magnitude relation.

For stars without spectroscopic [Fe/H], we use the empirical relation:
  [Fe/H] ≈ a + b × (Bp-Rp - ridge_line(M_G))
where the ridge_line is fit to the local disk (solar metallicity) population.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def estimate_metallicity_from_photometry(df, color_col='bp_rp1', mag_col='Mg1'):
    """
    Estimate photometric metallicity from color-magnitude deviation.
    
    The method uses the deviation from the main sequence ridge line as a
    metallicity proxy. Metal-poor stars are hotter (bluer) at fixed M_G.
    
    Calibration from Ivezic et al. (2008) and Covey et al. (2008):
    Δ[Fe/H] ≈ Δ(Bp-Rp) / 0.35 mag/dex for M dwarfs
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with color and absolute magnitude columns
    color_col : str
        Column name for color (Bp-Rp)
    mag_col : str
        Column name for absolute magnitude (M_G)
    
    Returns
    -------
    feh_est : pd.Series
        Estimated [Fe/H] values
    """
    # Define disk reference sample (solar metallicity)
    mask_disk_ref = (
        (np.abs(df.get('Z_gc', pd.Series(np.zeros(len(df)), index=df.index))) < 0.1) &
        (df[mag_col] > 4) & (df[mag_col] < 10) &
        df[color_col].notna()
    )
    
    if mask_disk_ref.sum() < 100:
        print(f"Warning: Insufficient disk stars for ridge line fit ({mask_disk_ref.sum()})")
        return pd.Series(np.nan, index=df.index)
    
    df_disk = df[mask_disk_ref]
    
    # Fit 3rd order polynomial to the main sequence color-magnitude track
    coeffs_ridge = np.polyfit(df_disk[mag_col], df_disk[color_col], 3)
    poly_ridge = np.poly1d(coeffs_ridge)
    
    # Calculate color residual from the solar-metallicity track
    c_ref = poly_ridge(df[mag_col])
    delta_c = df[color_col] - c_ref
    
    # Convert color residual to metallicity
    # Slope: 0.1 dex [Fe/H] ≈ 0.035 mag in Bp-Rp for M dwarfs (Covey et al. 2008)
    # Bluer stars (delta_c < 0) are metal-poor (feh < 0)
    # Therefore: [Fe/H] ≈ ΔC / 0.35
    feh_est = delta_c / 0.35
    
    # Clip to reasonable range
    feh_est = feh_est.clip(-2.0, 0.5)
    
    return feh_est


def add_metallicity_column(df, source_id_col='source_id1'):
    """
    Add metallicity column to dataframe.
    
    Priority order for metallicity:
    1. Spectroscopic [Fe/H] from LAMOST/APOGEE (if cached)
    2. Photometric estimate from color-magnitude deviation
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Gaia photometry
    source_id_col : str
        Column name for Gaia source_id
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'feh' column
    """
    print("Estimating metallicities from Gaia photometry...")
    
    # Try to load cached spectroscopic metallicities
    cache_paths = [
        PROJECT_ROOT / "data" / "processed" / "lamost_gaia_crossmatch.csv",
        PROJECT_ROOT / "data" / "processed" / "apogee_gaia_crossmatch.csv",
    ]
    
    feh_map = {}
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Loading cached spectroscopic metallicities from {cache_path}...")
            try:
                cached_df = pd.read_csv(cache_path)
                id_col = 'source_id' if 'source_id' in cached_df.columns else 'Source'
                feh_col = 'feh' if 'feh' in cached_df.columns else '[Fe/H]'
                
                if id_col in cached_df.columns and feh_col in cached_df.columns:
                    for _, row in cached_df.iterrows():
                        sid = row.get(id_col)
                        feh = row.get(feh_col)
                        if pd.notna(sid) and pd.notna(feh) and np.isfinite(feh):
                            feh_map[int(sid)] = float(feh)
                    print(f"  Loaded {len(feh_map)} spectroscopic [Fe/H] values")
            except Exception as e:
                print(f"  Failed to load cache: {e}")
    
    # Map spectroscopic metallicities
    if feh_map and source_id_col in df.columns:
        df['feh'] = df[source_id_col].map(feh_map)
        n_spec = df['feh'].notna().sum()
        df['feh_source'] = pd.Series(np.nan, index=df.index, dtype=object)
        df.loc[df['feh'].notna(), 'feh_source'] = 'spectroscopic'
        print(f"  Spectroscopic matches: {n_spec} stars")
    else:
        df['feh'] = np.nan
        df['feh_source'] = pd.Series(np.nan, index=df.index, dtype=object)
        n_spec = 0
    
    # Fill missing with photometric estimates
    mask_missing = df['feh'].isna()
    if mask_missing.sum() > 0:
        print(f"  Estimating photometric metallicities for {mask_missing.sum()} stars...")
        
        # Check if delta_c already exists from earlier step
        if 'delta_c' in df.columns:
            # Use existing delta_c
            feh_photo = df['delta_c'] / 0.35
            feh_photo = feh_photo.clip(-2.0, 0.5)
            df.loc[mask_missing, 'feh'] = feh_photo[mask_missing]
            df.loc[mask_missing, 'feh_source'] = 'photometric_proxy'
        else:
            # Compute from scratch
            feh_photo = estimate_metallicity_from_photometry(df, 'bp_rp1', 'Mg1')
            df.loc[mask_missing, 'feh'] = feh_photo[mask_missing]
            df.loc[mask_missing, 'feh_source'] = 'photometric_proxy'
        
        n_photo = df['feh'].notna().sum() - n_spec
        print(f"  Photometric estimates: {n_photo} stars")
    
    n_total = df['feh'].notna().sum()
    print(f"Total stars with [Fe/H]: {n_total}/{len(df)} ({100*n_total/len(df):.1f}%)")
    
    return df


if __name__ == "__main__":
    print("Gaia metallicity utility loaded")
