#!/usr/bin/env python3
"""
LAMOST Cross-match Utility for Metallicity Calibration

Cross-matches Gaia source_ids with LAMOST DR6 to obtain spectroscopic [Fe/H]
for empirical β_MLR calibration in step_002.

Data source: LAMOST DR6 (LRS) via CDS XMatch service or local catalog
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def query_lamost_crossmatch(gaia_source_ids, ra, dec, search_radius_arcsec=3.0):
    """
    Query LAMOST DR6 for spectroscopic metallicities via CDS XMatch service.
    
    Uses the CDS XMatch service to cross-match Gaia DR3 source_ids with LAMOST DR6.
    
    Parameters
    ----------
    gaia_source_ids : array-like
        Gaia DR3 source_ids
    ra : array-like  
        Right ascension (degrees)
    dec : array-like
        Declination (degrees)
    search_radius_arcsec : float
        Cross-match radius in arcseconds (default: 3.0")
    
    Returns
    -------
    feh_map : dict
        Mapping of Gaia source_id -> LAMOST [Fe/H]
    """
    # Check for cached cross-match
    cache_path = PROJECT_ROOT / "data" / "processed" / "lamost_gaia_crossmatch.csv"
    
    if cache_path.exists():
        print(f"Loading cached LAMOST cross-match from {cache_path}...")
        lamost_df = pd.read_csv(cache_path)
    else:
        # Use CDS XMatch service for Gaia-LAMOST cross-match
        print("Querying CDS XMatch for Gaia DR3 x LAMOST DR6 cross-match...")
        
        try:
            from astroquery.xmatch import XMatch
            
            # Create a table with our Gaia source IDs and positions
            sample_table = Table({
                'source_id': gaia_source_ids[:min(1000, len(gaia_source_ids))],  # Sample for testing
                'ra': ra[:min(1000, len(ra))],
                'dec': dec[:min(1000, len(dec))]
            })
            
            # Query XMatch with LAMOST DR6 (catalog: V/308/stellar)
            xmatch = XMatch()
            result = xmatch.query(
                cat1=sample_table,
                cat2='V/308/stellar',
                max_distance=search_radius_arcsec * u.arcsec,
                colRA1='ra',
                colDec1='dec'
            )
            
            if len(result) > 0:
                lamost_df = result.to_pandas()
                # Save cache
                lamost_df.to_csv(cache_path, index=False)
                print(f"Cached {len(lamost_df)} LAMOST matches to {cache_path}")
            else:
                print("No LAMOST matches found in sample")
                return {}
                
        except Exception as e:
            print(f"Failed to query XMatch: {e}")
            # Try alternative: APOGEE-Gaia cross-match which is more reliable
            return _query_apogee_crossmatch(gaia_source_ids, ra, dec, search_radius_arcsec)
    
    # Build mapping from source_id to feh
    feh_map = {}
    if 'source_id' in lamost_df.columns and ('feh' in lamost_df.columns or '[Fe/H]' in lamost_df.columns):
        feh_col = 'feh' if 'feh' in lamost_df.columns else '[Fe/H]'
        for _, row in lamost_df.iterrows():
            sid = int(row['source_id'])
            feh = row[feh_col]
            if pd.notna(feh) and np.isfinite(feh):
                feh_map[sid] = float(feh)
    
    return feh_map


def _query_apogee_crossmatch(gaia_source_ids, ra, dec, search_radius_arcsec=3.0):
    """
    Fallback: Query APOGEE DR17 for spectroscopic metallicities.
    APOGEE has better sky coverage and more reliable stellar parameters.
    """
    print("Trying APOGEE DR17 as fallback...")
    
    cache_path = PROJECT_ROOT / "data" / "processed" / "apogee_gaia_crossmatch.csv"
    
    if cache_path.exists():
        print(f"Loading cached APOGEE cross-match from {cache_path}...")
        apogee_df = pd.read_csv(cache_path)
    else:
        try:
            from astroquery.vizier import Vizier
            
            Vizier.ROW_LIMIT = 500000  # Large limit for APOGEE
            
            # Query APOGEE DR17 (IV/38/catalog)
            print("Querying APOGEE DR17 from VizieR...")
            catalogs = Vizier.get_catalogs('IV/38/catalog')
            apogee_table = catalogs[0]
            apogee_df = apogee_table.to_pandas()
            
            # Save cache
            apogee_df.to_csv(cache_path, index=False)
            print(f"Cached {len(apogee_df)} APOGEE stars to {cache_path}")
        except Exception as e:
            print(f"Failed to query APOGEE: {e}")
            return {}
    
    # Match by Gaia source_id if available
    feh_map = {}
    id_col = 'Source' if 'Source' in apogee_df.columns else 'source_id'
    feh_col = '[Fe/H]' if '[Fe/H]' in apogee_df.columns else 'FE_H'
    
    if id_col in apogee_df.columns:
        for _, row in apogee_df.iterrows():
            sid = row.get(id_col)
            feh = row.get(feh_col, np.nan)
            if pd.notna(sid) and pd.notna(feh) and np.isfinite(feh):
                feh_map[int(sid)] = float(feh)
    else:
        # Positional cross-match
        gaia_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        apogee_coords = SkyCoord(
            ra=apogee_df['RAJ2000'].values * u.deg,
            dec=apogee_df['DEJ2000'].values * u.deg
        )
        
        idx, sep2d, _ = gaia_coords.match_to_catalog_sky(apogee_coords)
        
        for i, (gaia_id, sep) in enumerate(zip(gaia_source_ids, sep2d)):
            if sep.arcsec < search_radius_arcsec:
                row = apogee_df.iloc[idx[i]]
                feh = row.get(feh_col, np.nan)
                if pd.notna(feh) and np.isfinite(feh):
                    feh_map[int(gaia_id)] = float(feh)
    
    print(f"APOGEE cross-match: {len(feh_map)} stars with [Fe/H]")
    return feh_map


def add_lamost_metallicities(df, source_id_col='source_id1'):
    """
    Add LAMOST [Fe/H] column to dataframe by cross-matching Gaia source_ids.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Gaia source_ids
    source_id_col : str
        Column name for Gaia source_id
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'feh' column
    """
    if source_id_col not in df.columns:
        print(f"Warning: {source_id_col} not found in dataframe")
        df['feh'] = np.nan
        return df
    
    # Get unique source IDs
    unique_ids = df[source_id_col].unique()
    
    # Query LAMOST
    feh_map = query_lamost_crossmatch(
        unique_ids,
        df['ra1'].values,
        df['dec1'].values
    )
    
    # Map to dataframe
    df['feh'] = df[source_id_col].map(feh_map)
    
    n_with_feh = df['feh'].notna().sum()
    print(f"LAMOST cross-match: {n_with_feh}/{len(df)} stars with [Fe/H] ({100*n_with_feh/len(df):.1f}%)")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    print("LAMOST cross-match utility loaded")
