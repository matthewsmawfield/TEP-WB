#!/usr/bin/env python3
"""
Step 000: Catalog Ingestion for TEP-WB

Downloads the El-Badry et al. (2021) Gaia eDR3 wide binary catalog from Zenodo.
This dataset serves as the foundational catalog for testing the TEP screening transition.

Scientific Context:
The El-Badry (2021) catalog is the gold standard for wide binary research. It uses 
probabilistic modeling of the 5-parameter astrometric space (position, parallax, 
proper motion) to identify bound stellar pairs while rigorously characterizing the 
chance-alignment rate of unbound interlopers.

This script pulls the raw FITS file, converts it into a pandas DataFrame, and 
pre-calculates foundational geometric parameters (like projecting 2D sky coordinates 
into the 3D Galactocentric reference frame).
"""

import os
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table

# Add the project root to the Python path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.downloader import smart_download

def ingest_catalog():
    print_status("Initializing Catalog Ingestion Pipeline", "TITLE")

    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URL for El-Badry et al. (2021) catalog on Zenodo
    # the pipeline uses the main catalog: all_columns_catalog.fits.gz
    catalog_url = "https://zenodo.org/records/4435257/files/all_columns_catalog.fits.gz?download=1"
    filename = "all_columns_catalog.fits.gz"
    target_path = data_dir / filename
    
    if not target_path.exists():
        print_status(f"Downloading {filename} from Zenodo...", "PROCESS")
        try:
            # Using smart_download with a minimum size of 100MB (catalog is 1.8GB)
            # Force n_workers=1 to avoid chunk assembly issues with large files in some envs
            success = smart_download(catalog_url, target_path, min_size_mb=100.0, n_workers=1)
            if success:
                print_status(f"Successfully downloaded {filename}", "SUCCESS")
            else:
                print_status(f"Download failed for {filename}", "ERROR")
                sys.exit(1)
        except Exception as e:
            print_status(f"Download exception: {e}", "ERROR")
            sys.exit(1)
    else:
        print_status(f"Found {filename} locally. Skipping download.", "INFO")
    
    # Convert FITS to CSV with derived quantities
    csv_path = data_dir / "gaia_dr3_wide_binaries_elbadry.csv"
    
    if not csv_path.exists():
        print_status("Converting FITS catalog to CSV...", "PROCESS")
        try:
            table = Table.read(target_path)
            print_status(f"Loaded FITS table with {len(table)} binaries", "INFO")
            
            # Convert to pandas DataFrame
            df = table.to_pandas()
            
            # Compute derived quantities needed for analysis
            print_status("Computing derived quantities...", "PROCESS")
            
            # Physical separation in AU (El-Badry catalog uses sep_AU directly)
            # We convert it to parsecs for standardized calculations.
            df['separation_au'] = df['sep_AU']
            df['separation_pc'] = df['separation_au'] / 206265
            
            # Distance from average parallax. 
            # The inverse of parallax (in milli-arcseconds) yields distance in parsecs.
            parallax_avg = (df['parallax1'] + df['parallax2']) / 2.0
            df['distance_pc'] = 1000.0 / parallax_avg
            
            # Store individual parallaxes and errors for quality cuts
            df['parallax_1'] = df['parallax1']
            df['parallax_error_1'] = df['parallax_error1']
            
            # Note: We assign a dummy mass here. 
            # True empirical mass derivation occurs rigorously in step_002.
            df['mass_total_solar'] = 1.0
            
            # Keplerian circular velocity expectation (baseline formula)
            G = 4.302e-3  # pc (km/s)^2 / M_sun
            df['v_circ_keplerian'] = np.sqrt(G * df['mass_total_solar'] / df['separation_pc'])
            
            # Compute sky-projected relative velocity from proper motion difference.
            # Following standard astrometric theory (e.g. Chae 2023 Eq 6):
            # v_p (km/s) = 4.74047 × Δμ (arcsec/yr) × d (pc)
            # With Gaia proper motions in mas/yr, the formula scales to:
            dpmra = df['pmra1'] - df['pmra2']  # mas/yr
            dpmdec = df['pmdec1'] - df['pmdec2']  # mas/yr
            dpm_total = np.sqrt(dpmra**2 + dpmdec**2)  # mas/yr
            
            # The 4.74 constant is the exact geometrical mapping of angular motion to physical speed.
            df['v_relative_observed'] = 4.74 * dpm_total * df['distance_pc']
            
            # Dimensionless velocity ratio (key observable for testing gravity)
            # This standardizes the kinematics independent of the exact mass or orbital scale.
            df['v_tilde'] = df['v_relative_observed'] / df['v_circ_keplerian']
            
            # Store raw sky coordinates
            df['ra_1'] = df['ra1']
            df['dec_1'] = df['dec1']
            df['source_id_1'] = df['source_id1']
            df['source_id_2'] = df['source_id2']
            
            # Galactic coordinates - proper conversion from ICRS (Equatorial) to Galactic
            print_status("Converting to Galactic coordinates...", "PROCESS")
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # the pipeline uses the average location of the pair to compute the Galactic frame position.
            ra_avg = (df['ra1'].values + df['ra2'].values) / 2.0
            dec_avg = (df['dec1'].values + df['dec2'].values) / 2.0
            dist = df['distance_pc'].values
            
            coords = SkyCoord(ra=ra_avg, dec=dec_avg, distance=dist,
                            unit=(u.degree, u.degree, u.pc), frame='icrs')
            gal_coords = coords.galactic
            
            df['l_gal'] = gal_coords.l.degree
            df['b_gal'] = gal_coords.b.degree
            
            # Galactocentric coordinate transformation
            # This is critical for testing the chameleon screening hypothesis in step_005.
            # We must know how deep the binary sits within the Galactic dark matter halo.
            R_sun = 8.2  # kpc
            z_sun = 0.025  # kpc
            
            x_gc = R_sun - gal_coords.cartesian.x.to(u.kpc).value
            y_gc = -gal_coords.cartesian.y.to(u.kpc).value  
            z_gc = gal_coords.cartesian.z.to(u.kpc).value - z_sun
            
            df['R_galactocentric_kpc'] = np.sqrt(x_gc**2 + y_gc**2)
            df['z_galactic_kpc'] = z_gc
            
            df.to_csv(csv_path, index=False)
            print_status(f"Saved {len(df)} binaries to CSV", "SUCCESS")
            
        except Exception as e:
            print_status(f"FITS conversion failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print_status("CSV catalog already exists. Skipping conversion.", "INFO")

    print_status("Catalog ingestion complete.", "SUCCESS")

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_000", str(log_dir / "step_000_catalog_ingestion.log"))
    set_step_logger(logger)
    
    ingest_catalog()
