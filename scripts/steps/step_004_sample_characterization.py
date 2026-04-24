#!/usr/bin/env python3
"""
Step 004: Sample Characterization for TEP-WB

Generates diagnostic plots to characterize the selected wide binary sample.

Scientific Rationale:
Before drawing profound cosmological conclusions, we must verify the statistical and 
astrophysical sanity of our post-cut data sample. This script generates four key diagnostics:

1. Distance Distribution: Verifies we are constrained to the local solar neighborhood 
   (typically < 200 pc) where Gaia astrometry is hyper-precise.
2. Separation Distribution: Confirms we have adequate sampling across the critical 
   screening transition boundary (~1,000 to ~10,000 AU). 
3. Mass Distribution: Ensures the sample consists of valid Main Sequence stars 
   (roughly 0.1 to 2.0 Solar Masses), avoiding white dwarfs or brown dwarfs where the 
   Mamajek empirical mass relations break down.
4. Sky Distribution: Checks for severe directional biases or artificial gaps in the 
   Gaia DR3 spatial coverage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

def characterize_sample():
    print_status("Initializing Sample Characterization", "TITLE")
    
    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status(f"Data not found at {input_path}. Run step_002 first.", "ERROR")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print_status(f"Loaded {len(df)} binaries.", "INFO")
    
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication-quality matplotlib defaults
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'font.serif': ['Georgia', 'Times New Roman', 'serif'],
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight'
    })
    
    # 1. Distance Distribution
    # The precision of transverse velocity v_tan = 4.74 * mu * d scales linearly with distance.
    # Therefore, a sample dominated by nearby stars is required for precision gravity tests.
    plt.figure(figsize=(8, 6))
    plt.hist(df['dist_pc'], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Distance [pc]')
    plt.ylabel('Count')
    plt.title('Distance Distribution of Wide Binary Sample')
    plt.grid(alpha=0.3)
    plt.savefig(figures_dir / "004_dist_distribution.png", dpi=600)
    print_status("Saved dist_distribution.png", "SUCCESS")
    
    # 2. Separation Distribution (Logarithmic)
    # Gravity transitions occur logarithmically across spatial scales. We must verify
    # a continuous, well-populated sampling across the expected ~2000 AU transition zone.
    plt.figure(figsize=(8, 6))
    plt.hist(np.log10(df['sep_AU']), bins=50, color='lightgreen', edgecolor='black')
    plt.xlabel('Log10(Projected Separation [AU])')
    plt.ylabel('Count')
    plt.title('Projected Separation Distribution')
    plt.grid(alpha=0.3)
    plt.savefig(figures_dir / "004_sep_distribution.png", dpi=600)
    print_status("Saved sep_distribution.png", "SUCCESS")
    
    # 3. Mass vs Mass (Component Hierarchy)
    # Confirms the empirical mass estimations from the Mamajek relations are bounded
    # within standard stellar physics. Most binaries tend towards a mass ratio q ~ 1.
    plt.figure(figsize=(8, 6))
    plt.hexbin(df['mass1_est'], df['mass2_est'], gridsize=50, cmap='inferno', mincnt=1, bins='log')
    plt.colorbar(label='Log10(Count)')
    plt.xlabel(r'Primary Mass [$M_\odot$]')
    plt.ylabel(r'Secondary Mass [$M_\odot$]')
    plt.title('Component Masses')
    plt.plot([0, 2], [0, 2], 'w--', alpha=0.5)
    plt.grid(alpha=0.3)
    plt.savefig(figures_dir / "004_mass_distribution.png", dpi=600)
    print_status("Saved mass_distribution.png", "SUCCESS")
    
    # 4. Sky Map (Equatorial Projection)
    # Ensures the pipeline did not accidentally filter out specific hemispheres.
    plt.figure(figsize=(10, 5))
    plt.subplot(111, projection='aitoff')
    
    if 'ra1' in df.columns:
        ra_rad = np.radians(df['ra1'] - 180)
        dec_rad = np.radians(df['dec1'])
        
        plt.hexbin(ra_rad, dec_rad, gridsize=50, cmap='viridis', mincnt=1, bins='log')
        plt.colorbar(label='Log10(Count)', orientation='horizontal', pad=0.1)
        plt.title('Sky Distribution (Equatorial)')
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / "004_sky_distribution.png", dpi=600)
        print_status("Saved sky_distribution.png", "SUCCESS")
    else:
        print_status("RA/Dec columns missing for sky map.", "WARNING")

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_004", str(log_dir / "step_004_sample_characterization.log"))
    set_step_logger(logger)
    
    characterize_sample()
