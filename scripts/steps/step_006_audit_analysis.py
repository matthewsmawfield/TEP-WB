#!/usr/bin/env python3
"""
Step 006: Audit and Systematic Analysis for TEP-WB

This script acts as the final sanity check on the derived astrophysical parameters.

Scientific Rationale:
If the TEP transition radius (R_s) derived from the kinematic data is physically meaningful, 
it should correspond to a physically sensible critical density scale. 
This audit evaluates the implied density of the binary system at the transition point,

rho_c = M_total / Volume_of_sphere(R_s)

Additionally, the methodology double-checks the basic population demographics (Disk vs Halo) to ensure the 
metallicity bias correction used in step 002 was applied to a sample that behaves as expected 
astrophysically (i.e., Halo stars should be systematically bluer/metal-poor for a given absolute magnitude).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

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

def perform_audit():
    print_status("Initializing Audit Analysis", "TITLE")
    
    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status("Data missing.", "ERROR")
        sys.exit(1)
        
    df = pd.read_parquet(input_path)
    
    summary_path = PROJECT_ROOT / "results" / "outputs" / "screening_fit_summary.csv"
    if not summary_path.exists():
        print_status("Screening fit summary missing. Run step_003 first.", "ERROR")
        sys.exit(1)

    fit_summary = pd.read_csv(summary_path)
    
    # Guard against empty or malformed fit summary
    if len(fit_summary) == 0 or 'r_s_au' not in fit_summary.columns:
        print_status("Screening fit summary is empty or missing required columns. Run step_003 first.", "ERROR")
        sys.exit(1)
    
    # 1. Critical Density Check
    # The derived global transition radius is read directly from step_003 output.
    Rs_au = float(fit_summary.loc[0, 'r_s_au'])
    Rs_cm = Rs_au * 1.495978707e13
    M_sol_g = 1.98847e33
    
    med_mass = df['mass_total'].median()
    print_status(f"Median Total Mass: {med_mass:.2f} M_sun", "INFO")
    
    # Vol = 4/3 * pi * r^3
    vol = (4/3) * np.pi * Rs_cm**3
    mass_g = med_mass * M_sol_g
    rho_c = mass_g / vol
    
    print_status(f"Implied Critical Density (at Rs={Rs_au:.0f} AU): {rho_c:.3e} g/cm^3", "RESULT")
    
    # 2. Population Demographics Check
    print_status("Assessing color-magnitude distributions to verify the expected photometric properties of the halo and disk populations.", "PROCESS")
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    mask_low_z = np.abs(df['Z_gc']) < 0.1 # Inner Disk
    mask_high_z = np.abs(df['Z_gc']) > 0.15 # Halo / Thick Disk
    
    disk = df[mask_low_z]
    halo = df[mask_high_z]
    
    print_status(f"Disk sample: {len(disk)}", "INFO")
    print_status(f"Halo sample: {len(halo)}", "INFO")
    
    # Verify the core assumption of the step 002 metallicity correction:
    # Are the Halo stars genuinely bluer than Disk stars at a given absolute magnitude?
    if 'bp_rp1' in df.columns:
        plt.figure(figsize=(10, 8))
        
        plt.scatter(halo['bp_rp1'], halo['Mg1'], s=1, color='#6A5ACD', alpha=0.5, label='Halo (|Z|>150pc)')  # SlateBlue
        plt.scatter(disk['bp_rp1'], disk['Mg1'], s=1, color='#4682B4', alpha=0.1, label='Disk (|Z|<100pc)')  # SteelBlue
        
        plt.gca().invert_yaxis()
        plt.xlabel('Bp - Rp (Color)')
        plt.ylabel('M_G (Absolute Magnitude)')
        plt.title('CMD: Disk vs Halo Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fig_path = figures_dir / "cmd_audit.png"
        plt.savefig(fig_path, dpi=300)
        print_status(f"Saved CMD audit to {fig_path}", "SUCCESS")
        
        # Check median color in a standard MS slice (Mg between 5 and 8)
        mask_mag = (disk['Mg1'] > 5) & (disk['Mg1'] < 8)
        med_color_disk = disk[mask_mag]['bp_rp1'].median()
        
        mask_mag_h = (halo['Mg1'] > 5) & (halo['Mg1'] < 8)
        med_color_halo = halo[mask_mag_h]['bp_rp1'].median()
        
        print_status(f"Median Color (5<Mg<8) Disk: {med_color_disk:.3f}", "INFO")
        print_status(f"Median Color (5<Mg<8) Halo: {med_color_halo:.3f}", "INFO")
        
        if med_color_halo < med_color_disk:
            print_status("Halo is Bluer -> Consistent with lower metallicity. Step 002 mass correction is physically justified.", "SUCCESS")
        else:
            print_status(f"Halo color ({med_color_halo}) >= Disk color ({med_color_disk})? Unexpected.", "WARNING")

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_006", str(log_dir / "step_006_audit_analysis.log"))
    set_step_logger(logger)

    perform_audit()
