#!/usr/bin/env python3
"""
TEP-WB: Full Canonical Pipeline

Executes the complete, rigorously vetted wide binary analysis pipeline sequentially.

Scientific Process:
1. Ingest raw Gaia DR3 catalog (El-Badry 2021).
2. Apply astrometric quality filters (e.g., RUWE < 1.2) to isolate a high-purity wide binary sample.
3. Compute kinematics, integrating metallicity-dependent mass corrections for the high-galactic-latitude sample.
4. Perform statistical fitting of the global kinematic profile to quantify deviations from Newtonian expectations.
5. Generate diagnostic validation plots.
6. Analyze environmental modulation (Disk vs. Halo) to evaluate local potential dependence.
7. Audit final parameters (e.g., universal critical density) for astrophysical validity.
"""

import sys
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60 + "\n")

def run_step(script_name):
    print_header(f"Running {script_name}")
    result = subprocess.run([sys.executable, f"scripts/steps/{script_name}"])
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline halted. {script_name} failed.")
        sys.exit(1)

def main():
    steps = [
        "step_001_sample_selection.py",
        "step_002_kinematic_analysis.py",
        "step_003_screening_test.py",
        "step_004_sample_characterization.py",
        "step_005_environment_test.py",
        "step_006_audit_analysis.py",
        "step_007_supplemental_controls.py"
    ]
    
    # Note: step_000 is usually run once to download the 1.8GB file. 
    # Skipping here to allow rapid pipeline re-runs on cached data.
    
    for step in steps:
        run_step(step)
        
    print_header("Pipeline Execution Complete")

if __name__ == "__main__":
    main()
