# The Wide Binary Anomaly: Resolving the Gaia DR3 Controversy via Temporal Screening

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Author:** Matthew Lukin Smawfield  
**Status:** In Development  
**Paper Series:** TEP Series: Paper 15 (Wide Binaries)

## Overview

The Gaia DR3 catalog of over one million wide binaries opens a precise test of gravity in the weak-field regime ($a \lesssim 10^{-10}$ m/s$^2$), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested. This paper shows that the Temporal Equivalence Principle (TEP), in which the conformal factor $A(\phi) = \exp(2\beta\phi/M_{\text{Pl}})$ modulates matter proper time as $\mathrm{d}\tau/\mathrm{d}t \propto A(\phi)^{1/2}$, resolves that tension through density-dependent screening.

From 341,315 high-purity systems, the analysis identifies a screening transition at $R_s = 2646 \pm 182$ AU (statistical; $\pm 563$ AU total), strongly preferred over both a flat Newtonian profile ($\Delta \chi^2 = 14{,}845$) and a constant boost ($\Delta \chi^2 = 3{,}583$). At large separation the profile saturates at $\alpha_{\rm sat} = 0.366 \pm 0.012$, roughly 35--40% above the Keplerian baseline.

The signal also shows the environmental ordering required by TEP. After metallicity-dependent mass corrections and bootstrap uncertainty estimation, the lower-density halo transitions at smaller radius than the higher-density disk ($R_s = 4673 \pm 194$ versus $7159 \pm 1573$ AU), confirmed by a solar-track control ($R_s = 4099 \pm 251$ versus $6885 \pm 1223$ AU; permutation $p = 0.0033$ in both cases). Scrambling tests fail to reproduce the observed screening preference ($p = 0.0066$). These results support the interpretation of the wide-binary anomaly as a kinematic signature of the conformal scalar field emerging from screening at sub-galactic scales.

## Repository Structure

```text
TEP-WB/
├── data/                    # Gaia DR3 catalogs and processed samples
├── logs/                    # Execution logs
├── manuscripts/             # Generated PDF/Markdown outputs
├── results/                 # Analytical outputs and figures
├── scripts/
│   ├── steps/               # Sequential analysis pipeline
│   └── utils/               # Shared utilities
├── site/
│   └── components/          # HTML source of truth for manuscript
└── README.md
```

## Reproduction Pipeline

To reproduce the analysis and generate the manuscript:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   npm install --prefix site
   ```

2. **Run Analysis Pipeline:**
   ```bash
   python3 scripts/steps/step_000_catalog_ingestion.py   # Download Gaia DR3 catalog
   python3 scripts/steps/step_001_sample_selection.py    # Filter for high-purity binaries
   python3 scripts/steps/step_002_kinematic_analysis.py  # Calculate v_tilde profiles
   python3 scripts/steps/step_003_screening_test.py      # Fit TEP screening model
   python3 scripts/steps/step_004_sample_characterization.py # Generate diagnostic plots
   python3 scripts/steps/step_005_environment_test.py    # Test environmental dependence
   python3 scripts/steps/step_006_audit_analysis.py      # Audit for systematic biases
   python3 scripts/steps/step_007_supplemental_controls.py # Run RV, stratification, and null-control diagnostics
   ```

3. **Build Manuscript:**
   ```bash
   npm run build:markdown --prefix site
   ```
   The final manuscript will be available at `15manuscript-tep-wb.md`.

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).
