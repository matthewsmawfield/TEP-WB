# The Temporal Equivalence Principle: Density-Dependent Screening in Gaia DR3 Wide Binaries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19102062.svg)](https://doi.org/10.5281/zenodo.19102062)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-WB Wide Binary Screening](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Kilifi)  
**Date:** 19 March 2026  
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.19102062](https://doi.org/10.5281/zenodo.19102062)  
**Website:** [https://mlsmawfield.com/tep/wb/](https://mlsmawfield.com/tep/wb/)  
**Paper Series:** TEP Series: Paper 15 (Wide Binaries)

## Overview

The Gaia DR3 catalog of over one million wide binaries opens a precise window onto gravity in the weak-field regime ($a \lesssim 10^{-10}$ m/s$^2$), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested. This paper shows that the Temporal Equivalence Principle (TEP)—in which a conformal scalar field modulates matter proper time as $\mathrm{d}\tau/\mathrm{d}t \approx A(\phi)^{1/2}$—addresses that tension through density-dependent screening.

From 341,315 high-purity systems, the analysis identifies a screening transition at $R_s = 2,646 \pm 182$ AU (statistical; $\pm 609$ AU total), strongly preferred over both a flat Newtonian profile ($\Delta \chi^2 = 14{,}845$) and a constant boost ($\Delta \chi^2 = 3{,}583$). At large separation the profile saturates at $\alpha_{\rm sat} = 0.366 \pm 0.012$, roughly 35--40% above the Keplerian baseline. Broader smooth-transition fits preserve the same few-thousand-AU onset.

The signal also shows the environmental ordering required by TEP. After metallicity-dependent mass corrections and bootstrap uncertainty estimation, the lower-density high-$|Z|$ population transitions at smaller radius than the higher-density midplane ($R_s = 4,673 \pm 194$ versus $7,159 \pm 1,573$ AU), confirmed by a solar-track control ($R_s = 4,099 \pm 239$ versus $6,885 \pm 984$ AU; permutation $p < 10^{-4}$ for the full sample and $p < 10^{-3}$ for the solar track). Scrambling tests fail to reproduce the observed screening preference ($p < 10^{-4}$). The wide-binary anomaly is therefore not a generic low-acceleration excess but a structured, environmentally modulated screening transition—one whose morphology, onset scale, and density dependence are quantitatively consistent with the conformal scalar field of TEP and are not reproduced by MOND with or without the External Field Effect.

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
   python3 scripts/steps/run_all_steps.py
   ```
   This canonical runner executes `step_000` through `step_011`, including catalog ingestion, kinematic analysis, screening and environmental fits, robustness diagnostics, and the authoritative MOND+EFE comparison.

3. **Build Manuscript:**
   ```bash
   npm run build:markdown --prefix site
   ```
   The final manuscript will be available at `15manuscript-tep-wb.md`.

## Citation

```bibtex
@article{smawfield2026wb,
  title={The Temporal Equivalence Principle: Density-Dependent Screening in Gaia DR3 Wide Binaries},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.19102062},
  note={Preprint v0.1 (Kilifi)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthewsmawfield@gmail.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
