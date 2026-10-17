# Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19102061.svg)](https://doi.org/10.5281/zenodo.19102061)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-WB: Temporal Shear Recovery in Gaia DR3 Wide Binaries](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.7 (Kilifi)  
**First published:** 19 March 2026 · **Last updated:** 17 October 2026  
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061)  
**Website:** [https://mlsmawfield.com/tep/wb/](https://mlsmawfield.com/tep/wb/)  
**Paper Series:** TEP Series: Paper 13 (Wide Binaries)

## Abstract

<!-- Generated from site/components/1_abstract.html; update the manuscript source, then run tools/sync_current_corpus_readmes.py. -->

The Gaia DR3 catalog of over one million wide binaries opens a precise window onto gravity in the weak-field regime ($a \lesssim 10^{-10}\,\mathrm{m\,s^{-2}}$), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested.

In the Temporal Equivalence Principle (TEP), a conformal scalar field modulates matter proper time as

\begin{equation} \label{eq:conformal_relation} \mathrm{d}\tau/\mathrm{d}t \approx A(\phi), \qquad A(\phi)=\exp(\beta_A\phi/M_{\rm Pl}). \end{equation}

The Cepheid-calibrated response scale is denoted $\kappa_{\rm Cep}$, while the wide-binary transition is parameterized independently by the velocity-profile saturation amplitude $\alpha_{\rm sat}$, not by a bare scalar coupling. This paper tests whether the Gaia wide-binary anomaly is better described as smooth Temporal Shear recovery in weak-field environments.

From 341,315 high-purity systems, the analysis identifies a screening transition at $R_s = 2{,}646 \pm 182$ AU (statistical; $\pm 609$ AU total), strongly preferred over both a flat Newtonian profile ($\Delta \chi² = 14{,}845$) and a constant boost ($\Delta \chi² = 3{,}583$). The transition is strongly preferred under diagonal, AR(1) and Gaussian-process covariance treatments; even the most conservative covariance-aware comparison gives $\Delta\chi² \ge 1{,}073$ against a flat Newtonian profile and $\Delta\chi² \ge 284$ against a constant boost. At large separation the profile saturates at $\alpha_{\rm sat} = 0.366 \pm 0.012$, roughly 35–40% above the Keplerian baseline. Broader smooth-transition fits preserve the same few-thousand-AU onset.

The signal also shows the environmental ordering required by TEP. With a non-circular metallicity guardrail that uses a conservative external $\beta_{\rm MLR}$ prior unless independent spectroscopic metallicities are cached, the lower-density high-$|Z|$ population transitions at smaller radius than the higher-density midplane. The headline comparison is the joint profile-likelihood fit, which shares one saturation amplitude across environments—the physically prescribed constraint, since screening modulates the recovery scale while the ambient-floor-limited plateau is a shared response at fixed environment—and returns $R_s = 3{,}447 \pm 366$ versus $4{,}912 \pm 792$ AU with $\Delta\chi² = 60.3$ ($p = 8.1 \times 10^{-15}$), the ordering preserved in 100% of resamples. The model-free permutation test confirms the ordering at $p \le 10^{-4}$ (0 exceedances in $10{,}000$ realizations; fixed-amplitude protocol: $R_s = 4{,}662 \pm 196$ versus $7{,}131 \pm 1{,}341$ AU), as does the solar-track control ($R_s = 4{,}145 \pm 276$ versus $6{,}856 \pm 920$ AU; $p \le 10^{-3}$), whose zero-metallicity-correction selection additionally breaks the $|Z|$–metallicity correlation by construction. The transition scale furthermore rises with system mass ($R_s = 2{,}778$, $2{,}966$, $4{,}848$ AU across mass terciles; measured exponent $n = 0.64$ with 86% of resamples positive), consistent with the derived radius law $R_s \propto M^{1/2}$. A five-bin $|Z|$ stratification confirms the density-dependent scaling: the environmental ordering is quantified by an empirical density-scaling exponent, $R_s \propto \rho_{\rm amb}^{p}$ with $p \approx 0.4 \pm 0.1$ under free fits of the same stratified radii—consistent with the square-root benchmark $p = 1/2$ and the $\rho^{1/3}$ ambient channel alike (TEP does not commit to a specific microscopic mechanism; the inverse-power chameleon form that motivated the parameterization is excluded for the TEP coupling $\beta_A = -1$, Smawfield 2025a, F1). The ordering itself is selection-independent: restricted to the window below 9 kAU where the purity cut removes fewer than 2% of pairs, the per-separation median-$\tilde v$ ordering is monotone in $|Z|$ in every bin, reaches $+0.043 \pm 0.006$ ($6.9\sigma$, 100% bootstrap-positive) at $s > 3$ kAU, and is unchanged with no purity cut applied at all. Under bound-probability weighting the profiles rise monotonically to the widest measured separations, the absolute high- versus low-$|Z|$ contrast grows to $+0.094 \pm 0.018$ ($5.3\sigma$) at $s = 15$–$30$ kAU, and a mixture deconvolution leaves the bound-component median elevated at $s = 20$–$50$ kAU under interloper fractions up to three times the prior estimate. Scrambling tests and phase-mixed Newtonian orbital forward models fail to reproduce the observed screening preference. The wide-binary anomaly is therefore not a generic low-acceleration excess but a structured, environmentally modulated screening transition—one whose morphology, onset scale, and environmental ordering are quantitatively consistent with the conformal scalar field of TEP and are not reproduced by the Newtonian orbital-projection or MOND/EFE parameterizations tested here.

Keywords: Temporal Equivalence Principle – wide binaries – Gaia DR3 – weak-field gravity – Temporal Shear recovery – environmental transition morphology – Temporal Topology – Temporal Shear – modified gravity – MOND

Current manuscript source: [1_abstract.html](site/components/1_abstract.html).

## Overview

The Gaia DR3 catalog of over one million wide binaries opens a precise window onto gravity in the weak-field regime ($a \lesssim 10^{-10}$ m/s$^2$), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested.

In the Temporal Equivalence Principle (TEP), a conformal scalar field modulates matter proper time as $\mathrm{d}\tau/\mathrm{d}t \approx A(\phi)$, with $A(\phi)=\exp(\beta\phi/M_{\rm Pl})$. The Cepheid-calibrated response scale is denoted $\kappa_{\rm Cep}$, while the wide-binary transition is parameterized independently by the velocity-profile saturation amplitude $\alpha_{\rm sat}$, not by a bare scalar coupling. This paper tests whether the Gaia wide-binary anomaly is better described as smooth Temporal Shear recovery in weak-field environments.

From 341,315 high-purity systems, the analysis identifies a screening transition at $R_s = 2{,}646 \pm 182$ AU (statistical; $\pm 609$ AU total), strongly preferred over both a flat Newtonian profile ($\Delta \chi^2 = 14{,}845$) and a constant boost ($\Delta \chi^2 = 3{,}583$). The transition is strongly preferred under diagonal, AR(1) and Gaussian-process covariance treatments; even the most conservative covariance-aware comparison gives $\Delta\chi^2 \ge 1{,}073$ against a flat Newtonian profile and $\Delta\chi^2 \ge 284$ against a constant boost. At large separation the profile saturates at $\alpha_{\rm sat} = 0.366 \pm 0.012$, roughly 35–40% above the Keplerian baseline. Broader smooth-transition fits preserve the same few-thousand-AU onset.

The signal also shows the environmental ordering required by TEP. With a non-circular metallicity guardrail that uses a conservative external $\beta_{\rm MLR}$ prior unless independent spectroscopic metallicities are cached, the lower-density high-$|Z|$ population transitions at smaller radius than the higher-density midplane ($R_s = 4{,}662 \pm 196$ versus $7{,}131 \pm 1{,}341$ AU), confirmed by a solar-track control ($R_s = 4{,}145 \pm 276$ versus $6{,}856 \pm 920$ AU; permutation $p < 10^{-4}$ for the full sample and $p < 10^{-3}$ for the solar track). Scrambling tests and phase-mixed Newtonian orbital forward models fail to reproduce the observed screening preference. The wide-binary anomaly is therefore not a generic low-acceleration excess but a structured, environmentally modulated screening transition—one whose morphology, onset scale, and environmental ordering are quantitatively consistent with the conformal scalar field of TEP and are not reproduced by the Newtonian orbital-projection or MOND/EFE parameterizations tested here.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Consistency Test | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Synthesis | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Cross-Scale Consistency of ρ_T | [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | **TEP-WB** (This repo) | The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454862](https://doi.org/10.5281/zenodo.19454862) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454619](https://doi.org/10.5281/zenodo.19454619) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446028](https://doi.org/10.5281/zenodo.19446028) |

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
   This canonical runner executes `step_000` through `step_013`, including catalog ingestion, non-circular kinematic calibration guardrails, screening and environmental fits, robustness diagnostics, the authoritative MOND+EFE comparison, a Newtonian orbital forward-model null, and the claim consistency audit.

3. **Build Manuscript:**
   ```bash
   npm run build:markdown --prefix site
   ```
   The final manuscript will be available at `D13-TEP-WB-v0.7-Kilifi.md`.

## Citation

```bibtex
@article{smawfield2026wb,
  title={The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.19102061},
  note={Preprint v0.7 (Kilifi)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthewsmawfield@gmail.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
