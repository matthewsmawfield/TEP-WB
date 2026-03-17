# TEP-WB: Wide Binary Stars as Probes of the Temporal Equivalence Principle

## TEP Series: Paper 15 (Wide Binaries)

This repository contains the analysis pipeline and manuscript for **Paper 15** of the TEP series, investigating gravitational dynamics in wide binary star systems using Gaia DR3 data.

### Abstract

Wide binary stars with separations exceeding ~3,000 AU have emerged as a critical testing ground for gravitational physics, with recent studies reporting conflicting evidence for deviations from Newtonian gravity. We present a systematic analysis of Gaia DR3 wide binary dynamics through the lens of the Temporal Equivalence Principle (TEP), which predicts a screening transition at separations where orbital accelerations approach the critical value a₀ ≈ 1.2×10⁻¹⁰ m/s². For solar-mass binaries, this corresponds to r_trans ≈ 7,000 AU.

### Key Findings

- **Screening Transition**: TEP predicts a gradual transition from Keplerian to enhanced dynamics at ~7,000 AU
- **Partial MOND Signal**: The ~22% velocity boost at >3,000 AU is consistent with unscreened TEP dynamics
- **Resolution of Controversy**: Neither pure MOND nor pure GR fully describes the data; TEP provides a unified framework

### Repository Structure

```
TEP-WB/
├── data/                    # Gaia DR3 wide binary catalogs
├── scripts/steps/           # Reproducible analysis pipeline
├── results/outputs/         # Analysis outputs and figures
├── site/components/         # Manuscript HTML sections
├── manuscripts/             # Generated PDF manuscripts
└── logs/                    # Execution logs
```

### Data Sources

- **Gaia DR3**: Wide binary catalog from El-Badry et al. (2021), updated for DR3
- **Separation Range**: 1,000 - 30,000 AU (probing screened and unscreened regimes)

### License

MIT License - See [LICENSE](LICENSE) for details.
