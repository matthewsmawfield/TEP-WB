# TEP-WB Analysis Scripts

This folder contains the reproducible analysis pipeline for Paper 15.

## Structure

```
scripts/
├── steps/           # Numbered analysis steps
│   ├── step_000_*.py # Data acquisition (catalog download)
│   ├── step_001_*.py # Sample selection
│   ├── step_002_*.py # Kinematic analysis
│   └── ...
└── utils/           # Shared utilities
```

## Execution

Steps should be run in numerical order. Each step produces outputs in `results/outputs/`.
