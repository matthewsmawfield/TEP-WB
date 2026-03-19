#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger
from scripts.utils.tep_prediction_and_mond_efe import main as run_mond_comparison


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_011", str(log_dir / "step_011_mond_comparison.log"))
    set_step_logger(logger)
    run_mond_comparison()
