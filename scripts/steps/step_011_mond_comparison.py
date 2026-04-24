#!/usr/bin/env python3

import sys
from pathlib import Path
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger
from scripts.utils.tep_prediction_and_mond_efe import main as run_mond_comparison


class LoggerStream:
    """Captures stdout and redirects to log file; console output preserved."""
    def __init__(self, log_file, stdout):
        self.log_file = log_file
        self.stdout = stdout
    
    def write(self, message):
        self.stdout.write(message)
        if message.strip():
            # Write to log file without echoing to console again
            with open(self.log_file, 'a') as f:
                f.write(message)
    
    def flush(self):
        self.stdout.flush()


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "step_011_mond_comparison.log")
    logger = TEPLogger("step_011", log_file)
    set_step_logger(logger)
    
    # Clear log file at start
    open(log_file, 'w').close()
    
    # Redirect stdout to capture print statements from tep_prediction_and_mond_efe
    original_stdout = sys.stdout
    sys.stdout = LoggerStream(log_file, original_stdout)
    try:
        run_mond_comparison()
    finally:
        sys.stdout = original_stdout  # Restore original stdout
