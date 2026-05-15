# src/logger.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized logging configuration.
#
# WHY logging over print()?
#   print() is for exploration. logging is for production.
#   Differences that matter:
#     - Log levels (DEBUG/INFO/WARNING/ERROR) let you filter noise
#     - Timestamps tell you when something happened
#     - File handlers persist logs after the process exits
#     - In cloud deployments, structured logs are searchable
#     - You can turn off DEBUG logs in production without changing code
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
from pathlib import Path
from config import ROOT_DIR

LOG_DIR  = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage in any src/ file:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Training started")
        logger.warning("Missing values detected: %d rows", count)
        logger.error("Pipeline failed: %s", str(e))

    Output format:
        2024-01-15 14:23:01 | INFO     | trainer    | Training RandomForest
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # File handler — DEBUG and above (captures everything)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger