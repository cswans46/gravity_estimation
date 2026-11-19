"""Backward-compatible entry point for the documented ICIO gravity pipeline.

Run from this directory with:
    python estimation.py --config config.yaml
"""
from run_estimation import main


if __name__ == "__main__":
    main()
