"""Analyze existing screens: --input R0016-screen.json --output new-summary.

Repeat --candidate and --baseline to select methods. Repeat --input to report
multiple runs separately. Writes new JSON and Markdown files only.
"""
import sys
from pathlib import Path

# Also work from a fresh checkout without changing packaging/entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sca3_compass.robustness_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
