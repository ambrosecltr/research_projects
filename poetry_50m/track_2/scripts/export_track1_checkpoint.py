"""Thin compatibility wrapper for ``genome export-track1-checkpoint``."""
import sys

from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "export-track1-checkpoint")
    app()
