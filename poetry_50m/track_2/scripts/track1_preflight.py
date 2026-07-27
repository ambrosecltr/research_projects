"""Thin compatibility wrapper for ``genome track1-preflight``."""
import sys

from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "track1-preflight")
    app()
