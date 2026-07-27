"""Thin compatibility wrapper for ``genome architecture-graph``."""
import sys
from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "architecture-graph")
    app()
