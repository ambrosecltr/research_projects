"""Thin compatibility wrapper for ``genome rate-distortion``."""
import sys
from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "rate-distortion")
    app()
