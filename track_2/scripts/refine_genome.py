"""Thin compatibility wrapper for ``genome refine-latent``."""
import sys
from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "refine-latent")
    app()
