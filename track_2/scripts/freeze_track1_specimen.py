"""Thin compatibility wrapper for ``genome freeze``."""
import sys
from genome.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "freeze")
    app()
