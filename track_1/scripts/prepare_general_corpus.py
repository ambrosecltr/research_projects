#!/usr/bin/env python3
"""Prepare the pinned Ultra-FineWeb corpus for the general 8M model."""

from __future__ import annotations

import argparse
from pathlib import Path

from poetry50m.data.general_corpus import prepare_general_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(
        prepare_general_corpus(
            config_path=args.config,
            output_directory=args.output,
            scratch_directory=args.scratch,
            tokenizer_path=args.tokenizer,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
