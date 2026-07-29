#!/usr/bin/env python3
"""Prepare response-only Fineweb-Instruct data for the general 8M model."""

from __future__ import annotations

import argparse
from pathlib import Path

from poetry50m.data.general_sft import prepare_general_sft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare_general_sft(
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            output_directory=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
