#!/usr/bin/env python3
"""Acquire Smol-Smoltalk and build the canonical Track 1 SFT mixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from poetry50m.data.sft_mixture import acquire_smoltalk, build_sft_mixture
from poetry50m.data.synthetic_sft import DEFAULT_TARGET_TOKENS


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    acquire = commands.add_parser("acquire-smoltalk")
    acquire.add_argument("--config", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)

    build = commands.add_parser("build")
    build.add_argument("--source-config", type=Path, required=True)
    build.add_argument("--acquisition", type=Path, required=True)
    build.add_argument("--synthetic-receipt", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--heldout", type=Path, nargs="+", required=True)
    build.add_argument("--evaluation-suite", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "acquire-smoltalk":
        print(acquire_smoltalk(args.config, args.output))
    elif args.command == "build":
        print(
            build_sft_mixture(
                source_config_path=args.source_config,
                acquisition_directory=args.acquisition,
                synthetic_receipt_path=args.synthetic_receipt,
                tokenizer_path=args.tokenizer,
                heldout_paths=args.heldout,
                evaluation_suite_path=args.evaluation_suite,
                output_directory=args.output,
                target_tokens=args.target_tokens,
            )
        )
    else:
        raise AssertionError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
