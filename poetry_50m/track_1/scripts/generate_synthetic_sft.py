#!/usr/bin/env python3
"""Plan, run, finalize, and summarize chunked synthetic poetry SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetry50m.data.synthetic_corpus import run_openai_compatible_batch
from poetry50m.data.synthetic_sft import (
    DEFAULT_TARGET_TOKENS,
    assemble_sft_dataset,
    finalize_sft_chunk,
    plan_sft_chunk,
    record_sft_dispatch,
    summarize_sft_chunks,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan-chunk")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--provider", required=True)
    plan.add_argument("--start-index", type=int, required=True)
    plan.add_argument("--examples", type=int, required=True)
    plan.add_argument("--seed", type=int, default=20260728)
    plan.add_argument("--temperature", type=float, default=0.9)
    plan.add_argument("--max-completion-tokens", type=int, default=1024)
    plan.add_argument(
        "--max-tokens-field",
        choices=("max_completion_tokens", "max_tokens"),
        default="max_completion_tokens",
    )

    run = commands.add_parser("run-openai-compatible")
    run.add_argument("--chunk", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--requests-per-minute", type=int, default=60)
    run.add_argument("--tokens-per-minute", type=int, default=100_000)
    run.add_argument("--timeout-seconds", type=float, default=180.0)

    finalize = commands.add_parser("finalize-chunk")
    finalize.add_argument("--chunk", type=Path, required=True)
    finalize.add_argument("--tokenizer", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--receipts", type=Path, nargs="+", required=True)
    summarize.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    summarize.add_argument("--output", type=Path)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--receipts", type=Path, nargs="+", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    assemble.add_argument(
        "--target-metric",
        choices=("formatted", "supervised"),
        default="formatted",
    )
    assemble.add_argument("--allow-under-target", action="store_true")

    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "plan-chunk":
        plan_sft_chunk(
            output_directory=args.output,
            model=args.model,
            provider=args.provider,
            start_index=args.start_index,
            example_count=args.examples,
            seed=args.seed,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            max_tokens_field=args.max_tokens_field,
        )
    elif args.command == "run-openai-compatible":
        record_sft_dispatch(
            plan_path=args.chunk / "plan.json",
            base_url=args.base_url,
            api_key_environment_variable=args.api_key_env,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            tokens_per_minute=args.tokens_per_minute,
            timeout_seconds=args.timeout_seconds,
        )
        run_openai_compatible_batch(
            args.chunk / "requests.jsonl",
            args.chunk / "results.jsonl",
            base_url=args.base_url,
            api_key_environment_variable=args.api_key_env,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            tokens_per_minute=args.tokens_per_minute,
            timeout_seconds=args.timeout_seconds,
            store_final_text_only=True,
        )
    elif args.command == "finalize-chunk":
        receipt_path = finalize_sft_chunk(
            plan_path=args.chunk / "plan.json",
            results_path=args.chunk / "results.jsonl",
            tokenizer_path=args.tokenizer,
            output_directory=args.output,
        )
        print(receipt_path)
    elif args.command == "summarize":
        summary = summarize_sft_chunks(
            args.receipts,
            target_tokens=args.target_tokens,
            output_path=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "assemble":
        receipt_path = assemble_sft_dataset(
            args.receipts,
            output_directory=args.output,
            target_tokens=args.target_tokens,
            target_metric=args.target_metric,
            allow_under_target=args.allow_under_target,
        )
        print(receipt_path)
    else:
        raise AssertionError(f"unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
