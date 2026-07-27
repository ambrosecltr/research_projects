#!/usr/bin/env python3
"""Plan, execute, and locally validate an auditable synthetic poetry corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from poetry50m.data.synthetic_corpus import (
    finalize_synthetic_corpus,
    ingest_generation_results,
    merge_corpus_artifacts,
    plan_generation,
    run_openai_compatible_batch,
    run_synchronous_batch,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan-generation")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--requests", type=int, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--model")
    plan.add_argument("--openai-compatible", action="store_true")
    plan.add_argument(
        "--response-format",
        choices=("json-schema", "json-object", "none"),
        default="json-schema",
    )
    plan.add_argument(
        "--max-tokens-field",
        choices=("max_completion_tokens", "max_tokens"),
        default="max_completion_tokens",
    )

    run_sync = commands.add_parser("run-sync")
    run_sync.add_argument("--requests", type=Path, required=True)
    run_sync.add_argument("--results", type=Path, required=True)
    run_sync.add_argument("--concurrency", type=int, default=8)
    run_sync.add_argument("--requests-per-minute", type=int, default=950)
    run_sync.add_argument("--tokens-per-minute", type=int, default=950_000)

    run_openai = commands.add_parser("run-openai-compatible")
    run_openai.add_argument("--requests", type=Path, required=True)
    run_openai.add_argument("--results", type=Path, required=True)
    run_openai.add_argument("--base-url", required=True)
    run_openai.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_openai.add_argument("--concurrency", type=int, default=8)
    run_openai.add_argument("--requests-per-minute", type=int, default=60)
    run_openai.add_argument("--tokens-per-minute", type=int, default=100_000)
    run_openai.add_argument("--timeout-seconds", type=float, default=180.0)

    ingest = commands.add_parser("ingest-generation")
    ingest.add_argument("--config", type=Path, required=True)
    ingest.add_argument("--requests", type=Path, required=True)
    ingest.add_argument("--results", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--skip-critic", action="store_true")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--candidates", type=Path, required=True)
    finalize.add_argument("--critic-results", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--reference-manifest", type=Path)

    finalize_local = commands.add_parser("finalize-local")
    finalize_local.add_argument("--config", type=Path, required=True)
    finalize_local.add_argument("--candidates", type=Path, required=True)
    finalize_local.add_argument("--output", type=Path, required=True)
    finalize_local.add_argument("--reference-manifest", type=Path)

    merge = commands.add_parser("merge")
    merge.add_argument("--base-manifest", type=Path, required=True)
    merge.add_argument("--base-prompts", type=Path, required=True)
    merge.add_argument("--base-thoughts", type=Path, required=True)
    merge.add_argument("--base-pairings", type=Path, required=True)
    merge.add_argument("--synthetic", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "plan-generation":
        plan_generation(
            args.config,
            request_count=args.requests,
            output_directory=args.output,
            model_override=args.model,
            openai_compatible=args.openai_compatible,
            response_format_mode=args.response_format,
            max_tokens_field=args.max_tokens_field,
        )
    elif args.command == "run-sync":
        run_synchronous_batch(
            args.requests,
            args.results,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            tokens_per_minute=args.tokens_per_minute,
        )
    elif args.command == "run-openai-compatible":
        run_openai_compatible_batch(
            args.requests,
            args.results,
            base_url=args.base_url,
            api_key_environment_variable=args.api_key_env,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            tokens_per_minute=args.tokens_per_minute,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "ingest-generation":
        ingest_generation_results(
            args.config,
            requests_path=args.requests,
            results_path=args.results,
            output_directory=args.output,
            create_critic_requests=not args.skip_critic,
        )
    elif args.command == "finalize":
        finalize_synthetic_corpus(
            args.config,
            candidates_path=args.candidates,
            critic_results_path=args.critic_results,
            output_directory=args.output,
            reference_manifest=args.reference_manifest,
        )
    elif args.command == "finalize-local":
        finalize_synthetic_corpus(
            args.config,
            candidates_path=args.candidates,
            critic_results_path=None,
            output_directory=args.output,
            reference_manifest=args.reference_manifest,
        )
    elif args.command == "merge":
        merge_corpus_artifacts(
            base_manifest=args.base_manifest,
            base_prompts=args.base_prompts,
            base_thoughts=args.base_thoughts,
            base_pairings=args.base_pairings,
            synthetic_directory=args.synthetic,
            output_directory=args.output,
        )
    else:
        raise AssertionError(f"unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
