from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import ensure_dir, read_json


def discover_evaluation_reports(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("evaluation.json"))


def flatten_report(report: dict[str, Any], source: str) -> dict[str, Any]:
    functional = report.get("functional_metrics", {})
    parameters = report.get("parameter_metrics", {})
    accounting = report.get("bit_accounting", report.get("bytes", {}))
    compute = report.get("compute", {})
    return {
        "source": source,
        "candidate_id": report.get("candidate_id"),
        "decision": report.get("decision"),
        "mgp_bytes": accounting.get("mgp_bytes", accounting.get("artifact_bytes")),
        "single_model_total_bytes": accounting.get("single_model_total_bytes"),
        "interpreter_bytes": accounting.get("interpreter_bytes", 0),
        "mean_loss": functional.get("candidate_mean_loss"),
        "loss_gap": functional.get("loss_gap"),
        "perplexity": functional.get("candidate_perplexity"),
        "anchor_logit_kl": functional.get("anchor_logit_kl"),
        "top1_agreement": functional.get("top1_agreement"),
        "relative_parameter_l2": parameters.get("relative_l2"),
        "parameter_mse": parameters.get("mse"),
        "fit_seconds": compute.get("fit_seconds"),
        "decode_seconds": compute.get("decode_seconds"),
        "evaluation_seconds": compute.get("evaluation_seconds"),
    }


def make_report(
    inputs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    title: str = "GENOME experiment report",
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    report_paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            report_paths.extend(discover_evaluation_reports(path))
        elif path.name == "evaluation.json":
            report_paths.append(path)
        else:
            raise ValueError(f"unsupported report input: {path}")
    unique = sorted(set(report_paths))
    rows = [flatten_report(read_json(path), str(path)) for path in unique]
    rows.sort(key=lambda row: (float("inf") if row["mgp_bytes"] is None else row["mgp_bytes"], row["candidate_id"] or ""))

    csv_path = output / "results.csv"
    fields = list(rows[0]) if rows else ["candidate_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output / "REPORT.md"
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No evaluation reports were found.")
    else:
        lines.extend(
            [
                "| Candidate | MGP bytes | Single-model bytes | Loss gap | Logit KL | Relative parameter L2 | Decision |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            def fmt(value: Any, precision: int = 6) -> str:
                if value is None:
                    return "—"
                if isinstance(value, int):
                    return f"{value:,}"
                return f"{float(value):.{precision}f}"

            lines.append(
                f"| {row['candidate_id']} | {fmt(row['mgp_bytes'])} | {fmt(row['single_model_total_bytes'])} | "
                f"{fmt(row['loss_gap'])} | {fmt(row['anchor_logit_kl'])} | "
                f"{fmt(row['relative_parameter_l2'])} | {row['decision']} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw_path = output / "results.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path, "json": raw_path}
