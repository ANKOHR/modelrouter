"""Command line interface for the offline, rule-based routing fixture."""

from __future__ import annotations

import argparse
import html
import json
import sys
from typing import Any

from .core import (
    NOTICE,
    append_audit,
    load_catalogue,
    load_workload,
    metrics_from_audit,
    read_audit,
    route_task,
    simulate_workload,
)


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelrouter",
        description="Deterministic rule-based routing over synthetic fixtures. No real model API calls are made.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="inspect the synthetic fixture catalogue")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="list fixture models")
    models_list.add_argument("--format", choices=("table", "json"), default="table")

    route = sub.add_parser("route", help="score a task, apply policy, and estimate fixture cost")
    route.add_argument("--task", required=True)
    route.add_argument("--latency-budget-ms", type=int)
    route.add_argument("--capability", action="append", default=[], help="required capability, repeatable")
    route.add_argument("--input-tokens", type=int)
    route.add_argument("--output-tokens", type=int)
    route.add_argument("--audit", default="mr-audit.jsonl")

    simulate = sub.add_parser("simulate", help="route and account for every task in a JSON workload")
    simulate.add_argument("--workload", help="JSON file, defaults to the bundled fixture workload")
    simulate.add_argument("--latency-budget-ms", type=int)
    simulate.add_argument("--audit", default="mr-audit.jsonl")

    metrics = sub.add_parser("metrics", help="summarize decisions from the JSONL audit log")
    metrics.add_argument("--audit", default="mr-audit.jsonl")

    report = sub.add_parser("report", help="render audit metrics as Markdown or HTML")
    report.add_argument("--format", choices=("md", "html"), default="md")
    report.add_argument("--audit", default="mr-audit.jsonl")
    report.add_argument("--output", help="optional destination file")

    policy = sub.add_parser("policy", help="inspect the named routing policy")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    policy_sub.add_parser("show", help="show ordered policy rules")
    return parser


def _models_output(as_json: bool) -> str:
    models = load_catalogue()
    if as_json:
        return _json({"honesty_boundary": NOTICE, "models": [model.to_dict() for model in models]})
    lines = [NOTICE, "Synthetic fixture models, illustrative costs per 1k tokens", "ID             INPUT/1K  OUTPUT/1K  P50 MS  PRIVACY     CAPABILITIES"]
    for model in models:
        caps = ", ".join(model.capabilities)
        lines.append(
            f"{model.id:<14} ${model.input_cost_per_1k:<8} ${model.output_cost_per_1k:<9} "
            f"{model.p50_latency_ms:<7} {model.privacy_class:<11} {caps}"
        )
    return "\n".join(lines)


def _policy_output() -> str:
    data = {
        "honesty_boundary": NOTICE,
        "policy": "Ordered deterministic filters, followed by lowest estimated task cost.",
        "rules": [
            {"order": 1, "name": "capability_match", "effect": "Exclude models missing a required or inferred capability."},
            {"order": 2, "name": "privacy_boundary", "effect": "At privacy score 50 or higher, allow local and private models only."},
            {"order": 3, "name": "latency_budget", "effect": "If a p50 cap is set, exclude models above it."},
            {"order": 4, "name": "lowest_estimated_cost", "effect": "Choose the lowest estimated cost, breaking ties by model id."},
        ],
    }
    return _json(data)


def _report_text(metrics: dict[str, Any], format_name: str) -> str:
    if format_name == "html":
        models = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
            for name, count in metrics["per_model"].items()
        ) or '<tr><td colspan="2">No decisions recorded</td></tr>'
        rules = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
            for name, count in metrics["rule_fire_counts"].items()
        ) or '<tr><td colspan="2">No rules fired</td></tr>'
        return "\n".join([
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8"><title>ModelRouter report</title>',
            "<style>body{font:16px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#18212f}"
            ".honesty{background:#fff3cd;border:1px solid #e1bd55;padding:1rem;font-weight:600}"
            "table{border-collapse:collapse;margin:1rem 0 2rem}td,th{border:1px solid #ccd3dc;padding:.5rem 1rem;text-align:left}</style></head><body>",
            "<h1>ModelRouter metrics report</h1>",
            f'<p class="honesty">{html.escape(NOTICE)}</p>',
            f"<p>Recorded decisions: {metrics['decision_count']}. Estimated routed cost: ${metrics['estimated_cost_total']:.8f}. "
            f"Always-flagship fixture baseline: ${metrics['always_flagship_cost_total']:.8f}. "
            f"Estimated cost saved: ${metrics['estimated_cost_saved']:.8f} ({metrics['estimated_savings_percent']:.4f}%).</p>",
            "<h2>Routing counts by model</h2><table><thead><tr><th>Model</th><th>Count</th></tr></thead><tbody>", models, "</tbody></table>",
            "<h2>Rule fire counts</h2><table><thead><tr><th>Rule</th><th>Count</th></tr></thead><tbody>", rules, "</tbody></table>",
            "</body></html>",
        ])
    lines = [
        "# ModelRouter metrics report",
        "",
        f"> {NOTICE}",
        "",
        f"Recorded decisions: **{metrics['decision_count']}**  ",
        f"Estimated routed cost: **${metrics['estimated_cost_total']:.8f}**  ",
        f"Always-flagship fixture baseline: **${metrics['always_flagship_cost_total']:.8f}**  ",
        f"Estimated cost saved: **${metrics['estimated_cost_saved']:.8f} ({metrics['estimated_savings_percent']:.4f}%)**",
        "",
        "## Routing counts by model",
        "",
        "| Model | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in metrics["per_model"].items())
    if not metrics["per_model"]:
        lines.append("| No decisions recorded | 0 |")
    lines.extend(["", "## Rule fire counts", "", "| Rule | Count |", "| --- | ---: |"])
    lines.extend(f"| {name} | {count} |" for name, count in metrics["rule_fire_counts"].items())
    if not metrics["rule_fire_counts"]:
        lines.append("| No rules fired | 0 |")
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> str:
    if args.command == "models":
        return _models_output(args.format == "json")
    if args.command == "policy":
        return _policy_output()
    if args.command == "route":
        decision = route_task(
            args.task,
            latency_budget_ms=args.latency_budget_ms,
            required_capabilities=args.capability,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
        append_audit(decision, args.audit)
        return _json(decision)
    if args.command == "simulate":
        workload = load_workload(args.workload)
        result = simulate_workload(workload, latency_budget_ms=args.latency_budget_ms)
        for decision in result["decisions"]:
            append_audit(decision, args.audit)
        return _json(result)
    if args.command == "metrics":
        return _json(metrics_from_audit(read_audit(args.audit)))
    if args.command == "report":
        rendered = _report_text(metrics_from_audit(read_audit(args.audit)), args.format)
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered + "\n")
            return f"{NOTICE}\nWrote {args.format} report to {args.output}"
        return rendered
    raise ValueError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        print(_run(args))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(NOTICE, file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
