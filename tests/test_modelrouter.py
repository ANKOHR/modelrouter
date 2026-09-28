import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from modelrouter.core import (
    estimate_cost,
    load_catalogue,
    load_workload,
    metrics_from_audit,
    route_task,
    score_task,
    simulate_workload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_catalogue_has_all_required_fields_and_valid_values():
    models = load_catalogue()
    assert len(models) >= 3
    assert {m.id for m in models} >= {"flagship", "local-safe"}
    for model in models:
        assert model.input_cost_per_1k >= 0
        assert model.output_cost_per_1k >= 0
        assert model.p50_latency_ms > 0
        assert model.capabilities
        assert model.privacy_class in {"public-api", "private", "local"}


@pytest.mark.parametrize("task", [
    "Send a note to alice@example.com",
    "Please use this password secret to log in",
    "Customer SSN and date of birth are in this record",
])
def test_planted_pii_and_secrets_score_privacy_high(task):
    _, privacy, _ = score_task(task)
    assert privacy.value >= 50
    assert privacy.signals


def test_code_heavy_task_scores_difficulty_and_explains_signals():
    difficulty, _, capabilities = score_task("Debug this Python function and fix the compiler error")
    assert difficulty.value >= 30
    assert "code" in capabilities
    assert any("code marker" in signal for signal in difficulty.signals)


def test_reasoning_and_multi_step_signals_are_explained():
    difficulty, _, capabilities = score_task("Compare these options, reason through them step by step")
    assert difficulty.value >= 40
    assert any("reasoning" in signal for signal in difficulty.signals)
    assert any("multi-step" in signal for signal in difficulty.signals)
    assert "reasoning" in capabilities


def test_high_difficulty_requires_reasoning_capability():
    result = route_task("Debug this Python function, analyze the cause, then explain the fix step by step")
    assert result["scores"]["difficulty"]["value"] >= 55
    assert "reasoning" in result["required_capabilities"]
    assert any("requires reasoning" in signal for signal in result["scores"]["difficulty"]["signals"])


def test_privacy_sensitive_task_never_routes_to_public_model():
    for task in (
        "Summarize this confidential internal record for the team",
        "Summarize the internal system configuration",
    ):
        result = route_task(task)
        assert result["scores"]["privacy"]["value"] >= 50
        assert result["chosen_model"]["privacy_class"] in {"local", "private"}
        assert all(model_id in {"local-safe", "private-pro"} for model_id in result["candidate_models"])


def test_latency_cap_is_respected():
    result = route_task("Say hello", latency_budget_ms=200)
    assert result["chosen_model"]["p50_latency_ms"] <= 200
    assert all(load_catalogue_by_id(model_id).p50_latency_ms <= 200 for model_id in result["candidate_models"])


def load_catalogue_by_id(model_id):
    return next(model for model in load_catalogue() if model.id == model_id)


def test_trivial_task_uses_cheapest_eligible_model_not_flagship():
    result = route_task("Say hello")
    assert result["chosen_model"]["id"] != "flagship"
    assert result["chosen_model"]["id"] == result["candidate_estimates"][0]["model_id"]
    assert result["chosen_model"]["id"] == "swift-small"
    assert result["candidate_estimates"] == sorted(
        result["candidate_estimates"], key=lambda row: (row["estimated_cost"], row["model_id"])
    )
    model = load_catalogue_by_id(result["chosen_model"]["id"])
    assert result["estimated_cost"] == float(estimate_cost(
        model,
        result["token_estimate"]["input"],
        result["token_estimate"]["output"],
    ))


def test_route_decision_is_deterministic():
    first = route_task("Compare a code design and explain the tradeoffs")
    second = route_task("Compare a code design and explain the tradeoffs")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_simulation_totals_are_deterministic():
    workload = load_workload()
    first = simulate_workload(workload)
    second = simulate_workload(workload)
    assert first["totals"] == second["totals"]
    assert first["per_model"] == second["per_model"]


def test_fixture_workload_saves_against_hand_calculated_flagship_baseline():
    workload = load_workload()
    result = simulate_workload(workload)
    models = {model.id: model for model in load_catalogue()}
    route_cost = Decimal("0")
    flagship_cost = Decimal("0")
    for decision in result["decisions"]:
        tokens = decision["token_estimate"]
        route_cost += Decimal(str(estimate_cost(
            models[decision["chosen_model"]["id"]], tokens["input"], tokens["output"]
        )))
        flagship_cost += Decimal(str(estimate_cost(models["flagship"], tokens["input"], tokens["output"])))
    assert result["workload_size"] == len(workload)
    assert result["totals"]["estimated_cost"] == float(route_cost)
    assert result["totals"]["always_flagship_cost"] == float(flagship_cost)
    assert result["totals"]["estimated_cost_saved"] == float(flagship_cost - route_cost)
    assert result["totals"]["estimated_cost_saved"] > 0
    assert sum(item["count"] for item in result["per_model"].values()) == len(workload)


def test_fixture_workload_has_rule_counts_and_latency_accounting():
    result = simulate_workload(load_workload())
    totals = result["totals"]
    assert totals["estimated_latency_ms"] == sum(
        item["estimated_latency_ms"] for item in result["decisions"]
    )
    assert totals["always_flagship_latency_ms"] == 700 * len(result["decisions"])
    assert result["rule_fire_counts"]["capability_match"] == len(result["decisions"])


@pytest.mark.parametrize("args", [
    ["models", "list"],
    ["route", "--task", "Say hello"],
    ["simulate"],
    ["metrics"],
    ["report", "--format", "md"],
    ["report", "--format", "html"],
    ["policy", "show"],
])
def test_cli_subcommand_smoke(args, tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "modelrouter", *args, "--audit", str(tmp_path / "audit.jsonl")]
        if args[0] in {"route", "simulate", "metrics", "report"}
        else [sys.executable, "-m", "modelrouter", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SYNTHETIC" in completed.stdout


def test_cli_route_logs_decision_and_metrics_reads_it(tmp_path):
    audit = tmp_path / "audit.jsonl"
    routed = subprocess.run(
        [sys.executable, "-m", "modelrouter", "route", "--task", "Say hello", "--audit", str(audit)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert routed.returncode == 0
    record = json.loads(audit.read_text(encoding="utf-8"))
    assert record["chosen_model"]["id"] == json.loads(routed.stdout)["chosen_model"]["id"]
    assert record["timestamp_utc"].endswith("Z")
    metrics = metrics_from_audit([record])
    assert metrics["decision_count"] == 1
    assert metrics["always_flagship_cost_total"] > metrics["estimated_cost_total"]
    assert metrics["estimated_cost_saved"] > 0
    assert metrics["per_model"][record["chosen_model"]["id"]] == 1


def test_cli_simulate_records_fixture_and_report_html_has_honesty_banner(tmp_path):
    audit = tmp_path / "audit.jsonl"
    simulated = subprocess.run(
        [sys.executable, "-m", "modelrouter", "simulate", "--audit", str(audit)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stderr
    result = json.loads(simulated.stdout)
    assert result["totals"]["estimated_cost_saved"] > 0
    report = subprocess.run(
        [sys.executable, "-m", "modelrouter", "report", "--format", "html", "--audit", str(audit)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert report.returncode == 0
    assert "SYNTHETIC FIXTURE DATA" in report.stdout
    assert "No real API calls are made" in report.stdout
    assert "Routing counts by model" in report.stdout


def test_models_json_states_catalogue_notice():
    completed = subprocess.run(
        [sys.executable, "-m", "modelrouter", "models", "list", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "SYNTHETIC" in json.loads(completed.stdout)["honesty_boundary"]
