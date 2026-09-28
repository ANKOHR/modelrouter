"""Rule-based scoring, policy application, and deterministic cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


NOTICE = (
    "SYNTHETIC FIXTURE DATA: prices and latencies are illustrative, not live provider quotes. "
    "No real API calls are made. Routing decisions are rule-based, not learned."
)
PRIVACY_THRESHOLD = 50


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    p50_latency_ms: int
    capabilities: tuple[str, ...]
    privacy_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "input_cost_per_1k": float(self.input_cost_per_1k),
            "output_cost_per_1k": float(self.output_cost_per_1k),
            "p50_latency_ms": self.p50_latency_ms,
            "capabilities": list(self.capabilities),
            "privacy_class": self.privacy_class,
        }


@dataclass(frozen=True)
class Score:
    value: int
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "signals": list(self.signals)}


def _data_file(name: str) -> Path:
    return Path(__file__).parent / "data" / name


def load_catalogue(path: str | Path | None = None) -> tuple[Model, ...]:
    source = Path(path) if path else _data_file("models.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    models = []
    for item in raw["models"]:
        model = Model(
            id=item["id"],
            label=item["label"],
            input_cost_per_1k=Decimal(str(item["input_cost_per_1k"])),
            output_cost_per_1k=Decimal(str(item["output_cost_per_1k"])),
            p50_latency_ms=int(item["p50_latency_ms"]),
            capabilities=tuple(sorted(set(item["capabilities"]))),
            privacy_class=item["privacy_class"],
        )
        if model.input_cost_per_1k < 0 or model.output_cost_per_1k < 0:
            raise ValueError(f"Model {model.id} has a negative illustrative price")
        if model.p50_latency_ms < 0:
            raise ValueError(f"Model {model.id} has a negative p50 latency")
        if model.privacy_class not in {"public-api", "private", "local"}:
            raise ValueError(f"Model {model.id} has an unsupported privacy class")
        models.append(model)
    if not models:
        raise ValueError("The synthetic model catalogue is empty")
    return tuple(models)


def score_task(task: str) -> tuple[Score, Score, tuple[str, ...]]:
    """Return difficulty, privacy, and inferred capability tags in stable order."""
    text = task.lower()
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    difficulty_points = 0
    difficulty_signals: list[str] = []
    if word_count >= 80:
        difficulty_points += 30
        difficulty_signals.append("long task: 80 or more words (+30)")
    elif word_count >= 30:
        difficulty_points += 18
        difficulty_signals.append("medium task: 30 to 79 words (+18)")
    elif word_count >= 12:
        difficulty_points += 8
        difficulty_signals.append("task has 12 to 29 words (+8)")
    if re.search(r"\b(code|python|javascript|typescript|function|class|debug|compiler|sql)\b|```|=>", text):
        difficulty_points += 30
        difficulty_signals.append("code marker detected (+30)")
    if re.search(r"\b(reason|reasoning|analy[sz]e|compare|tradeoffs?|derive|prove|evaluate|because)\b", text):
        difficulty_points += 24
        difficulty_signals.append("reasoning marker detected (+24)")
    if re.search(r"\b(step[- ]by[- ]step|first|then|finally|multiple steps|workflow)\b", text):
        difficulty_points += 18
        difficulty_signals.append("multi-step marker detected (+18)")
    difficulty_value = min(100, difficulty_points)
    if difficulty_value >= 55:
        difficulty_signals.append("difficulty score at least 55 requires reasoning capability")
    difficulty = Score(difficulty_value, tuple(difficulty_signals))

    privacy_points = 0
    privacy_signals: list[str] = []
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", task, re.IGNORECASE):
        privacy_points += 60
        privacy_signals.append("email address detected (+60)")
    if re.search(r"\b(?:\+?\d[\d .()/-]{7,}\d)\b", task):
        privacy_points += 55
        privacy_signals.append("phone-like number detected (+55)")
    if re.search(r"\b(?:api[_ -]?key|password|secret|token|credential|private key)\b", text):
        privacy_points += 70
        privacy_signals.append("secret or credential marker detected (+70)")
    if re.search(r"\b(?:ssn|social security number|date of birth|medical record|patient record)\b", text):
        privacy_points += 65
        privacy_signals.append("sensitive personal data marker detected (+65)")
    if re.search(r"\bconfidential\b", text):
        privacy_points += 55
        privacy_signals.append("confidential marker detected (+55)")
    if re.search(r"\b(?:internal|intranet|production|staging|company-only|proprietary|admin portal|company server)\b", text):
        privacy_points += 55
        privacy_signals.append("internal information marker detected (+55)")
    privacy = Score(min(100, privacy_points), tuple(privacy_signals))

    capabilities: set[str] = set()
    if re.search(r"\b(code|python|javascript|typescript|function|class|debug|compiler|sql)\b|```|=>", text):
        capabilities.add("code")
    if re.search(r"\b(reason|reasoning|analy[sz]e|compare|tradeoffs?|derive|prove|evaluate|because|step[- ]by[- ]step)\b", text):
        capabilities.add("reasoning")
    if difficulty.value >= 55:
        capabilities.add("reasoning")
    if re.search(r"\b(image|photo|picture|screenshot|visual|diagram)\b", text):
        capabilities.add("vision")
    if word_count > 450 or re.search(r"\blong[- ]context\b", text):
        capabilities.add("long-context")
    if re.search(r"\b(?:use|call|invoke) (?:a )?(?:tool|browser|web search|file system)\b", text):
        capabilities.add("tools")
    return difficulty, privacy, tuple(sorted(capabilities))


def estimate_tokens(task: str, output_tokens: int | None = None) -> tuple[int, int]:
    """Use a documented deterministic character heuristic, not a tokenizer."""
    input_estimate = max(1, (len(task.encode("utf-8")) + 3) // 4)
    difficulty, _, _ = score_task(task)
    words = len(re.findall(r"\b[\w'-]+\b", task))
    output_estimate = output_tokens if output_tokens is not None else 32 + min(160, (words * 3 + 3) // 4 + difficulty.value // 4)
    if output_estimate < 0:
        raise ValueError("Output token estimate cannot be negative")
    return input_estimate, output_estimate


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> Decimal:
    cost = (
        model.input_cost_per_1k * Decimal(input_tokens)
        + model.output_cost_per_1k * Decimal(output_tokens)
    ) / Decimal(1000)
    return cost.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def _sort_models(models: Iterable[Model], input_tokens: int, output_tokens: int) -> list[Model]:
    return sorted(models, key=lambda model: (estimate_cost(model, input_tokens, output_tokens), model.id))


def route_task(
    task: str,
    *,
    latency_budget_ms: int | None = None,
    required_capabilities: Iterable[str] = (),
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    models: tuple[Model, ...] | None = None,
) -> dict[str, Any]:
    if not task.strip():
        raise ValueError("Task text cannot be empty")
    if latency_budget_ms is not None and latency_budget_ms < 0:
        raise ValueError("Latency budget cannot be negative")
    difficulty, privacy, inferred = score_task(task)
    required = tuple(sorted(set(inferred).union(required_capabilities)))
    known_capabilities = {cap for model in (models or load_catalogue()) for cap in model.capabilities}
    unknown = sorted(set(required) - known_capabilities)
    if unknown:
        raise ValueError("Unknown or unsupported capability: " + ", ".join(unknown))
    in_tokens, out_tokens = estimate_tokens(task, output_tokens)
    if input_tokens is not None:
        if input_tokens < 0:
            raise ValueError("Input token estimate cannot be negative")
        in_tokens = input_tokens
    catalogue = models or load_catalogue()
    all_count = len(catalogue)
    candidates = [m for m in catalogue if set(required).issubset(m.capabilities)]
    capability_count = len(candidates)
    if privacy.value >= PRIVACY_THRESHOLD:
        candidates = [m for m in candidates if m.privacy_class in {"local", "private"}]
    privacy_count = len(candidates)
    if latency_budget_ms is not None:
        candidates = [m for m in candidates if m.p50_latency_ms <= latency_budget_ms]
    latency_count = len(candidates)
    if not candidates:
        raise ValueError("No model satisfies the capability, privacy, and latency policy filters")
    ordered = _sort_models(candidates, in_tokens, out_tokens)
    chosen = ordered[0]
    rules = [
        {
            "name": "capability_match",
            "description": "Keep models that provide every required capability.",
            "before": all_count,
            "after": capability_count,
            "fired": True,
        },
        {
            "name": "privacy_boundary",
            "description": "For privacy scores at or above 50, allow local and private models only.",
            "before": capability_count,
            "after": privacy_count,
            "fired": privacy.value >= PRIVACY_THRESHOLD,
        },
        {
            "name": "latency_budget",
            "description": "When set, keep models whose observed fixture p50 is at or below the cap.",
            "before": privacy_count,
            "after": latency_count,
            "fired": latency_budget_ms is not None,
        },
        {
            "name": "lowest_estimated_cost",
            "description": "Choose the eligible model with the lowest estimated task cost, then model id.",
            "before": latency_count,
            "after": latency_count,
            "fired": True,
        },
    ]
    return {
        "honesty_boundary": NOTICE,
        "task_hash": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "scores": {"difficulty": difficulty.to_dict(), "privacy": privacy.to_dict()},
        "required_capabilities": list(required),
        "token_estimate": {"input": in_tokens, "output": out_tokens, "method": "UTF-8 bytes rounded up per 4 for input; deterministic word and difficulty heuristic for output"},
        "policy": {"privacy_threshold": PRIVACY_THRESHOLD, "latency_budget_ms": latency_budget_ms},
        "candidate_models": [model.id for model in ordered],
        "candidate_estimates": [
            {
                "model_id": model.id,
                "estimated_cost": money(estimate_cost(model, in_tokens, out_tokens)),
                "p50_latency_ms": model.p50_latency_ms,
                "privacy_class": model.privacy_class,
            }
            for model in ordered
        ],
        "rules_fired": rules,
        "chosen_model": chosen.to_dict(),
        "estimated_cost": money(estimate_cost(chosen, in_tokens, out_tokens)),
        "estimated_latency_ms": chosen.p50_latency_ms,
    }


def load_workload(path: str | Path | None = None) -> list[dict[str, str]]:
    source = Path(path) if path else _data_file("workload.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Workload file must contain a JSON list")
    tasks = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            task_id, task = f"task-{index + 1}", item
        elif isinstance(item, dict) and isinstance(item.get("task"), str):
            task_id, task = str(item.get("id", f"task-{index + 1}")), item["task"]
        else:
            raise ValueError(f"Workload item {index + 1} must be a task string or object with a task string")
        if not task.strip():
            raise ValueError(f"Workload task {task_id} is empty")
        tasks.append({"id": task_id, "task": task})
    return tasks


def simulate_workload(
    workload: list[dict[str, str]], *, latency_budget_ms: int | None = None
) -> dict[str, Any]:
    catalogue = load_catalogue()
    flagship = next((model for model in catalogue if model.id == "flagship"), None)
    if flagship is None:
        raise ValueError("Synthetic catalogue must include the flagship baseline model")
    decisions = []
    per_model: dict[str, int] = {}
    route_total = Decimal("0")
    baseline_total = Decimal("0")
    route_latency_total = 0
    baseline_latency_total = 0
    rule_counts: dict[str, int] = {}
    for item in workload:
        decision = route_task(item["task"], latency_budget_ms=latency_budget_ms, models=catalogue)
        decision = {**decision, "workload_id": item["id"]}
        decisions.append(decision)
        selected = next(model for model in catalogue if model.id == decision["chosen_model"]["id"])
        in_tokens = decision["token_estimate"]["input"]
        out_tokens = decision["token_estimate"]["output"]
        route_total += estimate_cost(selected, in_tokens, out_tokens)
        baseline_total += estimate_cost(flagship, in_tokens, out_tokens)
        route_latency_total += selected.p50_latency_ms
        baseline_latency_total += flagship.p50_latency_ms
        per_model[selected.id] = per_model.get(selected.id, 0) + 1
        for rule in decision["rules_fired"]:
            if rule["fired"]:
                rule_counts[rule["name"]] = rule_counts.get(rule["name"], 0) + 1
    task_count = len(decisions)
    savings = baseline_total - route_total
    percent = (savings / baseline_total * 100) if baseline_total else Decimal("0")
    return {
        "honesty_boundary": NOTICE,
        "workload_size": task_count,
        "decisions": decisions,
        "totals": {
            "estimated_cost": money(route_total),
            "always_flagship_cost": money(baseline_total),
            "estimated_cost_saved": money(savings),
            "estimated_savings_percent": round(float(percent), 4),
            "estimated_latency_ms": route_latency_total,
            "always_flagship_latency_ms": baseline_latency_total,
            "estimated_latency_delta_ms": baseline_latency_total - route_latency_total,
        },
        "per_model": {
            model.id: {"count": per_model.get(model.id, 0), "share": round(per_model.get(model.id, 0) / task_count, 4) if task_count else 0.0}
            for model in catalogue
        },
        "rule_fire_counts": {name: rule_counts[name] for name in sorted(rule_counts)},
    }


def append_audit(decision: dict[str, Any], path: str | Path = "mr-audit.jsonl") -> None:
    from datetime import datetime, timezone

    record = {
        **decision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def read_audit(path: str | Path = "mr-audit.jsonl") -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid audit JSON on line {line_number}: {exc.msg}") from exc
    return records


def metrics_from_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    total_cost = Decimal("0")
    baseline_cost = Decimal("0")
    flagship = next((model for model in load_catalogue() if model.id == "flagship"), None)
    for record in records:
        model_id = record.get("chosen_model", {}).get("id", "unknown")
        per_model[model_id] = per_model.get(model_id, 0) + 1
        total_cost += Decimal(str(record.get("estimated_cost", 0)))
        token_estimate = record.get("token_estimate", {})
        if flagship is not None and "input" in token_estimate and "output" in token_estimate:
            baseline_cost += estimate_cost(flagship, int(token_estimate["input"]), int(token_estimate["output"]))
        for rule in record.get("rules_fired", []):
            if rule.get("fired"):
                name = rule.get("name", "unknown")
                rule_counts[name] = rule_counts.get(name, 0) + 1
    savings = baseline_cost - total_cost
    savings_percent = savings / baseline_cost * 100 if baseline_cost else Decimal("0")
    return {
        "honesty_boundary": NOTICE,
        "decision_count": len(records),
        "estimated_cost_total": money(total_cost),
        "always_flagship_cost_total": money(baseline_cost),
        "estimated_cost_saved": money(savings),
        "estimated_savings_percent": round(float(savings_percent), 4),
        "per_model": {key: per_model[key] for key in sorted(per_model)},
        "rule_fire_counts": {key: rule_counts[key] for key in sorted(rule_counts)},
    }
