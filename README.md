# ModelRouter

ModelRouter is a deterministic, cost-aware routing engine with an auditable rule-based policy. It scores each task, filters a synthetic model catalogue by capability, privacy, and optional latency budget, then selects the eligible model with the lowest estimated task cost.

## Honesty boundary

The model catalogue is **SYNTHETIC fixture data**. Prices and latency values are illustrative examples, not live provider quotes or current provider pricing. ModelRouter makes **NO real API calls**. Routing decisions are rule-based, not learned. The cost and latency results are estimates computed from the fixture values and token heuristics. They do not represent a model invocation or a bill.

Every CLI result, the metrics report header, and the HTML report state this boundary. The catalogue file also includes the notice. The tool runs offline and has no model-provider dependency.

## Quickstart

Requires Python 3.10 or newer. The implementation uses the Python standard library. Install pytest only if you want to run the test suite.

```powershell
python -m modelrouter models list
python -m modelrouter route --task "Summarize this short paragraph in two sentences."
python -m modelrouter route --task "Compare these designs and reason through the tradeoffs." --latency-budget-ms 500
python -m modelrouter simulate
python -m modelrouter metrics
python -m modelrouter report --format html --output report.html
python -m modelrouter policy show
python -m pytest
```

The `modelrouter` command is available after installing the project with `python -m pip install -e .`. All commands also work as `python -m modelrouter`. A route appends a JSON Lines record to `mr-audit.jsonl`; simulation appends one record per workload task. Use `--audit PATH` to choose another log file.

## Architecture

```text
Task text and optional constraints
              |
              v
Rule-based difficulty and privacy scoring
              |
              v
Capability filter -> privacy filter -> p50 latency filter
              |
              v
Lowest estimated eligible cost, then model id for ties
              |
              +----> deterministic decision JSON
              +----> append-only audit log
              +----> simulation, metrics, Markdown and HTML reports
```

The package is split into `core.py` for the catalogue, scoring, policy, estimates, simulation, and audit operations, and `cli.py` for the command interface and report rendering. Fixture definitions live in `modelrouter/data/`. Tests live in `tests/`.

## Policy rules

Rules run in this order:

1. **Capability match:** remove models that do not support all required or inferred tags. Inference recognizes code markers, reasoning language, vision terms, long-context requests, and explicit tool use. A difficulty score of 55 or higher also requires reasoning capability. `--capability TAG` adds a required tag.
2. **Privacy boundary:** a privacy score of 50 or more restricts candidates to `local` or `private` classes. Email addresses, phone-like values, credentials, sensitive personal data, confidential language, and internal information have named scoring signals.
3. **Latency budget:** when `--latency-budget-ms` is supplied, remove models whose fixture p50 latency is above the cap. If no model remains, the command returns an error instead of silently violating the cap.
4. **Lowest estimated cost:** select the eligible model with the lowest estimated task cost. Model id breaks a cost tie, so the outcome stays stable.

For example, `route --task "Summarize this confidential internal memo"` scores privacy at or above the threshold and picks only from local or private fixtures. `route --task "Say hello" --latency-budget-ms 200` can only choose a fixture with p50 latency at or below 200 ms. A trivial task without constraints goes to the cheapest eligible fixture, not automatically to the flagship reference.

Use `python -m modelrouter policy show` to see the rule names and their effects. Each route decision includes each rule's before and after candidate counts, the signals that fired, the remaining candidates, the selected fixture, and the estimated cost and p50 latency.

## Estimates and determinism

Input token count defaults to UTF-8 bytes rounded up per four bytes. Output token count uses a fixed word-count and difficulty formula. These are transparent accounting heuristics, not a model tokenizer. Users may provide `--input-tokens` and `--output-tokens` to override route estimates.

The same task, capabilities, and policy values produce the same route, scores, candidate ordering, and cost estimate. The simulator uses the same calculations and compares them with the synthetic `flagship` reference. No random number generator is used, so there is no random seed. Audit timestamps are the time of recording and are intentionally not part of the deterministic decision payload.

## Simulation and reporting

The default fixture workload is `modelrouter/data/workload.json`. You can pass a JSON list of task strings or objects with `id` and `task` fields:

```powershell
python -m modelrouter simulate --workload path/to/workload.json
python -m modelrouter metrics --audit mr-audit.jsonl
python -m modelrouter report --format md --audit mr-audit.jsonl
python -m modelrouter report --format html --audit mr-audit.jsonl --output report.html
```

Simulation reports estimated cost and p50 latency totals, the always-flagship reference totals, cost saved, latency delta, per-model workload share, and rule fire counts. The fixture has a hand-checkable positive cost saving because it compares every routed estimate with the same task's estimate at the flagship fixture rates.

## Development

Run the offline test suite with `python -m pytest`. No GPU is used, and no machine-learning framework is required. The local Git history contains the project implementation and test results are not stored as runtime state. Audit files and launcher residue are ignored by Git.
