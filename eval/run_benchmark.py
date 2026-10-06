from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from ncair_lms.models import ToolName
from ncair_lms.service import AssistantService

from .metrics import summarize
from .validate_benchmark import BENCHMARK_PATH, load_records, validate_records

RESULTS_DIR = Path(__file__).with_name("results")


def _actual_args(decision) -> dict:
    if decision.tool is ToolName.PORTAL_LINK:
        return {"action": decision.action.value if decision.action else None}
    if decision.tool is ToolName.STEP_GUIDANCE:
        return {"step": decision.step}
    return {"query_required": bool(decision.retrieval_query and decision.retrieval_query.strip())}


def _argument_correct(record: dict, actual: dict) -> bool:
    expected = record["expected_args"]
    if record["expected_tool"] == ToolName.KNOWLEDGE.value:
        return actual == {"query_required": True}
    return actual == expected


def run(version: str, mode: str, *, dataset: Path = BENCHMARK_PATH) -> dict:
    records = load_records(dataset)
    if dataset.resolve() == BENCHMARK_PATH.resolve():
        validate_records(records)
    service = AssistantService(version)
    results = []

    for record in records:
        started = time.perf_counter()
        result = {
            **record,
            "actual_tool": None,
            "actual_language": None,
            "actual_args": None,
            "argument_correct": False,
            "supported_actual": None,
            "support_score": None,
            "support_margin": None,
            "evidence_terms_found": False,
            "answer": None,
            "failure": False,
            "error": None,
        }

        try:
            decision = service.route(record["question"])
            tool_result = service.execute(decision, question=record["question"])
            evidence = tool_result.evidence.as_context().lower()

            result["actual_tool"] = decision.tool.value
            result["actual_language"] = decision.language.value
            result["actual_args"] = _actual_args(decision)
            result["argument_correct"] = _argument_correct(record, result["actual_args"])
            result["supported_actual"] = tool_result.evidence.supported
            result["support_score"] = tool_result.evidence.support_score
            result["support_margin"] = tool_result.evidence.support_margin
            result["evidence_terms_found"] = all(
                term.lower() in evidence for term in record["expected_evidence_terms"]
            )

            if mode == "full":
                response = service.respond(
                    question=record["question"],
                    decision=decision,
                    tool_result=tool_result,
                )
                result["answer"] = response.answer
        except Exception as exc:
            result["failure"] = True
            result["error"] = f"{type(exc).__name__}: {exc}"

        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        results.append(result)

    payload = {
        "version": version,
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": summarize(results),
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    dataset_label = "" if dataset.resolve() == BENCHMARK_PATH.resolve() else f"-{dataset.stem}"
    path = RESULTS_DIR / (
        f"{version}-{mode}{dataset_label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "metrics": payload["metrics"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    parser.add_argument("--mode", choices=["routing", "full"], default="routing")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BENCHMARK_PATH,
        help="JSONL evaluation dataset; defaults to the frozen 60-case regression set.",
    )
    args = parser.parse_args()

    output = run(args.version, args.mode, dataset=args.dataset)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
