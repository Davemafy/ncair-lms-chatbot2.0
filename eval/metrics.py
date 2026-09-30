from __future__ import annotations

from collections import defaultdict
from statistics import mean


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(results: list[dict]) -> dict:
    successful = [result for result in results if not result["failure"]]
    tool_accuracy = _rate([result["actual_tool"] == result["expected_tool"] for result in results])
    argument_accuracy = _rate(
        [result["argument_correct"] for result in results if result["actual_tool"]]
    )
    language_accuracy = _rate(
        [result["actual_language"] == result["language"] for result in successful]
    )
    trap_accuracy = _rate(
        [
            result["actual_tool"] == result["expected_tool"] and result["argument_correct"]
            for result in results
            if result["category"] == "trap"
        ]
    )
    undocumented_safety = _rate(
        [
            result["supported_actual"] is False
            for result in results
            if result["category"] == "undocumented"
        ]
    )
    evidence_hit_rate = _rate(
        [result["evidence_terms_found"] for result in results if result["expected_evidence_terms"]]
    )

    by_language: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_language[result["language"]].append(result["actual_tool"] == result["expected_tool"])

    semantic_groups: dict[str, list[str | None]] = defaultdict(list)
    for result in successful:
        if result.get("semantic_key"):
            semantic_groups[result["semantic_key"]].append(result["actual_tool"])
    comparable_groups = [tools for tools in semantic_groups.values() if len(tools) > 1]
    cross_language_consistency = _rate([len(set(tools)) == 1 for tools in comparable_groups])

    return {
        "records": len(results),
        "tool_accuracy": tool_accuracy,
        "argument_accuracy": argument_accuracy,
        "language_accuracy": language_accuracy,
        "trap_accuracy": trap_accuracy,
        "undocumented_safety": undocumented_safety,
        "evidence_hit_rate": evidence_hit_rate,
        "cross_language_tool_consistency": cross_language_consistency,
        "failure_rate": _rate([result["failure"] for result in results]),
        "mean_latency_ms": round(mean(result["latency_ms"] for result in results), 2),
        "tool_accuracy_by_language": {
            language: _rate(values) for language, values in sorted(by_language.items())
        },
    }
