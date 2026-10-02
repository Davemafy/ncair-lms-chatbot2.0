from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .validate_benchmark import (
    ALLOWED_TOOLS,
    BENCHMARK_PATH,
    PORTAL_ACTIONS,
    load_records as load_benchmark_records,
)

HOLDOUT_PATH = Path(__file__).with_name("holdout.jsonl")
EXPECTED_LANGUAGES = {"english": 8, "hausa": 8, "yoruba": 8, "igbo": 8}
EXPECTED_CATEGORIES = {
    "navigation": 8,
    "step_guidance": 4,
    "documented_policy": 12,
    "trap": 4,
    "undocumented": 4,
}


def load_holdout_records(path: Path = HOLDOUT_PATH) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on holdout line {line_number}.") from exc
    return records


def validate_holdout_records(records: list[dict]) -> None:
    errors: list[str] = []

    if len(records) != 32:
        errors.append(f"expected 32 holdout records, found {len(records)}")

    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("holdout record ids must be unique")

    questions = [record.get("question") for record in records]
    if len(questions) != len(set(questions)):
        errors.append("holdout questions must be unique")

    frozen_questions = {
        record["question"] for record in load_benchmark_records(BENCHMARK_PATH)
    }
    overlap = sorted(set(questions) & frozen_questions)
    if overlap:
        errors.append(f"holdout questions overlap frozen benchmark: {overlap!r}")

    language_counts = Counter(record.get("language") for record in records)
    if dict(language_counts) != EXPECTED_LANGUAGES:
        errors.append(
            f"holdout language distribution must be {EXPECTED_LANGUAGES}, "
            f"found {dict(language_counts)}"
        )

    category_counts = Counter(record.get("category") for record in records)
    if dict(category_counts) != EXPECTED_CATEGORIES:
        errors.append(
            f"holdout category distribution must be {EXPECTED_CATEGORIES}, "
            f"found {dict(category_counts)}"
        )

    semantic_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        record_id = record.get("id", "<missing id>")
        tool = record.get("expected_tool")
        args = record.get("expected_args")
        supported = record.get("expected_supported")
        language = record.get("language")
        semantic_key = record.get("semantic_key")

        if language not in EXPECTED_LANGUAGES:
            errors.append(f"{record_id}: invalid language {language!r}")
        if tool not in ALLOWED_TOOLS:
            errors.append(f"{record_id}: invalid expected_tool {tool!r}")
        if not isinstance(args, dict):
            errors.append(f"{record_id}: expected_args must be an object")
            continue

        if tool == "get_portal_link" and args.get("action") not in PORTAL_ACTIONS:
            errors.append(f"{record_id}: invalid portal action")
        if tool == "get_step_guidance" and args.get("step") not in {1, 2, 3, 4}:
            errors.append(f"{record_id}: step must be 1-4")
        if tool == "search_ncair_knowledge_base" and set(args) != {"query_required"}:
            errors.append(f"{record_id}: knowledge expected_args must declare query_required only")

        if not isinstance(supported, bool):
            errors.append(f"{record_id}: expected_supported must be boolean")
        if record.get("category") == "undocumented" and supported is not False:
            errors.append(f"{record_id}: undocumented questions must expect unsupported")
        if record.get("category") != "undocumented" and supported is not True:
            errors.append(f"{record_id}: documented records must expect supported")

        terms = record.get("expected_evidence_terms")
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            errors.append(f"{record_id}: expected_evidence_terms must be a string list")

        if not isinstance(semantic_key, str) or not semantic_key:
            errors.append(f"{record_id}: semantic_key must be a non-empty string")
        else:
            semantic_groups[semantic_key].append(record)

    if len(semantic_groups) != 8:
        errors.append(f"expected 8 semantic groups, found {len(semantic_groups)}")

    expected_language_set = set(EXPECTED_LANGUAGES)
    for semantic_key, group in semantic_groups.items():
        if len(group) != 4:
            errors.append(f"{semantic_key}: expected 4 cross-language variants")
            continue

        languages = {record["language"] for record in group}
        if languages != expected_language_set:
            errors.append(
                f"{semantic_key}: expected one variant per language, found {sorted(languages)}"
            )

        tools = {record["expected_tool"] for record in group}
        categories = {record["category"] for record in group}
        supported_values = {record["expected_supported"] for record in group}
        if len(tools) != 1:
            errors.append(f"{semantic_key}: expected tool must match across languages")
        if len(categories) != 1:
            errors.append(f"{semantic_key}: category must match across languages")
        if len(supported_values) != 1:
            errors.append(f"{semantic_key}: expected support must match across languages")

    if errors:
        raise ValueError("Holdout validation failed:\n- " + "\n- ".join(errors))


def main() -> None:
    records = load_holdout_records()
    validate_holdout_records(records)
    print(
        "holdout valid: 32 records, 8 per language, 8 cross-language semantic groups, "
        "no exact overlap with frozen benchmark"
    )


if __name__ == "__main__":
    main()
