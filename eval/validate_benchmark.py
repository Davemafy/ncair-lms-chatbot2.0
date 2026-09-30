from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BENCHMARK_PATH = Path(__file__).with_name("benchmark.jsonl")
EXPECTED_LANGUAGES = {"english": 15, "hausa": 15, "yoruba": 15, "igbo": 15}
EXPECTED_CATEGORIES = {
    "navigation": 10,
    "step_guidance": 10,
    "documented_policy": 20,
    "trap": 12,
    "undocumented": 8,
}
ALLOWED_TOOLS = {
    "get_portal_link",
    "get_step_guidance",
    "search_ncair_knowledge_base",
}
PORTAL_ACTIONS = {
    "main",
    "login",
    "signin",
    "ncair_home",
    "register",
    "profile",
    "courses",
    "track_selection",
    "support",
}


def load_records(path: Path = BENCHMARK_PATH) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}.") from exc
    return records


def validate_records(records: list[dict]) -> None:
    errors: list[str] = []

    if len(records) != 60:
        errors.append(f"expected 60 records, found {len(records)}")

    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("record ids must be unique")

    questions = [record.get("question") for record in records]
    if len(questions) != len(set(questions)):
        errors.append("questions must be unique")

    language_counts = Counter(record.get("language") for record in records)
    if dict(language_counts) != EXPECTED_LANGUAGES:
        errors.append(
            f"language distribution must be {EXPECTED_LANGUAGES}, found {dict(language_counts)}"
        )

    category_counts = Counter(record.get("category") for record in records)
    if dict(category_counts) != EXPECTED_CATEGORIES:
        errors.append(
            f"category distribution must be {EXPECTED_CATEGORIES}, found {dict(category_counts)}"
        )

    for record in records:
        record_id = record.get("id", "<missing id>")
        tool = record.get("expected_tool")
        args = record.get("expected_args")
        supported = record.get("expected_supported")

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
            errors.append(f"{record_id}: documented/trap/navigation/step records must be supported")

        terms = record.get("expected_evidence_terms")
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            errors.append(f"{record_id}: expected_evidence_terms must be a string list")

    if errors:
        raise ValueError("Benchmark validation failed:\n- " + "\n- ".join(errors))


def main() -> None:
    records = load_records()
    validate_records(records)
    print("benchmark valid: 60 records, 15 per language, required category totals satisfied")


if __name__ == "__main__":
    main()
