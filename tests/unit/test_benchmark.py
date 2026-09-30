from eval.validate_benchmark import (
    EXPECTED_CATEGORIES,
    EXPECTED_LANGUAGES,
    load_records,
    validate_records,
)


def test_benchmark_has_required_shape():
    records = load_records()
    validate_records(records)
    assert len(records) == 60
    assert sum(EXPECTED_LANGUAGES.values()) == 60
    assert sum(EXPECTED_CATEGORIES.values()) == 60
