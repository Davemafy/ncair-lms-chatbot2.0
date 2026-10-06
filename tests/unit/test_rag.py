import json

import pytest

from ncair_lms.errors import RetrievalError
from ncair_lms.rag import AtomicKnowledgeBase, _support_from_scores


def _kb(tmp_path):
    return AtomicKnowledgeBase(
        tmp_path,
        embedding_model="intfloat/multilingual-e5-base",
        embedding_device="cpu",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="cpu",
        candidate_k=64,
        min_score=0.5,
        min_margin=0.0,
    )


def test_support_threshold_accepts_clear_relevance():
    assert _support_from_scores([0.91, 0.42], min_score=0.5, min_margin=0.0) is True


def test_support_threshold_rejects_low_relevance():
    assert _support_from_scores([0.49, 0.2], min_score=0.5, min_margin=0.0) is False


def test_support_margin_can_require_separation():
    assert _support_from_scores([0.8, 0.77], min_score=0.5, min_margin=0.05) is False
    assert _support_from_scores([0.8, 0.6], min_score=0.5, min_margin=0.05) is True


def test_atomic_fact_loader_preserves_official_fact_text(tmp_path):
    path = tmp_path / "knowledge_facts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "password",
                "source": "guide.txt",
                "title": "Password rule",
                "text": "Passwords require eight characters.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    facts = _kb(tmp_path)._load_facts()

    assert len(facts) == 1
    assert facts[0].id == "password"
    assert facts[0].source == "guide.txt"
    assert facts[0].search_text == "Password rule. Passwords require eight characters."


def test_atomic_fact_loader_rejects_duplicate_ids(tmp_path):
    record = {
        "id": "duplicate",
        "source": "guide.txt",
        "title": "Rule",
        "text": "Some rule.",
    }
    path = tmp_path / "knowledge_facts.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalError, match="Duplicate atomic knowledge id"):
        _kb(tmp_path)._load_facts()
