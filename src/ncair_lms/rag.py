from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import RetrievalError
from .models import EvidencePassage, RetrievedEvidence

LOGGER = logging.getLogger(__name__)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class _Chunk:
    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class _AtomicFact:
    id: str
    source: str
    title: str
    text: str

    @property
    def search_text(self) -> str:
        return f"{self.title}. {self.text}"


def _chunk_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        additional = len(word) + (1 if current else 0)
        if current and current_length + additional > size:
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            overlap_words: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                overlap_words.append(previous)
                overlap_length += len(previous) + 1
                if overlap_length >= overlap:
                    break
            current = list(reversed(overlap_words))
            current_length = len(" ".join(current))

        current.append(word)
        current_length += additional

    final_chunk = " ".join(current).strip()
    if final_chunk:
        chunks.append(final_chunk)
    return chunks


class KnowledgeBase:
    """Preserved V1 FAISS knowledge base.

    V1 remains unchanged so comparisons against the original baseline stay meaningful.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        min_score: float = 0.30,
        embedding_device: str = "cpu",
    ):
        self._data_dir = data_dir
        self._min_score = min_score
        self._embedding_device = embedding_device
        self._lock = threading.Lock()
        self._index = None
        self._embedding_model = None
        self._chunks: list[_Chunk] = []

    def _load_source_chunks(self) -> list[_Chunk]:
        chunks: list[_Chunk] = []

        for text_path in sorted(self._data_dir.glob("*.txt")):
            text = text_path.read_text(encoding="utf-8")
            chunks.extend(
                _Chunk(source=text_path.name, page=None, text=chunk) for chunk in _chunk_text(text)
            )

        pdf_paths = sorted(self._data_dir.glob("*.pdf"))
        if pdf_paths:
            try:
                import pytesseract
                from pdf2image import convert_from_path
            except ImportError as exc:
                raise RetrievalError(
                    "PDF indexing requires pytesseract and pdf2image. "
                    "Install the runtime dependencies first."
                ) from exc

            for pdf_path in pdf_paths:
                try:
                    images = convert_from_path(str(pdf_path))
                except Exception as exc:
                    raise RetrievalError(
                        "Could not render the official NCAIR PDF. "
                        "Confirm Poppler is installed and the file is readable."
                    ) from exc

                for page_number, image in enumerate(images, start=1):
                    text = pytesseract.image_to_string(image)
                    chunks.extend(
                        _Chunk(source=pdf_path.name, page=page_number, text=chunk)
                        for chunk in _chunk_text(text)
                    )

        if not chunks:
            raise RetrievalError(f"No knowledge-base content found in {self._data_dir}.")

        return chunks

    def _ensure_ready(self) -> None:
        if self._index is not None:
            return

        with self._lock:
            if self._index is not None:
                return

            try:
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RetrievalError(
                    "FAISS retrieval requires faiss-cpu and sentence-transformers. "
                    "Install the runtime dependencies first."
                ) from exc

            chunks = self._load_source_chunks()
            LOGGER.info("building_v1_faiss_index chunks=%s", len(chunks))

            model = SentenceTransformer(EMBEDDING_MODEL, device=self._embedding_device)
            vectors = model.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype="float32")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)

            self._chunks = chunks
            self._embedding_model = model
            self._index = index

    def search(self, query: str, *, top_k: int) -> RetrievedEvidence:
        query = query.strip()
        if not query:
            return RetrievedEvidence()

        self._ensure_ready()
        assert self._index is not None
        assert self._embedding_model is not None

        try:
            vector = self._embedding_model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            scores, indexes = self._index.search(vector, min(top_k, len(self._chunks)))
        except Exception as exc:
            raise RetrievalError("FAISS retrieval failed.") from exc

        passages = tuple(
            EvidencePassage(
                source=self._chunks[index].source,
                page=self._chunks[index].page,
                text=self._chunks[index].text,
            )
            for score, index in zip(scores[0], indexes[0], strict=True)
            if score >= self._min_score and 0 <= index < len(self._chunks)
        )
        return RetrievedEvidence(passages)


def _support_from_scores(scores: list[float], *, min_score: float, min_margin: float) -> bool:
    if not scores:
        return False
    if scores[0] < min_score:
        return False
    return not (len(scores) > 1 and scores[0] - scores[1] < min_margin)


class AtomicKnowledgeBase:
    """V2 multilingual retrieval over atomic official facts.

    FAISS is retained as the candidate-retrieval layer. Because the corpus is tiny, the default
    candidate budget is intentionally larger than the fact count, giving the cross-encoder a
    chance to score every official fact instead of relying on dense top-3 recall.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        embedding_model: str,
        embedding_device: str,
        reranker_model: str,
        reranker_device: str,
        candidate_k: int,
        min_score: float,
        min_margin: float,
    ):
        self._data_dir = data_dir
        self._embedding_model_name = embedding_model
        self._embedding_device = embedding_device
        self._reranker_model_name = reranker_model
        self._reranker_device = reranker_device
        self._candidate_k = candidate_k
        self._min_score = min_score
        self._min_margin = min_margin

        self._lock = threading.Lock()
        self._index = None
        self._embedding_model = None
        self._reranker_tokenizer = None
        self._reranker_model = None
        self._facts: list[_AtomicFact] = []

    def _load_facts(self) -> list[_AtomicFact]:
        path = self._data_dir / "knowledge_facts.jsonl"
        if not path.exists():
            raise RetrievalError(f"Atomic knowledge file not found: {path}")

        facts: list[_AtomicFact] = []
        seen_ids: set[str] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                fact = _AtomicFact(
                    id=str(raw["id"]).strip(),
                    source=str(raw["source"]).strip(),
                    title=str(raw["title"]).strip(),
                    text=str(raw["text"]).strip(),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RetrievalError(
                    f"Invalid atomic knowledge record on line {line_number}."
                ) from exc

            if not all((fact.id, fact.source, fact.title, fact.text)):
                raise RetrievalError(f"Atomic knowledge record {line_number} has an empty field.")
            if fact.id in seen_ids:
                raise RetrievalError(f"Duplicate atomic knowledge id: {fact.id}")
            seen_ids.add(fact.id)
            facts.append(fact)

        if not facts:
            raise RetrievalError("Atomic knowledge base is empty.")
        return facts

    def _ensure_ready(self) -> None:
        if self._index is not None:
            return

        with self._lock:
            if self._index is not None:
                return

            try:
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RetrievalError(
                    "V2 retrieval requires faiss-cpu, sentence-transformers, and transformers."
                ) from exc

            facts = self._load_facts()
            LOGGER.info(
                "building_v2_atomic_index facts=%s embedding=%s reranker=%s",
                len(facts),
                self._embedding_model_name,
                self._reranker_model_name,
            )

            try:
                embedding_model = SentenceTransformer(
                    self._embedding_model_name,
                    device=self._embedding_device,
                )
                vectors = embedding_model.encode(
                    [f"passage: {fact.search_text}" for fact in facts],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                vectors = np.asarray(vectors, dtype="float32")
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)

                reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
                reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    self._reranker_model_name
                )
                if self._reranker_device != "auto":
                    reranker_model = reranker_model.to(self._reranker_device)
                reranker_model.eval()
            except Exception as exc:
                raise RetrievalError("Could not load the V2 embedding/reranker models.") from exc

            self._facts = facts
            self._embedding_model = embedding_model
            self._index = index
            self._reranker_tokenizer = reranker_tokenizer
            self._reranker_model = reranker_model

    def _rerank(self, query: str, candidate_indexes: list[int]) -> list[tuple[float, int]]:
        assert self._reranker_tokenizer is not None
        assert self._reranker_model is not None

        try:
            import torch

            candidate_texts = [self._facts[index].search_text for index in candidate_indexes]
            inputs = self._reranker_tokenizer(
                [query] * len(candidate_indexes),
                candidate_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            model_device = next(self._reranker_model.parameters()).device
            inputs = {name: value.to(model_device) for name, value in inputs.items()}

            with torch.no_grad():
                raw_scores = self._reranker_model(**inputs, return_dict=True).logits.view(-1)
                probabilities = torch.sigmoid(raw_scores.float()).cpu().tolist()

            return sorted(
                zip(probabilities, candidate_indexes, strict=True),
                key=lambda item: item[0],
                reverse=True,
            )
        except Exception as exc:
            raise RetrievalError("V2 cross-encoder reranking failed.") from exc

    def search(self, query: str, *, top_k: int) -> RetrievedEvidence:
        query = query.strip()
        if not query:
            return RetrievedEvidence(support_verified=False)

        self._ensure_ready()
        assert self._index is not None
        assert self._embedding_model is not None

        try:
            vector = self._embedding_model.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            candidate_count = min(self._candidate_k, len(self._facts))
            _, indexes = self._index.search(vector, candidate_count)
            candidate_indexes = [
                int(index) for index in indexes[0] if 0 <= index < len(self._facts)
            ]
        except Exception as exc:
            raise RetrievalError("V2 multilingual FAISS retrieval failed.") from exc

        ranked = self._rerank(query, candidate_indexes)
        scores = [score for score, _ in ranked]
        supported = _support_from_scores(
            scores,
            min_score=self._min_score,
            min_margin=self._min_margin,
        )

        LOGGER.info(
            "v2_rerank top_score=%s second_score=%s supported=%s top_fact=%s",
            round(scores[0], 4) if scores else None,
            round(scores[1], 4) if len(scores) > 1 else None,
            supported,
            self._facts[ranked[0][1]].id if ranked else None,
        )

        keep = min(max(top_k, 1), len(ranked))
        passages = tuple(
            EvidencePassage(
                source=self._facts[index].source,
                text=self._facts[index].text,
            )
            for _, index in ranked[:keep]
        )
        return RetrievedEvidence(passages, support_verified=supported)
