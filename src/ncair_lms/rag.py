from __future__ import annotations

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
    """Lazy FAISS knowledge base built from the repository's official NCAIR source files."""

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
            LOGGER.info("building_faiss_index chunks=%s", len(chunks))

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
