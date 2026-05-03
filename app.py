"""Orchestration: router → retriever → generator."""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.generator import generate
from core.retriever import RetrievedChunk, retrieve
from core.router import RouteDecision, RouteLabel, classify_query
from ingest.embedder import IngestBackend, IngestEmbedderConfig
from ingest.store import ChromaStoreConfig, collection_stats


CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "wikipedia_rag")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")

_EMB_RAW = os.environ.get("RAG_EMBEDDER_BACKEND", "ollama_nomic")
EMBEDDER_BACKEND: IngestBackend = (
    _EMB_RAW if _EMB_RAW in ("ollama_nomic", "sentence_transformers") else "ollama_nomic"
)
EMBEDDER_MODEL = os.environ.get("RAG_EMBEDDER_MODEL", "nomic-embed-text")
EMBEDDER_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDER_BATCH_SIZE", "8"))


def default_chroma_config() -> ChromaStoreConfig:
    return ChromaStoreConfig(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=CHROMA_COLLECTION_NAME,
    )


def default_embedder_config() -> IngestEmbedderConfig:
    return IngestEmbedderConfig(
        backend=EMBEDDER_BACKEND,
        model_name=EMBEDDER_MODEL,
        batch_size=EMBEDDER_BATCH_SIZE,
        ollama_host=OLLAMA_HOST,
    )


@dataclass(frozen=True)
class GenerationResult:
    answer_text: str
    retrieved_chunks: list[RetrievedChunk]
    route_label: RouteLabel


def _assert_store_ready(chroma_config: ChromaStoreConfig) -> None:
    try:
        stats = collection_stats(chroma_config)
    except Exception as e:
        raise RuntimeError(
            "Chroma vector store is missing or unreadable. "
            "Run `python -m ingest.run_ingest` (or your ingestion script) first "
            f"using persist_directory={chroma_config.persist_directory!r} and "
            f"collection_name={chroma_config.collection_name!r}.",
        ) from e
    if stats.get("chunk_count", 0) <= 0:
        raise RuntimeError(
            "Chroma vector store is empty. Run ingestion before calling ask().",
        )


def ask(
    question: str,
    *,
    k: int = 5,
    return_sources: bool = True,
    chroma_config: ChromaStoreConfig | None = None,
    embedder_config: IngestEmbedderConfig | None = None,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    temperature: float = 0.1,
) -> GenerationResult:
    """Single-turn RAG: classify → retrieve (with optional widen) → generate."""
    cc = chroma_config or default_chroma_config()
    ec0 = embedder_config or default_embedder_config()
    host = ollama_host or OLLAMA_HOST
    model = ollama_model or OLLAMA_LLM_MODEL

    if ec0.ollama_host != host:
        ec = IngestEmbedderConfig(
            backend=ec0.backend,
            model_name=ec0.model_name,
            batch_size=ec0.batch_size,
            ollama_host=host,
        )
    else:
        ec = ec0

    _assert_store_ready(cc)

    route = classify_query(question)
    chunks = retrieve(
        question,
        route,
        chroma_config=cc,
        embedder_config=ec,
        k=k,
    )

    if len(chunks) < k and route.label != "both":
        chunks = retrieve(
            question,
            RouteDecision(label="both"),
            chroma_config=cc,
            embedder_config=ec,
            k=k,
        )

    if not chunks:
        return GenerationResult(
            answer_text="I don't know",
            retrieved_chunks=[],
            route_label=route.label,
        )

    retrieval_for_answer = list(chunks)
    max_tokens = min(2048, 256 + k * 80)

    try:
        answer = generate(
            question,
            retrieval_for_answer,
            ollama_model=model,
            ollama_host=host,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e

    out_chunks = retrieval_for_answer if return_sources else []
    return GenerationResult(
        answer_text=answer.strip(),
        retrieved_chunks=out_chunks,
        route_label=route.label,
    )


__all__ = [
    "GenerationResult",
    "ask",
    "default_chroma_config",
    "default_embedder_config",
]
