"""Chroma persistent store for ingested chunks."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb


@dataclass(frozen=True)
class ChromaStoreConfig:
    persist_directory: str
    collection_name: str


_DIM_KEY = "embedding_dimension"


def _client(config: ChromaStoreConfig):
    return chromadb.PersistentClient(path=config.persist_directory)


def ensure_collection(config: ChromaStoreConfig, embedding_dimension: int) -> None:
    """Create collection if missing; verify embedding dimension matches if present."""
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")

    client = _client(config)
    meta: dict[str, str | int] = {
        "hnsw:space": "cosine",
        _DIM_KEY: embedding_dimension,
    }
    try:
        existing = client.get_collection(name=config.collection_name)
    except Exception:
        existing = None

    if existing is not None:
        cm = existing.metadata or {}
        prior = cm.get(_DIM_KEY)
        if prior is not None and int(prior) != embedding_dimension:
            raise RuntimeError(
                f"Collection {config.collection_name!r} expects embedding dim {prior}, "
                f"got {embedding_dimension}",
            )
        return

    client.get_or_create_collection(name=config.collection_name, metadata=meta)


def _collection(config: ChromaStoreConfig):
    client = _client(config)
    return client.get_collection(name=config.collection_name)


def upsert_chunks(
    config: ChromaStoreConfig,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    n = len(ids)
    if not (n == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError("ids, embeddings, documents, metadatas length mismatch")
    col = _collection(config)
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def collection_stats(config: ChromaStoreConfig) -> dict[str, int]:
    try:
        col = _collection(config)
    except Exception as e:
        # Chroma raises NotFoundError when the collection has never been created.
        # Avoid ``from chromadb.errors import …`` here: tests stub ``sys.modules["chromadb"]``.
        msg = str(e).lower()
        if type(e).__name__ == "NotFoundError" or "does not exist" in msg:
            return {"chunk_count": 0}
        raise
    return {"chunk_count": col.count()}
