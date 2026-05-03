"""Metadata-filtered similarity retrieval over Chroma."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from core.router import RouteDecision
from ingest.embedder import IngestEmbedderConfig, embed_texts, normalize_text_for_embedding
from ingest.store import ChromaStoreConfig


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


def _store_client(config: ChromaStoreConfig):
    return chromadb.PersistentClient(path=config.persist_directory)


def _tie_key(row: RetrievedChunk) -> tuple[float, int, str]:
    md = row.metadata or {}
    idx = md.get("chunk_index", 0)
    if isinstance(idx, str):
        try:
            idx = int(idx)
        except ValueError:
            idx = 0
    cid = str(md.get("chunk_id", ""))
    return (row.distance, idx, cid)


def retrieve(
    query: str,
    route: RouteDecision,
    chroma_config: ChromaStoreConfig,
    embedder_config: IngestEmbedderConfig,
    k: int,
) -> list[RetrievedChunk]:
    if k <= 0:
        return []

    q = normalize_text_for_embedding(query)
    if not q:
        return []

    qvec = embed_texts([query], embedder_config)[0]

    client = _store_client(chroma_config)
    col = client.get_collection(name=chroma_config.collection_name)

    where = None
    if route.label == "person":
        where = {"entity_type": "person"}
    elif route.label == "place":
        where = {"entity_type": "place"}

    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids_meta = res.get("metadatas") or []
    ids_docs = res.get("documents") or []
    ids_dist = res.get("distances") or []
    if not (ids_meta and ids_docs and ids_dist):
        return []

    metas = ids_meta[0] or []
    docs = ids_docs[0] or []
    dists = ids_dist[0] or []
    if not docs:
        return []

    out: list[RetrievedChunk] = []
    for d, m, dist in zip(docs, metas, dists):
        if d is None:
            continue
        md = dict(m) if isinstance(m, dict) else {}
        fv = float(dist) if dist is not None else float("inf")
        out.append(RetrievedChunk(text=str(d), metadata=md, distance=fv))

    out.sort(key=_tie_key)
    return out
