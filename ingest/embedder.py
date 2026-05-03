"""Local embeddings: Ollama ``nomic-embed-text`` or ``sentence-transformers``."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

IngestBackend = Literal["ollama_nomic", "sentence_transformers"]


@dataclass(frozen=True)
class IngestEmbedderConfig:
    backend: IngestBackend
    model_name: str
    batch_size: int = 32
    ollama_host: str = "http://127.0.0.1:11434"


OLLAMA_TIMEOUT_S = 120.0


def normalize_text_for_embedding(text: str) -> str:
    """Shared query/document normalization (must match at retrieval time)."""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def embedder_signature(config: IngestEmbedderConfig) -> str:
    if config.backend == "ollama_nomic":
        return f"ollama:{config.model_name}"
    return f"st:{config.model_name}"


def _ollama_embed_batch(
    texts: list[str],
    config: IngestEmbedderConfig,
) -> list[list[float]]:
    url = config.ollama_host.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": config.model_name, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama embed HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama embed request failed: {e.reason}") from e
    data = json.loads(raw.decode("utf-8"))
    embs = data.get("embeddings")
    if not isinstance(embs, list):
        single = data.get("embedding")
        if isinstance(single, list) and len(texts) == 1:
            embs = [single]
        else:
            raise RuntimeError("Malformed Ollama /api/embed response (no embeddings)")
    if len(embs) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embs)} for {len(texts)} texts",
        )
    out: list[list[float]] = []
    for row in embs:
        if not isinstance(row, list):
            raise RuntimeError("Malformed embedding vector")
        out.append([float(x) for x in row])
    return out


def _st_encode_many(texts: list[str], config: IngestEmbedderConfig) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed; pip install sentence-transformers",
        ) from e

    model = SentenceTransformer(config.model_name)
    bs = max(1, min(config.batch_size, len(texts)))
    vectors = model.encode(
        texts,
        batch_size=bs,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return [v.astype(float).tolist() for v in vectors]


def embed_texts(texts: list[str], config: IngestEmbedderConfig) -> list[list[float]]:
    if not texts:
        raise ValueError("embed_texts requires non-empty texts")
    normed = [normalize_text_for_embedding(t) for t in texts]
    if any(not x for x in normed):
        raise ValueError("Each text must be non-empty after strip/normalize")

    bs = max(1, int(config.batch_size))
    all_out: list[list[float]] = []

    if config.backend == "sentence_transformers":
        return _st_encode_many(normed, config)

    for i in range(0, len(normed), bs):
        batch = normed[i : i + bs]
        if config.backend == "ollama_nomic":
            all_out.extend(_ollama_embed_batch(batch, config))
        else:
            raise ValueError(f"Unknown backend {config.backend!r}")

    if len(all_out) != len(texts):
        raise RuntimeError("Internal: embedding output length mismatch")
    return all_out
