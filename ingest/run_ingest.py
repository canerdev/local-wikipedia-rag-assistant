"""End-to-end offline ingestion CLI / library entrypoint."""

from __future__ import annotations

import logging
import time
import uuid

from ingest.chunker import ChunkRecord, EntityType, chunk_article
from ingest.embedder import IngestEmbedderConfig, embed_texts, embedder_signature
from ingest.fetcher import ArticleNotFoundError, WikipediaHTTPError, fetch_wikipedia_plaintext
from ingest.store import ChromaStoreConfig, collection_stats, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)

# Space out article fetches to reduce HTTP 429 (Wikipedia API rate limiting).
FETCH_DELAY_S = 0.6

# PRD I-2 required + extras so |people| >= 20 and |places| >= 20 (PRD I-1)
DEFAULT_ENTITY_SPECS: list[tuple[str, EntityType]] = [
    # People (20)
    ("Albert Einstein", "person"),
    ("Marie Curie", "person"),
    ("Leonardo da Vinci", "person"),
    ("William Shakespeare", "person"),
    ("Ada Lovelace", "person"),
    ("Nikola Tesla", "person"),
    ("Lionel Messi", "person"),
    ("Cristiano Ronaldo", "person"),
    ("Taylor Swift", "person"),
    ("Frida Kahlo", "person"),
    ("Isaac Newton", "person"),
    ("Charles Darwin", "person"),
    ("Wolfgang Amadeus Mozart", "person"),
    ("Ludwig van Beethoven", "person"),
    ("Nelson Mandela", "person"),
    ("Cleopatra", "person"),
    ("Vincent van Gogh", "person"),
    ("Pablo Picasso", "person"),
    ("Oprah Winfrey", "person"),
    ("Michael Jordan", "person"),
    # Places (20)
    ("Eiffel Tower", "place"),
    ("Great Wall of China", "place"),
    ("Taj Mahal", "place"),
    ("Grand Canyon", "place"),
    ("Machu Picchu", "place"),
    ("Colosseum", "place"),
    ("Hagia Sophia", "place"),
    ("Statue of Liberty", "place"),
    ("Pyramids of Giza", "place"),
    ("Mount Everest", "place"),
    ("Great Barrier Reef", "place"),
    ("Niagara Falls", "place"),
    ("Burj Khalifa", "place"),
    ("Christ the Redeemer (statue)", "place"),
    ("Acropolis of Athens", "place"),
    ("Sydney Opera House", "place"),
    ("Petra", "place"),
    ("Angkor Wat", "place"),
    ("Notre-Dame de Paris", "place"),
    ("Alps", "place"),
]


def run_ingestion(
    *,
    chroma_config: ChromaStoreConfig,
    embedder_config: IngestEmbedderConfig,
    entity_specs: list[tuple[str, EntityType]],
) -> dict[str, int]:
    ingest_run_id = str(uuid.uuid4())
    sig = embedder_signature(embedder_config)

    n_specs = len(entity_specs)
    logger.info(
        "Starting ingestion: %d articles (Wikipedia → chunk → embed → Chroma). "
        "This can take several minutes (API spacing + Ollama embeddings).",
        n_specs,
    )

    attempted = 0
    succeeded = 0
    failed = 0

    staged: list[tuple[ChunkRecord, str]] = []

    for entity_name, entity_type in entity_specs:
        attempted += 1
        try:
            logger.info(
                "[%d/%d] Fetching from Wikipedia: %r (%s)",
                attempted,
                n_specs,
                entity_name,
                entity_type,
            )
            art = fetch_wikipedia_plaintext(entity_name, language="en")
            chunks = chunk_article(
                art.plaintext,
                entity_name=entity_name,
                entity_type=entity_type,
                source_url=art.canonical_url,
            )
            for c in chunks:
                staged.append((c, art.wikipedia_title))
            logger.info(
                "[%d/%d] Got %d chunk(s) for %r",
                attempted,
                n_specs,
                len(chunks),
                entity_name,
            )
            succeeded += 1
        except (ArticleNotFoundError, WikipediaHTTPError, OSError) as e:
            logger.warning("Ingest failed for %r: %s", entity_name, e)
            failed += 1
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected ingest error for %r: %s", entity_name, e)
            failed += 1
        finally:
            if attempted < len(entity_specs):
                time.sleep(FETCH_DELAY_S)

    if not staged:
        logger.error(
            "No chunks were staged (all Wikipedia fetches failed or produced no text). "
            "Fix network/SSL/Ollama and retry; Chroma was not updated.",
        )
        return {
            "entities_attempted": attempted,
            "entities_succeeded": succeeded,
            "entities_failed": failed,
            "chunks_upserted": 0,
        }

    logger.info(
        "Fetching done. Embedding %d chunks with %s/%s (Ollama/host as configured) …",
        len(staged),
        embedder_config.backend,
        embedder_config.model_name,
    )

    texts = [c.text for c, _ in staged]
    embeddings = embed_texts(texts, embedder_config)
    if len(embeddings) != len(staged):
        raise RuntimeError("Internal error: embedding count does not match chunk count")
    dim = len(embeddings[0])
    ensure_collection(chroma_config, embedding_dimension=dim)

    logger.info(
        "Writing vectors to Chroma: %r / collection %r …",
        chroma_config.persist_directory,
        chroma_config.collection_name,
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for (rec, wiki_title), emb in zip(staged, embeddings):
        ids.append(rec.chunk_id)
        documents.append(rec.text)
        metadatas.append(
            {
                "entity_name": rec.entity_name,
                "entity_type": rec.entity_type,
                "source_url": rec.source_url,
                "chunk_index": int(rec.chunk_index),
                "section_title": rec.section_title,
                "wikipedia_title": wiki_title,
                "chunk_id": rec.chunk_id,
                "ingest_run_id": ingest_run_id,
                "embedder_signature": sig,
            },
        )

    upsert_chunks(
        chroma_config,
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return {
        "entities_attempted": attempted,
        "entities_succeeded": succeeded,
        "entities_failed": failed,
        "chunks_upserted": len(ids),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cc = ChromaStoreConfig(persist_directory="./chroma_db", collection_name="wikipedia_rag")
    ec = IngestEmbedderConfig(
        backend="ollama_nomic",
        model_name="nomic-embed-text",
        batch_size=8,
    )
    stats = run_ingestion(
        chroma_config=cc,
        embedder_config=ec,
        entity_specs=list(DEFAULT_ENTITY_SPECS),
    )
    logger.info("Ingestion finished: %s", stats)
    logger.info("Store: %s", collection_stats(cc))


if __name__ == "__main__":
    main()
