"""Tests for core.retriever — Chroma query + tie-break (mocked, stdlib unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# retriever imports chromadb at module load; stub before importing core.retriever
sys.modules.setdefault(
    "chromadb",
    MagicMock(name="chromadb_stub"),
)

from core.retriever import RetrievedChunk, retrieve  # noqa: E402
from core.router import RouteDecision  # noqa: E402
from ingest.embedder import IngestEmbedderConfig  # noqa: E402
from ingest.store import ChromaStoreConfig  # noqa: E402


def _make_configs():
    chroma = ChromaStoreConfig(persist_directory="/tmp/chroma-test", collection_name="test_col")
    embedder = IngestEmbedderConfig(
        backend="ollama_nomic",
        model_name="nomic-embed-text",
        batch_size=8,
        ollama_host="http://127.0.0.1:11434",
    )
    return chroma, embedder


class TestRetrieveGuards(unittest.TestCase):
    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_k_zero_returns_empty_without_chroma(self, mock_embed, mock_client):
        chroma, embedder = _make_configs()
        out = retrieve(
            "Who is X",
            RouteDecision(label="person"),
            chroma,
            embedder,
            k=0,
        )
        self.assertEqual(out, [])
        mock_embed.assert_not_called()
        mock_client.assert_not_called()

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_whitespace_only_query_returns_empty(self, mock_embed, mock_client):
        chroma, embedder = _make_configs()
        out = retrieve(
            "   \t\n  ",
            RouteDecision(label="both"),
            chroma,
            embedder,
            k=3,
        )
        self.assertEqual(out, [])
        mock_embed.assert_not_called()
        mock_client.assert_not_called()


class TestRetrieveFilters(unittest.TestCase):
    def _patch_chroma_query(self, mock_client, query_result):
        mock_col = MagicMock()
        mock_col.query.return_value = query_result
        instance = MagicMock()
        instance.get_collection.return_value = mock_col
        mock_client.return_value = instance
        return mock_col

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_person_route_passes_person_metadata_filter(self, mock_embed, mock_client):
        mock_embed.return_value = [[0.0, 1.0]]
        mock_col = self._patch_chroma_query(
            mock_client,
            {
                "metadatas": [
                    [
                        {
                            "entity_type": "person",
                            "chunk_index": 0,
                            "chunk_id": "p-1",
                        }
                    ]
                ],
                "documents": [["chunk text"]],
                "distances": [[0.25]],
            },
        )
        chroma, embedder = _make_configs()
        retrieve("Who was Ada", RouteDecision(label="person"), chroma, embedder, k=5)
        mock_col.query.assert_called_once()
        call_kw = mock_col.query.call_args.kwargs
        self.assertEqual(call_kw["where"], {"entity_type": "person"})
        self.assertEqual(call_kw["n_results"], 5)

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_place_route_passes_place_metadata_filter(self, mock_embed, mock_client):
        mock_embed.return_value = [[0.0, 1.0]]
        mock_col = self._patch_chroma_query(
            mock_client,
            {
                "metadatas": [
                    [
                        {
                            "entity_type": "place",
                            "chunk_index": 0,
                            "chunk_id": "pl-1",
                        }
                    ]
                ],
                "documents": [["place chunk"]],
                "distances": [[0.1]],
            },
        )
        chroma, embedder = _make_configs()
        retrieve("Where is Taj Mahal", RouteDecision(label="place"), chroma, embedder, k=3)
        call_kw = mock_col.query.call_args.kwargs
        self.assertEqual(call_kw["where"], {"entity_type": "place"})

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_both_route_has_no_where_filter(self, mock_embed, mock_client):
        mock_embed.return_value = [[0.5, 0.5]]
        mock_col = self._patch_chroma_query(
            mock_client,
            {
                "metadatas": [
                    [
                        {"entity_type": "place", "chunk_index": 0, "chunk_id": "a"},
                        {"entity_type": "person", "chunk_index": 0, "chunk_id": "b"},
                    ]
                ],
                "documents": [["d1", "d2"]],
                "distances": [[0.2, 0.2]],
            },
        )
        chroma, embedder = _make_configs()
        retrieve("Which famous place is in Turkey", RouteDecision(label="both"), chroma, embedder, k=2)
        call_kw = mock_col.query.call_args.kwargs
        self.assertIsNone(call_kw["where"])


class TestRetrieveShapeAndTieBreak(unittest.TestCase):
    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_returns_chunk_objects_sorted_by_distance_then_index_then_id(self, mock_embed, mock_client):
        mock_embed.return_value = [[1.0, 0.0, 0.0]]
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "metadatas": [
                [
                    {"chunk_index": 5, "chunk_id": "z"},
                    {"chunk_index": 1, "chunk_id": "a"},
                    {"chunk_index": 1, "chunk_id": "m"},
                ]
            ],
            "documents": [["second", "first", "tie"]],
            "distances": [[0.8, 0.8, 0.9]],
        }
        mock_client.return_value.get_collection.return_value = mock_col
        chroma, embedder = _make_configs()
        out = retrieve("query", RouteDecision(label="both"), chroma, embedder, k=5)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], RetrievedChunk)
        self.assertEqual(out[0].text, "first")
        self.assertEqual(out[1].text, "second")
        self.assertEqual(out[2].text, "tie")

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_empty_chroma_response_returns_empty_list(self, mock_embed, mock_client):
        mock_embed.return_value = [[0.1]]
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "metadatas": None,
            "documents": None,
            "distances": None,
        }
        mock_client.return_value.get_collection.return_value = mock_col
        chroma, embedder = _make_configs()
        out = retrieve("x", RouteDecision(label="person"), chroma, embedder, k=4)
        self.assertEqual(out, [])

    @patch("core.retriever.chromadb.PersistentClient")
    @patch("core.retriever.embed_texts")
    def test_embedder_receives_original_query_string(self, mock_embed, mock_client):
        mock_embed.return_value = [[0.0]]
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "metadatas": [[{"chunk_index": 0, "chunk_id": "x"}]],
            "documents": [["doc"]],
            "distances": [[0.0]],
        }
        mock_client.return_value.get_collection.return_value = mock_col
        chroma, embedder = _make_configs()
        retrieve("  MixedCase Query ", RouteDecision(label="both"), chroma, embedder, k=1)
        mock_embed.assert_called_once()
        args, _kwargs = mock_embed.call_args
        self.assertEqual(args[0], ["  MixedCase Query "])


if __name__ == "__main__":
    unittest.main()
