"""Tests for ingest.chunker — structure-first chunking (stdlib unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Project root on path when tests are not installed as a package
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingest.chunker import (  # noqa: E402
    HARD_CAP_CHARS,
    MIN_CHARS_BEFORE_MERGE,
    TARGET_CHARS,
    ChunkRecord,
    chunk_article,
)


class TestChunkerSections(unittest.TestCase):
    def test_lead_and_level2_sections_yield_distinct_section_titles(self):
        # Bodies must meet MIN_CHARS_BEFORE_MERGE or short sections collapse under the first title.
        big = "x" * 300
        wikitext = (
            f"Lead paragraph here.\n{big}\n"
            "== Early life ==\n"
            f"Born in testville.\n{big}\n"
            "== Career ==\n"
            f"Did important things.\n{big}"
        )
        chunks = chunk_article(
            wikitext,
            entity_name="Test Person",
            entity_type="person",
            source_url="https://example.com/wiki/Test_Person",
        )
        titles = {c.section_title for c in chunks}
        self.assertIn("__LEAD__", titles)
        self.assertIn("Early life", titles)
        self.assertIn("Career", titles)

    def test_no_headings_entire_body_is_lead(self):
        text = "Only one block without headings."
        chunks = chunk_article(
            text,
            entity_name="X",
            entity_type="place",
            source_url="https://example.com/x",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_title, "__LEAD__")
        self.assertIn("without headings", chunks[0].text)


class TestChunkerMetadata(unittest.TestCase):
    def test_each_chunk_has_prd_metadata_fields(self):
        body = "== Intro ==\nHello world."
        chunks = chunk_article(
            body,
            entity_name="Ada Lovelace",
            entity_type="person",
            source_url="https://en.wikipedia.org/wiki/Ada_Lovelace",
        )
        self.assertTrue(chunks)
        for i, c in enumerate(chunks):
            self.assertIsInstance(c, ChunkRecord)
            self.assertEqual(c.entity_name, "Ada Lovelace")
            self.assertEqual(c.entity_type, "person")
            self.assertEqual(c.source_url, "https://en.wikipedia.org/wiki/Ada_Lovelace")
            self.assertEqual(c.chunk_index, i)
            self.assertTrue(c.chunk_id)
            self.assertTrue(c.text.strip())

    def test_chunk_index_monotonic_after_empty_segment_drop(self):
        wikitext = "Lead only.\n\n== A ==\n\n== B ==\nBody B."
        chunks = chunk_article(
            wikitext,
            entity_name="Y",
            entity_type="place",
            source_url="https://example.com/y",
        )
        for i, c in enumerate(chunks):
            self.assertEqual(c.chunk_index, i)


class TestChunkerWikiCleanup(unittest.TestCase):
    def test_wiki_links_reduced_to_display_text(self):
        wikitext = "See [[Foo|Bar]] and [[Baz]] for more."
        chunks = chunk_article(
            wikitext,
            entity_name="Z",
            entity_type="person",
            source_url="https://example.com/z",
        )
        joined = " ".join(c.text for c in chunks)
        self.assertIn("Bar", joined)
        self.assertIn("Baz", joined)
        self.assertNotIn("[[", joined)


class TestChunkerLongDocumentWindows(unittest.TestCase):
    def test_long_section_produces_multiple_chunks_under_hard_cap(self):
        filler = "word " * 800
        wikitext = f"== Big ==\n{filler}"
        chunks = chunk_article(
            wikitext,
            entity_name="Long Article",
            entity_type="place",
            source_url="https://example.com/long",
        )
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), HARD_CAP_CHARS + 50)

    def test_chunk_size_targets_documented_constants(self):
        paragraph = "x" * (TARGET_CHARS + 200)
        wikitext = f"== P ==\n{paragraph}"
        chunks = chunk_article(
            wikitext,
            entity_name="Doc",
            entity_type="person",
            source_url="https://example.com/d",
        )
        self.assertGreaterEqual(len(chunks), 2)


class TestChunkerShortMerge(unittest.TestCase):
    def test_very_short_sections_merge_into_single_chunk(self):
        tiny_a = "a" * 50
        tiny_b = "b" * 50
        wikitext = f"== S1 ==\n{tiny_a}\n== S2 ==\n{tiny_b}"
        chunks = chunk_article(
            wikitext,
            entity_name="Stub",
            entity_type="person",
            source_url="https://example.com/s",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_title, "S1")
        self.assertIn(tiny_a, chunks[0].text)
        self.assertIn(tiny_b, chunks[0].text)
        self.assertLess(len(chunks[0].text), MIN_CHARS_BEFORE_MERGE)


class TestChunkerDeterminism(unittest.TestCase):
    def test_same_input_same_chunk_ids(self):
        wikitext = "== A ==\nHello.\n== B ==\nWorld."
        a = chunk_article(
            wikitext,
            entity_name="Same",
            entity_type="place",
            source_url="https://example.com/same",
        )
        b = chunk_article(
            wikitext,
            entity_name="Same",
            entity_type="place",
            source_url="https://example.com/same",
        )
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])


if __name__ == "__main__":
    unittest.main()
