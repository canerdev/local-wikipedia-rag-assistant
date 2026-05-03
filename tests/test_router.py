"""Tests for core.router — keyword / rule-based query routing (stdlib unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.router import RouteDecision, classify_query  # noqa: E402


class TestRouterPRDExamples(unittest.TestCase):
    """Exact cases requested for QA (PRD / AC-style phrasing)."""

    def test_who_was_albert_einstein_is_person(self):
        d = classify_query("Who was Albert Einstein")
        self.assertIsInstance(d, RouteDecision)
        self.assertEqual(d.label, "person")

    def test_where_is_eiffel_tower_is_place(self):
        d = classify_query("Where is the Eiffel Tower")
        self.assertEqual(d.label, "place")

    def test_which_famous_place_in_turkey_is_both_or_unclear(self):
        d = classify_query("Which famous place is in Turkey")
        self.assertEqual(d.label, "both")

    def test_president_of_mars_router_still_person_not_idk(self):
        """Generation may return 'I don't know'; router only classifies intent."""
        d = classify_query("Who is the president of Mars")
        self.assertEqual(d.label, "person")


class TestRouterDecisionTable(unittest.TestCase):
    def test_balanced_weak_signals_yield_both(self):
        d = classify_query("Tell me something")
        self.assertEqual(d.label, "both")

    def test_where_pattern_boosts_place(self):
        d = classify_query("Where was the meeting")
        self.assertEqual(d.label, "place")

    def test_place_name_substring_boosts_place(self):
        d = classify_query("Height of Mount Everest")
        self.assertEqual(d.label, "place")

    def test_person_name_substring_boosts_person(self):
        d = classify_query("Nobel prizes Einstein discussion")
        self.assertEqual(d.label, "person")


class TestRouterFrozenDecision(unittest.TestCase):
    def test_route_decision_is_frozen_dataclass(self):
        d = RouteDecision(label="person")
        with self.assertRaises(Exception):
            d.label = "place"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
