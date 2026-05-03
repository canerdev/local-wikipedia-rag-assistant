"""Keyword / rule-based query routing (architecture §5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RouteLabel = Literal["person", "place", "both"]


@dataclass(frozen=True)
class RouteDecision:
    label: RouteLabel


THRESH_DELTA = 2

_PLACE_STRONG = frozenset(
    {
        "where",
        "location",
        "located",
        "mountain",
        "river",
        "canyon",
        "landmark",
        "monument",
        "building",
        "temple",
        "pyramid",
        "tower",
        "statue",
        "bridge",
        "cathedral",
        "mosque",
        "museum",
        "site",
        "geography",
        "country",
        "city",
        "capital",
        "island",
        "peak",
        "summit",
        "wall",
        "ruins",
        "park",
        "unesco",
    },
)

_PERSON_STRONG = frozenset(
    {
        "who",
        "biography",
        "born",
        "died",
        "nobel",
        "composer",
        "painter",
        "scientist",
        "inventor",
        "philosopher",
        "writer",
        "author",
        "artist",
        "athlete",
        "football",
        "soccer",
        "singer",
        "actor",
        "actress",
        "politician",
        "queen",
        "king",
        "emperor",
        "president",
    },
)

_GEO_SPILL = frozenset(
    {"in", "near", "north", "south", "east", "west", "border", "elevation", "altitude"},
)

_PERSON_LIGHT = frozenset(
    {"discovered", "invented", "won", "award", "team", "album", "film"},
)

# PRD I-2 + normalized aliases (architecture corpus dictionary)
PLACE_SUBSTRINGS: tuple[str, ...] = (
    "eiffel tower",
    "great wall of china",
    "great wall",
    "taj mahal",
    "grand canyon",
    "machu picchu",
    "colosseum",
    "hagia sophia",
    "statue of liberty",
    "pyramids of giza",
    "pyramids",
    "giza",
    "mount everest",
    "everest",
    "great barrier reef",
    "niagara falls",
    "burj khalifa",
    "christ the redeemer",
    "acropolis",
    "sydney opera house",
    "petra",
    "angkor wat",
    "notre-dame",
    "notre dame",
    "the alps",
    "alps",
)

PERSON_SUBSTRINGS: tuple[str, ...] = (
    "albert einstein",
    "einstein",
    "marie curie",
    "curie",
    "leonardo da vinci",
    "da vinci",
    "william shakespeare",
    "shakespeare",
    "ada lovelace",
    "lovelace",
    "nikola tesla",
    "tesla",
    "lionel messi",
    "messi",
    "cristiano ronaldo",
    "ronaldo",
    "taylor swift",
    "swift",
    "frida kahlo",
    "kahlo",
    "isaac newton",
    "newton",
    "charles darwin",
    "darwin",
    "mozart",
    "beethoven",
    "nelson mandela",
    "mandela",
    "cleopatra",
    "van gogh",
    "picasso",
    "oprah winfrey",
    "oprah",
    "michael jordan",
    "jordan",
)


def _preprocess(q: str) -> tuple[str, list[str], str]:
    s = q.strip().lower()
    s = re.sub(r"\s+", " ", s)
    tokens = re.findall(r"[\w]+", s, flags=re.UNICODE)
    return s, tokens, s


def classify_query(query: str) -> RouteDecision:
    s, tokens, _ = _preprocess(query)
    tok_set = frozenset(tokens)

    s_place = 0
    s_person = 0

    for w in _PLACE_STRONG:
        if w in tok_set:
            s_place += 2
    for w in _PERSON_STRONG:
        if w in tok_set:
            s_person += 2

    place_cue_hit = s_place >= 2
    place_name_hit = any(alias in s for alias in PLACE_SUBSTRINGS)
    person_name_hit = any(alias in s for alias in PERSON_SUBSTRINGS)

    if place_name_hit:
        s_place += 2
    if person_name_hit:
        s_person += 2

    if ("who" in tok_set or s_person >= 2) and bool(tok_set & _PERSON_LIGHT):
        s_person += 1

    geo_ok = place_cue_hit or place_name_hit
    if geo_ok and bool(tok_set & _GEO_SPILL):
        s_place += 1

    if s.startswith("who is ") or s.startswith("who was ") or s.startswith("who's "):
        s_person += 3
    if s.startswith("where is ") or s.startswith("where was ") or s.startswith("where are "):
        s_place += 3
    if "which country" in s or "which city" in s:
        s_place += 2
    if "which scientist" in s or "which writer" in s or "which player" in s:
        s_person += 2

    if abs(s_place - s_person) <= 1 and (s_place + s_person) <= 3:
        label: RouteLabel = "both"
    elif abs(s_place - s_person) <= 1 and max(s_place, s_person) >= 4:
        label = "both"
    elif s_place >= s_person + THRESH_DELTA:
        label = "place"
    elif s_person >= s_place + THRESH_DELTA:
        label = "person"
    else:
        label = "both"

    return RouteDecision(label=label)
