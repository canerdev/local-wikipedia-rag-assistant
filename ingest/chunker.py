"""Structure-first Wikipedia wikitext chunking (§3 architecture)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

EntityType = Literal["person", "place"]

TARGET_CHARS = 1100
HARD_CAP_CHARS = 1600
OVERLAP_CHARS = 150
MIN_CHARS_BEFORE_MERGE = 280

_LEAD = "__LEAD__"
_SECTION_RE = re.compile(r"^==\s*([^=\n][^=\n]*?)\s*==\s*$")


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    chunk_id: str
    entity_name: str
    entity_type: EntityType
    source_url: str
    chunk_index: int
    section_title: str


def _collapse_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", s.strip()))


def _slug(s: str, max_len: int = 72) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s[:max_len] or "x").rstrip("-")


def _strip_simple_wiki_noise(text: str) -> str:
    """Light cleanup for chunk text; stdlib only."""
    out = text
    out = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", out)
    out = re.sub(r"\[\[([^\]]+)\]\]", r"\1", out)
    out = re.sub(r"<ref[^>]*>[\s\S]*?</ref>", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"<[^>]+>", " ", out)
    out = re.sub(r"\{\{[^}]*\}\}", " ", out)
    out = _collapse_ws(out)
    return out


def _split_sections(wikitext: str) -> list[tuple[str, str]]:
    lines = wikitext.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = _LEAD
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if body or sections == []:
            sections.append((current_title, body))
        buf = []

    for line in lines:
        m = _SECTION_RE.match(line.strip())
        if m:
            flush()
            current_title = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    if not sections:
        return [(_LEAD, _collapse_ws(wikitext))]
    return sections


def _utf8_safe_rewind(text: str, end_exclusive: int, overlap: int) -> int:
    if end_exclusive <= 0 or overlap <= 0:
        return max(0, end_exclusive - overlap)
    start = max(0, end_exclusive - overlap)
    if start == 0:
        return 0
    # move to UTF-8 character boundary
    while start < end_exclusive and (text[start].encode("utf-8")[0] & 0b1100_0000) == 0b1000_0000:
        start += 1
    return start


def _next_window_end(text: str, start: int, target: int, hard: int) -> int:
    n = len(text)
    lo = min(start + target, n)
    hi = min(start + hard, n)
    if lo >= n:
        return n
    window = text[start:hi]
    rel_lo = target
    rel_hi = len(window)

    def cut_at(rel: int) -> int:
        return min(start + rel, n)

    for sep in ("\n\n", "\n", ". "):
        idx = window.rfind(sep, rel_lo, rel_hi)
        if idx != -1:
            return cut_at(idx + len(sep))

    if rel_hi > rel_lo:
        sp = window.rfind(" ", rel_lo, rel_hi)
        if sp != -1:
            return cut_at(sp + 1)
    return lo


def _window_split_long(body: str) -> list[str]:
    if not body.strip():
        return []
    parts: list[str] = []
    start = 0
    n = len(body)
    while start < n:
        end = _next_window_end(body, start, TARGET_CHARS, HARD_CAP_CHARS)
        if end <= start:
            end = min(start + TARGET_CHARS, n)
        chunk = body[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end >= n:
            break
        start = _utf8_safe_rewind(body, end, OVERLAP_CHARS)
        if start >= end:
            start = end
    return parts


def _section_to_pieces(section_title: str, body_raw: str) -> list[tuple[str, str, Literal["short", "ready"]]]:
    body = _strip_simple_wiki_noise(body_raw)
    if not body:
        return []
    if len(body) > TARGET_CHARS:
        windows = _window_split_long(body)
        return [(section_title, w, "ready") for w in windows]
    if len(body) < MIN_CHARS_BEFORE_MERGE:
        return [(section_title, body, "short")]
    return [(section_title, body, "ready")]


def chunk_article(
    plaintext: str,
    entity_name: str,
    entity_type: EntityType,
    source_url: str,
) -> list[ChunkRecord]:
    """Split wikitext into :class:`ChunkRecord` list (deterministic ids and indices)."""
    normalized = _collapse_ws(plaintext)
    sections = _split_sections(normalized)

    pieces: list[tuple[str, str, Literal["short", "ready"]]] = []
    for sec_title, body in sections:
        pieces.extend(_section_to_pieces(sec_title, body))

    merged_segments: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_titles: list[str] = []

    def flush_short_buffer() -> None:
        nonlocal buf, buf_titles
        if not buf:
            return
        merged = "\n\n".join(b.strip() for b in buf if b.strip()).strip()
        title = buf_titles[0] if buf_titles else _LEAD
        if len(merged) > TARGET_CHARS:
            for w in _window_split_long(merged):
                merged_segments.append((title, w))
        else:
            merged_segments.append((title, merged))
        buf = []
        buf_titles = []

    for sec_title, text, kind in pieces:
        if kind == "short":
            buf.append(text)
            buf_titles.append(sec_title)
            total_len = len("\n\n".join(buf))
            if total_len >= MIN_CHARS_BEFORE_MERGE:
                flush_short_buffer()
            continue
        flush_short_buffer()
        merged_segments.append((sec_title, text))

    flush_short_buffer()

    records: list[ChunkRecord] = []
    stem_counts: dict[str, int] = {}
    entity_slug = _slug(entity_name)
    for sec_title, chunk_text in merged_segments:
        if not chunk_text.strip():
            continue
        stem = f"{entity_slug}__{_slug(sec_title)}"
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
        ordinal = stem_counts[stem]
        chunk_id = f"{stem}__{ordinal:04d}"
        records.append(
            ChunkRecord(
                text=chunk_text,
                chunk_id=chunk_id,
                entity_name=entity_name,
                entity_type=entity_type,
                source_url=source_url,
                chunk_index=len(records),
                section_title=sec_title,
            )
        )

    # Re-assign chunk_index monotonically 0..n-1 after filter
    out: list[ChunkRecord] = []
    for i, r in enumerate(records):
        out.append(
            ChunkRecord(
                text=r.text,
                chunk_id=r.chunk_id,
                entity_name=r.entity_name,
                entity_type=r.entity_type,
                source_url=r.source_url,
                chunk_index=i,
                section_title=r.section_title,
            )
        )
    return out
