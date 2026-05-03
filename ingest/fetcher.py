"""Fetch Wikipedia article source via the MediaWiki Action API using urllib only.

Article body is revision wikitext (preserves ``== Section ==`` markers for ``chunker``).
The :class:`ArticleFetchResult` field ``plaintext`` holds this wikitext per the
architecture contract (suitable for structure-first chunking).
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_API_TIMEOUT_S = 60.0
# MediaWiki returns HTTP 429 when requests arrive too quickly; backoff and obey Retry-After.
_MAX_FETCH_RETRIES = 8
_INITIAL_BACKOFF_S = 3.0
_MAX_BACKOFF_S = 180.0
USER_AGENT = (
    "LocalWikipediaRAGAssistant/1.0 "
    "(local educational RAG; Python urllib; contact: local)"
)


class WikipediaHTTPError(Exception):
    """Non-200 response, timeout, I/O failure, or malformed API JSON envelope."""


class ArticleNotFoundError(Exception):
    """Missing page, invalid title, or disambiguation (ingest failure)."""


@dataclass(frozen=True)
class ArticleFetchResult:
    wikipedia_title: str
    canonical_url: str
    plaintext: str


def normalize_title_for_url(title: str) -> str:
    """Normalize a human or slug-derived title for the API ``titles`` parameter.

    Strips ends, collapses internal whitespace, and converts underscores to spaces.
    """
    t = title.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("_", " ")
    return t


def _api_endpoint(language: str) -> str:
    lang = (language or "en").strip().lower()
    if not lang:
        lang = "en"
    return f"https://{lang}.wikipedia.org/w/api.php"


def _canonical_wikipedia_article_url(language: str, wikipedia_title: str) -> str:
    slug = wikipedia_title.strip().replace(" ", "_")
    encoded = urllib.parse.quote(slug, safe="():%,'!~@$&*+=-./\\")
    lang = (language or "en").strip().lower() or "en"
    return f"https://{lang}.wikipedia.org/wiki/{encoded}"


def _ssl_context() -> ssl.SSLContext:
    """CA bundle suitable for urllib on macOS / many Python installers.

    Prefer certifi when installed (fixes ``CERTIFICATE_VERIFY_FAILED`` when the
    interpreter ships without usable system CAs).
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _retry_after_seconds(http_err: urllib.error.HTTPError) -> float | None:
    headers = getattr(http_err, "headers", None)
    if headers is None:
        return None
    raw_ra = headers.get("Retry-After")
    if not raw_ra:
        return None
    try:
        return float(raw_ra.strip())
    except ValueError:
        return None


def _should_retry_http_status(code: int) -> bool:
    return code in (429, 503)


def _sleep_before_retry(http_err: urllib.error.HTTPError, attempt: int) -> None:
    ra = _retry_after_seconds(http_err)
    if ra is not None and ra > 0:
        delay = min(ra + 0.5, _MAX_BACKOFF_S)
    else:
        delay = min(_INITIAL_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
    time.sleep(delay)


def _request_json(url: str) -> dict[str, Any]:
    """GET JSON once with retries when Wikipedia rate-limits (429) or flakes (503)."""
    for attempt in range(_MAX_FETCH_RETRIES):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=DEFAULT_API_TIMEOUT_S,
                context=_ssl_context(),
            ) as resp:
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise WikipediaHTTPError(f"HTTP status {status} for {url!r}")
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if (
                _should_retry_http_status(e.code)
                and attempt + 1 < _MAX_FETCH_RETRIES
            ):
                try:
                    e.read()
                except OSError:
                    pass
                _sleep_before_retry(e, attempt)
                continue
            raise WikipediaHTTPError(f"HTTP {e.code} for {url!r}") from e
        except urllib.error.URLError as e:
            raise WikipediaHTTPError(f"Request failed for {url!r}: {e.reason}") from e
        except TimeoutError as e:
            raise WikipediaHTTPError(f"Timeout fetching {url!r}") from e

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise WikipediaHTTPError("Malformed API response (not valid UTF-8 JSON)") from e

        if not isinstance(data, dict):
            raise WikipediaHTTPError(
                "Malformed API envelope (top-level JSON not an object)",
            )
        return data


def _parse_query_page(
    page_key: str,
    page: Mapping[str, Any],
) -> tuple[str, str]:
    """Return (resolved_title, wikitext) or raise ArticleNotFoundError."""
    if "missing" in page:
        t = page.get("title", page_key)
        raise ArticleNotFoundError(f"Missing Wikipedia page: {t!r}")

    title = str(page.get("title", "")).strip()
    if not title:
        title = str(page_key)

    pageprops = page.get("pageprops")
    if isinstance(pageprops, dict) and "disambiguation" in pageprops:
        raise ArticleNotFoundError(
            f"Disambiguation page is not ingestible: {title!r}",
        )

    revs = page.get("revisions")
    if not isinstance(revs, list) or not revs:
        raise ArticleNotFoundError(f"No revisions for page: {title!r}")

    rev0 = revs[0]
    if not isinstance(rev0, dict):
        raise WikipediaHTTPError("Malformed revisions[0] in API response")

    slots = rev0.get("slots")
    if isinstance(slots, dict) and "main" in slots:
        main = slots["main"]
        if isinstance(main, dict) and "content" in main:
            content = main["content"]
            if isinstance(content, str):
                return title, content
        raise WikipediaHTTPError("Malformed slots.main.content in API response")

    # Older API shape: * slots may be absent; try top-level "*" key
    if "*" in rev0 and isinstance(rev0["*"], str):
        return title, rev0["*"]

    raise WikipediaHTTPError("Could not locate revision content in API response")


def fetch_wikipedia_plaintext(title: str, language: str = "en") -> ArticleFetchResult:
    """Fetch one article's wikitext via the MediaWiki Action API.

    ``plaintext`` is revision wikitext (section headings preserved for chunking).
    """
    normalized = normalize_title_for_url(title)
    if not normalized:
        raise ArticleNotFoundError("Empty title after normalization")

    base = _api_endpoint(language)
    params: dict[str, str] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "revisions|pageprops",
        "rvprop": "content",
        "rvslots": "main",
        "titles": normalized,
    }
    query = urllib.parse.urlencode(params, safe="|", encoding="utf-8")
    url = f"{base}?{query}"
    envelope = _request_json(url)

    query_block = envelope.get("query")
    if not isinstance(query_block, dict):
        raise WikipediaHTTPError("Missing or invalid 'query' in API envelope")

    pages = query_block.get("pages")
    if not isinstance(pages, list) or not pages:
        raise WikipediaHTTPError("Missing or empty 'query.pages' in API envelope")

    page0 = pages[0]
    if not isinstance(page0, dict):
        raise WikipediaHTTPError("Invalid page object in API response")

    resolved_title, wikitext = _parse_query_page("0", page0)
    lang = (language or "en").strip().lower() or "en"
    return ArticleFetchResult(
        wikipedia_title=resolved_title,
        canonical_url=_canonical_wikipedia_article_url(lang, resolved_title),
        plaintext=wikitext,
    )