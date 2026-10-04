"""
Web search service for Hello AI using Tavily.
"""

import logging
import os
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_URL = "https://api.tavily.com/search"

DEFAULT_TIMEOUT = 20
MAX_RESULTS_LIMIT = 10
MAX_QUERY_LENGTH = 4000
MAX_TITLE_LENGTH = 300
MAX_CONTENT_LENGTH = 3000
MAX_URL_LENGTH = 2000

NEWS_PATTERN = re.compile(
    r"\b("
    r"news|latest news|breaking news|today's news|today news|"
    r"current news|recent news|headlines|top stories|"
    r"latest updates|"
    r")\b",
    re.IGNORECASE,
)

BENGALI_NEWS_PATTERN = re.compile(
    r"(খবর|সংবাদ|আজকের খবর|সর্বশেষ খবর|সাম্প্রতিক খবর)"
)

HINDI_NEWS_PATTERN = re.compile(
    r"(खबर|समाचार|आज की खबर|ताज़ा खबर|ताजा खबर)"
)

GENERIC_NEWS_PATHS = {
    "",
    "/india",
    "/world/india",
    "/news",
    "/news/india",
    "/india-news",
    "/latest",
    "/latest-news",
    "/us",
    "/world",
}


def _is_news_query(query: str) -> bool:
    """Return True when the query appears to request news."""

    return bool(
        NEWS_PATTERN.search(query)
        or BENGALI_NEWS_PATTERN.search(query)
        or HINDI_NEWS_PATTERN.search(query)
    )


def _is_generic_news_page(url: str) -> bool:
    """Identify obvious homepages and general news-section pages."""

    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").lower()
    except (TypeError, ValueError):
        return True

    if path in GENERIC_NEWS_PATHS:
        return True

    return False


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily.

    Returns a list of dictionaries containing title, content and url.
    For news queries, prefers direct article pages over general sections.
    """

    if not isinstance(query, str):
        raise ValueError("Search query must be a string.")

    query = query.strip()

    if not query:
        return []

    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    try:
        max_results = int(max_results)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_results must be an integer.") from exc

    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))
    is_news = _is_news_query(query)

    payload = {
        "query": query[:MAX_QUERY_LENGTH],
        "max_results": min(MAX_RESULTS_LIMIT, max_results * 2)
        if is_news else max_results,
        "search_depth": "advanced" if is_news else "basic",
    }

    if is_news:
        payload["topic"] = "news"

        if re.search(
            r"\b(today|latest|breaking|current|right now)\b|আজকের|সর্বশেষ|आज की|ताज़ा|ताजा",
            query,
            re.IGNORECASE,
        ):
            payload["time_range"] = "day"

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            TAVILY_URL,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.Timeout as exc:
        logger.warning("Tavily search timed out.")
        raise RuntimeError("Web search timed out.") from exc
    except requests.RequestException as exc:
        logger.warning("Could not connect to Tavily: %s", exc)
        raise RuntimeError("Could not connect to web search.") from exc

    if response.status_code >= 400:
        logger.error(
            "Tavily API returned HTTP %s: %s",
            response.status_code,
            response.text[:300],
        )
        raise RuntimeError(
            f"Tavily search failed with HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Invalid response from web search."
        ) from exc

    raw_results = data.get("results", [])

    if not isinstance(raw_results, list):
        logger.warning("Tavily returned an unexpected results format.")
        return []

    results = []
    seen_urls = set()

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        url = str(item.get("url") or "").strip()

        if not title or not url:
            continue

        if not url.startswith(("https://", "http://")):
            continue

        if url in seen_urls:
            continue

        if is_news and _is_generic_news_page(url):
            logger.info(
                "Skipping generic news page: %s",
                url,
            )
            continue

        seen_urls.add(url)

        results.append({
            "title": title[:MAX_TITLE_LENGTH],
            "content": content[:MAX_CONTENT_LENGTH],
            "url": url[:MAX_URL_LENGTH],
        })

        if len(results) >= max_results:
            break

    logger.info(
        "Tavily search completed: news_query=%s, usable_results=%d",
        is_news,
        len(results),
    )

    for result in results:
        logger.info(
            "Tavily result: title=%s | url=%s",
            result["title"],
            result["url"],
        )

    return results