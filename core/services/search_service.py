# core/services/search_service.py

import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_URL = "https://api.tavily.com/search"

DEFAULT_MAX_RESULTS = 5
MAX_QUERY_LENGTH = 500
REQUEST_TIMEOUT = 20


# Explicit years: historical + present + reasonable future range.
YEAR_PATTERN = re.compile(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2}|21\d{2})(?!\d)")


# ============================================================
# LANGUAGE / QUERY PATTERNS
# ============================================================

CURRENT_PATTERNS = [
    r"\bcurrent\b",
    r"\bright now\b",
    r"\btoday\b",
    r"\btonight\b",
    r"\bcurrently\b",
    r"\blatest\b",
    r"\bnow\b",
    r"\bnews\b",
    r"\brecent\b",
    r"\bupdate\b",
    r"\bupdates\b",

    r"আজ",
    r"এখন",
    r"বর্তমানে",
    r"সাম্প্রতিক",
    r"সর্বশেষ",
    r"খবর",
    r"আপডেট",
    r"আজকের",

    r"आज",
    r"अभी",
    r"वर्तमान",
    r"ताज़ा",
    r"लेटेस्ट",
    r"खबर",
    r"समाचार",
    r"अपडेट",
]


NEWS_PATTERNS = [
    r"\bnews\b",
    r"\bheadline\b",
    r"\bheadlines\b",
    r"\blatest news\b",
    r"\bbreaking\b",
    r"\bupdate\b",

    r"খবর",
    r"সংবাদ",
    r"সর্বশেষ খবর",
    r"ব্রেকিং",
    r"আপডেট",

    r"खबर",
    r"समाचार",
    r"ताज़ा खबर",
    r"ब्रेकिंग",
    r"अपडेट",
]


EVENT_PATTERNS = [
    "event",
    "events",
    "happened",
    "happened in",
    "what happened",
    "important events",
    "major events",
    "key events",
    "timeline",
    "history",
    "historical",
    "year in",
    "recap",
    "roundup",
    "milestones",
    "significant events",

    "ঘটনা",
    "ঘটেছিল",
    "কি হয়েছিল",
    "কী হয়েছিল",
    "গুরুত্বপূর্ণ ঘটনা",
    "উল্লেখযোগ্য ঘটনা",
    "ইতিহাস",
    "ঐতিহাসিক",
    "সালভিত্তিক",
    "ঘটনাবলি",
    "ফিরে দেখা",
    "মাইলস্টোন",

    "घटना",
    "क्या हुआ",
    "महत्वपूर्ण घटनाएं",
    "इतिहास",
    "ऐतिहासिक",
    "घटनाक्रम",
]


FUTURE_PATTERNS = [
    "could",
    "may",
    "might",
    "expected",
    "expect",
    "forecast",
    "forecasted",
    "prediction",
    "predicted",
    "planned",
    "plan",
    "scheduled",
    "schedule",
    "upcoming",
    "will",
    "likely",
    "potential",
    "prospect",

    "হতে পারে",
    "সম্ভাবনা",
    "সম্ভাব্য",
    "পরিকল্পনা",
    "পরিকল্পিত",
    "নির্ধারিত",
    "আসন্ন",
    "ভবিষ্যতে",
    "হওয়ার সম্ভাবনা",
    "হওয়ার সম্ভাবনা",

    "हो सकता है",
    "संभावना",
    "संभावित",
    "योजना",
    "नियोजित",
    "निर्धारित",
    "आगामी",
    "भविष्य",
]


# ============================================================
# ENTITY ALIASES
# ============================================================

ENTITY_ALIASES = {
    "india": [
        "india",
        "indian",
        "bharat",
        "ভারত",
        "ভারতে",
        "ভারতের",
        "ইন্ডিয়া",
        "ইন্ডিয়া",
        "भारत",
        "भारतीय",
        "इंडिया",
    ],

    "bangladesh": [
        "bangladesh",
        "bangladeshi",
        "বাংলাদেশ",
        "বাংলাদেশে",
        "বাংলাদেশের",
        "বাংলাদেশি",
        "बांग्लादेश",
        "बांग्लादेश में",
    ],

    "pakistan": [
        "pakistan",
        "pakistani",
        "পাকিস্তান",
        "পাকিস্তানে",
        "পাকিস্তানের",
        "पाकिस्तान",
        "पाकिस्तानी",
    ],

    "usa": [
        "usa",
        "u.s.a",
        "u.s.",
        "united states",
        "america",
        "american",
        "আমেরিকা",
        "যুক্তরাষ্ট্র",
        "अमेरिका",
        "संयुक्त राज्य",
    ],

    "uk": [
        "uk",
        "u.k.",
        "united kingdom",
        "britain",
        "british",
        "england",
        "যুক্তরাজ্য",
        "ব্রিটেন",
        "ইংল্যান্ড",
        "ब्रिटेन",
        "इंग्लैंड",
    ],

    "china": [
        "china",
        "chinese",
        "চীন",
        "চীনে",
        "চীনের",
        "चीन",
        "चीनी",
    ],

    "japan": [
        "japan",
        "japanese",
        "জাপান",
        "জাপানে",
        "জাপানের",
        "जापान",
        "जापानी",
    ],

    "russia": [
        "russia",
        "russian",
        "রাশিয়া",
        "রাশিয়া",
        "রাশিয়ায়",
        "রাশিয়ায়",
        "रूस",
        "रूसी",
    ],
}


# ============================================================
# LOW VALUE DOMAINS
# ============================================================

LOW_VALUE_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "fb.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
}


GENERIC_PAGE_DOMAINS = {
    "timeanddate.com",
}


# ============================================================
# HELPERS
# ============================================================

def _safe_text(value):
    if value is None:
        return ""

    try:
        return str(value).strip()
    except Exception:
        return ""


def _normalise_spaces(text):
    return re.sub(r"\s+", " ", _safe_text(text)).strip()


def _clean_query(query):
    query = _normalise_spaces(query)

    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH].rstrip()

    return query


def _extract_years(text):
    text = _safe_text(text)
    years = []

    for match in YEAR_PATTERN.findall(text):
        try:
            year = int(match)
            if year not in years:
                years.append(year)
        except Exception:
            continue

    return years


def _has_specific_year(query):
    return bool(_extract_years(query))


def _is_future_year(year):
    current_year = datetime.now(timezone.utc).year
    return year > current_year


def _is_current_query(query):
    query_lower = _safe_text(query).lower()

    if _has_specific_year(query):
        return False

    return any(
        re.search(pattern, query_lower, re.IGNORECASE)
        for pattern in CURRENT_PATTERNS
    )


def _is_news_query(query):
    query_lower = _safe_text(query).lower()

    if _has_specific_year(query):
        return False

    return any(
        re.search(pattern, query_lower, re.IGNORECASE)
        for pattern in NEWS_PATTERNS
    )


def _is_event_query(query):
    text = _safe_text(query).lower()

    return any(
        pattern.lower() in text
        for pattern in EVENT_PATTERNS
    )


def _is_future_query(query):
    text = _safe_text(query).lower()

    return any(
        pattern.lower() in text
        for pattern in FUTURE_PATTERNS
    )


def _detect_query_entities(query):
    text = _safe_text(query).lower()
    found = []

    for entity, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text:
                found.append(entity)
                break

    return found


def _entity_alias_present(entity, text):
    text = _safe_text(text).lower()

    aliases = ENTITY_ALIASES.get(entity, [])

    return any(alias.lower() in text for alias in aliases)


def _domain_from_url(url):
    try:
        domain = urlparse(_safe_text(url)).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception:
        return ""


def _is_low_value_url(url):
    domain = _domain_from_url(url)

    if not domain:
        return False

    for blocked in LOW_VALUE_DOMAINS:
        if domain == blocked or domain.endswith("." + blocked):
            return True

    return False


def _is_generic_page(url):
    domain = _domain_from_url(url)

    if not domain:
        return False

    return any(
        domain == d or domain.endswith("." + d)
        for d in GENERIC_PAGE_DOMAINS
    )


def _source_quality(url):
    domain = _domain_from_url(url)

    if not domain:
        return 0.20

    if _is_low_value_url(url):
        return 0.05

    if domain.endswith(".gov") or ".gov." in domain:
        return 1.00

    if domain.endswith(".edu") or ".edu." in domain:
        return 0.95

    high_quality = {
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "apnews.com",
        "theguardian.com",
        "nytimes.com",
        "history.com",
        "archives.gov",
        "congress.gov",
        "parliament.uk",
        "un.org",
        "who.int",
        "wikipedia.org",
        "wikimedia.org",
        "hrw.org",
    }

    if domain in high_quality or any(
        domain.endswith("." + item)
        for item in high_quality
    ):
        return 0.90

    return 0.60


def _title_score(title, query):
    title = _safe_text(title).lower()
    query = _safe_text(query).lower()

    if not title:
        return 0.0

    query_words = [
        word
        for word in re.findall(r"\w+", query, flags=re.UNICODE)
        if len(word) >= 3
    ]

    if not query_words:
        return 0.0

    matched = sum(
        1 for word in query_words
        if word in title
    )

    return min(1.0, matched / max(1, min(len(query_words), 8)))


def _keyword_relevance(title, content, query):
    title = _safe_text(title).lower()
    content = _safe_text(content).lower()
    query = _safe_text(query).lower()

    query_words = [
        word
        for word in re.findall(r"\w+", query, flags=re.UNICODE)
        if len(word) >= 3
    ]

    if not query_words:
        return 0.0

    title_hits = 0
    content_hits = 0

    for word in query_words:
        if word in title:
            title_hits += 1
        elif word in content:
            content_hits += 1

    total = min(len(query_words), 10)

    return min(
        1.0,
        ((title_hits * 2.0) + content_hits) / max(1.0, total * 2.0)
    )


# ============================================================
# YEAR RELEVANCE
# ============================================================

def _year_presence_score(years, title, content):
    if not years:
        return 0.0

    title_text = _safe_text(title)
    content_text = _safe_text(content)

    score_values = []

    for year in years:
        year_text = str(year)

        in_title = year_text in title_text
        in_content = year_text in content_text

        if in_title and in_content:
            score_values.append(1.0)
        elif in_title:
            score_values.append(0.92)
        elif in_content:
            score_values.append(0.65)
        else:
            score_values.append(0.0)

    return max(score_values) if score_values else 0.0


def _year_match_count(years, title, content):
    text = _safe_text(title) + " " + _safe_text(content)

    count = 0

    for year in years:
        count += len(
            re.findall(
                rf"(?<!\d){year}(?!\d)",
                text
            )
        )

    return count


def _competing_year_penalty(years, title, content):
    """
    Prevent a result from ranking highly just because it mentions
    the requested year once while actually focusing on another year.

    Example:
        Query: 1947 India
        Article: mostly about 1937 Burma partition,
        with 1947 mentioned only as a comparison.
    """

    if not years:
        return 0.0

    title = _safe_text(title)
    content = _safe_text(content)

    requested_count = 0

    for year in years:
        requested_count += len(
            re.findall(
                rf"(?<!\d){year}(?!\d)",
                title + " " + content
            )
        )

    all_years = Counter(
        int(year)
        for year in YEAR_PATTERN.findall(
            title + " " + content
        )
    )

    if not all_years:
        return 0.0

    competing = 0

    for year, count in all_years.items():
        if year not in years:
            competing = max(competing, count)

    if competing <= requested_count:
        return 0.0

    # Strong penalty when another year dominates.
    ratio = competing / max(1, requested_count)

    if ratio >= 5:
        return 0.35

    if ratio >= 3:
        return 0.25

    if ratio >= 2:
        return 0.15

    return 0.05


def _year_focus_score(years, title, content):
    if not years:
        return 0.0

    title = _safe_text(title)
    content = _safe_text(content)

    requested_count = _year_match_count(
        years,
        title,
        content
    )

    if requested_count == 0:
        return 0.0

    presence = _year_presence_score(
        years,
        title,
        content
    )

    penalty = _competing_year_penalty(
        years,
        title,
        content
    )

    return max(
        0.0,
        min(1.0, presence + min(0.25, requested_count * 0.02) - penalty)
    )


# ============================================================
# ENTITY RELEVANCE
# ============================================================

def _entity_relevance_score(entities, title, content):
    if not entities:
        return 0.0

    title = _safe_text(title)
    content = _safe_text(content)

    scores = []

    for entity in entities:
        title_match = _entity_alias_present(
            entity,
            title
        )

        content_match = _entity_alias_present(
            entity,
            content
        )

        if title_match and content_match:
            scores.append(1.0)
        elif title_match:
            scores.append(0.92)
        elif content_match:
            scores.append(0.65)
        else:
            scores.append(0.0)

    return max(scores) if scores else 0.0


def _entity_match_count(entities, title, content):
    if not entities:
        return 0

    text = _safe_text(title) + " " + _safe_text(content)

    return sum(
        1
        for entity in entities
        if _entity_alias_present(entity, text)
    )


# ============================================================
# TOPIC / BROAD QUERY RELEVANCE
# ============================================================

def _topic_relevance_score(query, title, content):
    query_lower = _safe_text(query).lower()
    title_lower = _safe_text(title).lower()
    content_lower = _safe_text(content).lower()

    score = 0.0

    if _is_event_query(query):
        event_words = [
            "event",
            "events",
            "happened",
            "history",
            "historical",
            "timeline",
            "important",
            "major",
            "recap",
            "roundup",
            "ঘটনা",
            "ঘটেছিল",
            "গুরুত্বপূর্ণ",
            "ইতিহাস",
            "ঐতিহাসিক",
            "ঘটনাবলি",
            "ফিরে দেখা",
            "महत्‍वपूर्ण",
            "घटना",
            "इतिहास",
        ]

        title_hits = sum(
            1
            for word in event_words
            if word in title_lower
        )

        content_hits = sum(
            1
            for word in event_words
            if word in content_lower
        )

        score += min(
            0.75,
            title_hits * 0.25 + content_hits * 0.05
        )

    # Explicitly reward broad year-overview titles.
    broad_markers = [
        "year in",
        "important events",
        "major events",
        "key events",
        "timeline",
        "history of",
        "events of",
        "recap",
        "roundup",
        "year in review",
        "ফিরে দেখা",
        "গুরুত্বপূর্ণ ঘটনা",
        "উল্লেখযোগ্য ঘটনা",
        "ঘটনাবলি",
        "ইতিহাস",
        "घटनाएं",
        "महत्वपूर्ण घटनाएं",
        "इतिहास",
    ]

    broad_title_hits = sum(
        1
        for marker in broad_markers
        if marker in title_lower
    )

    if broad_title_hits:
        score += min(
            0.60,
            broad_title_hits * 0.30
        )

    return min(1.0, score)


def _narrow_result_penalty(query, title, content):
    """
    Penalize single-topic results for broad questions such as:
        '2021 সালে ভারতে কী হয়েছিল?'
    """

    if not _is_event_query(query):
        return 0.0

    title_lower = _safe_text(title).lower()
    content_lower = _safe_text(content).lower()

    narrow_markers = [
        "cricket",
        "football",
        "match",
        "tournament",
        "movie",
        "film",
        "actor",
        "actress",
        "recipe",
        "stock",
        "share price",
        "product",
        "shopping",
        "single",
        "rath yatra",
        "রথযাত্রা",
        "ক্রিকেট",
        "ফুটবল",
        "ম্যাচ",
        "সিনেমা",
        "অভিনেতা",
        "অভিনেত্রী",
        "क्रिकेट",
        "फुटबॉल",
        "फिल्म",
        "अभिनेता",
    ]

    hits = sum(
        1
        for marker in narrow_markers
        if marker in title_lower
    )

    # Stronger penalty if title itself is clearly narrow.
    if hits >= 2:
        return 0.30

    if hits == 1:
        return 0.18

    return 0.0


def _future_relevance_score(query, title, content, years):
    if not years:
        return 0.0

    future_years = [
        year
        for year in years
        if _is_future_year(year)
    ]

    if not future_years:
        return 0.0

    text_title = _safe_text(title).lower()
    text_content = _safe_text(content).lower()

    score = 0.0

    future_markers = [
        "planned",
        "plan",
        "scheduled",
        "schedule",
        "upcoming",
        "expected",
        "forecast",
        "forecasted",
        "prediction",
        "predicted",
        "will",
        "could",
        "may",
        "might",
        "likely",
        "potential",
        "census",
        "economy",

        "পরিকল্পনা",
        "পরিকল্পিত",
        "নির্ধারিত",
        "আসন্ন",
        "সম্ভাবনা",
        "সম্ভাব্য",
        "হতে পারে",
        "জনগণনা",
        "অর্থনীতি",

        "योजना",
        "नियोजित",
        "निर्धारित",
        "आगामी",
        "संभावना",
        "हो सकता है",
        "जनगणना",
        "अर्थव्यवस्था",
    ]

    for marker in future_markers:
        if marker in text_title:
            score += 0.15
        elif marker in text_content:
            score += 0.04

    return min(1.0, score)


# ============================================================
# RESULT CLEANING
# ============================================================

def _clean_result(result, requested_years=None):
    if not isinstance(result, dict):
        return None

    title = _normalise_spaces(
        result.get("title")
        or result.get("name")
        or ""
    )

    url = _safe_text(
        result.get("url")
        or result.get("link")
        or ""
    )

    content = _normalise_spaces(
        result.get("content")
        or result.get("snippet")
        or result.get("text")
        or ""
    )

    if not title and not url and not content:
        return None

    if not url.startswith(("http://", "https://")):
        return None

    # Avoid enormous payloads.
    title = title[:500]
    content = content[:5000]

    source = _domain_from_url(url)

    cleaned = {
        "title": title,
        "url": url,
        "content": content,
        "source": source,
    }

    if requested_years:
        cleaned["_year_score"] = _year_focus_score(
            requested_years,
            title,
            content
        )

    return cleaned


def _dedupe_results(results):
    output = []
    seen_urls = set()
    seen_titles = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        url = _safe_text(item.get("url")).rstrip("/")
        title = _normalise_spaces(
            item.get("title", "")
        ).lower()

        if not url and not title:
            continue

        if url in seen_urls:
            continue

        if title and title in seen_titles:
            continue

        seen_urls.add(url)

        if title:
            seen_titles.add(title)

        output.append(item)

    return output


# ============================================================
# QUERY VARIANTS
# ============================================================

def _build_query_variants(query, entities, years):
    variants = []

    query = _clean_query(query)

    if query:
        variants.append(query)

    # Explicit year + entity queries.
    if years and entities:
        for entity in entities:
            canonical = entity

            for year in years:
                variants.extend([
                    f"{canonical} {year} major events",
                    f"{canonical} {year} important events",
                    f"{canonical} {year} history events",
                    f"{canonical} {year} timeline",
                ])

    # Explicit year without entity.
    elif years:
        for year in years:
            variants.extend([
                f"{year} major events",
                f"{year} important events",
                f"{year} events timeline",
                f"{year} year in review",
            ])

    # Future year.
    if years and any(_is_future_year(y) for y in years):
        for entity in entities:
            for year in years:
                if _is_future_year(year):
                    variants.extend([
                        f"{entity} {year} planned events",
                        f"{entity} {year} scheduled events",
                        f"{entity} {year} forecast",
                        f"{entity} {year} expected developments",
                    ])

    # Remove duplicates while preserving order.
    output = []
    seen = set()

    for item in variants:
        item = _clean_query(item)

        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output[:12]


# ============================================================
# TAVILY
# ============================================================

def _tavily_search(
    query,
    max_results=DEFAULT_MAX_RESULTS,
    search_depth="advanced",
    topic=None,
    time_range=None,
):
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY is not configured.")
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": _clean_query(query),
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    if topic:
        payload["topic"] = topic

    if time_range:
        payload["time_range"] = time_range

    try:
        response = requests.post(
            TAVILY_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not isinstance(results, list):
            return []

        return results

    except requests.RequestException as exc:
        logger.warning(
            "Tavily request failed: %s",
            exc
        )
        return []

    except Exception as exc:
        logger.exception(
            "Unexpected Tavily error: %s",
            exc
        )
        return []


# ============================================================
# YEAR / ENTITY FILTERING
# ============================================================

def _filter_year_entity_results(
    results,
    years,
    entities,
):
    if not years:
        return results

    matched = []
    fallback = []

    for item in results:
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        year_score = _year_focus_score(
            years,
            title,
            content
        )

        entity_score = _entity_relevance_score(
            entities,
            title,
            content
        )

        low_value = _is_low_value_url(url)

        # Explicit year + entity:
        # require meaningful evidence whenever possible.
        if entities:
            if year_score >= 0.65 and entity_score >= 0.60:
                matched.append(item)
            elif year_score >= 0.85:
                # Keep very strong year matches as fallback.
                fallback.append(item)
        else:
            if year_score >= 0.65:
                matched.append(item)
            else:
                fallback.append(item)

        # Never allow social/video pages into the preferred pool.
        if low_value:
            if item in matched:
                matched.remove(item)

            if item in fallback:
                fallback.remove(item)

    # Strong matched results first.
    if matched:
        return matched + fallback

    return fallback


# ============================================================
# RANKING
# ============================================================

def _rank_results(
    results,
    query,
    years=None,
    entities=None,
):
    years = years or []
    entities = entities or []

    ranked = []

    year_query = bool(years)

    for item in results:
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        year_score = _year_focus_score(
            years,
            title,
            content
        )

        entity_score = _entity_relevance_score(
            entities,
            title,
            content
        )

        keyword_score = _keyword_relevance(
            title,
            content,
            query
        )

        title_score = _title_score(
            title,
            query
        )

        source_score = _source_quality(url)

        topic_score = _topic_relevance_score(
            query,
            title,
            content
        )

        narrow_penalty = _narrow_result_penalty(
            query,
            title,
            content
        )

        future_score = _future_relevance_score(
            query,
            title,
            content,
            years
        )

        low_value_penalty = (
            0.50
            if _is_low_value_url(url)
            else 0.0
        )

        generic_penalty = (
            0.12
            if _is_generic_page(url)
            else 0.0
        )

        competing_year_penalty = _competing_year_penalty(
            years,
            title,
            content
        )

        if year_query:
            # Year-specific searches should be dominated by
            # exact year focus and entity relevance.
            score = (
                year_score * 0.34
                + entity_score * 0.24
                + topic_score * 0.16
                + keyword_score * 0.10
                + title_score * 0.06
                + source_score * 0.10
                + future_score * 0.10
                - narrow_penalty * 0.12
                - low_value_penalty
                - generic_penalty
                - competing_year_penalty * 0.40
            )

        else:
            score = (
                keyword_score * 0.35
                + title_score * 0.20
                + topic_score * 0.15
                + source_score * 0.15
                + entity_score * 0.10
                + future_score * 0.05
                - low_value_penalty
                - generic_penalty
            )

        item["_score"] = round(score, 6)
        item["_year_score"] = round(year_score, 6)
        item["_entity_score"] = round(entity_score, 6)

        ranked.append(item)

    ranked.sort(
        key=lambda x: x.get("_score", 0),
        reverse=True
    )

    return ranked


# ============================================================
# REMOVE INTERNAL FIELDS
# ============================================================

def _strip_internal_fields(results):
    clean = []

    for item in results:
        clean.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "source": item.get("source", ""),
        })

    return clean


# ============================================================
# MAIN SEARCH FUNCTION
# ============================================================

def search_web(query, max_results=DEFAULT_MAX_RESULTS):
    """
    Main web-search entry point used by Hello AI.

    Features:
    - Tavily search
    - multilingual queries
    - explicit year detection
    - historical/future year handling
    - entity/country anchoring
    - year-focus scoring
    - competing-year penalty
    - broad-event ranking
    - low-value domain filtering
    - source-quality ranking
    - deduplication
    """

    query = _clean_query(query)

    if not query:
        return []

    try:
        max_results = int(max_results)
    except Exception:
        max_results = DEFAULT_MAX_RESULTS

    max_results = max(
        1,
        min(max_results, 10)
    )

    years = _extract_years(query)
    entities = _detect_query_entities(query)

    current_query = _is_current_query(query)
    news_query = _is_news_query(query)

    year_query = bool(years)
    future_query = any(
        _is_future_year(year)
        for year in years
    )

    variants = _build_query_variants(
        query,
        entities,
        years
    )

    # Always search the user's exact query first.
    search_queries = variants[:]

    # Safety fallback.
    if not search_queries:
        search_queries = [query]

    all_results = []

    for index, search_query in enumerate(search_queries):
        # More results from first few variants.
        variant_limit = max(
            5,
            min(8, max_results + 2)
        )

        # Explicit year searches must NOT use current freshness.
        if year_query:
            topic = None
            time_range = None
        else:
            topic = "news" if news_query else None

            if current_query:
                time_range = "week"
            else:
                time_range = None

        # First query gets advanced search.
        depth = "advanced" if index < 4 else "basic"

        results = _tavily_search(
            search_query,
            max_results=variant_limit,
            search_depth=depth,
            topic=topic,
            time_range=time_range,
        )

        all_results.extend(results)

        # Avoid excessive API calls once enough results exist.
        if len(all_results) >= max_results * 4:
            break

    # Clean.
    cleaned = []

    for result in all_results:
        item = _clean_result(
            result,
            requested_years=years
        )

        if item:
            cleaned.append(item)

    # Deduplicate.
    cleaned = _dedupe_results(cleaned)

    if not cleaned:
        return []

    # Explicit year/entity filtering.
    if year_query:
        cleaned = _filter_year_entity_results(
            cleaned,
            years,
            entities
        )

    # Rank.
    ranked = _rank_results(
        cleaned,
        query,
        years=years,
        entities=entities,
    )

    # --------------------------------------------------------
    # LOW-VALUE DOMAIN RULE
    #
    # If at least 3 good results exist, remove YouTube/social
    # results completely for explicit-year queries.
    # --------------------------------------------------------

    if year_query:
        good_results = [
            item
            for item in ranked
            if not _is_low_value_url(
                item.get("url", "")
            )
        ]

        if len(good_results) >= 3:
            ranked = good_results

    # --------------------------------------------------------
    # Future broad question:
    # avoid generic holiday/calendar pages unless there are
    # too few useful results.
    # --------------------------------------------------------

    if future_query and _is_event_query(query):
        useful_future = []

        for item in ranked:
            title = item.get("title", "").lower()
            content = item.get("content", "").lower()

            strong_future_markers = [
                "planned",
                "scheduled",
                "census",
                "forecast",
                "expected",
                "economy",
                "will",
                "could",
                "may",
                "পরিকল্পনা",
                "নির্ধারিত",
                "জনগণনা",
                "সম্ভাবনা",
                "অর্থনীতি",
                "योजना",
                "निर्धारित",
                "जनगणना",
                "संभावना",
            ]

            marker_hits = sum(
                1
                for marker in strong_future_markers
                if marker in title or marker in content
            )

            if marker_hits > 0:
                useful_future.append(item)

        if len(useful_future) >= 3:
            ranked = useful_future

    # Final limit.
    ranked = ranked[:max_results]

    return _strip_internal_fields(ranked)


# ============================================================
# COMPATIBILITY HELPER
# ============================================================

def should_search_web(query):
    """
    Compatibility helper used by router/app code.

    Returns True for:
    - explicit years
    - current/latest questions
    - news
    - events
    - source requests
    - location/address/PIN/route questions that benefit
      from live web evidence
    """

    text = _safe_text(query).lower()

    if not text:
        return False

    if _has_specific_year(text):
        return True

    if _is_current_query(text):
        return True

    if _is_news_query(text):
        return True

    web_keywords = [
        "source",
        "sources",
        "according to",
        "reference",
        "official",
        "address",
        "location",
        "where",
        "pin code",
        "pincode",
        "postal code",
        "route",
        "road",
        "how to go",
        "train",
        "bus",
        "station",
        "weather",
        "price",
        "cost",
        "today",
        "tomorrow",

        "উৎস",
        "সূত্র",
        "ঠিকানা",
        "কোথায়",
        "কোথায়",
        "পিন কোড",
        "রুট",
        "রাস্তা",
        "কীভাবে যাব",
        "কিভাবে যাব",
        "ট্রেন",
        "বাস",
        "স্টেশন",
        "দাম",
        "মূল্য",
        "আবহাওয়া",
        "আবহাওয়া",

        "पता",
        "स्थान",
        "कहाँ",
        "पिन कोड",
        "रास्ता",
        "कैसे जाएं",
        "ट्रेन",
        "बस",
        "स्टेशन",
        "मौसम",
        "कीमत",
    ]

    return any(
        keyword in text
        for keyword in web_keywords
    )


__all__ = [
    "search_web",
    "should_search_web",
]