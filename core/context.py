"""
Hello AI - Conversation Context

কাজ:
- Conversation history পরিষ্কার করা
- সর্বশেষ ২০টি message রাখা
- Follow-up question বুঝতে সাহায্য করা
- আগের conversation থেকে place/topic/reference resolve করা
- আগের news item থেকে news follow-up reference resolve করা
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================
# LIMITS
# ============================================================

MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 12000
MAX_TOTAL_HISTORY_CHARS = 60000


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_role(role: Any) -> str:
    if not isinstance(role, str):
        return ""

    role = role.strip().casefold()

    if role in {"user", "assistant"}:
        return role

    return ""


def _clean_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, (int, float, bool)):
        return str(content).strip()

    try:
        return str(content).strip()
    except Exception:
        return ""


def _normalise_message(
    message: Any,
) -> dict[str, str] | None:

    if not isinstance(message, dict):
        return None

    role = _clean_role(
        message.get("role")
    )

    content = _clean_content(
        message.get("content")
    )

    if not role or not content:
        return None

    if len(content) > MAX_MESSAGE_LENGTH:
        content = (
            content[:MAX_MESSAGE_LENGTH]
            .rstrip()
            + "…"
        )

    return {
        "role": role,
        "content": content,
    }


# ============================================================
# HISTORY NORMALIZATION
# ============================================================

def normalize_history(
    history: Any,
    *,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_message_length: int = MAX_MESSAGE_LENGTH,
    max_total_chars: int = MAX_TOTAL_HISTORY_CHARS,
) -> list[dict[str, str]]:

    if not isinstance(history, list):
        return []

    try:
        max_messages = max(
            1,
            int(max_messages),
        )
    except Exception:
        max_messages = MAX_HISTORY_MESSAGES

    try:
        max_message_length = max(
            100,
            int(max_message_length),
        )
    except Exception:
        max_message_length = MAX_MESSAGE_LENGTH

    try:
        max_total_chars = max(
            1000,
            int(max_total_chars),
        )
    except Exception:
        max_total_chars = MAX_TOTAL_HISTORY_CHARS

    normalized: list[dict[str, str]] = []

    for raw_message in history:
        message = _normalise_message(
            raw_message
        )

        if message is None:
            continue

        if len(message["content"]) > max_message_length:
            message["content"] = (
                message["content"][
                    :max_message_length
                ]
                .rstrip()
                + "…"
            )

        normalized.append(message)

    if not normalized:
        return []

    if len(normalized) > max_messages:
        normalized = normalized[-max_messages:]

    while (
        normalized
        and normalized[0]["role"] == "assistant"
    ):
        normalized.pop(0)

    if not normalized:
        return []

    result: list[dict[str, str]] = []
    total_chars = 0

    for message in reversed(normalized):
        size = len(message["content"])

        if (
            total_chars + size
            > max_total_chars
        ):
            break

        result.append(message)
        total_chars += size

    result.reverse()

    while (
        result
        and result[0]["role"] == "assistant"
    ):
        result.pop(0)

    return result


# ============================================================
# HISTORY HELPERS
# ============================================================

def append_message(
    history: Any,
    role: str,
    content: Any,
) -> list[dict[str, str]]:

    normalized = normalize_history(history)

    message = _normalise_message(
        {
            "role": role,
            "content": content,
        }
    )

    if message is None:
        return normalized

    normalized.append(message)

    return normalize_history(normalized)


def last_user_message(
    history: Any,
) -> str:

    normalized = normalize_history(history)

    for message in reversed(normalized):
        if message["role"] == "user":
            return message["content"]

    return ""


def last_assistant_message(
    history: Any,
) -> str:

    normalized = normalize_history(history)

    for message in reversed(normalized):
        if message["role"] == "assistant":
            return message["content"]

    return ""


def history_has_user_message(
    history: Any,
) -> bool:
    return bool(
        last_user_message(history)
    )


def history_has_assistant_message(
    history: Any,
) -> bool:
    return bool(
        last_assistant_message(history)
    )


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

FOLLOWUP_PATTERNS = (
    # Bengali
    "আরও বল",
    "আরো বল",
    "আরও বিস্তারিত",
    "আরো বিস্তারিত",
    "বিস্তারিত বল",
    "এটা কেন",
    "এটা কীভাবে",
    "ওটা কেন",
    "ওটা কীভাবে",
    "এটা কি",
    "ওটা কি",
    "এটার",
    "ওটার",
    "এখানে",
    "ওখানে",
    "ওখান থেকে",
    "সেখানে",
    "সেখান থেকে",
    "এই জায়গা",
    "এই জায়গা",
    "ওই জায়গা",
    "ওই জায়গা",
    "এইটা",
    "ওইটা",
    "ওটা",
    "এটা",
    "তারপর",
    "তার মানে",
    "মানে কী",
    "আর কী",
    "আর কি",
    "লিংকটা দাও",
    "লিংকটা দিন",
    "লিংক দাও",
    "লিংক দিন",
    "ওটার লিংক",
    "এটার লিংক",
    "কতক্ষণ লাগবে",
    "কত সময় লাগবে",
    "কত সময় লাগবে",
    "তারপর কী",
    "তারপর কি",

    # News follow-up — Bengali
    "এই খবর",
    "ওই খবর",
    "এই নিউজ",
    "ওই নিউজ",
    "এই সংবাদ",
    "ওই সংবাদ",
    "খবরটা",
    "খবরটি",
    "নিউজটা",
    "নিউজটি",
    "সংবাদটা",
    "সংবাদটি",

    # Roman Bengali
    "aro bolo",
    "aro bistarito",
    "bistarito bolo",
    "eta keno",
    "eta kivabe",
    "ota keno",
    "ota kivabe",
    "etar",
    "otar",
    "ekhane",
    "okhane",
    "okhan theke",
    "sekhane",
    "sekhan theke",
    "ei jayga",
    "ei jagay",
    "oi jayga",
    "oi jagay",
    "eta",
    "ota",
    "link ta dao",
    "link dao",
    "otar link",
    "etar link",
    "koto khon lagbe",
    "koto somoy lagbe",
    "tarpor ki",
    "tarpor",

    # Roman Bengali — news
    "ei khobor",
    "oi khobor",
    "ei news",
    "oi news",
    "ei songbad",
    "oi songbad",
    "khoborta",
    "khoborti",
    "newsta",
    "newsti",

    # English
    "same",
    "more",
    "explain more",
    "tell me more",
    "why",
    "how",
    "then",
    "what about that",
    "what about it",
    "more details",
    "there",
    "here",
    "that place",
    "this place",
    "from there",
    "its",
    "it",
    "this news",
    "that news",
    "this story",
    "that story",
    "this article",
    "that article",
    "the news",
    "the story",
    "give me the link",
    "send the link",
    "link please",
    "how long",
    "what next",
    "then what",

    # Hindi
    "और बताओ",
    "और बताइए",
    "विस्तार से बताओ",
    "फिर",
    "क्यों",
    "कैसे",
    "यह",
    "वह",
    "वहाँ",
    "वहां",
    "उसका",
    "इसका",
    "लिंक दो",
    "लिंक दीजिए",
    "कितना समय लगेगा",
    "फिर क्या",

    # Hindi — news
    "यह खबर",
    "वह खबर",
    "इस खबर",
    "उस खबर",
    "यह न्यूज़",
    "वह न्यूज़",
    "इस न्यूज़",
    "उस न्यूज़",
)


def is_likely_followup(
    question: Any,
    history: Any,
) -> bool:

    if not isinstance(question, str):
        return False

    value = question.strip().casefold()

    if not value:
        return False

    if not history_has_user_message(history):
        return False

    compact = " ".join(
        value.split()
    )

    # --------------------------------------------------------
    # Explicit follow-up patterns
    # --------------------------------------------------------

    if any(
        pattern.casefold() in compact
        for pattern in FOLLOWUP_PATTERNS
    ):
        return True

    # --------------------------------------------------------
    # Strong reference / deictic indicators
    #
    # These clearly point back to something already discussed.
    # --------------------------------------------------------

    reference_patterns = (
        # Bengali
        r"(?<!\S)এটা(?!\S)",
        r"(?<!\S)ওটা(?!\S)",
        r"(?<!\S)এইটা(?!\S)",
        r"(?<!\S)ওইটা(?!\S)",
        r"(?<!\S)এটার(?!\S)",
        r"(?<!\S)ওটার(?!\S)",
        r"(?<!\S)এর(?!\S)",
        r"(?<!\S)ওর(?!\S)",
        r"(?<!\S)তার(?!\S)",
        r"(?<!\S)এখানে(?!\S)",
        r"(?<!\S)ওখানে(?!\S)",
        r"(?<!\S)সেখানে(?!\S)",
        r"ওখান থেকে",
        r"সেখান থেকে",
        r"এই লিংক",
        r"ওই লিংক",
        r"এই ভিডিও",
        r"ওই ভিডিও",
        r"এই গান",
        r"ওই গান",
        r"এই মুভি",
        r"ওই মুভি",
        r"এই খবর",
        r"ওই খবর",
        r"এই নিউজ",
        r"ওই নিউজ",

        # Roman Bengali
        r"(?<!\S)eta(?!\S)",
        r"(?<!\S)ota(?!\S)",
        r"(?<!\S)etar(?!\S)",
        r"(?<!\S)otar(?!\S)",
        r"(?<!\S)er(?!\S)",
        r"(?<!\S)or(?!\S)",
        r"(?<!\S)tar(?!\S)",
        r"(?<!\S)eikhane(?!\S)",
        r"(?<!\S)okhane(?!\S)",
        r"(?<!\S)sekhane(?!\S)",
        r"okhan theke",
        r"sekhan theke",
        r"ei link",
        r"oi link",
        r"ei video",
        r"oi video",
        r"ei gan",
        r"oi gan",
        r"ei movie",
        r"oi movie",

        # English
        r"(?<!\S)this(?!\S)",
        r"(?<!\S)that(?!\S)",
        r"(?<!\S)it(?!\S)",
        r"(?<!\S)its(?!\S)",
        r"(?<!\S)there(?!\S)",
        r"(?<!\S)here(?!\S)",
        r"that place",
        r"this place",
        r"that link",
        r"this link",
        r"that video",
        r"this video",
        r"that song",
        r"this song",
        r"that movie",
        r"this movie",
        r"that news",
        r"this news",

        # Hindi
        r"(?<!\S)यह(?!\S)",
        r"(?<!\S)वह(?!\S)",
        r"(?<!\S)इसका(?!\S)",
        r"(?<!\S)उसका(?!\S)",
        r"(?<!\S)इसकी(?!\S)",
        r"(?<!\S)उसकी(?!\S)",
        r"(?<!\S)यहाँ(?!\S)",
        r"(?<!\S)वहाँ(?!\S)",
        r"इस लिंक",
        r"उस लिंक",
        r"इस वीडियो",
        r"उस वीडियो",
        r"इस गाने",
        r"उस गाने",
    )

    if len(compact) <= 160:
        if any(
            re.search(
                pattern,
                compact,
                flags=re.I,
            )
            for pattern in reference_patterns
        ):
            return True

    # --------------------------------------------------------
    # Strong contextual question forms
    #
    # These are follow-ups when they ask about an already
    # mentioned subject without repeating its name.
    # --------------------------------------------------------

    contextual_patterns = (
        # Bengali
        r"^আরও?\b",
        r"^কোনটা\b",
        r"^কোনটি\b",
        r"^কতক্ষণ\b",
        r"^কত সময়\b",
        r"^কত সময়\b",
        r"^কীভাবে\b",
        r"^কিভাবে\b",
        r"^কোথায়\b",
        r"^কোথায়\b",
        r"^তারপর\b",
        r"^লিংক\b",
        r"^লিংকটা\b",
        r"^আরেকটা\b",
        r"^আরও একটা\b",
        r"^ওটার\b",
        r"^এটার\b",
        r"^এর\b",
        r"^ওর\b",
        r"^তার\b",

        # Common Bengali follow-up questions
        r"^এর গায়ক\b",
        r"^এর গায়ক\b",
        r"^এর পরিচালক\b",
        r"^এর অভিনেতা\b",
        r"^এর অভিনেত্রী\b",
        r"^এর নাম\b",
        r"^এর release date\b",
        r"^এর রিলিজ\b",
        r"^এটার নাম\b",
        r"^এটার পরিচালক\b",
        r"^এটার গায়ক\b",
        r"^এটার গায়ক\b",
        r"^ওটার পরিচালক\b",
        r"^ওটার নাম\b",
        r"^ওটার গায়ক\b",
        r"^ওটার গায়ক\b",

        # Roman Bengali
        r"^aro\b",
        r"^konta\b",
        r"^kothay\b",
        r"^kivabe\b",
        r"^koto khon\b",
        r"^koto somoy\b",
        r"^tarpor\b",
        r"^link\b",
        r"^link ta\b",
        r"^arekta\b",
        r"^aro ekta\b",
        r"^etar\b",
        r"^otar\b",
        r"^er\b",
        r"^or\b",
        r"^tar\b",

        # English
        r"^more\b",
        r"^another\b",
        r"^which one\b",
        r"^how long\b",
        r"^what about\b",
        r"^what next\b",
        r"^then\b",
        r"^the link\b",
        r"^its link\b",
        r"^release date\b",
        r"^who is the singer\b",
        r"^who directed\b",
        r"^what is its name\b",

        # Hindi
        r"^और\b",
        r"^और एक\b",
        r"^फिर\b",
        r"^कौन सा\b",
        r"^कितना समय\b",
        r"^कैसे\b",
        r"^कहाँ\b",
        r"^लिंक\b",
        r"^इसका\b",
        r"^उसका\b",
    )

    if len(compact) <= 160:
        if any(
            re.search(
                pattern,
                compact,
                flags=re.I,
            )
            for pattern in contextual_patterns
        ):
            return True

    # --------------------------------------------------------
    # Short contextual questions with a clear question marker
    #
    # Do NOT treat generic "who/what/how/when" alone as a
    # follow-up. That prevents new standalone questions from
    # being incorrectly attached to old conversation context.
    # --------------------------------------------------------

    if len(compact.split()) <= 10:

        short_contextual_patterns = (
            # Bengali
            r"^কতক্ষণ",
            r"^কত সময়",
            r"^কত সময়",
            r"^কোথায়",
            r"^কোথায়",
            r"^কীভাবে",
            r"^কিভাবে",
            r"^তারপর",
            r"^আরেকটা",
            r"^আরও একটা",
            r"^লিংকটা",
            r"^এটার",
            r"^ওটার",
            r"^এর ",
            r"^ওর ",
            r"^তার ",
            r"^এইটা",
            r"^ওইটা",

            # Roman Bengali
            r"^kotokhon",
            r"^koto khon",
            r"^koto somoy",
            r"^kothay",
            r"^kivabe",
            r"^tarpor",
            r"^arekta",
            r"^aro ekta",
            r"^link ta",
            r"^etar",
            r"^otar",
            r"^er ",
            r"^or ",
            r"^tar ",
            r"^eta",
            r"^ota",

            # English
            r"^how long",
            r"^what about",
            r"^what next",
            r"^then\b",
            r"^another\b",
            r"^the link",
            r"^its link",
            r"^this ",
            r"^that ",
            r"^who is the singer",
            r"^who directed",
            r"^release date",

            # Hindi
            r"^कितना समय",
            r"^कहाँ",
            r"^कैसे",
            r"^फिर",
            r"^और एक",
            r"^इसका",
            r"^उसका",
            r"^यह ",
            r"^वह ",
        )

        if any(
            re.search(
                pattern,
                compact,
                flags=re.I,
            )
            for pattern in short_contextual_patterns
        ):
            return True

    return False

# ============================================================
# REFERENCE EXTRACTION
# ============================================================

_REFERENCE_STOP_WORDS = {
    "কোথায়",
    "কোথায়",
    "কী",
    "কি",
    "কেমন",
    "কীভাবে",
    "কিভাবে",
    "কত",
    "কেন",
    "আছে",
    "আসছে",
    "যাব",
    "যাবে",
    "যেতে",
    "থেকে",
    "এখানে",
    "ওখানে",
    "সেখানে",
    "এটার",
    "ওটার",
    "এটা",
    "ওটা",
    "বলো",
    "বলুন",
    "দাও",
    "দেখাও",
}


def _clean_reference(
    value: str,
) -> str:

    value = re.sub(
        r"\s+",
        " ",
        value or "",
    ).strip(" \t\r\n.,!?;:।")

    if not value:
        return ""

    value = re.sub(
        r"^(এই|ওই|সেই|এইটা|ওইটা|ওটা|এটা)\s+",
        "",
        value,
        flags=re.I,
    )

    return value.strip()


def _extract_place_from_text(
    text: str,
) -> str:

    if not isinstance(text, str):
        return ""

    value = text.strip()

    if not value:
        return ""

    patterns = (
        r"(?:নাম|নামটা)\s+(?:কি|কী)\s*(?:হল|হয়|হয়)?\s*$",
        r"(?:কোথায়|কোথায়)\s*(?:আছে|অবস্থিত)?\s*$",
        r"(?:কীভাবে|কিভাবে)\s+যাব.*$",
        r"(?:কি|কী)\s+ভাবে\s+যাব.*$",
        r"(?:এর|র)\s+(?:ঠিকানা|address|পিন|pin|পোস্ট.*)$",
    )

    candidate = value

    for pattern in patterns:
        candidate = re.sub(
            pattern,
            "",
            candidate,
            flags=re.I,
        ).strip()

    candidate = re.sub(
        r"^(বলুন|বলো|আমাকে|দয়া করে|দয়া করে)\s+",
        "",
        candidate,
        flags=re.I,
    ).strip()

    if len(candidate) < 3:
        return ""

    if candidate.casefold() in {
        item.casefold()
        for item in _REFERENCE_STOP_WORDS
    }:
        return ""

    return _clean_reference(candidate)


# ============================================================
# NEWS REFERENCE EXTRACTION
# ============================================================

def _is_news_like_text(text: str) -> bool:
    """
    Text-টি news/headline ধরনের কি না।
    """

    if not isinstance(text, str):
        return False

    value = text.casefold()

    news_words = (
        "খবর",
        "নিউজ",
        "সংবাদ",
        "latest news",
        "breaking news",
        "news",
        "headline",
        "story",
        "article",
        " খবর ",
        " खबर",
        "न्यूज़",
        "समाचार",
    )

    return any(
        word in value
        for word in news_words
    )


def _clean_news_title(value: str) -> str:
    """
    News title থেকে numbering, Markdown এবং source-link
    অংশ বাদ দেয়।
    """

    if not isinstance(value, str):
        return ""

    title = value.strip()

    if not title:
        return ""

    title = re.sub(
        r"\[([^\]]+)\]\(\s*https?://[^)\s]+\s*\)",
        r"\1",
        title,
        flags=re.I,
    )

    title = re.sub(
        r"https?://\S+",
        "",
        title,
        flags=re.I,
    )

    title = re.sub(
        r"^\s*(?:[-•*]\s+|\d+\s*[\.)]\s+|[০-৯]+\s*[\.)]\s+)",
        "",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    ).strip(" \t\r\n-–—:।")

    return title


def _extract_recent_news_reference(
    history: Any,
) -> str:
    """
    সর্বশেষ assistant news answer থেকে সবচেয়ে সম্ভাব্য
    news headline/reference বের করে।

    Source/link line বা সাধারণ introductory sentence
    reference হিসেবে নেওয়া হবে না।
    """

    normalized = normalize_history(
        history,
        max_messages=MAX_HISTORY_MESSAGES,
    )

    for message in reversed(normalized):
        if message["role"] != "assistant":
            continue

        content = message["content"].strip()

        if not content or not _is_news_like_text(content):
            continue

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        candidates: list[str] = []

        for line in lines:
            lowered = line.casefold()

            if (
                "http://" in lowered
                or "https://" in lowered
                or "source:" in lowered
                or "সোর্স:" in lowered
                or "সূত্র:" in lowered
                or "read more" in lowered
                or "আরও পড়ুন" in lowered
                or "আরও পড়ুন" in lowered
            ):
                continue

            cleaned = _clean_news_title(line)

            if not cleaned:
                continue

            if cleaned.casefold() in {
                "আজকের খবর",
                "আজকের সর্বশেষ খবর",
                "latest news",
                "latest news today",
                "today's news",
                "খবর",
                "নিউজ",
                "সংবাদ",
            }:
                continue

            if re.match(
                r"^(?:\d+|[০-৯]+)\s*[\.)]",
                line,
            ):
                candidates.insert(0, cleaned)
                continue

            if len(cleaned) >= 12:
                candidates.append(cleaned)

        if candidates:
            reference = candidates[0]

            if len(reference) <= 300:
                return reference

    return ""


def _extract_recent_reference(
    history: Any,
) -> str:

    normalized = normalize_history(history)

    if normalized:
        recent_user = last_user_message(normalized)
        recent_assistant = last_assistant_message(normalized)

        if (
            _is_news_like_text(recent_user)
            or _is_news_like_text(recent_assistant)
        ):
            news_reference = _extract_recent_news_reference(
                normalized
            )

            if news_reference:
                return news_reference

    for message in reversed(normalized):
        if message["role"] != "user":
            continue

        content = message["content"].strip()

        reference = _extract_place_from_text(
            content
        )

        if reference:
            if len(reference) <= 160:
                return reference

    for message in reversed(normalized):
        if message["role"] != "assistant":
            continue

        content = message["content"].strip()

        match = re.search(
            r"^(.{3,120}?)\s+(?:হলো|হল|হয়|হয়|is|is located|অবস্থিত)",
            content,
            flags=re.I,
        )

        if match:
            reference = _clean_reference(
                match.group(1)
            )

            if reference:
                return reference

    return ""


# ============================================================
# FOLLOW-UP QUESTION RESOLUTION
# ============================================================

def resolve_followup_question(
    question: Any,
    history: Any,
) -> tuple[str, str]:
    """
    Follow-up question-এর জন্য history থেকে আগের reference
    resolve করে।

    সরাসরি pronoun/reference পাওয়া গেলে সেটি replace করা হয়।

    আর reference না পাওয়া গেলেও, প্রশ্নটি যদি conversation-এর
    continuation হয়, তাহলে সাম্প্রতিক conversation context
    effective question-এর মধ্যে দেওয়া হয়।

    Returns:
        (effective_question, reference)
    """

    if not isinstance(question, str):
        return "", ""

    original = question.strip()

    if not original:
        return "", ""

    normalized = normalize_history(history)

    if not normalized:
        return original, ""

    if not is_likely_followup(
        original,
        normalized,
    ):
        return original, ""

    # --------------------------------------------------------
    # NEWS FOLLOW-UP
    # --------------------------------------------------------

    news_deictic_patterns = (
        r"এই\s+খবর(?:টা|টি|টার|টির)?",
        r"ওই\s+খবর(?:টা|টি|টার|টির)?",
        r"এই\s+নিউজ(?:টা|টি|টার|টির)?",
        r"ওই\s+নিউজ(?:টা|টি|টার|টির)?",
        r"এই\s+সংবাদ(?:টা|টি|টার|টির)?",
        r"ওই\s+সংবাদ(?:টা|টি|টার|টির)?",
        r"ei\s+khobor(?:ta|ti|tar|tir)?",
        r"oi\s+khobor(?:ta|ti|tar|tir)?",
        r"ei\s+news(?:ta|ti|tar|tir)?",
        r"oi\s+news(?:ta|ti|tar|tir)?",
        r"this\s+news",
        r"that\s+news",
        r"this\s+story",
        r"that\s+story",
        r"this\s+article",
        r"that\s+article",
        r"এই\s+खबर",
        r"ওই\s+खबर",
        r"यह\s+खबर",
        r"वह\s+खबर",
        r"इस\s+खबर",
        r"उस\s+खबर",
    )

    is_news_followup = any(
        re.search(
            pattern,
            original,
            flags=re.I,
        )
        for pattern in news_deictic_patterns
    )

    if is_news_followup:
        news_reference = _extract_recent_news_reference(
            normalized
        )

        if news_reference:
            resolved_news = original

            news_replacements = (
                (
                    r"এই\s+খবর(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"ওই\s+খবর(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"এই\s+নিউজ(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"ওই\s+নিউজ(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"এই\s+সংবাদ(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"ওই\s+সংবাদ(?:টা|টি|টার|টির)?",
                    news_reference,
                ),
                (
                    r"\bei\s+khobor(?:ta|ti|tar|tir)?\b",
                    news_reference,
                ),
                (
                    r"\boi\s+khobor(?:ta|ti|tar|tir)?\b",
                    news_reference,
                ),
                (
                    r"\bei\s+news(?:ta|ti|tar|tir)?\b",
                    news_reference,
                ),
                (
                    r"\boi\s+news(?:ta|ti|tar|tir)?\b",
                    news_reference,
                ),
                (
                    r"\bthis\s+news\b",
                    news_reference,
                ),
                (
                    r"\bthat\s+news\b",
                    news_reference,
                ),
                (
                    r"\bthis\s+story\b",
                    news_reference,
                ),
                (
                    r"\bthat\s+story\b",
                    news_reference,
                ),
                (
                    r"\bthis\s+article\b",
                    news_reference,
                ),
                (
                    r"\bthat\s+article\b",
                    news_reference,
                ),
                (
                    r"यह\s+खबर",
                    news_reference,
                ),
                (
                    r"वह\s+खबर",
                    news_reference,
                ),
                (
                    r"इस\s+खबर",
                    news_reference,
                ),
                (
                    r"उस\s+खबर",
                    news_reference,
                ),
            )

            for pattern, replacement in news_replacements:
                resolved_news = re.sub(
                    pattern,
                    replacement,
                    resolved_news,
                    flags=re.I,
                )

            resolved_news = re.sub(
                r"\s+",
                " ",
                resolved_news,
            ).strip()

            return resolved_news, news_reference

    # --------------------------------------------------------
    # GENERAL REFERENCE
    # --------------------------------------------------------

    reference = _extract_recent_reference(
        normalized
    )

    if not reference:
        # ----------------------------------------------------
        # IMPORTANT:
        # নির্দিষ্ট reference পাওয়া না গেলেও conversation-এর
        # continuation হলে recent context attach করা হবে।
        #
        # এতে:
        #
        # User:
        # "MrBeast-এর latest video কী?"
        #
        # User:
        # "লিংকটা দাও"
        #
        # অথবা:
        #
        # User:
        # "Kanchrapara থেকে Halisahar কীভাবে যাব?"
        #
        # User:
        # "কতক্ষণ লাগবে?"
        #
        # এই ধরনের follow-up শুধু keyword-এর ওপর নির্ভর করবে না।
        # ----------------------------------------------------

        recent_context = normalized[-8:]

        context_lines: list[str] = []

        for message in recent_context:
            role = message["role"]

            if role == "user":
                label = "User"
            else:
                label = "Assistant"

            content = message["content"].strip()

            if not content:
                continue

            context_lines.append(
                f"{label}: {content}"
            )

        if context_lines:
            conversation_context = "\n".join(
                context_lines
            )

            effective_question = (
                "The user is continuing the previous "
                "conversation.\n\n"
                "Use the following recent conversation "
                "context to understand what the current "
                "question refers to.\n\n"
                f"{conversation_context}\n\n"
                "Current user question:\n"
                f"{original}\n\n"
                "Answer the current question directly. "
                "If the current question refers to something "
                "from the previous conversation, use that "
                "context. Do not repeat the entire "
                "conversation unless necessary."
            )

            return effective_question, ""

        return original, ""

    value = original

    # --------------------------------------------------------
    # Bengali / Roman Bengali deictic replacement
    # --------------------------------------------------------

    replacements = (
        (
            r"\bএই\s+জায়গা(?:টায়|তে|য়|য়|টার|টির|টা)?\b",
            reference,
        ),
        (
            r"\bএই\s+জায়গা(?:টায়|তে|য়|টার|টির|টা)?\b",
            reference,
        ),
        (
            r"\bওই\s+জায়গা(?:টায়|তে|য়|য়|টার|টির|টা)?\b",
            reference,
        ),
        (
            r"\bওই\s+জায়গা(?:টায়|তে|য়|টার|টির|টা)?\b",
            reference,
        ),
        (
            r"\bএইটা\b",
            reference,
        ),
        (
            r"\bওইটা\b",
            reference,
        ),
        (
            r"\bএটা\b",
            reference,
        ),
        (
            r"\bওটা\b",
            reference,
        ),
        (
            r"\bএটার\b",
            f"{reference}-এর",
        ),
        (
            r"\bওটার\b",
            f"{reference}-এর",
        ),
        (
            r"\bএখানে\b",
            f"{reference}-এ",
        ),
        (
            r"\bওখানে\b",
            f"{reference}-এ",
        ),
        (
            r"\bসেখানে\b",
            f"{reference}-এ",
        ),
        (
            r"\bওখান\s+থেকে\b",
            f"{reference} থেকে",
        ),
        (
            r"\bসেখান\s+থেকে\b",
            f"{reference} থেকে",
        ),

        # Roman Bengali
        (
            r"\bei\s+jayga(?:y|te)?\b",
            reference,
        ),
        (
            r"\bei\s+jagay\b",
            reference,
        ),
        (
            r"\boi\s+jayga(?:y|te)?\b",
            reference,
        ),
        (
            r"\beta\b",
            reference,
        ),
        (
            r"\bota\b",
            reference,
        ),
        (
            r"\betar\b",
            f"{reference}-er",
        ),
        (
            r"\botar\b",
            f"{reference}-er",
        ),
        (
            r"\bekhane\b",
            f"{reference}-e",
        ),
        (
            r"\bokhane\b",
            f"{reference}-e",
        ),
        (
            r"\bokhan\s+theke\b",
            f"{reference} theke",
        ),

        # English
        (
            r"\bthat place\b",
            reference,
        ),
        (
            r"\bthis place\b",
            reference,
        ),
        (
            r"\bthere\b",
            f"at {reference}",
        ),
    )

    resolved = value

    for pattern, replacement in replacements:
        resolved = re.sub(
            pattern,
            replacement,
            resolved,
            flags=re.I,
        )

    # --------------------------------------------------------
    # "More" type follow-up
    # --------------------------------------------------------

    if resolved.strip().casefold() in {
        "আরও বল",
        "আরো বল",
        "আরও বিস্তারিত",
        "আরো বিস্তারিত",
        "বিস্তারিত বল",
        "more",
        "more details",
        "tell me more",
        "explain more",
        "और बताओ",
        "विस्तार से बताओ",
    }:
        resolved = (
            f"{reference} সম্পর্কে আরও বিস্তারিত বলো।"
        )

    resolved = re.sub(
        r"\s+",
        " ",
        resolved,
    ).strip()

    return resolved, reference


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_conversation_context(
    history: Any,
    current_question: Any,
) -> list[dict[str, str]]:

    normalized = normalize_history(history)

    if not isinstance(
        current_question,
        str,
    ):
        return normalized

    question = current_question.strip()

    if not question:
        return normalized

    if normalized:
        last = normalized[-1]

        if (
            last["role"] == "user"
            and last["content"].strip()
            == question
        ):
            return normalized

    normalized.append(
        {
            "role": "user",
            "content": question[
                :MAX_MESSAGE_LENGTH
            ].strip(),
        }
    )

    return normalize_history(
        normalized
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "MAX_HISTORY_MESSAGES",
    "MAX_MESSAGE_LENGTH",
    "MAX_TOTAL_HISTORY_CHARS",
    "normalize_history",
    "append_message",
    "last_user_message",
    "last_assistant_message",
    "history_has_user_message",
    "history_has_assistant_message",
    "is_likely_followup",
    "resolve_followup_question",
    "build_conversation_context",
]