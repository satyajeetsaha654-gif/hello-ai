# core/response.py
"""
Hello AI - Response Utilities

কাজ:
- AI response পরিষ্কার করা
- অস্বাভাবিক repeated output আটকানো
- Empty/broken response-এর জন্য language-aware fallback দেওয়া
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def _normalize_unicode(text: str) -> str:
    """Unicode normalization করে, কিন্তু Bengali/Hindi text নষ্ট করে না."""
    return unicodedata.normalize("NFKC", text)


def _remove_control_characters(text: str) -> str:
    """
    Dangerous/control characters সরায়।
    Newline, tab এবং carriage return রাখা হয়।
    """
    return "".join(
        ch
        for ch in text
        if ch in ("\n", "\r", "\t")
        or not unicodedata.category(ch).startswith("C")
    )


def _collapse_excessive_spaces(text: str) -> str:
    """একই line-এর অপ্রয়োজনীয় বহু space কমায়."""
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]{2,}", " ", line)
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines)


def _collapse_excessive_blank_lines(text: str) -> str:
    """৩ বা তার বেশি blank line → সর্বোচ্চ ২টি."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ============================================================
# REPETITION DETECTION
# ============================================================

def _has_repeated_characters(text: str, threshold: int = 12) -> bool:
    """
    একই character অস্বাভাবিকভাবে repeated হয়েছে কি না।
    """

    if not text:
        return False

    return bool(
        re.search(
            rf"(.)\1{{{threshold - 1},}}",
            text,
            flags=re.DOTALL,
        )
    )


def _has_repeated_punctuation(text: str, threshold: int = 10) -> bool:
    """
    একই punctuation অনেকবার repeated হলে reject করবে।
    """

    if not text:
        return False

    return bool(
        re.search(
            rf"([!?.।,:;_\-])\1{{{threshold - 1},}}",
            text,
        )
    )


def _normalise_token_for_comparison(token: str) -> str:
    token = token.strip().casefold()
    token = re.sub(
        r"^[^\w\u0980-\u09ff\u0900-\u097f]+",
        "",
        token,
    )
    token = re.sub(
        r"[^\w\u0980-\u09ff\u0900-\u097f]+$",
        "",
        token,
    )
    return token


def _has_pathological_token_repetition(text: str) -> bool:
    """
    একটি ছোট token অস্বাভাবিকভাবে বারবার এসেছে কি না।
    """

    tokens = re.findall(
        r"[\w\u0980-\u09ff\u0900-\u097f]+",
        text,
        flags=re.UNICODE,
    )

    if len(tokens) < 10:
        return False

    previous = None
    count = 0

    for raw_token in tokens:
        token = _normalise_token_for_comparison(raw_token)

        if not token:
            continue

        if token == previous:
            count += 1
        else:
            previous = token
            count = 1

        if count >= 10:
            return True

    return False


def _has_repeated_lines(text: str) -> bool:
    """
    একই পুরো line অস্বাভাবিকভাবে repeated হয়েছে কি না।
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        return False

    previous = None
    count = 0

    for line in lines:
        if line == previous:
            count += 1
        else:
            previous = line
            count = 1

        if count >= 4:
            return True

    return False


def has_malformed_repetition(text: str) -> bool:
    """
    AI output pathological/repeated/malformed কি না।

    True হলে caller-এর উচিত response সরাসরি user-কে না দেখানো।
    """

    if not isinstance(text, str):
        return True

    value = text.strip()

    if not value:
        return True

    if _has_repeated_characters(value):
        return True

    if _has_repeated_punctuation(value):
        return True

    if _has_pathological_token_repetition(value):
        return True

    if _has_repeated_lines(value):
        return True

    return False


# ============================================================
# MARKDOWN / RESPONSE CLEANUP
# ============================================================

def _remove_accidental_wrappers(text: str) -> str:
    """
    কিছু provider মাঝে মাঝে response-এর চারপাশে unnecessary
    triple backticks বা quotes বসাতে পারে।
    """

    value = text.strip()

    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()

        if len(lines) >= 2:
            first = lines[0].strip()

            if first.startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            value = "\n".join(lines).strip()

    return value


def _remove_leading_assistant_labels(text: str) -> str:
    """
    Provider যদি 'Assistant:' বা 'AI:' লিখে দেয়,
    unnecessary label সরায়।
    """

    patterns = (
        r"^\s*assistant\s*:\s*",
        r"^\s*ai\s*:\s*",
        r"^\s*hello\s*ai\s*:\s*",
    )

    result = text

    for pattern in patterns:
        result = re.sub(
            pattern,
            "",
            result,
            flags=re.IGNORECASE,
        )

    return result.strip()


def _user_wants_links(question: str) -> bool:
    """
    User explicitly link / URL / website / source চাইছে কি না।

    গুরুত্বপূর্ণ:
    সাধারণ কথোপকথনে link allow করা হবে না।
    শুধুমাত্র user-এর প্রশ্নে explicit link intent থাকলে
    AI response-এর URL রাখা হবে।
    """

    if not isinstance(question, str):
        return False

    q = question.casefold().strip()

    patterns = (
        # Bengali — general
        "লিংক দাও",
        "লিংকটা দাও",
        "লিংক দিন",
        "লিংক চাই",
        "লিংকটা চাই",
        "ওয়েবসাইট দাও",
        "ওয়েবসাইট দাও",
        "ওয়েবসাইটের লিংক",
        "ওয়েবসাইটের লিংক",
        "ওয়েবসাইটটা কী",
        "ওয়েবসাইটটা কী",
        "url দাও",
        "url দিন",
        "url চাই",
        "source দাও",
        "source দিন",
        "source দেখাও",
        "সোর্স দাও",
        "সোর্স দিন",
        "সোর্স দেখাও",
        "অফিশিয়াল লিংক",
        "অফিসিয়াল লিংক",

        # Bengali — movie / film / video / YouTube
        "মুভির লিংক",
        "মুভিটার লিংক",
        "মুভি লিংক",
        "সিনেমার লিংক",
        "সিনেমাটা লিংক",
        "সিনেমার লিংক দাও",
        "এই মুভির লিংক",
        "এই মুভিটার লিংক",
        "এই সিনেমার লিংক",
        "ইউটিউব লিংক",
        "ইউটিউব থেকে লিংক",
        "ইউটিউবের লিংক",
        "ভিডিওর লিংক",
        "ভিডিও লিংক",
        "এই ভিডিওর লিংক",

        # Bengali — news
        "খবরের লিংক",
        "খবরটা লিংক",
        "খবরটির লিংক",
        "নিউজের লিংক",
        "নিউজটা লিংক",
        "এই খবরের লিংক",
        "এই খবরটার লিংক",
        "এই খবরটির লিংক",
        "এই নিউজের লিংক",
        "এই নিউজটার লিংক",

        # English — general
        "official link",
        "give me the link",
        "send me the link",
        "give me the url",
        "give me the website",
        "website link",
        "official website",
        "source link",

        # English — movie / film / video / YouTube
        "movie link",
        "film link",
        "youtube link",
        "youtube url",
        "youtube video link",
        "video link",
        "link to this movie",
        "link for this movie",
        "link to the movie",
        "link to this film",
        "link to this video",

        # English — news
        "news link",
        "link to this news",
        "link to the news",
        "link to this story",
        "link to this article",

        # Hindi
        "लिंक दो",
        "लिंक चाहिए",
        "लिंक भेजो",
        "यूट्यूब लिंक",
        "यूट्यूब से लिंक",
        "फिल्म का लिंक",
        "मूवी का लिंक",
        "वीडियो का लिंक",
        "खबर का लिंक",
        "न्यूज़ का लिंक",
    )

    return any(pattern in q for pattern in patterns)


def _remove_links(text: str) -> str:
    """সাধারণ উত্তরে Markdown/raw URL সরায়."""

    # Markdown links: [text](https://...)
    text = re.sub(
        r"\[([^\]]+)\]\(\s*https?://[^)\s]+\s*\)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Raw URLs
    text = re.sub(
        r"https?://[^\s<>\])}]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\(\s*\)",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# PUBLIC CLEANER
# ============================================================

def clean_response(
    response: Optional[str],
    *,
    question: str = "",
    max_length: int = 12000,
) -> str:
    """
    AI-generated response safely clean করে।

    Important:
    - Bengali/Hindi Unicode preserve করে
    - Markdown preserve করে
    - অস্বাভাবিক repetition হলে empty string return করে
    - খুব বড় response সীমিত করে
    """

    if response is None:
        return ""

    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return ""

    value = response.strip()

    if not value:
        return ""

    value = _normalize_unicode(value)
    value = _remove_control_characters(value)
    value = _remove_accidental_wrappers(value)

    # User explicitly link/source না চাইলে
    # AI-এর দেওয়া accidental URL সরিয়ে দেওয়া হবে।
    if not _user_wants_links(question):
        value = _remove_links(value)

    value = _remove_leading_assistant_labels(value)
    value = _collapse_excessive_spaces(value)
    value = _collapse_excessive_blank_lines(value)
    value = value.strip()

    if not value:
        return ""

    # Pathological output user-কে দেখাব না।
    if has_malformed_repetition(value):
        return ""

    # Final safety limit.
    if len(value) > max_length:
        value = value[:max_length].rstrip()

        if " " in value:
            value = value.rsplit(" ", 1)[0].rstrip()

        value += "…"

    return value


# ============================================================
# FALLBACK RESPONSES
# ============================================================

def fallback_response(
    language: str = "en",
    *,
    reason: Optional[str] = None,
) -> str:
    """
    Friendly fallback.

    reason optional; internal caller চাইলে ব্যবহার করতে পারে।
    User-কে technical traceback দেখানো হবে না।
    """

    lang = (language or "en").casefold().strip()

    # Bengali
    if lang in {"bn", "bengali", "bangla"}:
        if reason == "rate_limit":
            return (
                "এই মুহূর্তে AI service-এর সীমা সাময়িকভাবে পূর্ণ হয়েছে। "
                "একটু পরে আবার চেষ্টা করুন।"
            )

        if reason == "unavailable":
            return (
                "এই মুহূর্তে AI service-এ সাময়িক সমস্যা হচ্ছে। "
                "কিছুক্ষণ পরে আবার চেষ্টা করুন।"
            )

        if reason == "malformed":
            return (
                "উত্তরটি ঠিকভাবে তৈরি হয়নি। "
                "দয়া করে একই প্রশ্নটি আবার করুন।"
            )

        return (
            "দুঃখিত, এই মুহূর্তে উত্তরটি তৈরি করা যাচ্ছে না। "
            "দয়া করে আবার চেষ্টা করুন।"
        )

    # Hindi
    if lang in {"hi", "hindi"}:
        if reason == "rate_limit":
            return (
                "इस समय AI service की सीमा अस्थायी रूप से पूरी हो गई है। "
                "थोड़ी देर बाद फिर कोशिश करें।"
            )

        if reason == "unavailable":
            return (
                "इस समय AI service में अस्थायी समस्या है। "
                "कुछ देर बाद फिर कोशिश करें।"
            )

        if reason == "malformed":
            return (
                "उत्तर सही तरीके से तैयार नहीं हुआ। "
                "कृपया वही सवाल फिर से पूछें।"
            )

        return (
            "माफ़ कीजिए, इस समय उत्तर तैयार नहीं किया जा सकता। "
            "कृपया फिर कोशिश करें।"
        )

    # English
    if reason == "rate_limit":
        return (
            "The AI service is temporarily at its usage limit. "
            "Please try again in a little while."
        )

    if reason == "unavailable":
        return (
            "The AI service is temporarily unavailable. "
            "Please try again in a little while."
        )

    if reason == "malformed":
        return (
            "The response was not generated correctly. "
            "Please ask the same question again."
        )

    return (
        "Sorry, I couldn't generate a reliable answer right now. "
        "Please try again."
    )


# ============================================================
# LANGUAGE-AWARE SAFE RESPONSE
# ============================================================

def safe_response(
    response: Optional[str],
    language: str = "en",
    *,
    question: str = "",
) -> str:
    """
    Response clean করার পরে empty/malformed হলে
    appropriate fallback দেয়।

    question optional রাখা হয়েছে যাতে explicit link request
    হলে safe_response() থেকেও requested links preserve হয়।
    """

    cleaned = clean_response(
        response,
        question=question,
    )

    if cleaned:
        return cleaned

    return fallback_response(
        language,
        reason="malformed",
    )


__all__ = [
    "clean_response",
    "fallback_response",
    "safe_response",
    "has_malformed_repetition",
]