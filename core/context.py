"""
Hello AI - Conversation Context

কাজ:
- Conversation history পরিষ্কার করা
- সর্বশেষ ২০টি message রাখা
- Follow-up question বুঝতে সাহায্য করা
- আগের conversation থেকে place/topic/reference resolve করা
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

    # সর্বশেষ message-গুলো রাখি
    if len(normalized) > max_messages:
        normalized = normalized[-max_messages:]

    # Conversation assistant দিয়ে শুরু হলে
    # প্রথম assistant message বাদ।
    while (
        normalized
        and normalized[0]["role"] == "assistant"
    ):
        normalized.pop(0)

    if not normalized:
        return []

    # Total character limit.
    result: list[dict[str, str]] = []
    total_chars = 0

    # Latest 20-এর মধ্যে যতটা সম্ভব context রাখা।
    # প্রয়োজনে পুরোনো দিক থেকে বাদ দেওয়া হবে।
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
    "sekh an theke",
    "ei jayga",
    "ei jagay",
    "oi jayga",
    "oi jagay",
    "eta",
    "ota",

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

    if any(
        pattern in value
        for pattern in FOLLOWUP_PATTERNS
    ):
        return True

    compact = " ".join(
        value.split()
    )

    if len(compact) <= 60:

        pronoun_words = (
            "এটা",
            "ওটা",
            "এটির",
            "ওটির",
            "এখানে",
            "ওখানে",
            "ওখান থেকে",
            "এখান থেকে",
            "তার",
            "সেটা",
            "সেখানে",
            "এইটা",
            "ওইটা",
            "এই জায়গা",
            "এই জায়গা",
            "ওই জায়গা",
            "ওই জায়গা",
            "this",
            "that",
            "it",
            "there",
            "here",
            "that place",
            "this place",
            "from there",
            "यह",
            "वह",
            "वहाँ",
            "वहां",
            "उसका",
            "इसका",
        )

        if any(
            word in compact
            for word in pronoun_words
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

    # সাধারণ conversational prefix বাদ।
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

    # পরিচিত location-question pattern।
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

    # Question prefix বাদ।
    candidate = re.sub(
        r"^(বলুন|বলো|আমাকে|দয়া করে|দয়া করে)\s+",
        "",
        candidate,
        flags=re.I,
    ).strip()

    # খুব ছোট generic sentence হলে entity হিসেবে নেব না।
    if len(candidate) < 3:
        return ""

    # শুধু common question words হলে বাদ।
    if candidate.casefold() in {
        item.casefold()
        for item in _REFERENCE_STOP_WORDS
    }:
        return ""

    return _clean_reference(candidate)


def _extract_recent_reference(
    history: Any,
) -> str:

    normalized = normalize_history(
        history,
        max_messages=MAX_HISTORY_MESSAGES,
    )

    # সর্বশেষ user messages আগে দেখি।
    for message in reversed(normalized):
        if message["role"] != "user":
            continue

        content = message["content"].strip()

        # Location/subject-এর সম্ভাব্য sentence।
        reference = _extract_place_from_text(
            content
        )

        if reference:
            # অতিরিক্ত বড় বা পুরো paragraph হলে
            # entity হিসেবে ব্যবহার না করা।
            if len(reference) <= 160:
                return reference

    # User message-এ না পেলে assistant-এর সাম্প্রতিক উত্তর
    # থেকেও একটি সম্ভাব্য named reference নেওয়া যাবে।
    for message in reversed(normalized):
        if message["role"] != "assistant":
            continue

        content = message["content"].strip()

        # "হালিশহর রেলওয়ে স্টেশন হলো..." ধরনের উত্তর।
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

    Returns:
        (effective_question, reference)

    উদাহরণ:
        আগের:
            "হালিশহর রেলওয়ে স্টেশন কোথায়?"
        নতুন:
            "এই জায়গায় কীভাবে যাব?"

        ফল:
            "হালিশহর রেলওয়ে স্টেশনে কীভাবে যাব?"
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

    reference = _extract_recent_reference(
        normalized
    )

    if not reference:
        # Reference না পাওয়া গেলে original question-ই রাখি।
        # AI নিজে history দেখে উত্তর দিতে পারবে।
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

    # "আরও বিস্তারিত বলো" / "more details" হলে
    # সরাসরি আগের subject-এর উপর বিস্তারিত চাই।
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