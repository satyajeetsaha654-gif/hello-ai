"""Conversation context helpers for Hello AI."""

from typing import Any


MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 12000


def normalize_history(
    history: Any,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    """Validate, normalize, and limit conversation history."""

    if not isinstance(history, list):
        return []

    try:
        limit = int(max_messages)
    except (TypeError, ValueError, OverflowError):
        limit = MAX_HISTORY_MESSAGES

    if limit <= 0:
        return []

    cleaned: list[dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role") or item.get("type") or ""
        ).strip().lower()

        if role in ("assistant", "ai", "model", "bot"):
            role = "assistant"
        elif role in ("user", "human"):
            role = "user"
        else:
            continue

        content = item.get(
            "content",
            item.get("text", item.get("message", "")),
        )

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        cleaned.append({
            "role": role,
            "content": content[:MAX_MESSAGE_LENGTH],
        })

    cleaned = cleaned[-limit:]

    # History should begin with a user message.
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    return cleaned