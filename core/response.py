"""Response formatting helpers for Hello AI."""

from typing import Any


def clean_response(value: Any) -> str:
    """Convert an AI response into clean, trimmed text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    try:
        return str(value).strip()
    except Exception:
        return ""


def fallback_response(language: str = "en") -> str:
    """Return a localized message when no answer is available."""

    language = str(language or "en").strip().lower()

    if language.startswith(("bn", "bangla", "bengali")):
        return (
            "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা যায়নি। "
            "অনুগ্রহ করে একটু পরে আবার চেষ্টা করো।"
        )

    if language.startswith(("hi", "hindi")):
        return (
            "माफ़ कीजिए, अभी उत्तर तैयार नहीं हो पाया। "
            "कृपया थोड़ी देर बाद फिर से कोशिश करें।"
        )

    return (
        "Sorry, I couldn't generate an answer right now. "
        "Please try again shortly."
    )