"""AI provider service for Hello AI.

Provider order:
1. Groq
2. Cloudflare Workers AI
3. Gemini
"""

import logging
import os
from typing import Any, Optional

import requests

from core.context import normalize_history


logger = logging.getLogger(__name__)


# ============================================================
# API CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

CLOUDFLARE_API_TOKEN = (
    os.getenv("CLOUDFLARE_API_TOKEN")
    or os.getenv("CLOUDFLARE_API_KEY", "")
).strip()

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID", ""
).strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

CLOUDFLARE_MODEL = os.getenv(
    "CLOUDFLARE_MODEL",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
).strip()

REQUEST_TIMEOUT = 45
MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 12000

SYSTEM_PROMPT = (
    "You are Hello AI, a helpful, accurate, and honest assistant. "
    "Answer the user's actual question directly. "
    "Do not invent facts, sources, quotations, or current information. "
    "If you are uncertain, clearly say so. "
    "Respond in the user's language and use conversation history "
    "when it is relevant. Distinguish verified facts from estimates."
)


# ============================================================
# MESSAGE PREPARATION
# ============================================================

def _build_messages(
    question: str,
    history: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    """Validate and prepare messages for OpenAI-compatible APIs."""

    if not isinstance(question, str):
        raise ValueError("Question must be text")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")

    if len(question) > MAX_MESSAGE_LENGTH:
        question = question[:MAX_MESSAGE_LENGTH]

    cleaned_history = normalize_history(
        history,
        max_messages=MAX_HISTORY_MESSAGES,
    )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for item in cleaned_history:
        messages.append({
            "role": item["role"],
            "content": item["content"][:MAX_MESSAGE_LENGTH],
        })

    messages.append({
        "role": "user",
        "content": question,
    })

    return messages


def _post_json(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, str]] = None,
    payload: Optional[dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT,
    provider_name: str = "AI provider",
) -> dict[str, Any]:
    """Send a JSON request and validate the response."""

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.warning(
            "%s network request failed (%s)",
            provider_name,
            type(exc).__name__,
        )
        raise RuntimeError(
            f"{provider_name} network request failed"
        ) from exc

    if response.status_code >= 400:
        logger.warning(
            "%s returned HTTP %s",
            provider_name,
            response.status_code,
        )
        raise RuntimeError(
            f"{provider_name} API request failed "
            f"(HTTP {response.status_code})"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{provider_name} returned invalid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{provider_name} returned an unexpected response"
        )

    return data


# ============================================================
# OPENAI-COMPATIBLE RESPONSE EXTRACTION
# ============================================================

def _extract_openai_answer(data: dict[str, Any]) -> str:
    """Extract a text answer from an OpenAI-compatible response."""

    if not isinstance(data, dict):
        raise RuntimeError("Provider returned an invalid response")

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Provider returned no answer")

    choice = choices[0]

    if not isinstance(choice, dict):
        raise RuntimeError("Provider returned an invalid choice")

    message = choice.get("message", {})

    if not isinstance(message, dict):
        raise RuntimeError("Provider returned an invalid message")

    answer = message.get("content", "")

    if isinstance(answer, str):
        answer = answer.strip()

    elif isinstance(answer, list):
        text_parts = []

        for block in answer:
            if not isinstance(block, dict):
                continue

            block_text = block.get("text")

            if isinstance(block_text, str):
                text_parts.append(block_text)

        answer = "".join(text_parts).strip()

    else:
        answer = ""

    if not answer:
        raise RuntimeError("Provider returned an empty answer")

    return answer


# ============================================================
# GROQ — PRIMARY PROVIDER
# ============================================================

def ask_groq(
    question: str,
    history: Optional[list[dict]] = None,
    model: str = DEFAULT_GROQ_MODEL,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Ask Groq and return its text response."""

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    data = _post_json(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        payload={
            "model": model or DEFAULT_GROQ_MODEL,
            "messages": _build_messages(question, history),
            "temperature": 0.3,
        },
        timeout=timeout,
        provider_name="Groq",
    )

    return _extract_openai_answer(data)


# ============================================================
# CLOUDFLARE WORKERS AI — FALLBACK 1
# ============================================================

def ask_cloudflare(
    question: str,
    history: Optional[list[dict]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Ask Cloudflare Workers AI."""

    if not CLOUDFLARE_API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is not configured"
        )

    if not CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is not configured"
        )

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
    )

    data = _post_json(
        url,
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        payload={
            "model": CLOUDFLARE_MODEL,
            "messages": _build_messages(question, history),
            "temperature": 0.3,
        },
        timeout=timeout,
        provider_name="Cloudflare",
    )

    if isinstance(data.get("choices"), list):
        return _extract_openai_answer(data)

    # Also support the native Workers AI response format.
    result = data.get("result")

    if isinstance(result, dict):
        answer = result.get("response")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    raise RuntimeError("Cloudflare returned no usable answer")


# ============================================================
# GEMINI — FALLBACK 2
# ============================================================

def _prepare_gemini_contents(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert messages and merge consecutive Gemini roles."""

    contents: list[dict[str, Any]] = []

    for message in messages[1:]:
        role = (
            "user"
            if message["role"] == "user"
            else "model"
        )

        content = message["content"]

        if not content:
            continue

        # Gemini expects alternating user/model turns.
        if contents and contents[-1]["role"] == role:
            previous_parts = contents[-1]["parts"]
            previous_parts.append({"text": content})
        else:
            contents.append({
                "role": role,
                "parts": [{"text": content}],
            })

    return contents


def ask_gemini(
    question: str,
    history: Optional[list[dict]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Ask Gemini using the generateContent REST API."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    messages = _build_messages(question, history)

    contents = _prepare_gemini_contents(messages)

    if not contents:
        raise RuntimeError("No Gemini message content available")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    data = _post_json(
        url,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        payload={
            "systemInstruction": {
                "parts": [{"text": messages[0]["content"]}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
            },
        },
        timeout=timeout,
        provider_name="Gemini",
    )

    candidates = data.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no answer")

    candidate = candidates[0]

    if not isinstance(candidate, dict):
        raise RuntimeError("Gemini returned an invalid candidate")

    content = candidate.get("content", {})

    if not isinstance(content, dict):
        raise RuntimeError("Gemini returned invalid content")

    parts = content.get("parts", [])

    if not isinstance(parts, list):
        raise RuntimeError("Gemini returned invalid answer parts")

    answer_parts = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        text_part = part.get("text")

        if isinstance(text_part, str):
            answer_parts.append(text_part)

    answer = "".join(answer_parts).strip()

    if not answer:
        feedback = data.get("promptFeedback", {})

        if isinstance(feedback, dict) and feedback.get(
            "blockReason"
        ):
            raise RuntimeError("Gemini blocked the prompt")

        raise RuntimeError("Gemini returned an empty answer")

    return answer


# ============================================================
# PROVIDER FALLBACK ROUTER
# ============================================================

def get_ai_answer(
    question: str,
    history: Optional[list[dict]] = None,
) -> str:
    """Try Groq first, then Cloudflare, then Gemini."""

    # Validate the input before contacting any provider.
    _build_messages(question, history)

    providers = [
        ("Groq", ask_groq),
        ("Cloudflare", ask_cloudflare),
        ("Gemini", ask_gemini),
    ]

    errors = []

    for name, provider in providers:
        try:
            answer = provider(question, history)

            if isinstance(answer, str) and answer.strip():
                logger.info("%s generated an answer", name)
                return answer.strip()

            errors.append(f"{name}: empty response")

        except Exception as exc:
            # Avoid logging API keys, request headers, or user content.
            logger.warning(
                "%s provider failed (%s)",
                name,
                type(exc).__name__,
            )
            errors.append(f"{name}: {type(exc).__name__}")

    raise RuntimeError(
        "All AI providers failed. Check API keys, model availability, "
        "account limits, and network connectivity. "
        + "; ".join(errors)
    )