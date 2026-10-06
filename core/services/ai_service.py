"""AI provider service for Hello AI.

Provider order:
1. Groq
2. Cloudflare Workers AI
3. Gemini
"""

import logging
import os
import time
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

# Provider cooldown after temporary rate-limit/service errors.
# This prevents repeatedly hitting a provider that is currently unavailable.
PROVIDER_COOLDOWN_SECONDS = 60

# Stores the time until which each temporarily failed provider
# should be skipped.
_provider_cooldowns: dict[str, float] = {}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Hello AI, a helpful, accurate, and honest AI assistant.

Answer the user's actual question directly and clearly.

IMPORTANT RULES:

1. Use the user's language.
   - Bengali question -> Bengali answer.
   - Hindi question -> Hindi answer.
   - English question -> English answer.
   - Romanized Bengali -> Bengali script.
   - Romanized Hindi -> Hindi script.

2. When the user question contains supplied web-search evidence,
   treat that evidence as the primary source for current or
   location-specific facts.

3. If useful evidence is supplied, DO NOT say that you have no
   information merely because you do not already know the fact
   from your internal knowledge.

4. For location, address, road, locality, landmark, school,
   college, business, station, PIN/postal code, phone number,
   coordinates, route, or similar questions:
   - carefully inspect the supplied evidence;
   - identify the exact requested place;
   - use matching evidence even if it comes from a directory,
     search snippet, map-related result, or video description;
   - never invent a missing detail;
   - if different sources give different details, explicitly
     mention the conflict instead of choosing a value without
     evidence.

5. If a search result gives only partial information, provide the
   verified part and clearly say which part could not be verified.

6. Do not confuse similarly named places, especially places in
   different cities, districts, states, or countries.

7. Do not use unrelated search results just because they contain
   some of the same words as the question.

8. For current information, prefer recent and directly relevant
   supplied search evidence over old general knowledge.

9. Conversation history is useful for context, but it must not
   override directly relevant supplied search evidence.

10. Never fabricate facts, addresses, PIN codes, phone numbers,
    routes, timings, URLs, sources, quotations, or current events.

11. If the supplied evidence is conflicting, explain the conflict
    briefly and give the safest evidence-based answer.

12. Keep the answer focused. Do not repeat the same sentence,
    phrase, character, or punctuation many times.

13. If the user asks a simple factual question and the evidence
    clearly supports a simple answer, answer directly instead of
    giving unnecessary disclaimers.
    14. Match the answer length to the user's request.
    - For a simple question, give a short and direct answer.
    - Do not automatically provide extra address, PIN code,
      map information, coordinates, phone number, sources,
      or other details unless the user asks for them.
    - If the user asks for more details, then provide the
      relevant additional information.

15. Do not include links, URLs, website addresses, map links,
    or source links in a normal answer.
    Only provide a link or URL when the user explicitly asks
    for a link, URL, website, source, official website, or
    similar information.

16. Search evidence may be used internally to answer the
    question accurately, but do not expose search-result URLs
    or unnecessary source information unless the user asks
    for them.

17. Never turn a simple place-name or identification question
    into a full location report. If the user asks for the name
    of a place, answer with the name first and keep the answer
    concise.
    18. Never use citation markers, bracketed source names, source labels,
    Wikipedia-style references, or search-result citation notation in
    the answer unless the user explicitly asks for sources.
""".strip()


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
        role = item.get("role")
        content = item.get("content", "")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        messages.append({
            "role": role,
            "content": content[:MAX_MESSAGE_LENGTH],
        })

    messages.append({
        "role": "user",
        "content": question,
    })

    return messages


# ============================================================
# HTTP REQUEST HELPER
# ============================================================

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

def _extract_openai_answer(
    data: dict[str, Any],
) -> str:
    """Extract a text answer from an OpenAI-compatible response."""

    if not isinstance(data, dict):
        raise RuntimeError(
            "Provider returned an invalid response"
        )

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "Provider returned no answer"
        )

    choice = choices[0]

    if not isinstance(choice, dict):
        raise RuntimeError(
            "Provider returned an invalid choice"
        )

    message = choice.get("message", {})

    if not isinstance(message, dict):
        raise RuntimeError(
            "Provider returned an invalid message"
        )

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
        raise RuntimeError(
            "Provider returned an empty answer"
        )

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
        raise RuntimeError(
            "GROQ_API_KEY is not configured"
        )

    messages = _build_messages(
        question,
        history,
    )

    data = _post_json(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        payload={
            "model": model or DEFAULT_GROQ_MODEL,
            "messages": messages,
            "temperature": 0.2,
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

    messages = _build_messages(
        question,
        history,
    )

    data = _post_json(
        url,
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        payload={
            "model": CLOUDFLARE_MODEL,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=timeout,
        provider_name="Cloudflare",
    )

    if isinstance(data.get("choices"), list):
        return _extract_openai_answer(data)

    # Native Workers AI response format.
    result = data.get("result")

    if isinstance(result, dict):
        answer = result.get("response")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    raise RuntimeError(
        "Cloudflare returned no usable answer"
    )


# ============================================================
# GEMINI
# ============================================================

def _prepare_gemini_contents(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert messages to Gemini format."""

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

        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({
                "text": content
            })

        else:
            contents.append({
                "role": role,
                "parts": [
                    {
                        "text": content
                    }
                ],
            })

    # Gemini expects the conversation to start with a user message.
    while contents and contents[0]["role"] != "user":
        contents.pop(0)

    return contents


def ask_gemini(
    question: str,
    history: Optional[list[dict]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Ask Gemini using the generateContent REST API."""

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    messages = _build_messages(
        question,
        history,
    )

    contents = _prepare_gemini_contents(
        messages
    )

    if not contents:
        raise RuntimeError(
            "No Gemini message content available"
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    data = _post_json(
        url,
        params={
            "key": GEMINI_API_KEY
        },
        headers={
            "Content-Type": "application/json"
        },
        payload={
            "systemInstruction": {
                "parts": [
                    {
                        "text": messages[0]["content"]
                    }
                ]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
            },
        },
        timeout=timeout,
        provider_name="Gemini",
    )

    candidates = data.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(
            "Gemini returned no answer"
        )

    candidate = candidates[0]

    if not isinstance(candidate, dict):
        raise RuntimeError(
            "Gemini returned an invalid candidate"
        )

    content = candidate.get(
        "content",
        {}
    )

    if not isinstance(content, dict):
        raise RuntimeError(
            "Gemini returned invalid content"
        )

    parts = content.get(
        "parts",
        []
    )

    if not isinstance(parts, list):
        raise RuntimeError(
            "Gemini returned invalid answer parts"
        )

    answer_parts = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        text_part = part.get("text")

        if isinstance(text_part, str):
            answer_parts.append(text_part)

    answer = "".join(
        answer_parts
    ).strip()

    if not answer:
        feedback = data.get(
            "promptFeedback",
            {}
        )

        if isinstance(feedback, dict):
            if feedback.get("blockReason"):
                raise RuntimeError(
                    "Gemini blocked the prompt"
                )

        raise RuntimeError(
            "Gemini returned an empty answer"
        )

    return answer


# ============================================================
# PROVIDER FALLBACK ROUTER
# ============================================================

def get_ai_answer(
    question: str,
    history: Optional[list[dict]] = None,
) -> str:
    """Try available AI providers with temporary failure cooldown."""

    # Validate the input before contacting providers.
    _build_messages(
        question,
        history,
    )

    providers = [
        ("Groq", ask_groq),
        ("Cloudflare", ask_cloudflare),
        ("Gemini", ask_gemini),
    ]

    errors = []

    now = time.monotonic()

    for name, provider in providers:

        # ----------------------------------------------------
        # Skip a provider if it recently returned a temporary
        # rate-limit or service-unavailable error.
        # ----------------------------------------------------
        cooldown_until = _provider_cooldowns.get(
            name,
            0,
        )

        if cooldown_until > now:
            remaining = int(
                cooldown_until - now
            ) + 1

            logger.info(
                "%s is temporarily on cooldown "
                "(%ss remaining)",
                name,
                remaining,
            )

            errors.append(
                f"{name}: cooldown"
            )

            continue

        try:
            answer = provider(
                question,
                history,
            )

            if isinstance(answer, str):
                answer = answer.strip()

            if answer:
                # Provider is healthy again.
                _provider_cooldowns.pop(
                    name,
                    None,
                )

                logger.info(
                    "%s generated an answer",
                    name,
                )

                return answer

            errors.append(
                f"{name}: empty response"
            )

        except Exception as exc:
            error_text = str(exc).lower()

            # ------------------------------------------------
            # Temporary failures:
            # 429 = rate limit
            # 503 = service unavailable
            # ------------------------------------------------
            temporary_failure = (
                "429" in error_text
                or "503" in error_text
                or "rate limit" in error_text
                or "too many requests" in error_text
                or "temporarily unavailable" in error_text
                or "service unavailable" in error_text
            )

            if temporary_failure:
                _provider_cooldowns[name] = (
                    time.monotonic()
                    + PROVIDER_COOLDOWN_SECONDS
                )

                logger.warning(
                    "%s temporarily unavailable; "
                    "cooling down for %ss",
                    name,
                    PROVIDER_COOLDOWN_SECONDS,
                )

            logger.warning(
                "%s provider failed (%s)",
                name,
                type(exc).__name__,
            )

            errors.append(
                f"{name}: {type(exc).__name__}"
            )

    # All currently available providers failed.
    raise RuntimeError(
        "All AI providers failed. "
        + "; ".join(errors)
    )