import os
import base64
import tempfile
import subprocess
import shutil
import re
from datetime import datetime, timezone, timedelta

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from groq import Groq
from tavily import TavilyClient
from google import genai
from google.genai import types
import requests


# =========================================================
# APP
# =========================================================

app = Flask(__name__)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# =========================================================
# CLIENTS
# =========================================================

groq = None
tavily_client = None
gemini = None


# =========================================================
# GROQ
# =========================================================

if GROQ_API_KEY:
    try:
        groq = Groq(
            api_key=GROQ_API_KEY
        )

        print("Groq client ready")

    except Exception as e:

        print(
            "Groq client error:",
            e
        )


# =========================================================
# TAVILY
# =========================================================

if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(
            api_key=TAVILY_API_KEY
        )

        print("Tavily client ready")

    except Exception as e:

        print(
            "Tavily client error:",
            e
        )


# =========================================================
# GEMINI
# =========================================================

if GEMINI_API_KEY:
    try:
        gemini = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini client ready")

    except Exception as e:

        print(
            "Gemini client error:",
            e
        )


# =========================================================
# MODELS
# =========================================================

GEMINI_CHAT_MODEL = "gemini-3.8-flash"

GROQ_CHAT_MODEL = "openai/gpt-oss-20b"

GEMINI_VISION_MODEL = GEMINI_CHAT_MODEL

GROQ_VISION_MODEL = "qwen/qwen3.8-27b"

GEMINI_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"

TTS_MODEL = "bulbul:v3"


# =========================================================
# CONSTANTS
# =========================================================

DAILY_LIMIT_MESSAGE = (
    "Your daily limit has been reached."
)

INDIA_TIMEZONE = timezone(
    timedelta(
        hours=5,
        minutes=30
    )
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Hello AI.

You were created by Satya.

If the user asks who created you, answer:
"I was created by Satya."

Never say that you were created by OpenAI.

Answer the user's current question directly.

Understand the user's language automatically.

If the user asks in Bengali, answer in Bengali.
If the user asks in Hindi, answer in Hindi.
If the user asks in English, answer in English.

You can understand other languages too and normally reply
in the same language as the user.

Use previous conversation context when relevant.

Do not repeat old answers unnecessarily.

IMPORTANT WEB ACCURACY RULES:

When web search information is provided, use it carefully.

Never invent:
- addresses
- locations
- phone numbers
- business names
- school/college/institute locations
- opening hours
- prices
- current news
- current events
- dates
- statistics
- names

For location or institution questions, rely on the supplied
web search information.

If the search results do not clearly identify the exact place,
say that the exact location could not be verified instead of
guessing.

If multiple places have similar names, clearly explain the
ambiguity and do not choose one without evidence.

Never present an assumption as a verified fact.

For current date and time in India, use the server-provided
India date/time as authoritative.

Be helpful, clear and natural.
"""


# =========================================================
# INDIA TIME
# =========================================================

def get_india_datetime():

    return datetime.now(
        INDIA_TIMEZONE
    )


def get_india_time_text():

    now = get_india_datetime()

    return now.strftime(
        "%A, %d %B %Y, %I:%M:%S %p"
    )


# =========================================================
# TTS LANGUAGE DETECTION
# =========================================================

def detect_language(text):

    text = str(
        text or ""
    )

    bengali_count = 0
    devanagari_count = 0
    latin_count = 0

    for char in text:

        code = ord(char)

        if 0x0980 <= code <= 0x09FF:

            bengali_count += 1

        elif 0x0900 <= code <= 0x097F:

            devanagari_count += 1

        elif (
            ("A" <= char <= "Z")
            or
            ("a" <= char <= "z")
        ):

            latin_count += 1

    if (
        bengali_count >
        devanagari_count
        and
        bengali_count >
        latin_count
    ):

        return "bn-IN"

    if (
        devanagari_count >
        bengali_count
        and
        devanagari_count >
        latin_count
    ):

        return "hi-IN"

    return "en-IN"


# =========================================================
# WEB SEARCH DECISION
# =========================================================

def needs_web_search(question):

    q = str(
        question or ""
    ).lower().strip()

    if not q:
        return False

    # -----------------------------------------------------
    # Current / fresh information
    # -----------------------------------------------------

    current_keywords = [

        "latest",
        "today",
        "now",
        "current",
        "recent",
        "news",
        "weather",
        "price",
        "stock",
        "score",
        "result",
        "live",
        "schedule",
        "opening hours",
        "open now",
        "address",
        "location",
        "where is",
        "who is",
        "what happened",
        "this week",
        "this month",
        "2026",

        "আজ",
        "এখন",
        "বর্তমান",
        "সাম্প্রতিক",
        "খবর",
        "নিউজ",
        "আবহাওয়া",
        "আবহাওয়া",
        "দাম",
        "মূল্য",
        "স্কোর",
        "ঠিকানা",
        "কোথায়",
        "কোথায়",
        "কোথায় আছে",
        "কোথায় আছে",
        "কোথায় অবস্থিত",
        "কোথায় অবস্থিত",
        "আজকের",

        "आज",
        "अभी",
        "वर्तमान",
        "ताज़ा",
        "समाचार",
        "खबर",
        "मौसम",
        "कीमत",
        "दाम",
        "स्कोर",
        "पता",
        "कहाँ",
        "कहां",
        "कहाँ है",
        "कहां है"
    ]

    for keyword in current_keywords:

        if keyword in q:

            return True

    # -----------------------------------------------------
    # Institution / place / person style questions
    # -----------------------------------------------------

    location_patterns = [

        r"\bwhere\b",
        r"\bwhere is\b",
        r"\bwhere are\b",
        r"\baddress\b",
        r"\blocated\b",
        r"\bnear\b",

        r"কোথায়",
        r"কোথায়",
        r"কোথায় আছে",
        r"কোথায় আছে",
        r"ঠিকানা",
        r"অবস্থিত",
        r"কাছাকাছি",

        r"कहाँ",
        r"कहां",
        r"पता",
        r"स्थित",
        r"पास में"
    ]

    for pattern in location_patterns:

        if re.search(
            pattern,
            q
        ):

            return True

    # -----------------------------------------------------
    # Common institution words
    # -----------------------------------------------------

    institution_words = [

        "school",
        "college",
        "institute",
        "institution",
        "hospital",
        "clinic",
        "university",
        "station",
        "airport",
        "hotel",
        "restaurant",
        "market",
        "mall",
        "office",

        "স্কুল",
        "কলেজ",
        "ইনস্টিটিউট",
        "ইন্সটিটিউট",
        "প্রতিষ্ঠান",
        "হাসপাতাল",
        "ক্লিনিক",
        "বিশ্ববিদ্যালয়",
        "বিশ্ববিদ্যালয়",
        "স্টেশন",
        "বিমানবন্দর",
        "হোটেল",
        "রেস্টুরেন্ট",
        "বাজার",

        "स्कूल",
        "कॉलेज",
        "इंस्टीट्यूट",
        "अस्पताल",
        "क्लिनिक",
        "विश्वविद्यालय",
        "स्टेशन",
        "एयरपोर्ट",
        "होटल",
        "रेस्टोरेंट",
        "बाज़ार"
    ]

    for word in institution_words:

        if word in q:

            return True

    # -----------------------------------------------------
    # Explicit factual questions
    # -----------------------------------------------------

    factual_patterns = [

        r"\bwho is\b",
        r"\bwhat is\b",
        r"\bwhen is\b",
        r"\bhow much\b",
        r"\bhow many\b",

        r"কে ",
        r"কী ",
        r"কি ",
        r"কখন",
        r"কত",

        r"कौन",
        r"क्या",
        r"कब",
        r"कितना",
        r"कितने"
    ]

    for pattern in factual_patterns:

        if re.search(
            pattern,
            q
        ):

            return True

    return False


# =========================================================
# TAVILY SEARCH
# =========================================================

def tavily_search(question):

    if not tavily_client:

        print(
            "Tavily unavailable"
        )

        return ""

    try:

        print(
            "===================================="
        )

        print(
            "TAVILY SEARCH:",
            question
        )

        result = tavily_client.search(
            query=question,
            search_depth="advanced",
            max_results=8
        )

        results = result.get(
            "results",
            []
        )

        if not results:

            print(
                "TAVILY: No results"
            )

            return ""

        lines = []

        for index, item in enumerate(
            results,
            start=1
        ):

            title = str(
                item.get(
                    "title",
                    ""
                )
            ).strip()

            content = str(
                item.get(
                    "content",
                    ""
                )
            ).strip()

            url = str(
                item.get(
                    "url",
                    ""
                )
            ).strip()

            if (
                title
                or
                content
            ):

                lines.append(
                    f"""
SOURCE {index}

Title:
{title}

Information:
{content}

URL:
{url}
""".strip()
                )

        final_text = "\n\n".join(
            lines
        )

        print(
            "TAVILY RESULTS:",
            len(results)
        )

        print(
            "===================================="
        )

        return final_text

    except Exception as e:

        print(
            "Tavily search failed:",
            e
        )

        return ""


# =========================================================
# BUILD GEMINI PROMPT
# =========================================================

def build_gemini_prompt(
    question,
    history=None,
    web_context=""
):

    parts = []

    parts.append(
        SYSTEM_PROMPT
    )

    parts.append(
        "\nCURRENT INDIA DATE AND TIME:\n"
        +
        get_india_time_text()
    )

    if web_context:

        parts.append(
            """
IMPORTANT:
The following information was retrieved from web search.

Use it as evidence for current/factual questions.

For location questions, identify the exact place before
giving an address or location.

Do not mix information from similarly named places.

If the sources do not establish the exact answer,
say that the exact information could not be verified.

WEB SEARCH RESULTS:
"""
            +
            web_context
        )

    if history:

        parts.append(
            "\nPREVIOUS CONVERSATION:"
        )

        for item in history:

            if not isinstance(
                item,
                dict
            ):

                continue

            role = str(
                item.get(
                    "role",
                    ""
                )
            ).strip()

            content = str(
                item.get(
                    "content",
                    ""
                )
            ).strip()

            if not content:

                continue

            if role == "user":

                parts.append(
                    "User: "
                    +
                    content
                )

            elif role == "assistant":

                parts.append(
                    "Hello AI: "
                    +
                    content
                )

    parts.append(
        "\nCURRENT USER QUESTION:\n"
        +
        str(question)
    )

    return "\n\n".join(
        parts
    )


# =========================================================
# GEMINI CHAT
# =========================================================

def gemini_answer(
    question,
    history=None,
    web_context=""
):

    if not gemini:

        raise RuntimeError(
            "Gemini unavailable"
        )

    print(
        "MAIN AI: Gemini"
    )

    prompt = build_gemini_prompt(
        question=question,
        history=history,
        web_context=web_context
    )

    response = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4
        )
    )

    text = getattr(
        response,
        "text",
        ""
    )

    text = (
        text
        or ""
    ).strip()

    if not text:

        raise RuntimeError(
            "Gemini returned empty answer"
        )

    print(
        "GEMINI ANSWER:",
        repr(
            text[:300]
        )
    )

    return text


# =========================================================
# GROQ CHAT FALLBACK
# =========================================================

def groq_answer(
    question,
    history=None,
    web_context=""
):

    if not groq:

        raise RuntimeError(
            "Groq unavailable"
        )

    print(
        "FALLBACK AI: Groq"
    )

    messages = []

    messages.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    )

    messages.append(
        {
            "role": "system",
            "content":
                "Current India date and time: "
                +
                get_india_time_text()
        }
    )

    if web_context:

        messages.append(
            {
                "role": "system",
                "content":
                    """
Use the following web search information
for factual/current answers.

Do not invent locations, addresses or
current information.

WEB SEARCH:
"""
                    +
                    web_context
            }
        )

    if history:

        for item in history:

            if not isinstance(
                item,
                dict
            ):

                continue

            role = item.get(
                "role"
            )

            content = str(
                item.get(
                    "content",
                    ""
                )
            ).strip()

            if role not in [
                "user",
                "assistant"
            ]:

                continue

            if not content:

                continue

            messages.append(
                {
                    "role":
                        role,
                    "content":
                        content
                }
            )

    messages.append(
        {
            "role":
                "user",
            "content":
                str(question)
        }
    )

    response = groq.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=messages,
        temperature=0.4
    )

    text = (
        response.choices[0]
        .message.content
    )

    text = (
        text
        or ""
    ).strip()

    if not text:

        raise RuntimeError(
            "Groq returned empty answer"
        )

    print(
        "GROQ ANSWER:",
        repr(
            text[:300]
        )
    )

    return text


# =========================================================
# GEMINI TRANSCRIPTION
# =========================================================
# IMPORTANT:
# Voice transcription uses ONLY Gemini.
# There is NO Groq fallback here.
# =========================================================

def gemini_transcribe(
    file_path,
    mime_type
):

    if not gemini:

        raise RuntimeError(
            "Gemini unavailable"
        )

    print(
        "GEMINI TRANSCRIPTION STARTED"
    )

    print(
        "TRANSCRIBE MODEL:",
        GEMINI_TRANSCRIBE_MODEL
    )

    print(
        "TRANSCRIBE MIME:",
        mime_type
    )

    uploaded = gemini.files.upload(
        file=file_path
    )

    print(
        "GEMINI AUDIO UPLOADED:",
        uploaded.uri
    )

    uploaded_mime = (
        getattr(
            uploaded,
            "mime_type",
            None
        )
        or
        mime_type
    )

    interaction = gemini.interactions.create(
        model=GEMINI_TRANSCRIBE_MODEL,
        input=[
            {
                "type":
                    "audio",

                "uri":
                    uploaded.uri,

                "mime_type":
                    uploaded_mime
            }
        ],
        generation_config={
            "transcription_config": {
                "language_codes": []
            }
        }
    )

    text = getattr(
        interaction,
        "output_text",
        ""
    )

    text = (
        text
        or ""
    ).strip()

    print(
        "GEMINI RAW TRANSCRIPTION:",
        repr(text)
    )

    if not text:

        raise RuntimeError(
            "Gemini returned empty transcription"
        )

    print(
        "GEMINI TRANSCRIPT:",
        repr(text)
    )

    return text


# =========================================================
# SARVAM TTS
# =========================================================

def sarvam_tts(
    text,
    speaker="shubh"
):

    if not SARVAM_API_KEY:

        raise RuntimeError(
            "Sarvam unavailable"
        )

    text = str(
        text or ""
    ).strip()

    if not text:

        raise RuntimeError(
            "Empty TTS text"
        )

    speaker = str(
        speaker
        or
        "shubh"
    ).strip()

    if speaker.lower() == "anushka":

        speaker = "shubh"

    language_code = detect_language(
        text
    )

    url = (
        "https://api.sarvam.ai/"
        "text-to-speech"
    )

    headers = {
        "api-subscription-key":
            SARVAM_API_KEY,

        "Content-Type":
            "application/json"
    }

    payload = {
        "inputs": [
            text
        ],

        "target_language_code":
            language_code,

        "speaker":
            speaker,

        "model":
            TTS_MODEL
    }

    print(
        "SARVAM TTS:",
        language_code,
        speaker
    )

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:

        print(
            "SARVAM ERROR:",
            response.status_code,
            response.text
        )

        raise RuntimeError(
            "Sarvam TTS failed"
        )

    data = response.json()

    audios = data.get(
        "audios",
        []
    )

    if not audios:

        raise RuntimeError(
            "Sarvam returned no audio"
        )

    return audios[0]


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ASK
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        history = data.get(
            "history",
            []
        )

        # -------------------------------------------------
        # Voice mode
        # -------------------------------------------------
        # Frontend can send:
        # "voice_mode": true
        #
        # If true:
        # Gemini only.
        # Groq will NOT be used.
        # -------------------------------------------------

        voice_mode = bool(
            data.get(
                "voice_mode",
                False
            )
        )

        if not question:

            return jsonify(
                {
                    "answer":
                        "",
                    "provider":
                        "none"
                }
            )

        print(
            "\nQUESTION RECEIVED:",
            question
        )

        print(
            "VOICE MODE:",
            voice_mode
        )

        # =================================================
        # DIRECT INDIA TIME / DATE
        # =================================================

        q_lower = question.lower().strip()

        exact_time_patterns = [

            "what time is it",
            "what is the time",
            "current time",
            "tell me the time",
            "what's the time",
            "time in india",

            "এখন কয়টা বাজে",
            "এখন কয়টা বাজে",
            "এখন সময় কত",
            "এখন সময় কত",
            "ভারতে এখন কয়টা",
            "ভারতে এখন কয়টা",

            "अभी कितने बजे हैं",
            "अभी समय क्या है",
            "भारत में अभी कितने बजे हैं"
        ]

        exact_date_patterns = [

            "what is today's date",
            "what is the date today",
            "today's date",
            "todays date",
            "what date is it",

            "আজকের তারিখ কত",
            "আজ কত তারিখ",
            "আজকের তারিখ কী",
            "আজকের তারিখ কি",

            "आज की तारीख क्या है",
            "आज कितनी तारीख है"
        ]

        wants_time = any(
            phrase in q_lower
            for phrase in exact_time_patterns
        )

        wants_date = any(
            phrase in q_lower
            for phrase in exact_date_patterns
        )

        if (
            wants_time
            or
            wants_date
        ):

            now = get_india_datetime()

            if wants_time:

                answer = (
                    "India time: "
                    +
                    now.strftime(
                        "%I:%M:%S %p"
                    )
                    +
                    "\nDate: "
                    +
                    now.strftime(
                        "%d %B %Y"
                    )
                )

            else:

                answer = (
                    "Today's date in India is "
                    +
                    now.strftime(
                        "%d %B %Y"
                    )
                )

            return jsonify(
                {
                    "answer":
                        answer,

                    "provider":
                        "india-time"
                }
            )

        # =================================================
        # WEB SEARCH
        # =================================================

        web_context = ""

        if needs_web_search(
            question
        ):

            web_context = tavily_search(
                question
            )

            if web_context:

                print(
                    "WEB SEARCH DATA FOUND"
                )

            else:

                print(
                    "WEB SEARCH FOUND NO VERIFIED DATA"
                )

        # =================================================
        # VOICE MODE
        # =================================================
        # Gemini only.
        # NO Groq fallback.
        # =================================================

        if voice_mode:

            try:

                answer = gemini_answer(
                    question=
                        question,

                    history=
                        history,

                    web_context=
                        web_context
                )

                return jsonify(
                    {
                        "answer":
                            answer,

                        "provider":
                            "Gemini"
                    }
                )

            except Exception as gemini_error:

                print(
                    "VOICE GEMINI FAILED:"
                )

                print(
                    gemini_error
                )

                return jsonify(
                    {
                        "answer":
                            DAILY_LIMIT_MESSAGE,

                        "provider":
                            "daily-limit"
                    }
                )

        # =================================================
        # NORMAL CHAT
        # =================================================
        # Gemini -> Groq fallback
        # =================================================

        try:

            answer = gemini_answer(
                question=
                    question,

                history=
                    history,

                web_context=
                    web_context
            )

            return jsonify(
                {
                    "answer":
                        answer,

                    "provider":
                        "Gemini"
                }
            )

        except Exception as gemini_error:

            print(
                "Gemini failed."
            )

            print(
                "Gemini error:",
                gemini_error
            )

        # =================================================
        # GROQ FALLBACK
        # =================================================

        try:

            answer = groq_answer(
                question=
                    question,

                history=
                    history,

                web_context=
                    web_context
            )

            return jsonify(
                {
                    "answer":
                        answer,

                    "provider":
                        "Groq fallback"
                }
            )

        except Exception as groq_error:

            print(
                "Groq failed."
            )

            print(
                "Groq error:",
                groq_error
            )

        # =================================================
        # BOTH FAILED
        # =================================================

        return jsonify(
            {
                "answer":
                    DAILY_LIMIT_MESSAGE,

                "provider":
                    "daily-limit"
            }
        )

    except Exception as e:

        print(
            "ASK ROUTE ERROR:",
            e
        )

        return jsonify(
            {
                "answer":
                    DAILY_LIMIT_MESSAGE,

                "provider":
                    "daily-limit"
            }
        )


# =========================================================
# TRANSCRIBE
# =========================================================
# IMPORTANT:
# ONLY GEMINI
# NO GROQ
# =========================================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    temp_dir = None

    try:

        if "audio" not in request.files:

            return jsonify(
                {
                    "text":
                        "Voice transcription failed.",

                    "provider":
                        "Gemini"
                }
            )

        audio_file = request.files[
            "audio"
        ]

        if not audio_file:

            return jsonify(
                {
                    "text":
                        "Voice transcription failed.",

                    "provider":
                        "Gemini"
                }
            )

        temp_dir = tempfile.mkdtemp(
            prefix="hello_ai_audio_"
        )

        input_path = os.path.join(
            temp_dir,
            "voice.webm"
        )

        wav_path = os.path.join(
            temp_dir,
            "voice.wav"
        )

        audio_file.save(
            input_path
        )

        print(
            "AUDIO SIZE:",
            os.path.getsize(
                input_path
            ),
            "bytes"
        )

        # -------------------------------------------------
        # FFMPEG
        # -------------------------------------------------

        ffmpeg_path = shutil.which(
            "ffmpeg"
        )

        if not ffmpeg_path:

            raise RuntimeError(
                "ffmpeg not found"
            )

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        print(
            "AUDIO CONVERTED TO WAV"
        )

        # =================================================
        # ONLY GEMINI TRANSCRIPTION
        # =================================================

        print(
            "TRANSCRIPTION MAIN: GEMINI"
        )

        text = gemini_transcribe(
            wav_path,
            "audio/wav"
        )

        print(
            "GEMINI VOICE TEXT:",
            repr(text)
        )

        return jsonify(
            {
                "text":
                    text,

                "provider":
                    "Gemini"
            }
        )

    except Exception as e:

        print(
            "GEMINI VOICE ERROR:",
            e
        )

        return jsonify(
            {
                "text":
                    "Voice transcription failed.",

                "provider":
                    "Gemini"
            }
        )

    finally:

        if temp_dir:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# =========================================================
# TTS
# =========================================================

@app.route(
    "/tts",
    methods=["POST"]
)
def tts():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        text = str(
            data.get(
                "text",
                ""
            )
        ).strip()

        speaker = str(
            data.get(
                "speaker",
                "shubh"
            )
        ).strip()

        if not text:

            return jsonify(
                {
                    "audio":
                        None,

                    "error":
                        "Empty text"
                }
            ), 400

        if speaker.lower() == "anushka":

            speaker = "shubh"

        audio_base64 = sarvam_tts(
            text=
                text,

            speaker=
                speaker
        )

        return jsonify(
            {
                "audio":
                    audio_base64,

                "speaker":
                    speaker,

                "language":
                    detect_language(
                        text
                    )
            }
        )

    except Exception as e:

        print(
            "TTS ERROR:",
            e
        )

        return jsonify(
            {
                "audio":
                    None,

                "error":
                    str(e)
            }
        ), 500


# =========================================================
# VOICES
# =========================================================

@app.route(
    "/voices",
    methods=["GET"]
)
def voices():

    return jsonify(
        {
            "model":
                TTS_MODEL,

            "voices": [
                "aditya",
                "ritu",
                "ashutosh",
                "priya",
                "neha",
                "rahul",
                "pooja",
                "rohan",
                "simran",
                "kavya",
                "amit",
                "dev",
                "ishita",
                "shreya",
                "ratan",
                "varun",
                "manan",
                "sumit",
                "roopa",
                "kabir",
                "aayan",
                "shubh",
                "advait",
                "anand",
                "tanya",
                "tarun",
                "sunny",
                "mani",
                "gokul",
                "vijay",
                "shruti",
                "suhani",
                "mohit",
                "kavitha",
                "rehan",
                "soham",
                "rupali"
            ]
        }
    )


# =========================================================
# VISION
# =========================================================

@app.route(
    "/vision",
    methods=["POST"]
)
def vision():

    temp_file = None

    try:

        image_file = request.files.get(
            "image"
        )

        question = ""

        if request.form:

            question = str(
                request.form.get(
                    "question",
                    "Describe this image."
                )
            ).strip()

        # -------------------------------------------------
        # JSON BASE64 SUPPORT
        # -------------------------------------------------

        if not image_file:

            data = (
                request.get_json(
                    silent=True
                )
                or {}
            )

            image_data = data.get(
                "image"
            )

            question = str(
                data.get(
                    "question",
                    "Describe this image."
                )
            ).strip()

            if image_data:

                if "," in image_data:

                    image_data = (
                        image_data.split(
                            ",",
                            1
                        )[1]
                    )

                raw = base64.b64decode(
                    image_data
                )

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                )

                temp_file.write(
                    raw
                )

                temp_file.close()

                image_path = (
                    temp_file.name
                )

            else:

                return jsonify(
                    {
                        "answer":
                            DAILY_LIMIT_MESSAGE,

                        "provider":
                            "daily-limit"
                    }
                )

        else:

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )

            image_file.save(
                temp_file.name
            )

            temp_file.close()

            image_path = (
                temp_file.name
            )

        # -------------------------------------------------
        # IMAGE BYTES
        # -------------------------------------------------

        with open(
            image_path,
            "rb"
        ) as f:

            image_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=
                image_bytes,

            mime_type=
                "image/jpeg"
        )

        vision_prompt = (
            SYSTEM_PROMPT
            +
            "\n\nAnalyze this image carefully."
            +
            "\nAnswer the user's request."
            +
            "\n\nUser request:\n"
            +
            question
        )

        # =================================================
        # GEMINI VISION
        # =================================================

        try:

            if not gemini:

                raise RuntimeError(
                    "Gemini unavailable"
                )

            print(
                "VISION MAIN: Gemini"
            )

            response = (
                gemini.models.generate_content(
                    model=
                        GEMINI_VISION_MODEL,

                    contents=[
                        vision_prompt,
                        image_part
                    ],

                    config=
                        types.GenerateContentConfig(
                            temperature=0.4
                        )
                )
            )

            answer = getattr(
                response,
                "text",
                ""
            )

            answer = (
                answer
                or
                ""
            ).strip()

            if not answer:

                raise RuntimeError(
                    "Gemini vision returned empty answer"
                )

            return jsonify(
                {
                    "answer":
                        answer,

                    "provider":
                        "Gemini vision"
                }
            )

        except Exception as gemini_error:

            print(
                "Gemini vision failed:",
                gemini_error
            )

        # =================================================
        # GROQ VISION FALLBACK
        # =================================================

        try:

            if not groq:

                raise RuntimeError(
                    "Groq unavailable"
                )

            print(
                "VISION FALLBACK: Groq"
            )

            encoded = base64.b64encode(
                image_bytes
            ).decode(
                "utf-8"
            )

            response = (
                groq.chat.completions.create(
                    model=
                        GROQ_VISION_MODEL,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                SYSTEM_PROMPT
                        },

                        {
                            "role":
                                "user",

                            "content": [
                                {
                                    "type":
                                        "text",

                                    "text":
                                        question
                                },

                                {
                                    "type":
                                        "image_url",

                                    "image_url": {
                                        "url":
                                            "data:image/jpeg;base64,"
                                            +
                                            encoded
                                    }
                                }
                            ]
                        }
                    ],

                    temperature=0.4
                )
            )

            answer = (
                response
                .choices[0]
                .message.content
            )

            answer = (
                answer
                or
                ""
            ).strip()

            if not answer:

                raise RuntimeError(
                    "Groq vision returned empty answer"
                )

            return jsonify(
                {
                    "answer":
                        answer,

                    "provider":
                        "Groq fallback vision"
                }
            )

        except Exception as groq_error:

            print(
                "Groq vision failed:",
                groq_error
            )

        return jsonify(
            {
                "answer":
                    DAILY_LIMIT_MESSAGE,

                "provider":
                    "daily-limit"
            }
        )

    except Exception as e:

        print(
            "VISION ERROR:",
            e
        )

        return jsonify(
            {
                "answer":
                    DAILY_LIMIT_MESSAGE,

                "provider":
                    "daily-limit"
            }
        )

    finally:

        if temp_file:

            try:

                os.unlink(
                    temp_file.name
                )

            except Exception:

                pass


# =========================================================
# SEARCH
# =========================================================

@app.route(
    "/search",
    methods=["POST"]
)
def search():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        query = str(
            data.get(
                "query",
                ""
            )
        ).strip()

        if not query:

            return jsonify(
                {
                    "results":
                        []
                }
            )

        results_text = tavily_search(
            query
        )

        if not results_text:

            return jsonify(
                {
                    "results":
                        []
                }
            )

        return jsonify(
            {
                "results":
                    results_text
            }
        )

    except Exception as e:

        print(
            "SEARCH ERROR:",
            e
        )

        return jsonify(
            {
                "results":
                    []
            }
        )


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status":
                "ok",

            "gemini":
                bool(gemini),

            "groq":
                bool(groq),

            "tavily":
                bool(tavily_client),

            "sarvam":
                bool(SARVAM_API_KEY),

            "india_time":
                get_india_time_text()
        }
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print(
        "===================================="
    )

    print(
        "HELLO AI SERVER STARTED"
    )

    print(
        "===================================="
    )

    print(
        "Gemini:",
        bool(gemini)
    )

    print(
        "Groq:",
        bool(groq)
    )

    print(
        "Tavily:",
        bool(tavily_client)
    )

    print(
        "Sarvam:",
        bool(SARVAM_API_KEY)
    )

    print(
        "===================================="
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )