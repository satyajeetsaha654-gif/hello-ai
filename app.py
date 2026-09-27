import os
import datetime
import tempfile
import time
import subprocess
import shutil
import requests

from flask import Flask, render_template, request, jsonify
from groq import Groq
from tavily import TavilyClient

try:
    from google import genai
except ImportError:
    genai = None


app = Flask(__name__)


# =========================================================
# API KEYS
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# =========================================================
# CLIENTS
# =========================================================

groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

tavily = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY
    else None
)

gemini = None

if genai and GEMINI_API_KEY:
    try:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("GEMINI CLIENT ERROR:", repr(e))


# =========================================================
# MODELS
# =========================================================

CHAT_MODEL = "openai/gpt-oss-20b"

VISION_MODEL = "qwen/qwen3.8-27b"

GEMINI_MODEL = "gemini-3.8-flash"

TRANSCRIBE_MODEL = "gemini-3.5-transcribe"

GROQ_TRANSCRIBE_MODEL = "whisper-large-v3"

TTS_MODEL = "bulbul:v3"


# =========================================================
# SARVAM VOICES
# =========================================================

MALE = {
    "shubh",
    "aditya",
    "rahul",
    "rohan",
    "amit",
    "dev",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "kabir",
    "aayan",
    "ashutosh",
    "advait",
    "anand",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "mohit",
    "rehan",
    "soham"
}

FEMALE = {
    "ritu",
    "priya",
    "neha",
    "pooja",
    "simran",
    "kavya",
    "ishita",
    "shreya",
    "roopa",
    "tanya",
    "shruti",
    "suhani",
    "kavitha",
    "rupali"
}


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM = """You are Hello AI, a helpful AI assistant.

Answer the user's current question directly.

Do not unnecessarily repeat previous answers.

Always respond naturally in the user's language.

You were created by Satya.

If asked who created you, answer exactly:
I was created by Satya.

Be helpful, clear and concise."""


# =========================================================
# INDIA REAL-TIME DATE AND TIME
# =========================================================

def current_time():

    now = datetime.datetime.now(
        datetime.timezone(
            datetime.timedelta(hours=5, minutes=30)
        )
    )

    return now.strftime(
        "%A, %d %B %Y, %I:%M %p"
    )


def is_time_or_date_question(question):

    q = question.lower().strip()

    time_words = [

        # English
        "what time",
        "what's the time",
        "whats the time",
        "time now",
        "current time",
        "time right now",
        "what is the time",
        "what is time",
        "current date",
        "today's date",
        "todays date",
        "what date",
        "what is today's date",
        "what is todays date",
        "today date",

        # Bengali
        "আজ কত",
        "আজ কয়টা",
        "আজ কয়টা",
        "এখন কয়টা",
        "এখন কয়টা",
        "এখন কটা",
        "এখন কয়টা বাজে",
        "এখন কয়টা বাজে",
        "এখন কটা বাজে",
        "এখন সময় কত",
        "এখন সময় কত",
        "বর্তমান সময়",
        "বর্তমান সময়",
        "আজকের তারিখ",
        "আজ কত তারিখ",
        "আজকের তারিখ কত",

        # Hindi
        "समय क्या है",
        "अभी कितना बजा है",
        "अभी कितने बजे हैं",
        "अभी कितने बजे है",
        "अभी समय क्या है",
        "आज की तारीख",
        "आज तारीख क्या है"
    ]

    return any(
        word in q
        for word in time_words
    )


# =========================================================
# WAIT FOR GEMINI AUDIO FILE
# =========================================================

def wait_for_gemini_file(name, timeout=45):

    if not gemini:
        raise Exception(
            "Gemini is not configured."
        )

    start = time.time()

    while time.time() - start < timeout:

        f = gemini.files.get(
            name=name
        )

        state = str(
            getattr(
                getattr(
                    f,
                    "state",
                    None
                ),
                "name",
                ""
            )
        ).upper()

        print(
            "GEMINI FILE STATE:",
            state
        )

        if state == "ACTIVE":
            return f

        if state in (
            "FAILED",
            "ERROR"
        ):

            error = getattr(
                f,
                "error",
                None
            )

            print(
                "GEMINI FILE ERROR:",
                error
            )

            raise Exception(
                "Gemini audio processing failed."
            )

        time.sleep(1)

    raise Exception(
        "Gemini audio file processing timeout."
    )


# =========================================================
# WEBM -> WAV
# =========================================================

def convert_audio(input_file):

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:

        print(
            "FFMPEG NOT FOUND - USING ORIGINAL AUDIO"
        )

        return (
            input_file,
            "audio/webm"
        )

    output = input_file + ".wav"

    try:

        result = subprocess.run(

            [
                ffmpeg,
                "-y",
                "-i",
                input_file,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                output
            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            timeout=30
        )

        if (
            result.returncode == 0
            and os.path.exists(output)
        ):

            print(
                "AUDIO CONVERTED TO WAV"
            )

            return (
                output,
                "audio/wav"
            )

        print(
            "FFMPEG ERROR:",
            result.stderr.decode(
                errors="ignore"
            )
        )

    except Exception as e:

        print(
            "AUDIO CONVERSION ERROR:",
            repr(e)
        )

    return (
        input_file,
        "audio/webm"
    )


# =========================================================
# GEMINI AUTOMATIC LANGUAGE TRANSCRIPTION
# =========================================================

def gemini_transcribe(uri, mime):

    if not gemini:
        return ""

    try:

        print(
            "GEMINI AUTO LANGUAGE DETECTION STARTED"
        )

        print(
            "GEMINI TRANSCRIBE MODEL:",
            TRANSCRIBE_MODEL
        )

        result = gemini.interactions.create(

            model=TRANSCRIBE_MODEL,

            input=[
                {
                    "type": "audio",
                    "uri": uri,
                    "mime_type": mime
                }
            ],

            generation_config={

                "transcription_config": {

                    "language_codes": [],

                    "mode": "smart"
                }
            }
        )

        text = str(
            getattr(
                result,
                "output_text",
                ""
            ) or ""
        ).strip()

        print(
            "GEMINI AUTO-DETECT OUTPUT:",
            repr(text)
        )

        return text

    except Exception as e:

        print(
            "GEMINI TRANSCRIBE ERROR:",
            repr(e)
        )

        return ""


# =========================================================
# GROQ WHISPER FALLBACK
# =========================================================

def groq_transcribe(audio_file):

    if not groq:
        return ""

    try:

        print(
            "GROQ TRANSCRIBE FALLBACK STARTED"
        )

        with open(
            audio_file,
            "rb"
        ) as f:

            result = groq.audio.transcriptions.create(

                file=f,

                model=GROQ_TRANSCRIBE_MODEL,

                response_format="json",

                temperature=0
            )

        text = str(
            getattr(
                result,
                "text",
                ""
            ) or ""
        ).strip()

        print(
            "GROQ TRANSCRIPT:",
            repr(text)
        )

        return text

    except Exception as e:

        print(
            "GROQ TRANSCRIBE ERROR:",
            repr(e)
        )

        return ""


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ASK AI
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get(
            "question",
            ""
        )
    ).strip()

    if not question:

        return jsonify({
            "answer":
            "Please ask something."
        })


    # =====================================================
    # REAL-TIME INDIA DATE / TIME
    # =====================================================

    if is_time_or_date_question(
        question
    ):

        return jsonify({

            "answer":
            f"The current India time is {current_time()}.",

            "provider":
            "real-time-india-clock"
        })


    lower = question.lower()


    # =====================================================
    # WEB SEARCH
    # =====================================================

    web_context = ""

    search_words = [

        "latest",
        "today",
        "news",
        "current",
        "recent",
        "breaking",
        "live",

        "আজ",
        "এখন",
        "সর্বশেষ",
        "খবর",

        "आज",
        "ताज़ा",
        "खबर",
        "अभी"
    ]


    if (
        tavily
        and any(
            word in lower
            for word in search_words
        )
    ):

        try:

            result = tavily.search(

                query=question,

                search_depth="advanced",

                max_results=5
            )

            parts = [
                "WEB SEARCH RESULTS:"
            ]

            for item in result.get(
                "results",
                []
            ):

                parts.append(

                    f"\nTitle: "
                    f"{item.get('title', '')}"

                    f"\nContent: "
                    f"{item.get('content', '')}"
                )

            web_context = "\n".join(
                parts
            )

        except Exception as e:

            print(
                "TAVILY ERROR:",
                repr(e)
            )


    prompt = question

    if web_context:

        prompt += (

            "\n\nUse this web information "
            "when useful. Do not mention "
            "the internal web-search process "
            "unless necessary:\n"

            + web_context
        )


    # =====================================================
    # GROQ CHAT
    # =====================================================

    if groq:

        try:

            response = (
                groq.chat.completions.create(

                    model=CHAT_MODEL,

                    messages=[

                        {
                            "role":
                            "system",

                            "content":
                            SYSTEM
                        },

                        {
                            "role":
                            "user",

                            "content":
                            prompt
                        }
                    ],

                    temperature=0.7,

                    max_tokens=2048
                )
            )


            answer = str(
                response
                .choices[0]
                .message
                .content
                or ""
            ).strip()


            if answer:

                return jsonify({

                    "answer":
                    answer,

                    "provider":
                    "groq"
                })


        except Exception as e:

            print(
                "GROQ CHAT ERROR:",
                repr(e)
            )


    # =====================================================
    # GEMINI CHAT FALLBACK
    # =====================================================

    if gemini:

        try:

            response = (
                gemini.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=[
                        SYSTEM,
                        prompt
                    ]
                )
            )


            answer = str(
                getattr(
                    response,
                    "text",
                    ""
                ) or ""
            ).strip()


            if answer:

                return jsonify({

                    "answer":
                    answer,

                    "provider":
                    "gemini"
                })


        except Exception as e:

            print(
                "GEMINI CHAT ERROR:",
                repr(e)
            )


    return jsonify({

        "answer":
        "Sorry, AI service is temporarily unavailable."

    }), 503


# =========================================================
# VOICE TRANSCRIPTION
# =========================================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    audio = request.files.get(
        "audio"
    )

    if not audio:

        return jsonify({

            "error":
            "No audio received."

        }), 400


    original = None
    converted = None


    try:

        # =================================================
        # SAVE AUDIO
        # =================================================

        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        ) as f:

            audio.save(
                f.name
            )

            original = f.name


        size = os.path.getsize(
            original
        )


        print(
            "GEMINI TRANSCRIBE FILE:",
            audio.filename or "voice.webm"
        )

        print(
            "AUDIO SIZE:",
            size,
            "bytes"
        )


        if size < 1000:

            return jsonify({

                "error":
                "Audio recording is too small or empty."

            }), 400


        # =================================================
        # CONVERT AUDIO
        # =================================================

        converted, mime = convert_audio(
            original
        )


        print(
            "TRANSCRIBE MIME:",
            mime
        )


        text = ""
        provider = ""


        # =================================================
        # GEMINI
        # =================================================

        if gemini:

            try:

                print(
                    "STARTING GEMINI TRANSCRIPTION..."
                )


                uploaded = gemini.files.upload(
                    file=converted
                )


                print(
                    "GEMINI AUDIO UPLOADED:",
                    uploaded.uri
                )


                active = wait_for_gemini_file(
                    uploaded.name
                )


                text = gemini_transcribe(

                    active.uri,

                    mime
                )


                if text:

                    provider = (
                        "gemini-3.5-transcribe"
                    )


            except Exception as e:

                print(
                    "GEMINI TRANSCRIBE ERROR:",
                    repr(e)
                )


        # =================================================
        # GROQ FALLBACK
        # =================================================

        if not text and groq:

            print(
                "GEMINI RETURNED EMPTY TEXT."
            )

            print(
                "STARTING GROQ FALLBACK..."
            )


            text = groq_transcribe(
                converted
            )


            if text:

                provider = (
                    "groq-whisper"
                )


        text = str(
            text or ""
        ).strip()


        print(
            "FINAL TRANSCRIPT:",
            repr(text)
        )


        # =================================================
        # NO SPEECH
        # =================================================

        if not text:

            return jsonify({

                "error":
                "No speech detected. "
                "Please speak clearly and try again."

            }), 400


        return jsonify({

            "text":
            text,

            "provider":
            provider
        })


    except Exception as e:

        print(
            "TRANSCRIBE ERROR:",
            repr(e)
        )


        # =================================================
        # LAST GROQ FALLBACK
        # =================================================

        if converted and groq:

            fallback = groq_transcribe(
                converted
            )


            if fallback:

                return jsonify({

                    "text":
                    fallback,

                    "provider":
                    "groq-whisper-fallback"
                })


        return jsonify({

            "error":
            "Voice transcription failed."

        }), 500


    finally:

        for file in (
            original,
            converted
        ):

            try:

                if (
                    file
                    and os.path.exists(file)
                ):

                    os.remove(file)

            except Exception:

                pass


# =========================================================
# TEXT TO SPEECH - SARVAM
# =========================================================

@app.route(
    "/tts",
    methods=["POST"]
)
def tts():

    if not SARVAM_API_KEY:

        return jsonify({

            "error":
            "Sarvam API key not configured."

        }), 503


    data = request.get_json(
        silent=True
    ) or {}


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
    ).lower()


    if not text:

        return jsonify({
            "error":
            "No text."
        }), 400


    # =====================================================
    # LANGUAGE
    # =====================================================

    if any(
        "\u0980" <= c <= "\u09ff"
        for c in text
    ):

        language = "bn-IN"

    elif any(
        "\u0900" <= c <= "\u097f"
        for c in text
    ):

        language = "hi-IN"

    else:

        language = "en-IN"


    # =====================================================
    # VOICE CHECK
    # =====================================================

    if speaker not in (
        MALE | FEMALE
    ):

        speaker = "shubh"


    try:

        response = requests.post(

            "https://api.sarvam.ai/text-to-speech",

            headers={

                "api-subscription-key":
                SARVAM_API_KEY,

                "Content-Type":
                "application/json"
            },

            json={

                "text":
                text,

                "target_language_code":
                language,

                "speaker":
                speaker,

                "model":
                TTS_MODEL,

                "enable_preprocessing":
                True
            },

            timeout=60
        )


        if not response.ok:

            print(
                "SARVAM ERROR:",
                response.text
            )

            return jsonify({

                "error":
                "TTS service failed."

            }), response.status_code


        result = response.json()


        audios = result.get(
            "audios",
            []
        )


        if not audios:

            return jsonify({

                "error":
                "No audio returned."

            }), 500


        audio = audios[0]


        if not audio:

            return jsonify({

                "error":
                "No audio returned."

            }), 500


        return jsonify({

            "audio":
            audio,

            "language":
            language,

            "speaker":
            speaker
        })


    except Exception as e:

        print(
            "TTS ERROR:",
            repr(e)
        )

        return jsonify({

            "error":
            "Voice generation failed."

        }), 500


# =========================================================
# VISION
# =========================================================

@app.route(
    "/vision",
    methods=["POST"]
)
def vision():

    if not groq:

        return jsonify({

            "error":
            "Groq is not configured."

        }), 503


    data = request.get_json(
        silent=True
    ) or {}


    image = data.get(
        "image"
    )


    question = str(
        data.get(
            "question",
            "Describe this image."
        )
    )


    if not image:

        return jsonify({

            "error":
            "No image received."

        }), 400


    try:

        response = (
            groq.chat.completions.create(

                model=VISION_MODEL,

                messages=[

                    {
                        "role":
                        "system",

                        "content":
                        SYSTEM
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
                                    image
                                }
                            }
                        ]
                    }
                ],

                temperature=0.5,

                max_tokens=2048
            )
        )


        answer = str(
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()


        return jsonify({

            "answer":
            answer,

            "provider":
            "groq-vision"
        })


    except Exception as e:

        print(
            "VISION ERROR:",
            repr(e)
        )

        return jsonify({

            "error":
            "Image processing failed."

        }), 500


# =========================================================
# SEARCH
# =========================================================

@app.route(
    "/search",
    methods=["POST"]
)
def search():

    if not tavily:

        return jsonify({

            "error":
            "Tavily is not configured."

        }), 503


    data = request.get_json(
        silent=True
    ) or {}


    query = str(
        data.get(
            "query",
            ""
        )
    ).strip()


    if not query:

        return jsonify({

            "error":
            "No search query."

        }), 400


    try:

        result = tavily.search(

            query=query,

            search_depth="advanced",

            max_results=5
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "SEARCH ERROR:",
            repr(e)
        )

        return jsonify({

            "error":
            "Search failed."

        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health"
)
def health():

    return jsonify({

        "status":
        "ok",

        "groq":
        bool(GROQ_API_KEY),

        "tavily":
        bool(TAVILY_API_KEY),

        "gemini":
        bool(GEMINI_API_KEY),

        "gemini_transcribe":
        bool(gemini),

        "sarvam":
        bool(SARVAM_API_KEY),

        "chat_model":
        CHAT_MODEL,

        "vision_model":
        VISION_MODEL,

        "gemini_model":
        GEMINI_MODEL,

        "transcribe_model":
        TRANSCRIBE_MODEL,

        "groq_transcribe_model":
        GROQ_TRANSCRIBE_MODEL,

        "tts_model":
        TTS_MODEL
    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "      HELLO AI SERVER"
    )

    print(
        "=============================="
    )

    print(
        "Groq:",
        bool(GROQ_API_KEY)
    )

    print(
        "Tavily:",
        bool(TAVILY_API_KEY)
    )

    print(
        "Gemini:",
        bool(GEMINI_API_KEY)
    )

    print(
        "Gemini Transcribe:",
        bool(gemini)
    )

    print(
        "Sarvam:",
        bool(SARVAM_API_KEY)
    )

    print(
        "Chat Model:",
        CHAT_MODEL
    )

    print(
        "Gemini Model:",
        GEMINI_MODEL
    )

    print(
        "Transcribe:",
        TRANSCRIBE_MODEL
    )

    print(
        "Groq Transcribe:",
        GROQ_TRANSCRIBE_MODEL
    )

    print(
        "TTS Model:",
        TTS_MODEL
    )

    print(
        "==============================\n"
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )