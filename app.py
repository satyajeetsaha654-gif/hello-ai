import os
import base64
from datetime import datetime, timezone, timedelta

import requests
from flask import Flask, render_template, request, jsonify
from groq import Groq
from tavily import TavilyClient
from google import genai
from google.genai import types


# =========================================================
# HELLO AI
# =========================================================

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

groq_client = Groq(api_key=GROQ_API_KEY)

tavily_client = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY
    else None
)

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# =========================================================
# LIMIT MESSAGE
# =========================================================

DAILY_LIMIT_MESSAGE = "Your daily AI limit has expired."


# =========================================================
# MAIN AI PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Hello AI, a helpful multilingual AI assistant.

You were created by Satya.

Rules:

1. Understand Bengali, English, Hindi and mixed-language messages.
2. Reply naturally in the same language as the user whenever possible.
3. Bengali input -> Bengali script response.
4. Hindi input -> Devanagari Hindi response.
5. English input -> English response.
6. Mixed language -> understand naturally and answer clearly.
7. Maintain the conversation context.
8. Understand references such as:
   ও, ওর, তার, he, she, it, this, that.
9. Give clear and useful answers.
10. Do not guess the current date or time.
11. For current date/time questions, use the current date/time
    information provided with the request.
"""


# =========================================================
# CURRENT INDIA TIME
# =========================================================

def get_current_datetime():

    # India Standard Time = UTC + 5:30
    ist = timezone(timedelta(hours=5, minutes=30))

    now = datetime.now(ist)

    return now.strftime(
        "%A, %d %B %Y, %I:%M:%S %p IST"
    )


# =========================================================
# SYSTEM PROMPT WITH LIVE DATE/TIME
# =========================================================

def get_system_prompt():

    current_datetime = get_current_datetime()

    return (
        SYSTEM_PROMPT
        + "\n\n"
        + "Current date and time in India:\n"
        + current_datetime
        + "\n\n"
        + "When the user asks for the current date, current time, "
        + "today's date, or day of the week, use the information above."
    )


# =========================================================
# CLEAN ANSWER
# =========================================================

def clean_answer(answer):

    if not answer:
        return "Sorry, I could not generate a response."

    return str(answer).strip()


# =========================================================
# CHECK API LIMIT
# =========================================================

def is_limit_error(error):

    text = str(error).lower()

    keywords = [
        "rate limit",
        "ratelimit",
        "quota",
        "too many requests",
        "429",
        "limit reached",
        "resource exhausted",
        "exceeded",
        "daily limit"
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# =========================================================
# GEMINI FALLBACK
# =========================================================

def ask_gemini(messages):

    if not gemini_client:
        raise Exception(
            "Gemini API key is not configured."
        )

    conversation = []

    for item in messages:

        role = item.get("role")
        content = item.get("content", "")

        if not isinstance(content, str):
            continue

        if role == "system":
            continue

        if role == "assistant":

            conversation.append(
                "Assistant: " + content
            )

        elif role == "user":

            conversation.append(
                "User: " + content
            )

    prompt = "\n".join(conversation)

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=get_system_prompt()
        )
    )

    return clean_answer(response.text)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# ASK AI
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        message = str(
            data.get("message", "")
        ).strip()

        history = data.get(
            "history",
            []
        )

        if not message:

            return jsonify({
                "error": "Please enter a message."
            }), 400


        if not isinstance(history, list):

            history = []


        # Keep only recent conversation
        history = history[-12:]


        # Current system prompt
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            }
        ]


        # Previous conversation
        for item in history:

            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role in ["user", "assistant"] and content:

                messages.append({
                    "role": role,
                    "content": str(content)
                })


        # Current user message
        messages.append({
            "role": "user",
            "content": message
        })


        # =================================================
        # GROQ
        # =================================================

        try:

            response = groq_client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=messages,

                temperature=0.7,

                max_completion_tokens=1024
            )


            answer = (
                response
                .choices[0]
                .message
                .content
            )


            return jsonify({
                "answer": clean_answer(answer),
                "provider": "groq"
            })


        except Exception as groq_error:

            print(
                "GROQ ERROR:",
                groq_error
            )


            # =================================================
            # GEMINI FALLBACK
            # =================================================

            if is_limit_error(groq_error):

                try:

                    answer = ask_gemini(
                        messages
                    )

                    return jsonify({
                        "answer": answer,
                        "provider": "gemini"
                    })


                except Exception as gemini_error:

                    print(
                        "GEMINI ERROR:",
                        gemini_error
                    )


                    if is_limit_error(
                        gemini_error
                    ):

                        return jsonify({
                            "limit": True,
                            "error":
                                DAILY_LIMIT_MESSAGE
                        }), 429


                    return jsonify({
                        "error":
                            "AI service is temporarily unavailable."
                    }), 500


            return jsonify({
                "error":
                    "AI service is temporarily unavailable."
            }), 500


    except Exception as error:

        print(
            "ASK ERROR:",
            error
        )

        return jsonify({
            "error":
                "Something went wrong."
        }), 500


# =========================================================
# VOICE TO TEXT
# =========================================================

@app.route("/transcribe", methods=["POST"])
def transcribe():

    try:

        if "audio" not in request.files:

            return jsonify({
                "error":
                    "No audio file received."
            }), 400


        audio = request.files["audio"]


        if not audio:

            return jsonify({
                "error":
                    "Invalid audio file."
            }), 400


        transcription = (
            groq_client
            .audio
            .transcriptions
            .create(
                file=(
                    audio.filename or "audio.webm",
                    audio.stream,
                    audio.mimetype or "audio/webm"
                ),
                model="whisper-large-v3-turbo"
            )
        )


        text = getattr(
            transcription,
            "text",
            ""
        )


        return jsonify({
            "text": text
        })


    except Exception as error:

        print(
            "TRANSCRIBE ERROR:",
            error
        )


        if is_limit_error(error):

            return jsonify({
                "limit": True,
                "error":
                    DAILY_LIMIT_MESSAGE
            }), 429


        return jsonify({
            "error":
                "Voice transcription is temporarily unavailable."
        }), 500


# =========================================================
# SARVAM TEXT TO SPEECH
# =========================================================

@app.route("/tts", methods=["POST"])
def tts():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        text = str(
            data.get("text", "")
        ).strip()


        if not text:

            return jsonify({
                "error":
                    "No text provided."
            }), 400


        if not SARVAM_API_KEY:

            return jsonify({
                "error":
                    "Sarvam API key is not configured."
            }), 500


        # Language from frontend
        language = str(
            data.get("language", "")
        ).strip()


        # Auto detect language
        if not language:

            # Bengali
            if any(
                "\u0980" <= char <= "\u09ff"
                for char in text
            ):

                language = "bn-IN"

            # Hindi
            elif any(
                "\u0900" <= char <= "\u097f"
                for char in text
            ):

                language = "hi-IN"

            # English
            else:

                language = "en-IN"


        payload = {

            "text": text,

            "target_language_code": language,

            "speaker": "shubh",

            "model": "bulbul:v3",

            "speech_sample_rate": 24000,

            "enable_preprocessing": True,

            "output_audio_codec": "wav"
        }


        headers = {

            "api-subscription-key":
                SARVAM_API_KEY,

            "Content-Type":
                "application/json"
        }


        response = requests.post(

            "https://api.sarvam.ai/text-to-speech",

            json=payload,

            headers=headers,

            timeout=60
        )


        if not response.ok:

            print(
                "SARVAM ERROR:",
                response.status_code,
                response.text
            )

            return jsonify({
                "error":
                    "Voice generation is temporarily unavailable."
            }), 500


        result = response.json()


        audio_base64 = (
            result
            .get("audios", [None])[0]
        )


        if not audio_base64:

            return jsonify({
                "error":
                    "No audio was returned by Sarvam."
            }), 500


        return jsonify({

            "audio":
                audio_base64,

            "language":
                language
        })


    except Exception as error:

        print(
            "TTS ERROR:",
            error
        )

        return jsonify({
            "error":
                "Voice generation is temporarily unavailable."
        }), 500


# =========================================================
# TAVILY WEB SEARCH
# =========================================================

@app.route("/search", methods=["POST"])
def search():

    try:

        if not tavily_client:

            return jsonify({
                "error":
                    "Tavily API key is not configured."
            }), 500


        data = request.get_json(
            silent=True
        ) or {}


        query = str(
            data.get("query", "")
        ).strip()


        if not query:

            return jsonify({
                "error":
                    "Please enter a search query."
            }), 400


        result = tavily_client.search(

            query=query,

            search_depth="basic",

            max_results=5
        )


        return jsonify({

            "results":
                result.get(
                    "results",
                    []
                )
        })


    except Exception as error:

        print(
            "SEARCH ERROR:",
            error
        )

        return jsonify({
            "error":
                "Search is temporarily unavailable."
        }), 500


# =========================================================
# IMAGE UNDERSTANDING
# =========================================================

@app.route("/vision", methods=["POST"])
def vision():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        image_data = data.get(
            "image"
        )


        question = str(
            data.get(
                "question",
                "Describe this image."
            )
        ).strip()


        if not image_data:

            return jsonify({
                "error":
                    "No image received."
            }), 400


        if not isinstance(
            image_data,
            str
        ):

            return jsonify({
                "error":
                    "Invalid image data."
            }), 400


        if not image_data.startswith(
            "data:image/"
        ):

            return jsonify({
                "error":
                    "Invalid image format."
            }), 400


        try:

            header, encoded = (
                image_data.split(",", 1)
            )

        except ValueError:

            return jsonify({
                "error":
                    "Invalid image data."
            }), 400


        mime_type = (
            header
            .split(";")[0]
            .replace("data:", "")
        )


        allowed_types = [

            "image/jpeg",

            "image/png",

            "image/webp",

            "image/gif"
        ]


        if mime_type not in allowed_types:

            return jsonify({
                "error":
                    "Unsupported image format."
            }), 400


        try:

            image_bytes = base64.b64decode(
                encoded,
                validate=True
            )

        except Exception:

            return jsonify({
                "error":
                    "Invalid image encoding."
            }), 400


        # Maximum 20 MB
        if len(image_bytes) > (
            20 * 1024 * 1024
        ):

            return jsonify({
                "error":
                    "Image is too large. Maximum size is 20 MB."
            }), 413


        # =================================================
        # GROQ VISION
        # =================================================

        response = (
            groq_client
            .chat
            .completions
            .create(

                model="qwen/qwen3.8-27b",

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            get_system_prompt()
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
                                        image_data
                                }
                            }
                        ]
                    }
                ],

                temperature=0.4,

                max_completion_tokens=1024
            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        return jsonify({

            "answer":
                clean_answer(answer),

            "provider":
                "groq-vision"
        })


    except Exception as error:

        print(
            "VISION ERROR:",
            error
        )


        if is_limit_error(error):

            return jsonify({

                "limit":
                    True,

                "error":
                    DAILY_LIMIT_MESSAGE
            }), 429


        return jsonify({
            "error":
                "Image analysis is temporarily unavailable."
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "service":
            "Hello AI",

        "current_time":
            get_current_datetime()
    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )