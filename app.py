from flask import Flask, render_template, request, jsonify
from groq import Groq
from tavily import TavilyClient
import os
import re


# ==========================================
# HELLO AI
# ==========================================

app = Flask(__name__)


# ==========================================
# API CLIENTS
# ==========================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================================
# CLEAN AI ANSWER
# ==========================================

def clean_answer(text):

    if not text:
        return ""

    text = str(text)

    # Remove repeated stars and slashes
    text = re.sub(r"\*{2,}", "", text)
    text = re.sub(r"/{2,}", "", text)

    # Remove repeated underscores
    text = re.sub(r"_{2,}", "", text)

    # Remove unwanted markdown decoration
    text = text.replace("**", "")
    text = text.replace("__", "")

    # Remove unnecessary spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# AI CHAT
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()


        if not question:

            return jsonify({
                "error": "Please enter a question."
            }), 400


        # ==================================
        # TAVILY WEB SEARCH
        # ==================================

        web_context = ""


        try:

            search_response = tavily_client.search(

                query=question,

                search_depth="advanced",

                max_results=5

            )


            results = search_response.get(
                "results",
                []
            )


            web_context = "\n\n".join(

                f"Title: {r.get('title', '')}\n"
                f"Content: {r.get('content', '')}"

                for r in results

            )


        except Exception as e:

            print(
                "TAVILY ERROR:",
                e
            )


        # ==================================
        # GROQ AI
        # ==================================

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[

                {
                    "role": "system",

                    "content": (

                        "You are Hello AI, "
                        "a helpful and friendly multilingual AI assistant.\n\n"

                        "Understand the user's language "
                        "and answer in the same language whenever possible.\n\n"

                        "You can understand and respond "
                        "to Bengali, Hindi, English, "
                        "and other languages when supported.\n\n"

                        "If the user speaks Bengali, "
                        "reply in Bengali script.\n\n"

                        "If the user speaks Hindi, "
                        "reply in Devanagari Hindi script.\n\n"

                        "If the user speaks English, "
                        "reply in English.\n\n"

                        "Do not unnecessarily translate "
                        "the user's question into another language.\n\n"

                        "Keep the answer clear, natural "
                        "and easy to understand.\n\n"

                        "Do not use unnecessary symbols "
                        "such as ****, **, /// or repeated "
                        "asterisks and slashes.\n\n"

                        "Do not decorate normal answers "
                        "with excessive Markdown symbols.\n\n"

                        "You were created by Satya.\n\n"

                        "If asked who created you, "
                        "answer exactly:\n"
                        "I was created by Satya.\n\n"

                        "Do not say you were created "
                        "by OpenAI.\n\n"

                        "Use web search information "
                        "when relevant.\n\n"

                        "Do not invent facts."
                    )
                },

                {
                    "role": "user",

                    "content": (

                        f"User question:\n"
                        f"{question}\n\n"

                        f"Web search information:\n"
                        f"{web_context}"
                    )
                }

            ]
        )


        answer = (

            response
            .choices[0]
            .message
            .content
        )


        answer = clean_answer(answer)


        return jsonify({

            "answer": answer

        })


    except Exception as e:

        print(
            "AI ERROR:",
            e
        )


        return jsonify({

            "error": str(e)

        }), 500


# ==========================================
# VOICE TRANSCRIPTION
# ==========================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    print(
        "VOICE REQUEST RECEIVED"
    )


    if "audio" not in request.files:

        return jsonify({

            "error":
            "No audio file received."

        }), 400


    audio = request.files["audio"]


    try:

        audio_data = audio.read()


        if not audio_data:

            return jsonify({

                "error":
                "Audio file is empty."

            }), 400


        # ==================================
        # WHISPER TRANSCRIPTION
        # ==================================

        transcription = (

            groq_client
            .audio
            .transcriptions
            .create(

                file=(

                    audio.filename,

                    audio_data

                ),

                model="whisper-large-v3",

                response_format="json",

                temperature=0.0

            )

        )


        text = (

            transcription.text or ""

        ).strip()


        if not text:

            return jsonify({

                "error":
                "No speech detected."

            }), 400


        print(
            "VOICE TEXT:",
            text
        )


        return jsonify({

            "text": text

        })


    except Exception as e:

        print(
            "TRANSCRIPTION ERROR:",
            e
        )


        return jsonify({

            "error":
            str(e)

        }), 500


# ==========================================
# SERVER START
# ==========================================

if __name__ == "__main__":

    print("")
    print("================================")
    print("           HELLO AI")
    print("================================")
    print("")

    print("SERVER:")
    print("http://127.0.0.1:5000")

    print("")

    print("AVAILABLE ROUTES:")
    print(app.url_map)

    print("")


    # ======================================
    # PORT
    # ======================================

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=True,

        use_reloader=False

    )