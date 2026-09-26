from flask import Flask, render_template, request, jsonify
from groq import Groq
from tavily import TavilyClient
import os


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
                        "a helpful and friendly AI assistant. "

                        "Answer the user's current "
                        "question directly. "

                        "Give clear and simple answers. "

                        "You were created by Satya. "

                        "If asked who created you, "
                        "answer: "
                        "'I was created by Satya.' "

                        "Do not say you were created "
                        "by OpenAI. "

                        "Use web search information "
                        "when relevant. "

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

            "text":
                text

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

    print(
        "SERVER:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("")

    print(
        "AVAILABLE ROUTES:"
    )

    print(
        app.url_map
    )

    print("")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True,

        use_reloader=False
    )