from flask import Flask, render_template, request, jsonify
from groq import Groq

client = Groq()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify({"error": "Please enter a question."}), 400

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Hello AI, a helpful and friendly AI assistant. "
                        "Answer the user's current question directly. "
                        "Do not repeat old answers. "
                        "Give clear and simple answers. "
                        "You were created by Satya. "
                        "If asked who created you, answer: "
                        "'I was created by Satya.' "
                        "Do not say you were created by OpenAI."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return jsonify({
            "answer": response.choices[0].message.content
        })

    except Exception as e:
        print("AI ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received."}), 400

    audio = request.files["audio"]

    try:
        audio_data = audio.read()

        if not audio_data:
            return jsonify({"error": "Audio file is empty."}), 400

        transcription = client.audio.transcriptions.create(
            file=(audio.filename, audio_data),
            model="whisper-large-v3",
            response_format="json",
            temperature=0.0
        )

        text = (transcription.text or "").strip()

        if not text:
            return jsonify({"error": "No speech detected."}), 400

        return jsonify({"text": text})

    except Exception as e:
        print("TRANSCRIPTION ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("HELLO AI SERVER STARTED")
    print("http://127.0.0.1:5000")

    app.run(
        debug=True,
        use_reloader=False
    )