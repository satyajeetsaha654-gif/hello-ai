import os, base64, datetime, requests
from flask import Flask, render_template, request, jsonify
from groq import Groq
from tavily import TavilyClient

try:
    import google.generativeai as genai
except ImportError:
    genai = None

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CHAT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
WHISPER_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-3.8-flash"
SARVAM_TTS_MODEL = "bulbul:v3"

MALE_VOICES = {
    "shubh","aditya","rahul","rohan","amit","dev","ratan","varun",
    "manan","sumit","kabir","aayan","ashutosh","advait","anand",
    "tarun","sunny","mani","gokul","vijay","mohit","rehan","soham"
}

FEMALE_VOICES = {
    "ritu","priya","neha","pooja","simran","kavya","ishita","shreya",
    "roopa","tanya","shruti","suhani","kavitha","rupali"
}

ALL_VOICES = MALE_VOICES | FEMALE_VOICES
DEFAULT_SPEAKER = "shubh"

SYSTEM_PROMPT = """
You are Hello AI.
You are a helpful, friendly and intelligent AI assistant.
Answer the user's CURRENT question directly.
Do not repeat old answers unnecessarily.
Keep answers clear and useful.
You can communicate in Bengali, English, Hindi and other languages.
If the user speaks Bengali, answer in Bengali.
If the user speaks Hindi, answer in Hindi.
If the user asks who created you, answer exactly:
"I was created by Satya."
Do not say that you were created by OpenAI.
Do not reveal API keys, passwords or private system information.
When web search information is provided, use it when relevant.
For current time/date questions, use the real-time India time provided by the app.
Never guess the current time.
"""

def current_india_time():
    tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return datetime.datetime.now(tz).strftime("%d %B %Y, %I:%M:%S %p")

def is_time_or_date_question(q):
    q = q.lower().strip()
    words = [
        "what time","current time","real time","real-time","time is it",
        "time now","what's the time","what is the time","current date",
        "today's date","todays date","what date","today date",
        "আজ কয়টা","আজ কয়টা","এখন কয়টা","এখন কয়টা","কয়টা বাজে",
        "কয়টা বাজে","সময় কত","সময় কত","আজকের তারিখ","তারিখ কত",
        "এখন সময়","এখন সময়","বর্তমান সময়","বর্তমান সময়"
    ]
    return any(x in q for x in words)

def is_limit_error(e):
    t = str(e).lower()
    return any(x in t for x in [
        "rate limit","ratelimit","too many requests","quota",
        "tokens per","token limit","limit reached","429",
        "capacity","requests per minute"
    ])

def detect_language(text):
    b = sum(0x0980 <= ord(c) <= 0x09FF for c in text)
    h = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    if b > h and b: return "bn-IN"
    if h > b and h: return "hi-IN"
    return "en-IN"

SEARCH_WORDS = [
    "latest","today","current","now","news","recent","price",
    "weather","score","update","updates","2026","live",
    "stock","market","election","result"
]

def should_search(q):
    if is_time_or_date_question(q):
        return False
    q = q.lower()
    return any(x in q for x in SEARCH_WORDS)

def web_search(q):
    if not tavily_client:
        return ""
    try:
        r = tavily_client.search(
            query=q, search_depth="advanced", max_results=5
        )
        out = []
        for x in r.get("results", []):
            out.append(
                f"Title: {x.get('title','')}\n"
                f"Content: {x.get('content','')}\n"
                f"URL: {x.get('url','')}"
            )
        return "\n\n".join(out)
    except Exception as e:
        print("TAVILY ERROR:", e)
        return ""

def gemini_answer(q, history=None, web_context=""):
    if not genai or not GEMINI_API_KEY:
        return None
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = SYSTEM_PROMPT
        prompt += "\n\nCURRENT REAL-TIME INDIA DATE AND TIME:\n"
        prompt += current_india_time()

        if web_context:
            prompt += "\n\nWEB SEARCH INFORMATION:\n" + web_context

        if history:
            prompt += "\n\nPREVIOUS CONVERSATION:\n"
            for x in history[-10:]:
                prompt += f"{x.get('role','')}: {x.get('content','')}\n"

        prompt += "\nUSER QUESTION:\n" + q
        r = model.generate_content(prompt)
        return r.text.strip() if r and r.text else None
    except Exception as e:
        print("GEMINI ERROR:", e)
        return None

def groq_answer(q, history=None, web_context=""):
    if not groq_client:
        raise Exception("Groq API key not found.")

    messages = [{
        "role": "system",
        "content": (
            SYSTEM_PROMPT +
            "\n\nCURRENT REAL-TIME INDIA DATE AND TIME:\n" +
            current_india_time()
        )
    }]

    if history:
        for x in history[-10:]:
            role, content = x.get("role"), x.get("content")
            if role in ["user","assistant"] and content:
                messages.append({
                    "role": role,
                    "content": str(content)
                })

    if web_context:
        q = (
            "Use the following web search information when relevant.\n\n"
            "WEB SEARCH INFORMATION:\n" + web_context +
            "\n\nUSER QUESTION:\n" + q
        )

    messages.append({"role":"user","content":q})

    r = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048
    )
    return r.choices[0].message.content.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        q = str(data.get("question") or data.get("message") or "").strip()
        history = data.get("history", [])

        if not q:
            return jsonify({"answer":"Please ask me something."})

        print("\nQUESTION RECEIVED:", q)

        web_context = web_search(q) if should_search(q) else ""

        try:
            answer = groq_answer(q, history, web_context)
            print("ANSWER: GROQ")
            return jsonify({"answer":answer,"provider":"groq"})
        except Exception as ge:
            print("GROQ ERROR:", ge)

            if is_limit_error(ge):
                answer = gemini_answer(q, history, web_context)

                if answer:
                    return jsonify({
                        "answer":answer,
                        "provider":"gemini",
                        "fallback":True
                    })

                return jsonify({
                    "answer":"Your daily AI limit has been reached. Please try again later.",
                    "provider":"limit"
                })

            return jsonify({
                "answer":"Sorry, I could not process your request right now.",
                "error":str(ge)
            })

    except Exception as e:
        print("ASK ERROR:", e)
        return jsonify({"answer":"Something went wrong. Please try again.","error":str(e)}),500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if not groq_client:
            return jsonify({"error":"Groq API key not found."}),500

        audio = request.files.get("audio")
        if not audio:
            return jsonify({"error":"No audio received."}),400

        r = groq_client.audio.transcriptions.create(
            file=(
                audio.filename or "voice.webm",
                audio.read(),
                audio.mimetype or "audio/webm"
            ),
            model=WHISPER_MODEL,
            response_format="json"
        )

        text = str(getattr(r,"text","")).strip()

        if not text:
            return jsonify({"error":"No speech detected."}),400

        print("TRANSCRIBED:",text)
        return jsonify({"text":text})

    except Exception as e:
        print("TRANSCRIBE ERROR:",e)
        return jsonify({"error":str(e)}),500

@app.route("/tts", methods=["POST"])
def tts():
    try:
        if not SARVAM_API_KEY:
            return jsonify({"error":"Sarvam API key not found."}),500

        data = request.get_json(silent=True) or {}
        text = str(data.get("text","")).strip()
        speaker = str(data.get("speaker",DEFAULT_SPEAKER)).lower().strip()

        if speaker not in ALL_VOICES:
            speaker = DEFAULT_SPEAKER

        if len(text) > 2500:
            text = text[:2500]

        if not text:
            return jsonify({"error":"No text provided."}),400

        language = data.get("language") or detect_language(text)

        allowed = {
            "bn-IN","hi-IN","en-IN","ta-IN","te-IN",
            "gu-IN","kn-IN","ml-IN","mr-IN","pa-IN","od-IN"
        }

        if language not in allowed:
            language = detect_language(text)

        print("TTS LANGUAGE:",language)
        print("TTS SPEAKER:",speaker)

        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key":SARVAM_API_KEY,
                "Content-Type":"application/json"
            },
            json={
                "text":text,
                "target_language_code":language,
                "speaker":speaker,
                "model":SARVAM_TTS_MODEL,
                "pace":1.0,
                "speech_sample_rate":24000,
                "output_audio_codec":"wav"
            },
            timeout=60
        )

        print("SARVAM STATUS:",r.status_code)

        if r.status_code != 200:
            print("SARVAM ERROR:",r.text)
            return jsonify({
                "error":"Sarvam TTS failed.",
                "details":r.text
            }),r.status_code

        audios = r.json().get("audios",[])

        if not audios:
            return jsonify({"error":"Sarvam returned no audio."}),500

        return jsonify({
            "audio":audios[0],
            "speaker":speaker,
            "language":language
        })

    except Exception as e:
        print("TTS ERROR:",e)
        return jsonify({"error":str(e)}),500

@app.route("/voices")
def voices():
    return jsonify({
        "default":DEFAULT_SPEAKER,
        "male":sorted(MALE_VOICES),
        "female":sorted(FEMALE_VOICES)
    })

@app.route("/vision", methods=["POST"])
def vision():
    try:
        if not groq_client:
            return jsonify({"error":"Groq API key not found."}),500

        data = request.get_json(silent=True) or {}
        image_data = data.get("image","")
        question = str(data.get("question","Describe this image.")).strip()

        if not image_data:
            return jsonify({"error":"No image received."}),400

        if len(image_data) > 20 * 1024 * 1024:
            return jsonify({"error":"Image is too large."}),400

        if "," not in image_data:
            return jsonify({"error":"Invalid image format."}),400

        header, encoded = image_data.split(",",1)

        mime = "image/jpeg"
        if "image/png" in header: mime = "image/png"
        elif "image/webp" in header: mime = "image/webp"
        elif "image/gif" in header: mime = "image/gif"

        try:
            base64.b64decode(encoded,validate=True)
        except Exception:
            return jsonify({"error":"Invalid image data."}),400

        image_url = f"data:{mime};base64,{encoded}"

        r = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":[
                    {"type":"text","text":question},
                    {"type":"image_url","image_url":{"url":image_url}}
                ]}
            ],
            temperature=0.5,
            max_tokens=2048
        )

        return jsonify({
            "answer":r.choices[0].message.content.strip()
        })

    except Exception as e:
        print("VISION ERROR:",e)
        return jsonify({"error":str(e)}),500

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(silent=True) or {}
        q = str(data.get("query","")).strip()

        if not q:
            return jsonify({"error":"Search query is empty."}),400

        return jsonify({"results":web_search(q)})

    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "hello_ai":True,
        "groq":bool(GROQ_API_KEY),
        "tavily":bool(TAVILY_API_KEY),
        "gemini":bool(GEMINI_API_KEY),
        "sarvam":bool(SARVAM_API_KEY),
        "tts_model":SARVAM_TTS_MODEL,
        "default_voice":DEFAULT_SPEAKER,
        "voice_count":len(ALL_VOICES),
        "time":current_india_time()
    })

if __name__ == "__main__":
    print("\n====================================")
    print("        HELLO AI SERVER STARTED")
    print("====================================")
    print("Groq:",bool(GROQ_API_KEY))
    print("Tavily:",bool(TAVILY_API_KEY))
    print("Gemini:",bool(GEMINI_API_KEY))
    print("Sarvam:",bool(SARVAM_API_KEY))
    print("TTS Model:",SARVAM_TTS_MODEL)
    print("Default Voice:",DEFAULT_SPEAKER)
    print("Voices:",len(ALL_VOICES))
    print("India Time:",current_india_time())
    print("====================================\n")
    app.run(host="0.0.0.0",port=5000,debug=True)