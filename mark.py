
import os
os.environ["SDL_AUDIODRIVER"] = "pulse"
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["ALSA_PCM_DEVICE"] = "0"

import json
import edge_tts
import asyncio
import tempfile
import speech_recognition as sr
recognizer = sr.Recognizer()

# 🔥 COMMAND MODE SETTINGS
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.4
recognizer.phrase_threshold = 0.1
recognizer.non_speaking_duration = 0.2

from datetime import datetime
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


SESSION_MEMORY = []
SESSION_LIMIT = 6  # keep last 6 exchanges only

import speech_recognition as sr

recognizer = sr.Recognizer()

# FAST COMMAND MODE
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.4
recognizer.phrase_threshold = 0.1
recognizer.non_speaking_duration = 0.2

def listen_and_transcribe():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.15)

        audio = recognizer.listen(
            source,
            timeout=2,
            phrase_time_limit=4
        )

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# =========================
# 1. LOAD ENV
# =========================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "paradox-memory"



# =========================
# 2. MEMORY FILES
# =========================

MEMORY_DIR = "memory"
IDENTITY_FILE = os.path.join(MEMORY_DIR, "identity.json")
LONG_TERM_FILE = os.path.join(MEMORY_DIR, "long_term.json")


def load_identity():
    with open(IDENTITY_FILE, "r") as f:
        return json.load(f)


def load_long_term():
    with open(LONG_TERM_FILE, "r") as f:
        return json.load(f)


def save_long_term(data):
    with open(LONG_TERM_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_long_term_event(category, content):
    data = load_long_term()
    event = {
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    if category in data:
        data[category].append(event)
        save_long_term(data)


# =========================
# 3. PINECONE (SAFE INIT)
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

vector_store = None  # lazy init


def get_vector_store():
    global vector_store
    if vector_store is None:
        print("Paradox: Initializing memory engine (one-time)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
    return vector_store


# =========================
# 4. LLM
# =========================

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral",
    temperature=0.25,   # lower = professional, less rambling
    top_p=0.9,
    num_ctx=2048        # smaller context = faster responses
)




# =========================
# 5. SYSTEM PROMPT
# =========================

def build_system_prompt(identity, long_term, mode="strict"):
    base = f"""
You are {identity['assistant_name']}, a professional AI assistant.

USER CONTEXT:
Name: {long_term['user']['name']}
Main Goal: {long_term['user']['main_goal']}
Side Project: {long_term['user']['side_project']}

GLOBAL RULES:
- Be accurate
- No hallucinations
- STRICTLY follow the language rule below
- NEVER answer in a different language than instructed
- If Telugu is requested, answer ONLY in spoken Telugu

"""

    modes = {
        "strict": """
MODE: STRICT
- Senior engineer tone
- Short, precise answers
- No explanations unless asked
""",
        "exam": """
MODE: EXAM
- Definition first
- Bullet points
- Simple language
- Exam-ready answers
""",
        "dev": """
MODE: DEV
- Think like a developer
- Give steps, logic, examples
- Code-friendly explanations
""",
        "simple": """
MODE: SIMPLE
- Explain like to a beginner
- Very short sentences
- No jargon
"""
    }

    return base + modes.get(mode, modes["strict"])


# =========================
# 6. CORE CHAT FUNCTION
# =========================
def is_memory_worthy(question, answer):
    keywords = [
        "my goal", "i want", "i decided", "remember this",
        "my project", "my startup", "my plan",
        "from now on", "in future", "i prefer"
    ]

    q = question.lower()
    return any(k in q for k in keywords)

def summarize_session():
    if not SESSION_MEMORY:
        return "No session activity."

    return "Session covered:\n" + "\n".join(
        f"- {m['question']}" for m in SESSION_MEMORY
    )


def is_memory_worthy(question):
    keywords = [
        "my goal", "i want", "i decided", "remember this",
        "my project", "my startup", "my plan",
        "from now on", "in future", "i prefer"
    ]
    return any(k in question.lower() for k in keywords)

def listen_and_transcribe():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.15)

        audio = recognizer.listen(
            source,
            timeout=2,
            phrase_time_limit=4
        )

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


def detect_language(text):
    for c in text:
        if '\u0C00' <= c <= '\u0C7F':
            return "telugu"
        if '\u0900' <= c <= '\u097F':
            return "hindi"
    return "english"
def make_voice_friendly(text, lang):
    prompt = f"""
Rewrite this to sound like a chill, casual human.
Rules:
- Friendly, slightly crazy tone
- Easy to listen, not formal
- Mix Telugu/English or Hindi/English naturally if suitable
- No textbook language

Language: {lang}
Text:
{text}

Rewritten version:
"""
    r = llm.invoke(prompt)
    return r.content.strip()
async def speak_async(text, voice):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(path)

os.system(f"paplay '{path}'")

    os.remove(path)




def speak(answer, user_question):
    lang = detect_language(user_question)

    if lang == "telugu":
        voice = "te-IN-ShrutiNeural"
    elif lang == "hindi":
        voice = "hi-IN-MadhurNeural"
    else:
        voice = "en-US-GuyNeural"  # chill male

    friendly = make_voice_friendly(answer, lang)
    asyncio.run(speak_async(friendly, voice))

def ask_paradox(question):
    # ----------------------------
    # INPUT NORMALIZATION
    # ----------------------------
    question = question.strip()

    for g in ["hi", "hello", "hey", "bro", "buddy"]:
        if question.lower().startswith(g + " "):
            question = question[len(g):].strip()

    # ----------------------------
    # LANGUAGE DETECTION
    # ----------------------------
    lang = detect_language(question)

    # ----------------------------
    # HARD OVERRIDE: NAME
    # ----------------------------
    if "ne peru" in question.lower() or "your name" in question.lower():
        answer = "My name is Paradox."
        print(answer)
        speak(answer, question)
        return

    # ----------------------------
    # LOAD CONTEXT
    # ----------------------------
    identity = load_identity()
    long_term = load_long_term()

    # ----------------------------
    # MEMORY
    # ----------------------------
    store = get_vector_store()
    try:
        docs = store.similarity_search(question, k=2)
        context = "\n".join(d.page_content for d in docs) if docs else ""
    except Exception:
        context = ""

    # ----------------------------
    # PROMPT (ENGLISH THINKING)
    # ----------------------------
    system_prompt = build_system_prompt(identity, long_term, "strict")

    prompt = f"""
{system_prompt}

SYSTEM RULES:
- Answer in English
- No emojis
- Be concise
- Do not overexplain

QUESTION:
{question}

FINAL ANSWER:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(answer)
    speak(answer, question)

    # ----------------------------
    # MEMORY SAVE
    # ----------------------------
    if is_memory_worthy(question):
        store.add_texts([
            f"IMPORTANT USER MEMORY:\n{question}\n{answer}"
        ])
        print("\n[Memory saved]")

    SESSION_MEMORY.append({
        "question": question,
        "answer": answer
    })
# =========================
# 7. MAIN LOOP
# =========================

if __name__ == "__main__":
    print("\n--- PARADOX CORE ONLINE ---")
    print("LLM: ChatGPT | Memory: Ready")
    print("Say 'exit' to standby.\n")

    while True:
        user_input = listen_and_transcribe()

        if not user_input:
            print("Didn't catch that. Try again.")
            continue

        print(f"You (voice): {user_input}")

        if user_input.lower() in ["exit", "quit", "sleep"]:
            SESSION_MEMORY.clear()
            print("Paradox: Session cleared. Standing by.")
            break

        ask_paradox(user_input)
        print()



