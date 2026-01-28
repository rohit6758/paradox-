import speech_recognition as sr
import asyncio
import edge_tts
import tempfile
import os
from langchain_ollama import ChatOllama

# -------- LLM --------
llm = ChatOllama(model="mistral", temperature=0.3)

# -------- MIC --------
r = sr.Recognizer()
r.energy_threshold = 150
r.dynamic_energy_threshold = False
r.pause_threshold = 0.4

WAKE_WORDS = ["hey paradox", "paradox"]

def is_wake_word(text):
    return any(w in text.lower() for w in WAKE_WORDS)

def listen():
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=3, phrase_time_limit=4)
        return r.recognize_google(audio)
    except:
        return ""

# -------- SPEAK --------
async def speak(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name

    tts = edge_tts.Communicate(text=text, voice="en-US-GuyNeural")
    await tts.save(path)

    os.system(f"ffplay -nodisp -autoexit -loglevel quiet '{path}'")
    os.remove(path)

# -------- LOOP --------
print("Say: Hey Paradox")

active = False

while True:
    heard = listen()
    if not heard:
        continue

    print("You:", heard)

    if not active:
        if is_wake_word(heard):
            active = True
            asyncio.run(speak("Yeah bro, cheppu"))
        continue

    if heard.lower() in ["exit", "sleep"]:
        active = False
        asyncio.run(speak("Okay, waiting"))
        continue

    response = llm.invoke(heard).content
    print("Paradox:", response)
    asyncio.run(speak(response))
    active = False
