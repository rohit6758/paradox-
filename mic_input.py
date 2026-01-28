import speech_recognition as sr

recognizer = sr.Recognizer()
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.4
recognizer.phrase_threshold = 0.1
recognizer.non_speaking_duration = 0.2

WAKE_WORDS = ["hey paradox", "hi paradox", "paradox"]
STOP_WORDS = ["over paradox", "stop paradox"]

def is_wake_word(text: str) -> bool:
    text = text.lower()
    return any(w in text for w in WAKE_WORDS)

def is_stop_word(text: str) -> bool:
    text = text.lower()
    return any(w in text for w in STOP_WORDS)

def listen_and_transcribe() -> str:
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.15)
            audio = recognizer.listen(
                source,
                timeout=2,
                phrase_time_limit=4
            )
        return recognizer.recognize_google(audio)
    except:
        return ""
