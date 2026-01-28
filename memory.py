import json
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("memory")
IDENTITY_FILE = MEMORY_DIR / "identity.json"
LONG_TERM_FILE = MEMORY_DIR / "long_term.json"

def load_identity():
    with open(IDENTITY_FILE, "r") as f:
        return json.load(f)

def load_long_term():
    with open(LONG_TERM_FILE, "r") as f:
        return json.load(f)

def save_long_term(data):
    with open(LONG_TERM_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_event(category, content):
    data = load_long_term()
    event = {
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    data[category].append(event)
    save_long_term(data)
