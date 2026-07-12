import json
import os
from datetime import datetime

KB_PATH = "knowledge_base/knowledge_base.json"

URGENCY_WORDS = ["stolen", "fraud", "blocked", "declined", "compromised",
                 "unauthorized", "unauthorised", "lost", "urgent", "emergency"]


SECURITY_INTENT_TOKENS = {"stolen", "lost", "compromised", "fraud", "blocked",
                          "declined", "hacked", "unrecognised", "unrecognized"}

_STORED_FIELDS = [
    "input_text", "predicted_intent", "predicted_label", "confidence",
    "confidence_explanation", "explanation", "priority", "priority_explanation",
    "recommended_action", "needs_human_review", "human_review_reason",
]


def load_kb(path: str = KB_PATH) -> list:
    """Return the list of stored cases (empty list if the KB doesn't exist yet).
    Used only for storing/updating and for the size counter — never to answer
    a new ticket."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def kb_size(path: str = KB_PATH) -> int:
    return len(load_kb(path))


def save_case(output: dict, path: str = KB_PATH) -> int:
    """Store one triaged ticket's response. A repeat of the exact same input
    text UPDATES its existing entry (the model's fresh prediction wins) rather
    than duplicating it. Returns the KB size after saving."""
    kb = load_kb(path)
    case = {k: output[k] for k in _STORED_FIELDS}
    case["stored_at"] = datetime.now().isoformat(timespec="seconds")

    kb = [c for c in kb if c["input_text"] != case["input_text"]]
    kb.append(case)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(kb, f, indent=2)
    return len(kb)
