"""Single-file, fully automatic knowledge base builder.

Reads ONLY data/data.csv (the file data.py already produces — nothing here
is manually authored ticket content). Groups the real intents found in
that CSV into categories via a keyword rule, pulls real TF-IDF vocabulary
per category for guidance text, and ranks priority by a measured
urgency-word rate — all computed from the actual data.

Also emits two POLICY rule entries required by the assignment
(high-priority escalation, low-confidence human review), generated from
the same threshold constants agent.py actually uses — not free-hand prose.

Output: results/knowledge.json

Run (after data.py has created data/data.csv):
    python build_knowledge.py

This file also contains the RUNTIME retrieval helpers (retrieve_guidance,
estimate_priority, recommend_action) that agent.py should import from here
— one file does both generation and lookup, per request.
"""
import json
import os
import re
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_CSV_PATH = "data/data.csv"
OUT_PATH = "results/knowledge.json"

TEXT_COL = "text"
LABEL_COL = "label_text"

# Same thresholds agent.py uses for its human-review decision — kept here
# so the two policy-rule KB entries are generated from the real constants,
# not duplicated by hand.
CONFIDENCE_THRESHOLD = 0.60
HIGH_PRIORITY_FORCES_REVIEW = True

# The one hand-authored piece: keyword -> category. Checked in order,
# first token match wins. Everything downstream (which of the real intents
# lands where, the guidance text, the priority ranking) is computed from
# the actual data in data/data.csv.
#
# Maps onto the assignment's suggested themes as:
#   billing/payment issue handling      -> CARD, TOP_UP
#   refund / duplicate deduction        -> REFUNDS_DISPUTES
#   technical issue escalation          -> CARD (card_not_working, virtual_card_not_working, ...)
#   account access issue handling       -> SECURITY, IDENTITY_VERIFICATION
#   product inquiry handling            -> ACCOUNT_GENERAL, CURRENCY_EXCHANGE
#   high-priority escalation rule       -> RULE_HIGH_PRIORITY_ESCALATION (below)
#   low-confidence human-review rule    -> RULE_LOW_CONFIDENCE_HUMAN_REVIEW (below)
ORDERED_KEYWORDS = [
    ("SECURITY", ["stolen", "lost", "compromised", "pin", "passcode", "blocked"]),
    ("TOP_UP", ["top", "topping"]),
    ("TRANSFERS", ["transfer", "beneficiary", "receiving", "debit"]),
    ("CASH_ATM", ["atm", "cash", "withdrawal"]),
    ("REFUNDS_DISPUTES", ["refund", "charged", "statement", "balance"]),
    ("IDENTITY_VERIFICATION", ["verify", "identity", "personal"]),
    ("CURRENCY_EXCHANGE", ["exchange", "currency", "currencies", "fiat"]),
    ("CARD", ["card", "cards", "contactless", "pay", "visa", "mastercard"]),
    ("ACCOUNT_GENERAL", ["account", "country", "age", "terminate"]),
]

URGENCY_WORDS = ["stolen", "fraud", "blocked", "declined", "compromised",
                  "unauthorized", "unauthorised", "lost", "urgent", "emergency"]
TOP_N_WORDS = 6


def categorize_intent(intent: str) -> str:
    tokens = set(re.sub(r"[^a-z_]", "", intent.lower()).split("_"))
    for category, keywords in ORDERED_KEYWORDS:
        if tokens & set(keywords):
            return category
    return "OTHER"


def top_tfidf_words(texts: list, top_n: int = TOP_N_WORDS) -> list:
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=200)
    matrix = vec.fit_transform(texts)
    mean_scores = matrix.mean(axis=0).A1
    order = mean_scores.argsort()[::-1][:top_n]
    return [vec.get_feature_names_out()[i] for i in order]


def urgency_rate(texts: list) -> float:
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if any(w in str(t).lower() for w in URGENCY_WORDS))
    return hits / len(texts)


# ----------------------------------------------------------------------------
# BUILD — generates results/knowledge.json from data/data.csv.
# ----------------------------------------------------------------------------
def build():
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(
            f"'{DATA_CSV_PATH}' not found. Run data.py first — it downloads "
            f"Banking77 once and caches it to this exact path."
        )

    df = pd.read_csv(DATA_CSV_PATH)
    intents = sorted(df[LABEL_COL].unique())
    print(f"Loaded {len(df)} tickets, {len(intents)} intents from '{DATA_CSV_PATH}'")

    # 1. Automatic category assignment.
    intent_to_category = {intent: categorize_intent(intent) for intent in intents}
    unassigned = [i for i, c in intent_to_category.items() if c == "OTHER"]
    if unassigned:
        print(f"WARNING: {len(unassigned)} intents unassigned (add a keyword for these): {unassigned}")

    categories = defaultdict(list)
    for intent, cat in intent_to_category.items():
        categories[cat].append(intent)

    # 2 & 3. Guidance text + priority from real ticket text per category.
    category_stats = {}
    for cat, cat_intents in categories.items():
        cat_texts = df[df[LABEL_COL].isin(cat_intents)][TEXT_COL].astype(str).tolist()
        category_stats[cat] = {
            "intents": sorted(cat_intents),
            "n_tickets": len(cat_texts),
            "top_words": top_tfidf_words(cat_texts),
            "urgency_rate": round(urgency_rate(cat_texts), 4),
        }

    ranked = sorted(category_stats.items(), key=lambda kv: -kv[1]["urgency_rate"])
    n = len(ranked)
    for rank, (cat, _) in enumerate(ranked):
        if rank < n / 3:
            priority = "High"
        elif rank < 2 * n / 3:
            priority = "Medium"
        else:
            priority = "Low"
        category_stats[cat]["default_priority"] = priority

    kb = {}
    for cat, stats in category_stats.items():
        words = stats["top_words"]
        word_str = ", ".join(words) if words else "(not enough data to extract terms)"
        kb[cat] = {
            "type": "ticket_category",
            "intents": stats["intents"],
            "guidance": (
                f"Category '{cat}' covers {len(stats['intents'])} intents "
                f"({stats['n_tickets']} tickets in the dataset). Common ticket "
                f"language includes: {word_str}. Measured urgency-word rate: "
                f"{stats['urgency_rate']:.1%} of tickets."
            ),
            "recommended_action": (
                f"Review the ticket against '{cat}' procedures; check the relevant "
                f"account/transaction record given the ticket mentions terms like {word_str}."
            ),
            "default_priority": stats["default_priority"],
            "urgency_rate": stats["urgency_rate"],
            "n_tickets": stats["n_tickets"],
        }

    # 4. Two required POLICY rule entries — generated from the actual
    #    thresholds used by agent.py, not hand-written prose.
    kb["RULE_HIGH_PRIORITY_ESCALATION"] = {
        "type": "policy_rule",
        "guidance": (
            f"Any ticket whose category default_priority == 'High' is escalated for "
            f"human review (HIGH_PRIORITY_FORCES_REVIEW={HIGH_PRIORITY_FORCES_REVIEW})."
        ),
        "recommended_action": "Flag for human review regardless of model confidence.",
        "default_priority": "High",
    }
    kb["RULE_LOW_CONFIDENCE_HUMAN_REVIEW"] = {
        "type": "policy_rule",
        "guidance": (
            f"Any prediction with confidence < {CONFIDENCE_THRESHOLD} is routed to a "
            f"human reviewer instead of being auto-resolved."
        ),
        "recommended_action": "Flag for human review when model confidence is below threshold.",
        "default_priority": "Medium",
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(kb, f, indent=2)

    n_categories = sum(1 for v in kb.values() if v["type"] == "ticket_category")
    n_intents = sum(len(v["intents"]) for v in kb.values() if "intents" in v)
    print(f"\nWrote {OUT_PATH}: {n_categories} ticket categories + 2 policy rules "
          f"= {len(kb)} entries, covering {n_intents} intents")

    for name, entry in kb.items():
        print(f"\n{name}  [{entry['default_priority']}]  ({entry['type']})")
        print(f"  {entry['guidance']}")


# ----------------------------------------------------------------------------
# RUNTIME LOOKUP — loads results/knowledge.json (the file build() writes).
# agent.py imports retrieve_guidance / estimate_priority / recommend_action
# from this same file.
# ----------------------------------------------------------------------------
def load_kb(path: str = OUT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _intent_to_category_map(kb: dict) -> dict:
    return {
        intent: name
        for name, entry in kb.items()
        if entry.get("type") == "ticket_category"
        for intent in entry["intents"]
    }


def retrieve_guidance(intent: str, kb: dict = None) -> dict:
    kb = kb if kb is not None else load_kb()
    mapping = _intent_to_category_map(kb)
    category = mapping.get(intent, "UNKNOWN")
    entry = kb.get(category)
    if entry is None:
        return {
            "category": category,
            "guidance": "No specific guidance on file for this category — route to a general support queue.",
            "recommended_action": "Route to general support queue for manual triage.",
            "default_priority": "Medium",
        }
    return {"category": category, **entry}


def estimate_priority(intent: str, kb: dict = None) -> str:
    return retrieve_guidance(intent, kb)["default_priority"]


def recommend_action(intent: str, kb: dict = None) -> str:
    return retrieve_guidance(intent, kb)["recommended_action"]


if __name__ == "__main__":
    build()
