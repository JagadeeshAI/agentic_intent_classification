"""Agentic ticket-triage pipeline — the model runs fresh on every ticket, and
ALL FOUR outputs explain themselves with evidence from THIS prediction:

    predicted intent -> which words / which learned tickets drove it
    confidence       -> the model's own probability vs. the runner-up intent
    priority         -> urgency terms found in the ticket + security signals
                        in the predicted intent + model confidence
    human review     -> the exact thresholds and numbers that triggered it

Every response is then stored (or updated, for a repeated input) into
knowledge_base/knowledge_base.json — a write-only case store that is never
read back to answer a new ticket.
"""
import json

from src.explainableAI import load_artifacts, explain_prediction, humanize_label
from src.knowledge import URGENCY_WORDS, SECURITY_INTENT_TOKENS, save_case

CONFIDENCE_THRESHOLD = 0.60      # below this -> flag for human review
LOW_PRIORITY_CONFIDENCE = 0.80   # confident + no urgency evidence -> Low
HIGH_PRIORITY_FORCES_REVIEW = True


def estimate_priority(text: str, intent: str, confidence: float):
    """Per-prediction priority with the evidence spelled out. Returns
    (priority, explanation) — the explanation cites what was actually found
    in THIS ticket and THIS prediction, not a pre-built table."""
    readable = humanize_label(intent)
    text_hits = sorted({w for w in URGENCY_WORDS if w in text.lower()})
    intent_hits = sorted(set(intent.lower().split("_")) & SECURITY_INTENT_TOKENS)

    if text_hits or intent_hits:
        reasons = []
        if text_hits:
            reasons.append(f"the ticket itself contains the urgency term(s) {text_hits}")
        if intent_hits:
            reasons.append(f"the model predicted '{readable}', whose meaning is "
                           f"security/loss-critical ({intent_hits})")
        return "High", "High — " + " and ".join(reasons) + "."

    if confidence >= LOW_PRIORITY_CONFIDENCE:
        return "Low", (f"Low — no urgency terms in the ticket, nothing "
                       f"security-critical about the predicted intent "
                       f"'{readable}', and the model is confident "
                       f"({confidence:.1%} >= {LOW_PRIORITY_CONFIDENCE:.0%}), so this "
                       f"reads as a routine/informational request.")

    return "Medium", (f"Medium — no urgency evidence was found in the ticket or the "
                      f"predicted intent '{readable}', but model confidence "
                      f"{confidence:.1%} is below {LOW_PRIORITY_CONFIDENCE:.0%}, so it "
                      f"is kept above Low pending a closer look.")


def recommend_action(intent: str, priority: str) -> str:
    readable = humanize_label(intent)
    if priority == "High":
        return f"Escalate immediately to the team handling '{readable}'."
    if priority == "Low":
        return f"Respond with standard guidance for '{readable}' (routine request)."
    return f"Route to the queue handling '{readable}' for normal processing."


def run_agent(text: str, pipe, id2label: dict, store: bool = True) -> dict:
    """Run the full agent pipeline on one ticket and return structured JSON.
    Every step is recorded in `trace` so the reasoning is visible, not hidden."""
    trace = []

    # Step 1 — classify + explain with model evidence.
    result = explain_prediction(pipe, id2label, text)
    predicted_intent = result["predicted_label"]
    predicted_readable = result["predicted_label_readable"]
    confidence = result["confidence"]
    trace.append({
        "step": 1, "name": "classify",
        "detail": f"Predicted intent='{predicted_readable}' "
                  f"(confidence={confidence:.4f}). {result['explanation']}",
    })

    # Step 2 — explain the confidence number itself (model's probability margin).
    trace.append({
        "step": 2, "name": "explain_confidence",
        "detail": result["confidence_explanation"],
    })

    # Step 3 — priority from THIS prediction, evidence cited.
    priority, priority_explanation = estimate_priority(text, predicted_intent, confidence)
    trace.append({
        "step": 3, "name": "estimate_priority",
        "detail": priority_explanation,
    })

    # Step 4 — recommended action.
    action = recommend_action(predicted_intent, priority)
    trace.append({
        "step": 4, "name": "recommend_action",
        "detail": f"Recommended action: \"{action}\"",
    })

    # Step 5 — decide human review, numbers and reasons cited.
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    is_high_priority = (priority == "High") and HIGH_PRIORITY_FORCES_REVIEW
    needs_human_review = low_confidence or is_high_priority
    reasons = []
    if low_confidence:
        reasons.append(f"the model's confidence {confidence:.1%} is below the "
                       f"{CONFIDENCE_THRESHOLD:.0%} auto-resolve threshold")
    if is_high_priority:
        reasons.append(f"priority is High ({priority_explanation.removeprefix('High — ')})")
    review_reason = (
        "Needs human review because " + " and ".join(reasons) + "."
        if reasons else
        f"No human review needed — the model's confidence {confidence:.1%} clears the "
        f"{CONFIDENCE_THRESHOLD:.0%} threshold and no High-priority evidence was found."
    )
    trace.append({
        "step": 5, "name": "decide_human_review",
        "detail": f"needs_human_review={needs_human_review}. {review_reason}",
    })

    # Step 6 — assemble structured output and store/update it in the KB.
    output = {
        "input_text": text,
        "predicted_intent": predicted_intent,
        "predicted_label": predicted_readable,
        "confidence": confidence,
        "confidence_explanation": result["confidence_explanation"],
        "explanation": result["explanation"],
        "priority": priority,
        "priority_explanation": priority_explanation,
        "recommended_action": action,
        "needs_human_review": needs_human_review,
        "human_review_reason": review_reason,
        "agent_trace": trace,
    }
    if store:
        n_cases = save_case(output)
        trace.append({
            "step": 6, "name": "store_knowledge_base_case",
            "detail": f"Response stored to knowledge base ({n_cases} case(s) total).",
        })
    return output


def print_trace(output: dict):
    print(f"\n{'='*70}")
    print(f"Ticket: {output['input_text']!r}")
    print(f"{'='*70}")
    print("Agent trace:")
    for step in output["agent_trace"]:
        print(f"  [{step['step']}] {step['name']:<28} {step['detail']}")
    print("\nFinal structured output:")
    printable = {k: v for k, v in output.items() if k != "agent_trace"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    import pandas as pd

    pipe, id2label, target, model_name = load_artifacts()
    print(f"Loaded '{model_name}' pipeline  |  target='{target}'  |  classes={len(id2label)}")

    example_tickets = (pd.read_csv("data/data.csv")["text"]
                       .sample(6).astype(str).tolist())
    for ticket in example_tickets:
        output = run_agent(ticket, pipe, id2label)
        print_trace(output)
