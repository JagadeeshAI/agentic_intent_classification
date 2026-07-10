"""Agentic ticket-triage pipeline.

Chains the explicit steps required by the assignment, with a VISIBLE trace
(a list of steps actually executed, printed and included in the output) —
not a black box that just returns a final answer.

Pipeline:
    1. Classify ticket text -> predicted intent (+ confidence)      [explainableAI.py]
    2. Map intent -> category, retrieve matching KB guidance         [build_knowledge.py]
    3. Estimate priority (rule-based on category)                    [build_knowledge.py]
    4. Decide recommended action                                     [build_knowledge.py]
    5. Decide if this needs human review (low confidence OR High priority)
    6. Emit structured JSON output

Requirements:
    pip install scikit-learn joblib
Run classification.py once first — it saves ./checkpoint/best_pipeline.joblib,
which this script loads (no retraining here). Run build_knowledge.py once
first too — it saves ./results/knowledge.json, which this script reads via
build_knowledge.py's retrieve_guidance/estimate_priority/recommend_action.
"""
import json

from src.explainableAI import load_artifacts, explain_prediction
from src.build_knowledge import retrieve_guidance, estimate_priority, recommend_action

CONFIDENCE_THRESHOLD = 0.60     # below this -> flag for human review
HIGH_PRIORITY_FORCES_REVIEW = True   # High-priority tickets always get a human glance


def run_agent(text: str, pipe, id2label: dict, top_n_words: int = 5) -> dict:
    """Run the full agent pipeline on one ticket and return structured JSON.
    Every step is recorded in `trace` so the reasoning is visible, not hidden."""
    trace = []

    # Step 1 — classify + explain.
    result = explain_prediction(pipe, id2label, text, top_n=top_n_words)
    predicted_intent = result["predicted_label"]
    confidence = result["confidence"]
    trace.append({
        "step": 1, "name": "classify",
        "detail": f"Predicted intent='{predicted_intent}' with confidence={confidence:.4f}",
    })

    # Step 2 — retrieve KB guidance (also gives us the category).
    kb_entry = retrieve_guidance(predicted_intent)
    trace.append({
        "step": 2, "name": "retrieve_knowledge_base",
        "detail": f"Mapped intent -> category='{kb_entry['category']}', "
                  f"retrieved guidance: \"{kb_entry['guidance']}\"",
    })

    # Step 3 — estimate priority (rule-based; documented heuristic, not learned).
    priority = estimate_priority(predicted_intent)
    trace.append({
        "step": 3, "name": "estimate_priority",
        "detail": f"Priority='{priority}' (rule-based on category='{kb_entry['category']}')",
    })

    # Step 4 — recommended action.
    action = recommend_action(predicted_intent)
    trace.append({
        "step": 4, "name": "recommend_action",
        "detail": f"Recommended action: \"{action}\"",
    })

    # Step 5 — decide human review.
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    is_high_priority = (priority == "High") and HIGH_PRIORITY_FORCES_REVIEW
    needs_human_review = low_confidence or is_high_priority
    reasons = []
    if low_confidence:
        reasons.append(f"confidence {confidence:.4f} < threshold {CONFIDENCE_THRESHOLD}")
    if is_high_priority:
        reasons.append("priority is High")
    review_reason = "; ".join(reasons) if reasons else "confidence sufficient and priority not High"
    trace.append({
        "step": 5, "name": "decide_human_review",
        "detail": f"needs_human_review={needs_human_review} ({review_reason})",
    })

    # Step 6 — assemble final structured output.
    output = {
        "input_text": text,
        "predicted_intent": predicted_intent,
        "predicted_category": kb_entry["category"],
        "priority": priority,
        "confidence": confidence,
        "explanation": result["explanation"],
        "top_contributing_words": result["top_contributing_words"],
        "retrieved_guidance": kb_entry["guidance"],
        "recommended_action": action,
        "needs_human_review": needs_human_review,
        "agent_trace": trace,
    }
    return output


def print_trace(output: dict):
    print(f"\n{'='*70}")
    print(f"Ticket: {output['input_text']!r}")
    print(f"{'='*70}")
    print("Agent trace:")
    for step in output["agent_trace"]:
        print(f"  [{step['step']}] {step['name']:<24} {step['detail']}")
    print("\nFinal structured output:")
    printable = {k: v for k, v in output.items() if k != "agent_trace"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    pipe, id2label, target, model_name = load_artifacts()
    print(f"Loaded '{model_name}' pipeline  |  target='{target}'  |  classes={len(id2label)}")

    example_tickets = [
        "my card payment was declined twice, why is this happening",
        "i think someone stole my card and used it",
        "how long does it take for a transfer to arrive",
        "i was charged twice for the same transaction",
        "how do i verify my identity on the app",
        "what is the minimum age to open an account",
    ]
    for ticket in example_tickets:
        output = run_agent(ticket, pipe, id2label)
        print_trace(output)
