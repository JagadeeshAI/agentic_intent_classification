"""Entry point for the ticket-triage system.

Loads the trained pipeline from checkpoint/ if it exists; ONLY if the
checkpoint is missing does it run the one-time training (src/classify.py).
The knowledge base is NOT pre-built: it starts empty and grows at
knowledge_base/knowledge_base.json as tickets are triaged — each response is
stored (or updated for a repeated ticket); the model runs fresh every time.

Two ways to run (both from the project root):
    streamlit run app.py    -> web UI at localhost:8501
    python app.py           -> terminal demo + interactive prompt
"""
import os

from src.classify import CKPT_DIR, train
from src.explainableAI import load_artifacts, ensure_similarity_index
from src.knowledge import KB_PATH, kb_size
from src.agent import run_agent, print_trace

PIPELINE_PATH = os.path.join(CKPT_DIR, "best_pipeline.joblib")
LABEL_MAPS_PATH = os.path.join(CKPT_DIR, "label_maps.json")
DATA_CSV_PATH = "data/data.csv"
N_EXAMPLES = 6


def sample_example_tickets(n: int = N_EXAMPLES) -> list:
    """Draw n random real tickets from the cached dataset."""
    import pandas as pd
    return pd.read_csv(DATA_CSV_PATH)["text"].sample(n).astype(str).tolist()


def ensure_artifacts():
    """Train only if the checkpoint is missing; load and return the artifacts."""
    if os.path.exists(PIPELINE_PATH) and os.path.exists(LABEL_MAPS_PATH):
        print(f"Found existing checkpoint at {CKPT_DIR}/ — loading it (no training).")
    else:
        print(f"No checkpoint at {PIPELINE_PATH} — training from scratch (one-time)...")
        train()

    pipe, id2label, target, model_name = load_artifacts()
    # Embedding-based best model: build the similarity-explanation index once
    # (encodes the training set; no-op for TF-IDF models or if already cached).
    ensure_similarity_index(pipe)
    print(f"Loaded '{model_name}' pipeline  |  target='{target}'  |  classes={len(id2label)}")
    return pipe, id2label, model_name


# ----------------------------------------------------------------------------
# Streamlit UI — used when launched via `streamlit run app.py`.
# ----------------------------------------------------------------------------
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Ticket Triage Agent", page_icon="🎫", layout="wide")
    st.title("🎫 Banking Ticket Triage Agent")
    st.caption("Classifies a support ticket and explains every output — intent, "
               "confidence, priority, and the human-review decision — with "
               "evidence from the model's own prediction. Every response is "
               "stored into a growing knowledge base (the model still runs "
               "fresh on each ticket).")

    @st.cache_resource(show_spinner="Loading model checkpoint (training only if missing)...")
    def get_artifacts():
        return ensure_artifacts()

    pipe, id2label, model_name = get_artifacts()
    st.sidebar.success(f"Model: **{model_name}**\n\nClasses: **{len(id2label)}**")
    st.sidebar.markdown(f"Checkpoint: `{PIPELINE_PATH}`")
    st.sidebar.info(f"Knowledge base: **{kb_size()}** stored case(s)\n\n`{KB_PATH}`")

    if "examples" not in st.session_state:
        st.session_state.examples = sample_example_tickets()
    if st.button("🔀 New examples"):
        st.session_state.examples = sample_example_tickets()

    example = st.selectbox("Try a real ticket drawn randomly from the dataset:",
                           ["(type your own below)"] + st.session_state.examples)
    default_text = "" if example == "(type your own below)" else example
    text = st.text_area("Ticket text", value=default_text,
                        placeholder="e.g. i was charged twice for the same transaction")

    if st.button("Triage ticket", type="primary") and text.strip():
        output = run_agent(text.strip(), pipe, id2label)

        # Full-width readable label — st.metric truncates long intent names.
        st.markdown(f"### Predicted intent: {output['predicted_label']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Priority", output["priority"])
        col2.metric("Confidence", f"{output['confidence']:.2%}")
        col3.metric("Human review", "Yes" if output["needs_human_review"] else "No")

        st.subheader("Why this intent")
        st.write(output["explanation"])

        st.subheader("Why this confidence")
        st.write(output["confidence_explanation"])

        st.subheader("Why this priority")
        st.write(output["priority_explanation"])

        st.subheader("Why human review" if output["needs_human_review"]
                     else "Why no human review")
        if output["needs_human_review"]:
            st.warning("⚠️ " + output["human_review_reason"])
        else:
            st.success("✅ " + output["human_review_reason"])

        st.markdown(f"**Recommended action:** {output['recommended_action']}")

        with st.expander("Agent trace (all steps)"):
            for step in output["agent_trace"]:
                st.markdown(f"**[{step['step']}] {step['name']}** — {step['detail']}")
        with st.expander("Raw JSON output"):
            st.json(output)


# ----------------------------------------------------------------------------
# CLI mode — used when launched via `python app.py`.
# ----------------------------------------------------------------------------
def run_cli():
    pipe, id2label, _ = ensure_artifacts()

    for ticket in sample_example_tickets():
        print_trace(run_agent(ticket, pipe, id2label))

    print(f"\nKnowledge base now holds {kb_size()} case(s) at {KB_PATH}")
    print("Enter your own tickets (blank line or Ctrl-C to quit):")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print_trace(run_agent(text, pipe, id2label))


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False


if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli()
