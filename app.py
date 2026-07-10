"""Entry point for the ticket-triage system.

Loads the trained pipeline from checkpoint/ if it exists; ONLY if the
checkpoint (or the knowledge base) is missing does it run the corresponding
build step (src/classify.py training, src/build_knowledge.py KB build).

Two ways to run (both from the project root):
    streamlit run app.py    -> web UI at localhost:8501
    python app.py           -> terminal demo + interactive prompt
"""
import os

from src.classify import CKPT_DIR, train
from src.build_knowledge import OUT_PATH as KB_PATH, build as build_kb
from src.explainableAI import load_artifacts
from src.agent import run_agent, print_trace

PIPELINE_PATH = os.path.join(CKPT_DIR, "best_pipeline.joblib")
LABEL_MAPS_PATH = os.path.join(CKPT_DIR, "label_maps.json")

EXAMPLE_TICKETS = [
    "my card payment was declined twice, why is this happening",
    "i think someone stole my card and used it",
    "how long does it take for a transfer to arrive",
    "i was charged twice for the same transaction",
    "how do i verify my identity on the app",
    "what is the minimum age to open an account",
]


def ensure_artifacts():
    """Train / build only what is missing; load and return the artifacts."""
    if os.path.exists(PIPELINE_PATH) and os.path.exists(LABEL_MAPS_PATH):
        print(f"Found existing checkpoint at {CKPT_DIR}/ — loading it (no training).")
    else:
        print(f"No checkpoint at {PIPELINE_PATH} — training from scratch (one-time)...")
        train()

    if os.path.exists(KB_PATH):
        print(f"Found existing knowledge base at {KB_PATH}.")
    else:
        print(f"No knowledge base at {KB_PATH} — building it (one-time)...")
        build_kb()

    pipe, id2label, target, model_name = load_artifacts()
    print(f"Loaded '{model_name}' pipeline  |  target='{target}'  |  classes={len(id2label)}")
    return pipe, id2label, model_name


# ----------------------------------------------------------------------------
# Streamlit UI — used when launched via `streamlit run app.py`.
# ----------------------------------------------------------------------------
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Ticket Triage Agent", page_icon="🎫", layout="wide")
    st.title("🎫 Banking Ticket Triage Agent")
    st.caption("Classifies a support ticket, retrieves knowledge-base guidance, "
               "estimates priority, recommends an action, and decides whether a "
               "human needs to review it — with a visible step-by-step trace.")

    @st.cache_resource(show_spinner="Loading model checkpoint (training only if missing)...")
    def get_artifacts():
        return ensure_artifacts()

    pipe, id2label, model_name = get_artifacts()
    st.sidebar.success(f"Model: **{model_name}**\n\nClasses: **{len(id2label)}**")
    st.sidebar.markdown(f"Checkpoint: `{PIPELINE_PATH}`")

    example = st.selectbox("Try an example ticket:", ["(type your own below)"] + EXAMPLE_TICKETS)
    default_text = "" if example == "(type your own below)" else example
    text = st.text_area("Ticket text", value=default_text,
                        placeholder="e.g. i was charged twice for the same transaction")

    if st.button("Triage ticket", type="primary") and text.strip():
        output = run_agent(text.strip(), pipe, id2label)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted intent", output["predicted_intent"])
        col2.metric("Category", output["predicted_category"])
        col3.metric("Priority", output["priority"])
        col4.metric("Confidence", f"{output['confidence']:.2%}")

        if output["needs_human_review"]:
            st.warning("⚠️ Needs human review — " + output["agent_trace"][4]["detail"])
        else:
            st.success("✅ Can be auto-resolved (confident and not high priority).")

        st.subheader("Explanation")
        st.write(output["explanation"])
        if output["top_contributing_words"]:
            st.table(output["top_contributing_words"])

        st.subheader("Knowledge-base guidance")
        st.info(output["retrieved_guidance"])
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

    for ticket in EXAMPLE_TICKETS:
        print_trace(run_agent(ticket, pipe, id2label))

    print("\nEnter your own tickets (blank line or Ctrl-C to quit):")
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
