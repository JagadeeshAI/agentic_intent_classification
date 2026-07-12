# AI Ticket Triage Copilot

Agentic pipeline that takes a customer support ticket, classifies its intent,
estimates priority, recommends an action, and decides whether it needs human
review — with **every one of those outputs explained from the model's own
prediction** (not a black box), and every response stored into a growing
knowledge base.

> 🎬 **A screen recording of the whole thing working — `demo.webm` (project root)** —
> shows the Streamlit UI: picking random example tickets, triaging, the four "why"
> explanations, and the knowledge base growing.

## Project structure

```
agentic/
├── demo.webm                     🎬 screen recording of the working app
├── app.py                        entry point — Streamlit UI (or CLI); loads the
│                                 checkpoint if present, trains ONLY if it's missing
├── checkpoint/                   (generated locally, not committed)
│   ├── best_pipeline.joblib      best pipeline (SBERT embeddings + LogisticRegression)
│   ├── label_maps.json           label <-> id mappings + best-model metadata
│   └── explain_index.npz         training-ticket embeddings for similarity-based
│                                 explanations (auto-built once on first run)
├── data/                         (generated locally, not committed)
│   └── data.csv                  Banking77, auto-downloaded once and cached here
├── results/                      (generated locally, not committed)
│   ├── model_comparison.csv      4-model baseline-vs-stronger comparison table
│   ├── classification_report_*.txt   per-model precision/recall/F1 (77 classes)
│   └── confusion_matrix_*.{csv,png}  per-model confusion matrices
├── knowledge_base/
│   └── knowledge_base.json       grows at runtime — every triaged ticket's full
│                                 explained response is stored (or updated) here
├── src/
│   ├── data.py                   downloads/caches Banking77, official train/test splits
│   ├── classify.py               trains 4 pipelines, saves results/ + checkpoint/
│   ├── explainableAI.py          single explanation method: SBERT + trained classifier
│   ├── knowledge.py              write-only knowledge-base case store
│   └── agent.py                  classify -> explain confidence -> priority -> action
│                                 -> human review -> store, with a visible trace
├── requirements.txt
├── README.md
└── research_note.md              model comparison, explainability, design decisions
```

## Setup

```bash
# 1. Get the code
git clone https://github.com/JagadeeshAI/agentic_intent_classification.git
cd agentic_intent_classification

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

No dataset download step is needed — Banking77 auto-downloads from Hugging Face on
first run and is cached to `data/data.csv` (fully offline afterwards). No API keys.

## Run — all commands

Everything runs from the **project root**. The only command you need:

```bash
streamlit run app.py             # 🎬 web UI at http://localhost:8501 (see demo.webm)
# or
python app.py                    # terminal demo + interactive prompt
```

`app.py` checks `checkpoint/` first — if the trained pipeline exists it **loads it and
does no training**; only on a missing checkpoint does it run the one-time 4-model
training/comparison (~10-20 min on CPU, once). On the very first prediction it also
builds the explanation index (~1-2 min, once).

Every step can also be run standalone (all from the project root):

```bash
python src/data.py               # download/cache dataset + sanity checks
                                 #   (class distribution, ambiguity, split-overlap)
python -m src.classify           # force a full retrain: trains all 4 models, writes
                                 #   results/ (reports, confusion matrices,
                                 #   model_comparison.csv) + checkpoint/
python -m src.explainableAI      # explainability standalone demo on 5 sample tickets
python -m src.agent              # full agent pipeline on 6 random tickets (CLI),
                                 #   stores responses to knowledge_base/
python -m py_compile app.py src/*.py   # quick syntax check of all code
```

To rebuild anything from scratch, delete its artifact and rerun:

```bash
rm -rf checkpoint/               # next run retrains the 4 models
rm -f  checkpoint/explain_index.npz    # next prediction re-encodes the index
rm -f  data/data.csv             # next run re-downloads Banking77
rm -f  knowledge_base/knowledge_base.json   # start the case store empty
```

## Demo video

**`demo.webm`** (project root) is a screen recording of the app in action: launching
`streamlit run app.py`, picking one of the 6 randomly-drawn real tickets, triaging it,
reading the four model-derived explanations (intent, confidence, priority, human
review), and watching the knowledge base grow in the sidebar.

## Dataset

**Used: [Banking77](https://huggingface.co/datasets/mteb/banking77)** — 13,083 real,
anonymized banking customer support queries, manually annotated with intent labels by
human annotators (Casanueva et al., ACL NLP4ConvAI 2020). 77 intents (e.g.
`declined_card_payment`, `lost_or_stolen_card`, `age_limit`). Uses the dataset's
official train (10,003) / test (3,080) split — zero text overlap between splits,
confirmed programmatically in `data.py`. Downloaded once from Hugging Face and cached
to `data/data.csv`; every later run is fully offline.

**Not used: the originally-assigned Kaggle "Customer Support Ticket Dataset."**
This dataset's `Ticket Type` label was found to be **statistically independent of the
ticket text**. In short: 85 exact-duplicate input texts mapped to different labels
(some spanning all 5 classes), and every classical/LLM model tested topped out at ~20%
accuracy — exactly chance level for 5 balanced classes. This was verified, not
assumed, before switching datasets.

**Columns used as model input:** `text` only (the raw customer query — Banking77 is
single-utterance, no separate Subject/Description/metadata fields).
**No leakage columns exist in this dataset** (unlike the original Kaggle CSV, which had
`Resolution`, `Ticket Status`, `First Response Time`, `Time to Resolution`,
`Customer Satisfaction Rating` — all only known *after* a ticket is resolved, and
correctly excluded during that dataset's investigation).

## Models — baseline vs. stronger (4 pipelines compared)

Per the assignment's >=2-model requirement, a TF-IDF **baseline family** is compared
against a **stronger** sentence-embedding model, all trained/evaluated identically
(`src/classify.py`, full table in `results/model_comparison.csv`):

| Model | Train acc | Val acc | Macro F1 |
|---|---|---|---|
| **stronger: SBERT (all-MiniLM-L6-v2) + LogisticRegression** | 0.9254 | **0.9083** | 0.9082 |
| baseline: TF-IDF + LogisticRegression (C=5.0) | 0.9675 | 0.8469 | 0.8465 |
| baseline: TF-IDF + LinearSVC | 0.9710 | 0.8371 | 0.8364 |
| baseline: TF-IDF + MultinomialNB | 0.8801 | 0.7760 | 0.7674 |

Chance baseline for 77 classes: 1.3%. The best model on validation accuracy (the SBERT
pipeline, +6.1 points over the best baseline, with a much smaller train/val gap) is
persisted to `checkpoint/best_pipeline.joblib` and is what `app.py` serves. Per-model
classification reports and confusion matrices are saved in `results/`.

## Explainability — every output explains itself

One method (`src/explainableAI.py`): **SBERT sentence embedding + the trained
classifier**, and all four outputs carry a model-derived explanation:

- **Why this intent** — similarity evidence in the same embedding space the classifier
  uses: *"Predicted 'Top up by cash or cheque' because this ticket's meaning is closest
  to real customer tickets the model learned from: 4 of the 5 most similar training
  tickets were labeled 'Top up by cash or cheque' (closest match similarity 0.93)."*
  The 10,003 training-ticket embeddings are encoded once and cached to
  `checkpoint/explain_index.npz` (auto-built on first run, ~1-2 min).
- **Why this confidence** — the model's own probability distribution: *"'Top up by
  cash or cheque' received 37.1%, the runner-up 'Top up by card' received 21.4% — a
  narrow margin, the model is torn between these two intents."* Confidence is a real
  `predict_proba()` probability.
- **Why this priority** — the urgency terms actually found in *this* ticket, whether
  the predicted intent is security/loss-critical, and the confidence value.
- **Why (no) human review** — the exact thresholds and numbers that triggered or
  cleared it.

Intent names are shown human-readable everywhere (`top_up_by_cash_or_cheque` →
"Top up by cash or cheque"); the raw label is kept in the JSON output.

## Agentic workflow

Every ticket runs through 6 explicit, logged steps (`src/agent.py`):

```
[1] classify                   -> predicted intent + similarity evidence
[2] explain_confidence         -> probability vs. runner-up intent (margin)
[3] estimate_priority          -> High/Medium/Low from THIS prediction, evidence cited
[4] recommend_action           -> escalate / route / standard response, by priority
[5] decide_human_review        -> confidence < 0.60 OR priority == High, numbers cited
[6] store_knowledge_base_case  -> response saved to knowledge_base/knowledge_base.json
```

The full trace + structured JSON is returned together and shown live in the UI.

## Example input / output

Input (typed into the Streamlit UI or CLI prompt):

```
i was charged twice for the same transaction
```

Output (structured JSON returned by the agent — values illustrative, exact numbers
vary slightly by run):

```json
{
  "input_text": "i was charged twice for the same transaction",
  "predicted_intent": "transaction_charged_twice",
  "predicted_label": "Transaction charged twice",
  "confidence": 0.91,
  "confidence_explanation": "The model distributes probability across all 77 intents: 'Transaction charged twice' received 91.0%, the runner-up 'Request refund' received 4.2% — a clear-cut decision.",
  "explanation": "Predicted 'Transaction charged twice' because this ticket's meaning is closest to real customer tickets the model learned from: 5 of the 5 most similar training tickets were labeled 'Transaction charged twice' (closest match similarity 0.93 out of 1.00).",
  "priority": "Medium",
  "priority_explanation": "Medium — no urgency evidence was found in the ticket or the predicted intent 'Transaction charged twice', but model confidence 91.0% ... ",
  "recommended_action": "Route to the queue handling 'Transaction charged twice' for normal processing.",
  "needs_human_review": false,
  "human_review_reason": "No human review needed — the model's confidence 91.0% clears the 60% threshold and no High-priority evidence was found.",
  "agent_trace": [
    {"step": 1, "name": "classify", "detail": "..."},
    {"step": 2, "name": "explain_confidence", "detail": "..."},
    {"step": 3, "name": "estimate_priority", "detail": "..."},
    {"step": 4, "name": "recommend_action", "detail": "..."},
    {"step": 5, "name": "decide_human_review", "detail": "..."},
    {"step": 6, "name": "store_knowledge_base_case", "detail": "Response stored to knowledge base (12 case(s) total)."}
  ]
}
```

## Knowledge base — built from the model's own predictions

`knowledge_base/knowledge_base.json` is **not pre-written**: it starts empty and grows
as tickets are triaged. Every response — prediction, all four explanations, priority,
action, review decision — is stored as a case (`src/knowledge.py`); triaging the exact
same text again **updates** its entry with the fresh prediction instead of duplicating
it. The store is write-only: the model runs fresh on every ticket, never answering
from the JSON.

## Demo

`streamlit run app.py` — pick one of **6 real tickets drawn randomly from the dataset**
(🔀 button resamples) or type your own, then see the readable predicted intent,
priority, confidence, the human-review decision, the four "why" explanations, the
recommended action, the full agent trace, and the raw JSON. The sidebar shows the
current knowledge-base size.

## Known limitations

- The stored knowledge base is write-only — past cases are not retrieved into new
  responses (a deliberate choice: the model must run fresh on every ticket). Guidance
  in the response comes from the current prediction's evidence instead.
- Priority and human-review rules cite model evidence but are still heuristics — the
  dataset has no priority/review labels to learn from or validate against.
- `checkpoint/`, `data/`, and `results/` are generated locally (gitignored); a fresh
  clone downloads the data and trains once automatically (metrics land in `results/`).
