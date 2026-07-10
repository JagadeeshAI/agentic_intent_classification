# AI Ticket Triage Copilot

Agentic pipeline that takes a customer support ticket, classifies its intent,
estimates priority, retrieves matching guidance from a local knowledge base,
recommends an action, decides whether it needs human review, and explains
*why* — with every step of that reasoning visible, not a black box.

## Project structure

```
agentic/
├── ai-engineer-assignment!.pdf   assignment brief
├── app.py                        entry point — Streamlit UI (or CLI); loads the
│                                 checkpoint if present, trains ONLY if it's missing
├── checkpoint/
│   ├── best_pipeline.joblib      best pipeline (SBERT embeddings + LogisticRegression)
│   └── label_maps.json           label <-> id mappings + best-model metadata
├── data/
│   └── data.csv                  Banking77, auto-downloaded once and cached here
├── results/
│   ├── model_comparison.csv      4-model baseline-vs-stronger comparison table
│   ├── classification_report_*.txt   per-model precision/recall/F1 (77 classes)
│   ├── confusion_matrix_*.{csv,png}  per-model confusion matrices
│   └── knowledge.json            auto-generated knowledge base (9 categories + 2 policy rules)
└── src/
    ├── data.py                   downloads/caches Banking77, official train/test splits
    ├── classify.py               trains 4 pipelines, saves results/ + checkpoint/
    ├── explainableAI.py          loads checkpoint, explains predictions
    ├── build_knowledge.py        generates results/knowledge.json + runtime retrieval helpers
    └── agent.py                  chains classify -> KB -> priority -> action -> human-review,
                                  with a visible step-by-step trace + structured JSON output
```

## Setup

```bash
pip install datasets pandas scikit-learn joblib sentence-transformers matplotlib streamlit
```

## Run

Everything runs from the **project root**. The only command you need:

```bash
streamlit run app.py    # web UI at localhost:8501
# or
python app.py           # terminal demo + interactive prompt
```

`app.py` checks `checkpoint/` and `results/knowledge.json` first — if they exist it
**loads them and does no training**; only on a missing checkpoint does it run the full
one-time training/comparison (and only on a missing knowledge.json does it rebuild the KB).

Individual steps can also be run standalone (all from the project root):

```bash
python src/data.py             # dataset sanity checks (stats, ambiguity, split overlap)
python -m src.classify         # force a full retrain + 4-model comparison
python -m src.explainableAI    # explainability standalone demo
python src/build_knowledge.py  # (re)generate results/knowledge.json
python -m src.agent            # full agent pipeline on example tickets (CLI)
```

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
ticket text** — see `Reserch_note.md` for the full investigation. In short: 85 exact-
duplicate input texts mapped to different labels (some spanning all 5 classes), and
every classical/LLM model tested topped out at ~20% accuracy — exactly chance level for
5 balanced classes. This was verified, not assumed, before switching datasets.

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

Chance baseline for 77 classes: 1.3%. The best model on validation accuracy (currently
the SBERT pipeline, +6.1 points over the best baseline, with a much smaller train/val
gap) is persisted to `checkpoint/best_pipeline.joblib` and is what `app.py` serves.
Per-model classification reports and confusion matrices are saved in `results/`.

## Explainability

`explainableAI.py` computes, for TF-IDF-based pipelines:

```
contribution(word) = tfidf_weight(word in this ticket) x class_coefficient(word, predicted_class)
```

This is exact for LogisticRegression/LinearSVC — not an approximation. Only words
actually present in that specific ticket are shown, ranked by how much they pushed the
model toward its prediction.

**Honest limitation:** the current best model is SBERT-based, and dense sentence
embeddings have no clean per-word decomposition — so for it, the explanation output
explicitly says word-level attribution is **not available** rather than faking one.
Confidence is still a real `predict_proba()` probability (the classifier on top is
LogisticRegression). See `Reserch_note.md` §4 for the accuracy-vs-explainability
trade-off discussion.

## Agentic workflow

Every prediction runs through 5 explicit, logged steps (`agent.py`):

```
[1] classify                -> predicted intent + confidence
[2] retrieve_knowledge_base -> map intent to category, pull guidance
[3] estimate_priority       -> rule-based, from measured urgency-word rate per category
[4] recommend_action        -> category-specific next step
[5] decide_human_review     -> confidence < 0.60  OR  priority == High
```

Each ticket's full trace + structured JSON (intent, category, priority, confidence,
explanation, guidance, action, `needs_human_review`) is returned together — shown live
in the Streamlit UI.

## Knowledge base

**Fully auto-generated from real data**, not hand-written — see `src/build_knowledge.py`
and `Reserch_note.md` §5 for the exact methodology (keyword-based category grouping over
real intent names, guidance text from real TF-IDF terms per category, priority ranked
from a measured urgency-word rate). Output: `results/knowledge.json` — 9 categories
covering all 77 intents, plus 2 policy-rule entries (high-priority escalation,
low-confidence human review) generated from the same threshold constants `agent.py`
actually uses.

## Demo

`streamlit run app.py` — type a ticket or pick an example, see predicted intent,
category, priority, confidence, the explanation, retrieved guidance, recommended
action, human-review flag, the full agent trace, and the raw JSON output.

## Known gaps (before final submission)

- **No `requirements.txt`** — dependencies are listed above but not pinned to a file yet.
- **Knowledge-base code lives in `src/build_knowledge.py`** (with its JSON in `results/`),
  not a separate top-level `knowledge_base/` folder as the brief's suggested structure
  shows. Low-priority — the content and retrieval logic are complete either way.
