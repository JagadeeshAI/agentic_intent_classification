# Research Note

## 1. Model comparison — baseline vs. stronger

Per the assignment's >=2-model requirement, a TF-IDF **baseline family** (3 classical
classifiers on identical TF-IDF features) is compared against a **stronger**
sentence-embedding model (SBERT `all-MiniLM-L6-v2` embeddings + LogisticRegression),
all trained and evaluated identically in `src/classify.py`
(full table: `results/model_comparison.csv`):

| Model | Train acc | Val acc | Macro F1 |
|---|---|---|---|
| **stronger: SBERT + LogisticRegression** | 0.9254 | **0.9083** | 0.9082 |
| baseline: TF-IDF + LogisticRegression (C=5.0) | 0.9675 | 0.8469 | 0.8465 |
| baseline: TF-IDF + LinearSVC | 0.9710 | 0.8371 | 0.8364 |
| baseline: TF-IDF + MultinomialNB | 0.8801 | 0.7760 | 0.7674 |

The SBERT pipeline wins by **+6.1 points** val accuracy over the best baseline, with a
much smaller train/val gap (92.5% vs 90.8%, against TF-IDF LogReg's 96.8% vs 84.7%) —
pretrained sentence embeddings generalize to unseen phrasings instead of overfitting
training-specific n-grams. It is persisted to `checkpoint/best_pipeline.joblib` and is
what `app.py` serves. LogisticRegression as the classifier head (in both the baseline
and the stronger model) exposes real `predict_proba()` — needed for honest confidence
scores in the human-review decision, rather than an approximated softmax over
`decision_function()`.

**Trade-off accepted knowingly:** the SBERT model gives up exact per-word
explainability in exchange for the accuracy gain — it is explained via nearest
training examples and its own probability margins instead (see §2).

**Train/val gap on the TF-IDF baselines (96.75% vs 84.69%):** expected, not a bug. With
77 fine-grained, sometimes semantically overlapping classes (e.g. `verify_my_identity`
vs `why_verify_identity`), and zero train/test text overlap, some overfitting on
training-specific phrasing is normal. The lowest-F1 classes in the classification
reports are exactly the semantically confusable pairs, which is the expected error
pattern for a model that's genuinely learning language patterns rather than memorizing.

## 2. Explainability — one method, four self-explaining outputs

A single method (`src/explainableAI.py`): **SBERT embedding + the trained
classifier**, chosen for simplicity and because it matches the deployed best model.
Dense sentence embeddings have no clean per-word decomposition, so instead of
fabricating a plausible-looking word attribution (e.g. via a surrogate that doesn't
reflect what the model actually computed), each output is explained with the model's
own evidence (a similarity-based explanation is one of the assignment's accepted
methods):

- **Intent** — cosine similarity in the *same* embedding space the classifier operates
  on: how many of the 5 nearest training tickets share the predicted label, and the
  closest match's similarity. The 10,003 training-ticket embeddings are encoded once
  and cached to `checkpoint/explain_index.npz` (auto-built on first run; no retraining).
- **Confidence** — the classifier's real `predict_proba()` distribution, quoted with
  the runner-up intent so the number is interpretable: a 37% top probability with a
  21% runner-up is honestly described as "a narrow margin — the model is torn between
  these two intents".
- **Priority** and **human review** — explained in §4 (rules over model outputs, with
  the triggering evidence and thresholds cited verbatim).

Intent names are humanized for display (`top_up_by_cash_or_cheque` -> "Top up by cash
or cheque"); raw labels are preserved in the structured output. An earlier iteration
also carried an exact TF-IDF coefficient decomposition for the baseline models; it was
removed to keep one clear method matching the served model — the trade-off discussed
in §1 stands.

## 3. Knowledge base — built from the model's own predictions

`knowledge_base/knowledge_base.json` is **not pre-written and not derived from static
dataset statistics**: it starts empty and grows at runtime (`src/knowledge.py`). Every
triaged ticket's full response — prediction, all four explanations, priority, action,
review decision, timestamp — is stored as a case; triaging the exact same text again
**updates** its entry with the fresh prediction instead of duplicating it.

The store is deliberately **write-only**: the model runs fresh on every ticket and
never answers from the JSON, so a stale stored prediction can never leak into a new
response. An earlier iteration pre-built a rule-based KB (keyword categories, per-
category TF-IDF guidance, urgency-rate priority tertiles); it was replaced because its
guidance was static and disconnected from what the model actually predicted for the
ticket at hand.

**Known limitation, kept from the earlier design and still true:** urgency detection
counts literal urgency *words* in the ticket ("stolen", "fraud", "urgent", ...).
Duplicate-charge complaints are typically phrased calmly ("I was charged twice")
without alarm language even when genuinely time-sensitive — lexical urgency is not
situational urgency. Documented rather than hidden.

## 4. Agent workflow

Six explicit, logged steps per ticket (`src/agent.py`):
classify -> explain confidence -> estimate priority -> recommend action ->
decide human review -> store the response as a knowledge-base case.

Priority is computed **per prediction** with the evidence cited: High if urgency terms
appear in the ticket text or the predicted intent is security/loss-critical (the
matched terms are listed in the explanation); Low if there is no urgency evidence and
confidence >= 0.80; Medium otherwise. `needs_human_review` is `True` if confidence
< 0.60 OR priority == High — the review explanation quotes the exact numbers and
thresholds. All thresholds are simple, disclosed heuristics, not tuned against a
validation objective (no ground-truth priority or human-review labels exist to tune
against).

## 5. Summary of what's learned vs. designed

| Component | Learned from data | Hand-designed |
|---|---|---|
| Intent classification | Yes - SBERT embeddings + LogReg (best of 4 compared) | - |
| Confidence scores + margin explanation | Yes - real predict_proba() | - |
| Intent explanation | Yes - nearest real training tickets by embedding similarity | - |
| Knowledge-base content | Yes - the model's own stored responses | Stored-field schema |
| Priority | Uses model outputs (predicted intent, confidence) per prediction | Urgency word list + thresholds authored; no ground-truth priority labels exist to validate against |
| Human-review thresholds | - | Confidence 0.60 / High-priority rule, disclosed heuristic |

## 6. Reproducibility & runtime behavior

- All artifacts are cached and checked before any expensive work: `data/data.csv`
  (downloaded once from Hugging Face), `checkpoint/best_pipeline.joblib` +
  `label_maps.json` (trained once), `checkpoint/explain_index.npz` (training
  embeddings for similarity explanations, encoded once).
  `knowledge_base/knowledge_base.json` is the one artifact that grows at runtime.
- `app.py` is the single entry point (Streamlit UI via `streamlit run app.py`, or CLI
  via `python app.py`). It **loads the existing checkpoint and does no training** when
  one is present; training runs only on a missing checkpoint. `python -m src.classify`
  forces a full retrain/re-comparison.
- Importing `src/classify.py` is side-effect free by design — the training run lives
  behind `train()` — because unpickling the saved SBERT pipeline requires importing
  `src.classify.SBERTEncoder`, and loading a checkpoint must never trigger a retrain.

## 7. Limitations & future work

- Single-domain (banking) and single-utterance inputs — narrower than a full
  Subject/Description ticket schema.
- Explanations are similarity- and probability-based, not per-word attribution (§2).
  Future work: a model-agnostic per-word explainer (e.g. LIME/SHAP on the SBERT
  pipeline), clearly labeled as approximate.
- The explainability path assumes the deployed checkpoint is the SBERT pipeline; if a
  retrain ever crowned a TF-IDF model, `explainableAI.py` would need its word-level
  path restored.
- The knowledge base is write-only — past cases are not retrieved into new responses
  (deliberate: the model runs fresh every time, see §3).
- Priority heuristic captures lexical, not situational, urgency (§3).
- No ground-truth priority or human-review labels exist for this dataset, so those two
  components can't be evaluated quantitatively — only sanity-checked by inspection.
