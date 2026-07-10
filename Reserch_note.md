# Research Note

## 1. Original dataset investigation (and why it was abandoned)

The assignment originally pointed to a Kaggle "Customer Support Ticket Dataset"
(`suraj520/customer-support-ticket-dataset`). Before building anything on it, its label
quality was tested directly rather than assumed:

**Test 1 — exact-duplicate inputs.** Grouping rows by the combined `Subject + Description`
text and checking how many distinct `Ticket Type` labels each group had:

- 85 exact-duplicate text groups mapped to more than one label.
- The worst case: one identical text string ("I'm having an issue with the
  {product_purchased}. Please assist. I need assistance as soon as possible...")
  appeared 25 times across **all 5 classes** (Technical issue x8, Cancellation request
  x6, Billing inquiry x5, Refund request x4, Product inquiry x2).
- Checking `Description` alone (ignoring Subject) made this worse: 49 duplicate groups,
  445 tickets (5.25% of the dataset) inside an ambiguous group.

**Test 2 — subject-category base rates.** All 526 tickets with Subject = "Payment issue"
were checked against their `Ticket Type` label:

| Ticket Type | Count |
|---|---|
| Cancellation request | 113 |
| Technical issue | 111 |
| Billing inquiry | 107 |
| Refund request | 104 |
| Product inquiry | 91 |

Essentially a uniform 5-way split — statistically indistinguishable from assigning the
label at random, independent of the actual ticket content.

**Test 3 — model performance.** TF-IDF + LogisticRegression and sentence-transformer
(all-MiniLM-L6-v2) embeddings + LogisticRegression were both trained on this dataset:
macro F1 ~0.20-0.21, accuracy ~20-21% — chance level for 5 balanced classes. A
label-shuffle sanity check (randomly permuting the training labels) produced the same
~20% accuracy, confirming the ceiling wasn't a modeling failure — the labels genuinely
carry no information correlated with the text.

**Test 4 — adding metadata didn't help.** Concatenating `Product Purchased`,
`Ticket Channel`, `Customer Gender`, and `Customer Age` into the input text made every
row's *combination* unique (0 duplicate full-input groups, vs. 85 for text alone) — but
this is coincidental uniqueness from high-cardinality fields (53 distinct ages), not
real signal. Training with this full input let the model overfit toward ~90%+ train
accuracy while validation stayed near chance — the classic memorization-without-
generalization signature, confirming (not fixing) the underlying problem.

**Conclusion:** `Ticket Type` in this dataset is statistically independent of the input.
No architecture — classical ML or LLM — can learn a mapping that doesn't exist in the
data. This was documented and the dataset was switched, rather than reporting a
misleadingly "improved" accuracy from overfitting on spurious metadata correlations.

## 2. Dataset switch: Banking77

**Selected:** [Banking77](https://huggingface.co/datasets/mteb/banking77) (Casanueva et
al., ACL NLP4ConvAI 2020) — real, anonymized online banking customer queries, manually
labeled by human annotators (not template-generated, not LLM-generated). 13,083
examples, 77 intents, official train/test split with zero text overlap.

The same ambiguity tests run against the original dataset were re-run here as a
sanity check before committing to it:
- 0 duplicate-text groups mapping to conflicting labels (vs. 85 on the original dataset).
- 0 texts overlapping between the official train and test splits.

Trade-off accepted knowingly: Banking77 is single-domain (retail banking) and
single-utterance (no Subject/Description/metadata split), which is a narrower shape than
the original ticket schema. This was judged a reasonable trade for having a genuinely
learnable task, and is disclosed here rather than left implicit.

## 3. Model comparison — baseline vs. stronger

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
explainability (see §4) in exchange for the accuracy gain. Selection is automatic on
val accuracy, so if a TF-IDF model ever wins, exact word-level explanations come back
for free.

**Train/val gap on the TF-IDF baselines (96.75% vs 84.69%):** expected, not a bug. With
77 fine-grained, sometimes semantically overlapping classes (e.g. `verify_my_identity`
vs `why_verify_identity`), and zero train/test text overlap, some overfitting on
training-specific phrasing is normal. The lowest-F1 classes in the classification
reports are exactly the semantically confusable pairs, which is the expected error
pattern for a model that's genuinely learning language patterns rather than memorizing.

## 4. Explainability

For TF-IDF-based pipelines:
`contribution(word) = tfidf_weight(word) x class_coefficient(word, predicted_class)`,
computed only over words actually present in that ticket, sorted descending. This is
exact for LogisticRegression (no approximation), and the same mechanism is used for
LinearSVC's `coef_` (with MultinomialNB handled via `feature_log_prob_`).

**For the currently-deployed SBERT model, per-word attribution is honestly reported as
NOT AVAILABLE.** Dense sentence embeddings mix all words into one vector with no clean
per-word decomposition, so `explainableAI.py` says so explicitly in its output instead
of fabricating a plausible-looking attribution (e.g. via a surrogate that doesn't
reflect what the model actually computed). Confidence remains a real `predict_proba()`
probability either way, since the classifier head is LogisticRegression. This
accuracy-vs-explainability trade-off is the deliberate choice discussed in §3.

## 5. Knowledge base — auto-generated, not hand-written

`src/build_knowledge.py` derives the entire knowledge base (written to
`results/knowledge.json`) from the real, loaded dataset rather than fabricated policy
text, and also emits two policy-rule entries (high-priority escalation, low-confidence
human review) generated from the same threshold constants `agent.py` actually uses:

- **Category grouping:** all 77 intent names (pulled live from the dataset, not
  hardcoded) are tokenized and matched against a small ordered keyword table
  (~30 keywords, 9 categories). This keyword *table* is the one hand-authored
  component; the *assignment* of each of the 77 intents to a category is programmatic.
- **Guidance text:** for each category, the top 6 TF-IDF terms across that category's
  real ticket texts are extracted and dropped into a template sentence. The vocabulary
  is real, pulled from the dataset — not invented.
- **Priority:** each category's real tickets are scanned for a small urgency-word list
  (stolen, fraud, blocked, declined, compromised, unauthorized, lost, urgent,
  emergency); the fraction of tickets containing at least one such word is the
  category's measured "urgency rate." The 9 categories are then ranked into
  High/Medium/Low tertiles by that measured rate — not manually assigned.

**Known limitation, found and worth reporting honestly:** `REFUNDS_DISPUTES`
(covering `transaction_charged_twice`, `request_refund`, etc.) measured only a 2.2%
urgency-word rate and landed at Medium priority — even though a duplicate charge
intuitively feels time-sensitive to a human reviewer. This is because the heuristic
counts literal urgency *words*, and duplicate-charge complaints are typically phrased
calmly ("I was charged twice") without alarm language, even when the underlying issue
is genuinely urgent. This is a real gap between lexical urgency and situational
urgency — the automatic heuristic captures the former, not the latter. Documented here
rather than hidden or manually overridden after the fact.

## 6. Agent workflow

Five explicit, logged steps per ticket (`agent.py`): classify -> retrieve KB guidance ->
estimate priority -> recommend action -> decide human review. `needs_human_review` is
`True` if confidence < 0.60 OR priority == High — both thresholds are simple, disclosed
heuristics, not tuned against a validation objective (no ground-truth human-review
labels exist to tune against).

## 7. Summary of what's learned vs. designed

| Component | Learned from data | Hand-designed |
|---|---|---|
| Intent classification | Yes - SBERT embeddings + LogReg (best of 4 compared) | - |
| Confidence scores | Yes - real predict_proba() | - |
| Explainability words | Yes for TF-IDF models (exact coefficient decomposition); disclosed as unavailable for the SBERT best model | - |
| Category grouping | Assignment is automatic | Keyword table is authored |
| KB guidance text | Yes - real TF-IDF terms per category | Template sentence structure |
| Priority ranking | Yes - measured urgency-word rate | Urgency word list is authored; no ground-truth priority labels exist to validate against |
| Human-review thresholds | - | Confidence 0.60 / High-priority rule, undisclosed-truth heuristic |

## 8. Reproducibility & runtime behavior

- All artifacts are cached and checked before any expensive work: `data/data.csv`
  (downloaded once from Hugging Face), `checkpoint/best_pipeline.joblib` +
  `label_maps.json` (trained once), `results/knowledge.json` (built once).
- `app.py` is the single entry point (Streamlit UI via `streamlit run app.py`, or CLI
  via `python app.py`). It **loads the existing checkpoint and does no training** when
  one is present; training runs only on a missing checkpoint. `python -m src.classify`
  forces a full retrain/re-comparison.
- Importing `src/classify.py` is side-effect free by design — the training run lives
  behind `train()` — because unpickling the saved SBERT pipeline requires importing
  `src.classify.SBERTEncoder`, and loading a checkpoint must never trigger a retrain.

## 9. Limitations & future work

- Single-domain (banking) and single-utterance inputs — narrower than a full
  Subject/Description ticket schema (trade-off disclosed in §2).
- The best model (SBERT) has no per-word explanation; only the TF-IDF baselines do
  (§4). Future work: a model-agnostic explainer (e.g. LIME/SHAP on the SBERT pipeline),
  clearly labeled as approximate.
- Priority heuristic captures lexical, not situational, urgency (§5).
- No ground-truth priority or human-review labels exist for this dataset, so those two
  components can't be evaluated quantitatively — only sanity-checked by inspection.
- No pinned `requirements.txt` yet (see README "Known gaps").
