"""Explainability for the ticket/intent classifier.

Loads the pipeline saved by classification.py and, for any input ticket,
shows WHICH WORDS drove the prediction — for TF-IDF-based models, this is
exact (not approximated). For the SBERT-embeddings model, per-word
attribution isn't mathematically meaningful (embeddings mix all words into
a dense vector with no clean per-word decomposition) — this is disclosed
explicitly in the output rather than faked.

How it works (for LogisticRegression / LinearSVC on TF-IDF features):
    contribution(word) = tfidf_weight(word in this ticket) * coef(word, predicted_class)
A word only shows up as a "top contributor" if it (a) is present in this
specific ticket AND (b) the model's learned weight for that word pushes
toward the predicted class.

For MultinomialNB, the equivalent is feature_log_prob_ (log P(word | class)).

Requirements:
    pip install scikit-learn joblib
Run this after classification.py (which saves ./checkpoint/best_pipeline.joblib).
"""
import json
import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Imported so joblib can unpickle a saved pipeline that used this class as
# its feature step (only relevant if the SBERT model was the best one).
from src.classify import SBERTEncoder  # noqa: F401

CKPT_DIR = "./checkpoint"
TOP_N = 8


def load_artifacts(ckpt_dir: str = CKPT_DIR):
    pipe = joblib.load(os.path.join(ckpt_dir, "best_pipeline.joblib"))
    with open(os.path.join(ckpt_dir, "label_maps.json")) as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    return pipe, id2label, maps["target"], maps["best_model_name"]


def _get_confidence(pipe, text: str):
    """Return (predicted_id, confidence)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba([text])[0]
        pred_id = int(np.argmax(proba))
        return pred_id, float(proba[pred_id])
    scores = np.atleast_1d(pipe.decision_function([text])[0])
    exp = np.exp(scores - scores.max())
    proba = exp / exp.sum()
    pred_id = int(np.argmax(scores))
    return pred_id, float(proba[pred_id])


def explain_prediction(pipe, id2label: dict, text: str, top_n: int = TOP_N) -> dict:
    """Return a structured explanation for one input ticket."""
    features_step = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]

    pred_id, confidence = _get_confidence(pipe, text)
    pred_label = id2label[pred_id]
    confidence_is_exact = hasattr(clf, "predict_proba")

    has_vocabulary = hasattr(features_step, "get_feature_names_out")

    if not has_vocabulary:
        # SBERT (or any non-vocabulary embedding) — no honest per-word
        # attribution is possible; say so rather than fabricating one.
        return {
            "input_text": text,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "confidence_is_exact_probability": confidence_is_exact,
            "top_contributing_words": [],
            "explanation_method": "NOT AVAILABLE — this model uses dense sentence embeddings "
                                   "(SBERT), which have no clean per-word decomposition. "
                                   "Word-level explanation only works for the TF-IDF-based models.",
            "explanation": f"Predicted '{pred_label}' (embedding-based model — "
                            f"per-word explanation not available for this model type).",
        }

    vectorizer = features_step
    feature_names = np.array(vectorizer.get_feature_names_out())
    x_vec = vectorizer.transform([text])
    present_idx = x_vec.nonzero()[1]
    tfidf_values = np.asarray(x_vec[0, present_idx].todense()).ravel()

    if isinstance(clf, (LogisticRegression, LinearSVC)):
        coef = clf.coef_
        class_row = coef[pred_id] if coef.shape[0] > 1 else coef[0] * (1 if pred_id == 1 else -1)
        contributions = tfidf_values * class_row[present_idx]
        method = "tfidf_weight x class_coefficient (exact, per-prediction)"

    elif isinstance(clf, MultinomialNB):
        log_prob = clf.feature_log_prob_
        other_mean = (log_prob.sum(axis=0) - log_prob[pred_id]) / (log_prob.shape[0] - 1)
        word_scores = log_prob[pred_id] - other_mean
        contributions = tfidf_values * word_scores[present_idx]
        method = "tfidf_weight x (log P(word|class) - mean log P(word|other classes))"

    elif isinstance(clf, RandomForestClassifier):
        importances = clf.feature_importances_
        contributions = tfidf_values * importances[present_idx]
        method = "GLOBAL feature_importances_ (not a true per-instance explanation)"

    else:
        contributions = tfidf_values
        method = "fallback: raw tfidf weight only (classifier type not specifically supported)"

    order = np.argsort(-contributions)
    top_idx = present_idx[order][:top_n]
    top_contrib = contributions[order][:top_n]
    top_words = [
        {"word": feature_names[i], "contribution": round(float(c), 4)}
        for i, c in zip(top_idx, top_contrib)
    ]

    explanation_text = (
        f"Predicted '{pred_label}' mainly because of: "
        + ", ".join(f"'{w['word']}'" for w in top_words[:5]) + "."
        if top_words else
        f"Predicted '{pred_label}', but no input words matched the model's vocabulary."
    )

    return {
        "input_text": text,
        "predicted_label": pred_label,
        "confidence": round(confidence, 4),
        "confidence_is_exact_probability": confidence_is_exact,
        "top_contributing_words": top_words,
        "explanation_method": method,
        "explanation": explanation_text,
    }


def print_explanation(result: dict):
    print(f"\nInput: {result['input_text']!r}")
    print(f"Predicted label: {result['predicted_label']}")
    conf_note = "" if result["confidence_is_exact_probability"] else "  (approximate — model has no predict_proba)"
    print(f"Confidence: {result['confidence']:.4f}{conf_note}")
    if result["top_contributing_words"]:
        print("Top contributing words (word: contribution score):")
        for w in result["top_contributing_words"]:
            print(f"  {w['word']:<20} {w['contribution']:+.4f}")
    print(f"Explanation: {result['explanation']}")
    print(f"(method: {result['explanation_method']})")


if __name__ == "__main__":
    pipe, id2label, target, model_name = load_artifacts()
    print(f"Loaded '{model_name}' pipeline  |  target='{target}'  |  classes={len(id2label)}")

    examples = [
        "my card payment was declined twice, why is this happening",
        "i think someone stole my card and used it",
        "how long does it take for a transfer to arrive",
        "i was charged twice for the same transaction",
        "how do i verify my identity on the app",
    ]
    for ex in examples:
        result = explain_prediction(pipe, id2label, ex)
        print_explanation(result)
