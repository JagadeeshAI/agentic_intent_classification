
import os
import json

from sklearn.base import BaseEstimator, TransformerMixin

RESULTS_DIR = "./results"
CKPT_DIR = "./checkpoint"


class SBERTEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, X, y=None):
        self._get_model()
        return self

    def transform(self, X):
        return self._get_model().encode(list(X), show_progress_bar=False)


def train():
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 classification_report, confusion_matrix)

    from src.data import load_splits, PRIMARY_TARGET

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- 1. Data — official Banking77 train/test split (cached by data.py). ---
    train_df, val_df, label2id, id2label = load_splits()
    num_labels = len(label2id)
    print(f"target: '{PRIMARY_TARGET}'  |  classes: {num_labels}")
    print(f"train={len(train_df)}  val={len(val_df)}")

    X_train, y_train = train_df["text"].tolist(), train_df["labels"].tolist()
    X_val, y_val = val_df["text"].tolist(), val_df["labels"].tolist()
    target_names = [id2label[i] for i in range(num_labels)]

    # --- 2. Candidate pipelines: BASELINE (TF-IDF family) + STRONGER (SBERT). ---
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                            sublinear_tf=True, stop_words="english")

    pipelines = {
        "baseline_tfidf_logreg": Pipeline([("features", tfidf), ("clf", LogisticRegression(max_iter=1000, C=5.0))]),
        "baseline_tfidf_linearsvc": Pipeline([("features", tfidf), ("clf", LinearSVC(C=1.0))]),
        "baseline_tfidf_nb": Pipeline([("features", tfidf), ("clf", MultinomialNB())]),
        "stronger_sbert_logreg": Pipeline([("features", SBERTEncoder()), ("clf", LogisticRegression(max_iter=1000))]),
    }

    summary_rows = []
    best_name, best_val_acc, best_pipe = None, -1.0, None

    for name, pipe in pipelines.items():
        print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
        pipe.fit(X_train, y_train)

        train_preds = pipe.predict(X_train)
        val_preds = pipe.predict(X_val)

        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0)
        weighted_f1 = precision_recall_fscore_support(
            y_val, val_preds, average="weighted", zero_division=0)[2]

        print(f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")

        summary_rows.append({
            "model": name, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
            "macro_precision": round(macro_p, 4), "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
        })

        report = classification_report(y_val, val_preds, target_names=target_names, zero_division=0)
        with open(os.path.join(RESULTS_DIR, f"classification_report_{name}.txt"), "w") as f:
            f.write(f"Model: {name}\ntrain_acc={train_acc:.4f}  val_acc={val_acc:.4f}\n\n")
            f.write(report)

        cm = confusion_matrix(y_val, val_preds)
        np.savetxt(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.csv"), cm, fmt="%d", delimiter=",")

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion matrix — {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.png"), dpi=120)
        plt.close(fig)

        if val_acc > best_val_acc:
            best_name, best_val_acc, best_pipe = name, val_acc, pipe

    # --- 4. Save the comparison summary table. ---
    summary_df = pd.DataFrame(summary_rows).sort_values("val_acc", ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    print(f"\n{'='*60}\n=== MODEL COMPARISON SUMMARY ===\n{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nBest model: {best_name}  (val_acc={best_val_acc:.4f})")
    print(f"All results saved to {RESULTS_DIR}/")

    # --- 5. Persist the single best pipeline + label maps. ---
    joblib.dump(best_pipe, os.path.join(CKPT_DIR, "best_pipeline.joblib"))
    with open(os.path.join(CKPT_DIR, "label_maps.json"), "w") as f:
        json.dump({"target": PRIMARY_TARGET,
                   "label2id": label2id,
                   "id2label": {str(k): v for k, v in id2label.items()},
                   "best_model_name": best_name}, f, indent=2)
    print(f"Saved best pipeline ({best_name}) to {CKPT_DIR}/best_pipeline.joblib")


if __name__ == "__main__":
    train()
