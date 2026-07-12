import json
import os
import re
import numpy as np
import joblib

CKPT_DIR = "./checkpoint"
INDEX_PATH = os.path.join(CKPT_DIR, "explain_index.npz")
TOP_K_SIMILAR = 5

_INDEX_CACHE = {}


def humanize_label(label: str) -> str:
    clean = re.sub(r"[_:]+", " ", str(label)).strip()
    return clean[:1].upper() + clean[1:]


def build_similarity_index(pipe, index_path: str = INDEX_PATH):
    from src.data import load_splits
    train_df, _, _, id2label = load_splits()
    texts = train_df["text"].astype(str).tolist()
    labels = [id2label[i] for i in train_df["labels"].tolist()]

    print(f"Building one-time similarity index ({len(texts)} training tickets)...")
    emb = np.asarray(pipe.named_steps["features"].transform(texts), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    np.savez_compressed(index_path, embeddings=emb, labels=np.array(labels))
    print(f"Saved similarity index to {index_path}")


def ensure_similarity_index(pipe, index_path: str = INDEX_PATH):
    if not os.path.exists(index_path):
        build_similarity_index(pipe, index_path)

def _load_index(pipe, index_path: str = INDEX_PATH):
    if index_path not in _INDEX_CACHE:
        ensure_similarity_index(pipe, index_path)
        data = np.load(index_path, allow_pickle=False)
        _INDEX_CACHE[index_path] = (data["embeddings"], data["labels"])
    return _INDEX_CACHE[index_path]


def load_artifacts(ckpt_dir: str = CKPT_DIR):
    pipe = joblib.load(os.path.join(ckpt_dir, "best_pipeline.joblib"))
    with open(os.path.join(ckpt_dir, "label_maps.json")) as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    return pipe, id2label, maps["target"], maps["best_model_name"]


def _confidence_explanation(n_classes: int, confidence: float, pred_readable: str,
                            second_readable: str, second_conf: float) -> str:
    margin = confidence - second_conf
    if margin >= 0.50:
        verdict = "a clear-cut decision"
    elif margin >= 0.15:
        verdict = "a solid margin"
    else:
        verdict = "a narrow margin — the model is torn between these two intents"
    return (f"The model distributes probability across all {n_classes} intents: "
            f"'{pred_readable}' received {confidence:.1%}, the runner-up "
            f"'{second_readable}' received {second_conf:.1%} — {verdict}.")


def explain_prediction(pipe, id2label: dict, text: str) -> dict:
    """Single method: SBERT embedding + trained classifier.

    - confidence: the classifier's own predict_proba, with the runner-up
      intent quoted so the number explains itself.
    - intent: similarity evidence — how many of the nearest training tickets
      (cosine similarity in the same embedding space) share the predicted label.
    """
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(-proba)
    pred_id, second_id = int(order[0]), int(order[1])
    confidence, second_conf = float(proba[pred_id]), float(proba[second_id])

    pred_label = id2label[pred_id]
    pred_readable = humanize_label(pred_label)
    second_readable = humanize_label(id2label[second_id])
    confidence_explanation = _confidence_explanation(
        len(id2label), confidence, pred_readable, second_readable, second_conf)

    emb, labels = _load_index(pipe)
    q = np.asarray(pipe.named_steps["features"].transform([text]), dtype=np.float32)[0]
    q /= np.linalg.norm(q) + 1e-12
    sims = emb @ q
    top_idx = np.argsort(-sims)[:TOP_K_SIMILAR]
    n_match = sum(1 for i in top_idx if str(labels[i]) == pred_label)

    explanation_text = (
        f"Predicted '{pred_readable}' because this ticket's meaning is closest "
        f"to real customer tickets the model learned from: {n_match} of the "
        f"{len(top_idx)} most similar training tickets were labeled "
        f"'{pred_readable}' (closest match similarity "
        f"{float(sims[top_idx[0]]):.2f} out of 1.00)."
    )

    return {
        "input_text": text,
        "predicted_label": pred_label,
        "predicted_label_readable": pred_readable,
        "confidence": round(confidence, 4),
        "confidence_explanation": confidence_explanation,
        "explanation_method": "SBERT sentence embedding + trained classifier; confidence "
                               "from predict_proba, intent evidence from cosine similarity "
                               "to training tickets in the same embedding space.",
        "explanation": explanation_text,
    }


def print_explanation(result: dict):
    print(f"\nInput: {result['input_text']!r}")
    print(f"Predicted label: {result['predicted_label_readable']}  ({result['predicted_label']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Why this confidence: {result['confidence_explanation']}")
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
