"""Data preparation for Banking77 intent classification.

Real, MANUALLY-LABELED dataset: anonymized online banking customer service
queries, annotated with intents by human annotators (Casanueva et al.,
ACL NLP4ConvAI 2020 — "Efficient Intent Detection with Dual Sentence
Encoders"). 13,083 examples, 77 intents. Uses the dataset's OFFICIAL
train (10,003) / test (3,080) split — zero text overlap between splits.

CACHING: on first run, downloads from Hugging Face and saves the full
(train+test, with a `split` column) dataset to DATA_CSV_PATH. Every run
after that loads directly from that local CSV — no network call, no
re-download. Delete the CSV (or call load_prepared(force_download=True))
to refresh from source.

Requirements:
    pip install datasets pandas scikit-learn
"""
import os
import re
import pandas as pd

DATA_CSV_PATH = "data/data.csv"
HF_DATASET = "mteb/banking77"

PRIMARY_TARGET = "label_text"      # human-readable intent name (77 classes)

# Optional: cap examples per class for a smaller/faster dataset. None = full.
SAMPLES_PER_CLASS = None


def clean_text(s) -> str:
    """Collapse newlines/whitespace so the text field is one clean line."""
    return re.sub(r"\s+", " ", str(s)).strip()


def _download_split(split: str) -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily — only needed on cache miss
    ds = load_dataset(HF_DATASET, split=split)
    df = ds.to_pandas()
    df["text"] = df["text"].map(clean_text)
    return df[["text", "label_text"]].copy()


def _download_full() -> pd.DataFrame:
    print(f"'{DATA_CSV_PATH}' not found — downloading '{HF_DATASET}' from Hugging Face...")
    train_df = _download_split("train")
    train_df["split"] = "train"
    test_df = _download_split("test")
    test_df["split"] = "test"
    full = pd.concat([train_df, test_df], ignore_index=True)

    os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
    full.to_csv(DATA_CSV_PATH, index=False)
    print(f"Saved {len(full)} rows to '{DATA_CSV_PATH}' for future runs.")
    return full


def load_prepared(force_download: bool = False) -> pd.DataFrame:
    """Return the full dataset (official train+test, with a `split` column).
    Loads from DATA_CSV_PATH if it exists; otherwise downloads once and
    caches it there. Set force_download=True to bypass the cache.
    """
    if not force_download and os.path.exists(DATA_CSV_PATH):
        df = pd.read_csv(DATA_CSV_PATH)
    else:
        df = _download_full()

    if SAMPLES_PER_CLASS is not None:
        df = (df.groupby(PRIMARY_TARGET, group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), SAMPLES_PER_CLASS), random_state=42))
                .reset_index(drop=True))
    return df


def get_label_maps(df: pd.DataFrame, target: str = PRIMARY_TARGET):
    """Return (labels, label2id, id2label) with a stable, sorted label order."""
    labels = sorted(df[target].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


def load_splits(test_size=None, seed: int = 42, target: str = PRIMARY_TARGET):
    """Banking77's OFFICIAL train/test split (loaded from the local cache,
    downloading once on first run — see load_prepared). `test_size`/`seed`
    kept in the signature for interface compatibility but ignored — this
    dataset's own official split is used instead of a random re-split.
    Returns (train_df, val_df, label2id, id2label), each with `text` +
    integer `labels` columns.
    """
    full = load_prepared()
    labels, label2id, id2label = get_label_maps(full, target)

    train_raw = full[full["split"] == "train"]
    test_raw = full[full["split"] == "test"]

    train_df = train_raw[["text"]].copy()
    train_df["labels"] = train_raw[target].map(label2id)
    val_df = test_raw[["text"]].copy()
    val_df["labels"] = test_raw[target].map(label2id)

    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            label2id, id2label)


if __name__ == "__main__":
    df = load_prepared()
    cached = os.path.exists(DATA_CSV_PATH)
    print(f"Loaded {len(df)} rows  |  source: {'local cache (' + DATA_CSV_PATH + ')' if cached else 'downloaded'}")
    print(f"Split sizes: {df['split'].value_counts().to_dict()}")

    print("\n=== WHAT THE MODEL TAKES AS INPUT ===")
    print("Input field: text  (raw customer banking query)")
    print("Example input:", repr(df["text"].iloc[0]))

    print(f"\n=== WHAT THE MODEL MUST PREDICT ===")
    print(f"Target field: '{PRIMARY_TARGET}'")
    labels, label2id, id2label = get_label_maps(df)
    print(f"Number of classes: {len(labels)}")
    print(f"Example label for the input above:", df[PRIMARY_TARGET].iloc[0])
    print(f"First 10 classes: {labels[:10]} ...")

    print("\n=== CLASS DISTRIBUTION (label_text, 77 classes) ===")
    counts = df[PRIMARY_TARGET].value_counts()
    print(f"min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

    print("\n=== TEXT LENGTH STATS (characters) ===")
    lens = df["text"].astype(str).str.len()
    print(f"min={lens.min()}  max={lens.max()}  mean={lens.mean():.1f}  median={lens.median()}")

    print("\n=== AMBIGUITY CHECK (same test we ran on prior datasets) ===")
    dup = df.groupby("text")[PRIMARY_TARGET].nunique()
    ambiguous = dup[dup > 1]
    print(f"Unique texts: {df['text'].nunique()} / {len(df)} rows")
    print(f"Duplicate texts mapping to >1 label: {len(ambiguous)}")

    print("\n=== TRAIN/TEST OVERLAP CHECK ===")
    train_texts = set(df[df["split"] == "train"]["text"])
    test_texts = set(df[df["split"] == "test"]["text"])
    print(f"Texts appearing in BOTH train and test: {len(train_texts & test_texts)}")
