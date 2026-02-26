from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import kagglehub
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


TEXT_CANDIDATES = ("headline", "title", "text", "news", "sentence")
LABEL_CANDIDATES = ("sentiment", "label", "labels", "target", "class")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_to_real:
            return lower_to_real[cand]
    raise ValueError(f"Could not infer {kind} column. Available columns: {list(df.columns)}")


def load_kaggle_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    try:
        infer_column(df, TEXT_CANDIDATES, "text")
        infer_column(df, LABEL_CANDIDATES, "label")
        return df
    except ValueError:
        pass

    # Fallback for headerless files like: <headline>,<label>
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] >= 2:
        raw = raw.iloc[:, :2].copy()
        raw.columns = ["text", "label"]
        return raw
    return df


def map_labels(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
        "-1": 0,
        "0": 1,
        "1": 2,
        "bearish": 0,
        "bullish": 2,
    }
    mapped = raw.map(mapping)
    if mapped.isna().any():
        unknown = sorted(raw[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unmapped label values found: {unknown}")
    return mapped.astype(int)


def split_df(df: pd.DataFrame, seed: int = 42) -> Splits:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    i1 = int(0.8 * n)
    i2 = int(0.9 * n)
    return Splits(train=df.iloc[:i1], val=df.iloc[i1:i2], test=df.iloc[i2:])


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_macro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "precision_negative": p_cls[0],
        "recall_negative": r_cls[0],
        "f1_negative": f1_cls[0],
        "precision_neutral": p_cls[1],
        "recall_neutral": r_cls[1],
        "f1_neutral": f1_cls[1],
        "precision_positive": p_cls[2],
        "recall_positive": r_cls[2],
        "f1_positive": f1_cls[2],
    }


def to_hf_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)

    def tok(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    return ds.map(tok, batched=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on Kaggle labeled news dataset.")
    parser.add_argument("--dataset", default="haojie98/news-headline")
    parser.add_argument("--model", default="ProsusAI/finbert")
    parser.add_argument("--output-dir", default="models/finbert_kaggle")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset folder: {path}")

    # Pick the largest CSV by size as default training table.
    csv_files = sorted(csv_files, key=lambda p: os.path.getsize(p), reverse=True)
    csv_path = csv_files[0]
    print(f"Using dataset file: {csv_path}")

    df = load_kaggle_table(csv_path)
    text_col = infer_column(df, TEXT_CANDIDATES + ("text",), "text")
    label_col = infer_column(df, LABEL_CANDIDATES + ("label",), "label")
    print(f"Inferred columns -> text: {text_col}, label: {label_col}")

    clean = df[[text_col, label_col]].copy()
    clean = clean.rename(columns={text_col: "text", label_col: "label"})
    clean = clean.dropna(subset=["text", "label"])
    clean["text"] = clean["text"].astype(str).str.strip()
    clean = clean[clean["text"].ne("")]
    clean["labels"] = map_labels(clean["label"])
    clean = clean[["text", "labels"]].drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Training rows after cleaning: {len(clean)}")

    splits = split_df(clean, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    train_ds = to_hf_dataset(splits.train, tokenizer, max_length=args.max_length)
    val_ds = to_hf_dataset(splits.val, tokenizer, max_length=args.max_length)
    test_ds = to_hf_dataset(splits.test, tokenizer, max_length=args.max_length)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_ds)
    print("Test metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model/tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
