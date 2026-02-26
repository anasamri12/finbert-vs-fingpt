from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer


TEXT_CANDIDATES = ("headline", "title", "text", "news", "sentence")
LABEL_CANDIDATES = ("sentiment", "label", "labels", "target", "class")
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


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


def build_train_text(text: str, label_id: int) -> str:
    label = ID2LABEL[int(label_id)]
    return (
        "You are a financial sentiment classifier.\n"
        "Return only one label: positive, neutral, or negative.\n\n"
        f"Headline: {text}\n"
        f"Sentiment: {label}"
    )


def build_eval_prompt(text: str) -> str:
    return (
        "You are a financial sentiment classifier.\n"
        "Return only one label: positive, neutral, or negative.\n\n"
        f"Headline: {text}\n"
        "Sentiment:"
    )


def normalize_label_text(generated: str) -> int:
    out = generated.strip().lower()
    if "positive" in out:
        return 2
    if "negative" in out:
        return 0
    if "neutral" in out:
        return 1
    # fallback: conservative to neutral if model output is noisy
    return 1


def evaluate_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    max_new_tokens: int = 4,
    max_eval_rows: int | None = 1000,
) -> tuple[dict[str, float], list[int], list[int]]:
    eval_df = df if max_eval_rows is None else df.iloc[:max_eval_rows]
    preds: list[int] = []
    gold: list[int] = []

    model.eval()
    for _, row in eval_df.iterrows():
        prompt = build_eval_prompt(str(row["text"]))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = normalize_label_text(gen)
        preds.append(pred)
        gold.append(int(row["labels"]))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold, preds, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        gold, preds, labels=[0, 1, 2], average=None, zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(gold, preds),
        "f1_macro": f1_macro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
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
    return metrics, gold, preds


def save_test_reports(output_dir: str, labels: list[int], preds: list[int], metrics: dict[str, float]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_text = classification_report(
        labels, preds, labels=[0, 1, 2], target_names=["negative", "neutral", "positive"], digits=6, zero_division=0
    )
    report_json = classification_report(
        labels,
        preds,
        labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, preds, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        matrix,
        index=["true_negative", "true_neutral", "true_positive"],
        columns=["pred_negative", "pred_neutral", "pred_positive"],
    )

    with (out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, sort_keys=True)
    with (out / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")
    with (out / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, sort_keys=True)
    matrix_df.to_csv(out / "confusion_matrix.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinGPT-style model on Kaggle labeled headlines.")
    parser.add_argument("--dataset", default="haojie98/news-headline")
    parser.add_argument("--model", required=True, help="Base causal LM checkpoint (FinGPT-compatible).")
    parser.add_argument("--output-dir", default="models/fingpt_kaggle_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-max-rows", type=int, default=1000)
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    csv_files = sorted(glob.glob(os.path.join(path, "*.csv")), key=lambda p: os.path.getsize(p), reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset folder: {path}")
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
    train_df = splits.train.copy()
    val_df = splits.val.copy()
    test_df = splits.test.copy()
    train_df["train_text"] = train_df.apply(lambda r: build_train_text(str(r["text"]), int(r["labels"])), axis=1)
    val_df["train_text"] = val_df.apply(lambda r: build_train_text(str(r["text"]), int(r["labels"])), axis=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_pandas(train_df[["train_text"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["train_text"]], preserve_index=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        bf16=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="train_text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    test_metrics, gold, preds = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        max_eval_rows=args.eval_max_rows,
    )
    print("Test metrics:", test_metrics)
    save_test_reports(args.output_dir, gold, preds, test_metrics)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter/tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
