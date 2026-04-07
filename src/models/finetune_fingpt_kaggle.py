from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from .ucl_cache import configure_ucl_scratch_cache
except ImportError:
    from src.models.ucl_cache import configure_ucl_scratch_cache

configure_ucl_scratch_cache()

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
    Trainer,
    TrainingArguments,
)


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


def load_table(csv_path: str) -> pd.DataFrame:
    return load_kaggle_table(csv_path)


def prepare_labeled_df(df: pd.DataFrame, text_column: str = "", label_column: str = "") -> tuple[pd.DataFrame, str, str]:
    text_col = text_column or infer_column(df, TEXT_CANDIDATES + ("text",), "text")
    label_col = label_column or infer_column(df, LABEL_CANDIDATES + ("label",), "label")

    clean = df[[text_col, label_col]].copy()
    clean = clean.rename(columns={text_col: "text", label_col: "label"})
    clean = clean.dropna(subset=["text", "label"])
    clean["text"] = clean["text"].astype(str).str.strip()
    clean = clean[clean["text"].ne("")]
    clean["labels"] = map_labels(clean["label"])
    clean = clean[["text", "labels"]].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return clean, text_col, label_col


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


def build_messages(text: str, label_id: int | None = None) -> list[dict[str, str]]:
    content = f"Headline: {text}\nSentiment:"
    if label_id is not None:
        content = f"{content} {ID2LABEL[int(label_id)]}"
    return [
        {
            "role": "system",
            "content": "You are a financial sentiment classifier. Return only one label: positive, neutral, or negative.",
        },
        {"role": "user", "content": content},
    ]


def render_prompt_for_tokenizer(tokenizer: AutoTokenizer, text: str, label_id: int | None = None) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            build_messages(text, label_id),
            tokenize=False,
            add_generation_prompt=label_id is None,
        )
    if label_id is None:
        return build_eval_prompt(text)
    return build_train_text(text, label_id)


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


def tokenize_for_causal_lm(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["train_text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


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
        prompt = render_prompt_for_tokenizer(tokenizer, str(row["text"]))
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


def save_metrics_json(output_dir: str, filename: str, metrics: dict[str, float]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / filename).open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinGPT-style model on a labeled headlines dataset.")
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional Kaggle dataset slug. Ignored when --input-csv is provided.",
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Optional local labeled CSV path. Use this for pseudo-labeled adaptation datasets.",
    )
    parser.add_argument("--train-csv", default="", help="Optional explicit training CSV for fixed-split runs.")
    parser.add_argument("--val-csv", default="", help="Optional explicit validation CSV for fixed-split runs.")
    parser.add_argument("--test-csv", default="", help="Optional explicit test CSV for fixed-split runs.")
    parser.add_argument(
        "--text-column",
        default="",
        help="Optional explicit text column when using --input-csv.",
    )
    parser.add_argument(
        "--label-column",
        default="",
        help="Optional explicit label column when using --input-csv.",
    )
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

    using_explicit_splits = any([args.train_csv, args.val_csv, args.test_csv])
    if using_explicit_splits and not all([args.train_csv, args.val_csv, args.test_csv]):
        raise ValueError("When using explicit split files, provide --train-csv, --val-csv, and --test-csv together.")
    if using_explicit_splits and (args.input_csv or args.dataset):
        raise ValueError("Use either explicit split files or --input-csv/--dataset, not both.")

    if using_explicit_splits:
        train_path = Path(args.train_csv)
        val_path = Path(args.val_csv)
        test_path = Path(args.test_csv)
        for path in (train_path, val_path, test_path):
            if not path.exists():
                raise FileNotFoundError(f"Split CSV not found: {path}")
        print(f"Using explicit split files: train={train_path}, val={val_path}, test={test_path}")
    elif args.input_csv:
        csv_path = args.input_csv
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"--input-csv not found: {csv_path}")
        print(f"Using local dataset file: {csv_path}")
    else:
        if not args.dataset:
            raise ValueError("Provide either --input-csv or --dataset.")
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError("kagglehub is required for --dataset. Install with: pip install kagglehub") from exc

        path = kagglehub.dataset_download(args.dataset)
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")), key=lambda p: os.path.getsize(p), reverse=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in downloaded dataset folder: {path}")
        csv_path = csv_files[0]
        print(f"Using Kaggle dataset file: {csv_path}")

    if using_explicit_splits:
        train_clean, text_col, label_col = prepare_labeled_df(load_table(args.train_csv), args.text_column, args.label_column)
        val_clean, _, _ = prepare_labeled_df(load_table(args.val_csv), args.text_column, args.label_column)
        test_clean, _, _ = prepare_labeled_df(load_table(args.test_csv), args.text_column, args.label_column)
        print(f"Inferred columns -> text: {text_col}, label: {label_col}")
        print(
            "Rows after cleaning:"
            f" train={len(train_clean)}, val={len(val_clean)}, test={len(test_clean)}"
        )
        splits = Splits(train=train_clean, val=val_clean, test=test_clean)
    else:
        df = load_table(csv_path)
        clean, text_col, label_col = prepare_labeled_df(df, args.text_column, args.label_column)
        print(f"Inferred columns -> text: {text_col}, label: {label_col}")
        print(f"Training rows after cleaning: {len(clean)}")
        splits = split_df(clean, seed=args.seed)

    train_df = splits.train.copy()
    val_df = splits.val.copy()
    test_df = splits.test.copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df["train_text"] = train_df.apply(
        lambda r: render_prompt_for_tokenizer(tokenizer, str(r["text"]), int(r["labels"])), axis=1
    )
    val_df["train_text"] = val_df.apply(
        lambda r: render_prompt_for_tokenizer(tokenizer, str(r["text"]), int(r["labels"])), axis=1
    )

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
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_ds = Dataset.from_pandas(train_df[["train_text"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["train_text"]], preserve_index=False)
    train_ds = tokenize_for_causal_lm(train_ds, tokenizer, args.max_length)
    val_ds = tokenize_for_causal_lm(val_ds, tokenizer, args.max_length)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    val_metrics, val_gold, val_preds = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        df=val_df,
        max_eval_rows=args.eval_max_rows,
    )
    print("Validation metrics:", val_metrics)
    save_metrics_json(args.output_dir, "validation_metrics.json", val_metrics)
    save_test_reports(str(Path(args.output_dir) / "validation_eval"), val_gold, val_preds, val_metrics)

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
