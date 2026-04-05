from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.models.test_sentiment_models import (
    BODY_CANDIDATES,
    TEXT_CANDIDATES,
    infer_column,
    normalize_whitespace,
    predict_finbert,
    predict_fingpt,
)


LABEL_CANDIDATES = ("sentiment", "label", "sentiment_label", "gold_label", "llm_consensus_label")
LABEL_ORDER = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face sentiment models on a labeled CSV.")
    parser.add_argument(
        "--input-csv",
        default="data/news_labels/news_human_label_sample_2025_100_cleaned.csv",
        help="Labeled CSV to evaluate on.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluations/sentiment_eval",
        help="Directory to store metrics, reports, and predictions.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick testing.")
    parser.add_argument("--text-column", default="", help="Optional explicit text column.")
    parser.add_argument("--label-column", default="", help="Optional explicit gold label column.")
    parser.add_argument("--use-body", action="store_true", help="Append body text to the text column.")
    parser.add_argument("--skip-finbert", action="store_true")
    parser.add_argument("--skip-fingpt", action="store_true")
    parser.add_argument("--finbert-model", default="anasamri12/finbert-malaysia-sentiment-run2")
    parser.add_argument("--finbert-zeroshot-model", default="")
    parser.add_argument("--fingpt-base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--fingpt-adapter", default="anasamri12/fingpt-malaysia-lora-run2")
    parser.add_argument("--fingpt-zeroshot-base-model", default="")
    parser.add_argument("--finbert-batch-size", type=int, default=32)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    text = normalize_whitespace(value).lower()
    if not text:
        return ""

    direct_map = {
        "-1": "negative",
        "0": "neutral",
        "1": "positive",
        "neg": "negative",
        "neut": "neutral",
        "neu": "neutral",
        "pos": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "bearish": "negative",
        "mixed": "neutral",
        "bullish": "positive",
    }
    if text in direct_map:
        return direct_map[text]

    if re.search(r"\bnegative\b|\bneg\b|\bbearish\b", text):
        return "negative"
    if re.search(r"\bneutral\b|\bneut\b|\bmixed\b", text):
        return "neutral"
    if re.search(r"\bpositive\b|\bpos\b|\bbullish\b", text):
        return "positive"
    return ""


def compute_metrics(gold_labels: list[str], pred_labels: list[str]) -> dict[str, float]:
    gold_idx = [LABEL_ORDER.index(label) for label in gold_labels]
    pred_idx = [LABEL_ORDER.index(label) for label in pred_labels]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_idx,
        pred_idx,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold_idx,
        pred_idx,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(gold_idx, pred_idx)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_eval_artifacts(
    model_dir: Path,
    predictions_df: pd.DataFrame,
    gold_labels: list[str],
    pred_labels: list[str],
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    report_text = classification_report(
        gold_labels,
        pred_labels,
        labels=LABEL_ORDER,
        digits=6,
        zero_division=0,
    )
    report_json = classification_report(
        gold_labels,
        pred_labels,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(gold_labels, pred_labels, labels=LABEL_ORDER)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in LABEL_ORDER],
        columns=[f"pred_{label}" for label in LABEL_ORDER],
    )

    predictions_df.to_csv(model_dir / "predictions.csv", index=False, encoding="utf-8")
    with (model_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({**metadata, **metrics}, f, indent=2, sort_keys=True)
    with (model_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")
    with (model_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, sort_keys=True)
    matrix_df.to_csv(model_dir / "confusion_matrix.csv")


def main() -> None:
    args = parse_args()

    run_any = not (args.skip_finbert and args.skip_fingpt) or bool(args.finbert_zeroshot_model) or bool(
        args.fingpt_zeroshot_base_model
    )
    if not run_any:
        raise ValueError("No model run selected.")

    df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    text_col = infer_column(df, TEXT_CANDIDATES, "text") if not args.text_column else args.text_column
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available columns: {list(df.columns)}")

    label_col = infer_column(df, LABEL_CANDIDATES, "gold label") if not args.label_column else args.label_column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available columns: {list(df.columns)}")

    body_col = infer_column(df, BODY_CANDIDATES, "body") if args.use_body else None

    text_values = df[text_col].fillna("").astype(str).map(normalize_whitespace)
    if body_col:
        body_values = df[body_col].fillna("").astype(str).map(normalize_whitespace)
        text_values = (text_values + " " + body_values).map(normalize_whitespace)

    work_df = df.copy()
    work_df["input_text"] = text_values
    work_df["gold_label"] = work_df[label_col].map(normalize_label)
    eval_df = work_df.loc[work_df["gold_label"].isin(LABEL_ORDER)].copy().reset_index(drop=True)
    if eval_df.empty:
        raise ValueError(f"No valid gold labels found in column '{label_col}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using text column: {text_col}")
    print(f"Using gold label column: {label_col}")
    if body_col:
        print(f"Appending body column: {body_col}")
    print(f"Running on device: {device}")
    print(f"Rows scored: {len(eval_df)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    if not args.skip_finbert:
        print(f"Evaluating FinBERT model: {args.finbert_model}")
        labels, p_neg, p_neu, p_pos = predict_finbert(
            texts=eval_df["input_text"].tolist(),
            model_id=args.finbert_model,
            device=device,
            batch_size=args.finbert_batch_size,
        )
        pred_df = eval_df.copy()
        pred_df["pred_label"] = labels
        pred_df["prob_negative"] = p_neg
        pred_df["prob_neutral"] = p_neu
        pred_df["prob_positive"] = p_pos

        metrics = compute_metrics(pred_df["gold_label"].tolist(), pred_df["pred_label"].tolist())
        metadata = {
            "model_family": "finbert",
            "model_source": args.finbert_model,
            "input_csv": args.input_csv,
            "text_column": text_col,
            "label_column": label_col,
            "rows_scored": int(len(pred_df)),
        }
        save_eval_artifacts(
            output_dir / "finbert",
            pred_df,
            pred_df["gold_label"].tolist(),
            pred_df["pred_label"].tolist(),
            metrics,
            metadata,
        )
        summary_rows.append({**metadata, **metrics})

    if args.finbert_zeroshot_model:
        print(f"Evaluating FinBERT zero-shot model: {args.finbert_zeroshot_model}")
        labels, p_neg, p_neu, p_pos = predict_finbert(
            texts=eval_df["input_text"].tolist(),
            model_id=args.finbert_zeroshot_model,
            device=device,
            batch_size=args.finbert_batch_size,
        )
        pred_df = eval_df.copy()
        pred_df["pred_label"] = labels
        pred_df["prob_negative"] = p_neg
        pred_df["prob_neutral"] = p_neu
        pred_df["prob_positive"] = p_pos

        metrics = compute_metrics(pred_df["gold_label"].tolist(), pred_df["pred_label"].tolist())
        metadata = {
            "model_family": "finbert_zeroshot",
            "model_source": args.finbert_zeroshot_model,
            "input_csv": args.input_csv,
            "text_column": text_col,
            "label_column": label_col,
            "rows_scored": int(len(pred_df)),
        }
        save_eval_artifacts(
            output_dir / "finbert_zeroshot",
            pred_df,
            pred_df["gold_label"].tolist(),
            pred_df["pred_label"].tolist(),
            metrics,
            metadata,
        )
        summary_rows.append({**metadata, **metrics})

    if not args.skip_fingpt:
        print(f"Evaluating FinGPT base model: {args.fingpt_base_model}")
        print(f"Evaluating FinGPT adapter: {args.fingpt_adapter}")
        labels, raw_outputs = predict_fingpt(
            texts=eval_df["input_text"].tolist(),
            base_model_id=args.fingpt_base_model,
            adapter_id=args.fingpt_adapter,
            device=device,
        )
        pred_df = eval_df.copy()
        pred_df["pred_label"] = labels
        pred_df["raw_output"] = raw_outputs

        metrics = compute_metrics(pred_df["gold_label"].tolist(), pred_df["pred_label"].tolist())
        metadata = {
            "model_family": "fingpt",
            "model_source": args.fingpt_base_model,
            "adapter_source": args.fingpt_adapter,
            "input_csv": args.input_csv,
            "text_column": text_col,
            "label_column": label_col,
            "rows_scored": int(len(pred_df)),
        }
        save_eval_artifacts(
            output_dir / "fingpt",
            pred_df,
            pred_df["gold_label"].tolist(),
            pred_df["pred_label"].tolist(),
            metrics,
            metadata,
        )
        summary_rows.append({**metadata, **metrics})

    if args.fingpt_zeroshot_base_model:
        print(f"Evaluating FinGPT zero-shot base model: {args.fingpt_zeroshot_base_model}")
        labels, raw_outputs = predict_fingpt(
            texts=eval_df["input_text"].tolist(),
            base_model_id=args.fingpt_zeroshot_base_model,
            adapter_id=None,
            device=device,
        )
        pred_df = eval_df.copy()
        pred_df["pred_label"] = labels
        pred_df["raw_output"] = raw_outputs

        metrics = compute_metrics(pred_df["gold_label"].tolist(), pred_df["pred_label"].tolist())
        metadata = {
            "model_family": "fingpt_zeroshot",
            "model_source": args.fingpt_zeroshot_base_model,
            "adapter_source": "",
            "input_csv": args.input_csv,
            "text_column": text_col,
            "label_column": label_col,
            "rows_scored": int(len(pred_df)),
        }
        save_eval_artifacts(
            output_dir / "fingpt_zeroshot",
            pred_df,
            pred_df["gold_label"].tolist(),
            pred_df["pred_label"].tolist(),
            metrics,
            metadata,
        )
        summary_rows.append({**metadata, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
