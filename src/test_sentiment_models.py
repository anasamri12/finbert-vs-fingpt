from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


LABEL_ID_TO_TEXT = {0: "negative", 1: "neutral", 2: "positive"}
TEXT_CANDIDATES = ("title", "headline", "text", "news", "sentence", "body")
BODY_CANDIDATES = ("body", "content", "article", "article_body")


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_to_real:
            return lower_to_real[cand]
    raise ValueError(f"Could not infer {kind} column. Available columns: {list(df.columns)}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def build_fingpt_prompt(text: str) -> str:
    return (
        "You are a financial sentiment classifier.\n"
        "Return only one label: positive, neutral, or negative.\n\n"
        f"Headline: {text}\n"
        "Sentiment:"
    )


def normalize_label_text(generated: str) -> str:
    out = generated.strip().lower()
    if "positive" in out:
        return "positive"
    if "negative" in out:
        return "negative"
    if "neutral" in out:
        return "neutral"
    return "neutral"


def predict_finbert(
    texts: list[str],
    model_id: str,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 32,
) -> tuple[list[str], list[float], list[float], list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    labels: list[str] = []
    p_neg: list[float] = []
    p_neu: list[float] = []
    p_pos: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = torch.argmax(probs, dim=-1).tolist()

            labels.extend(LABEL_ID_TO_TEXT[int(idx)] for idx in pred_ids)
            p_neg.extend(probs[:, 0].tolist())
            p_neu.extend(probs[:, 1].tolist())
            p_pos.extend(probs[:, 2].tolist())

    return labels, p_neg, p_neu, p_pos


def predict_fingpt(
    texts: list[str],
    base_model_id: str,
    adapter_id: str | None,
    device: torch.device,
    max_length: int = 512,
    max_new_tokens: int = 4,
) -> tuple[list[str], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model = base_model if adapter_id is None else PeftModel.from_pretrained(base_model, adapter_id)
    model.to(device)
    model.eval()

    labels: list[str] = []
    raw_outputs: list[str] = []
    with torch.no_grad():
        for idx, text in enumerate(texts, start=1):
            prompt = build_fingpt_prompt(text)
            enc = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
            raw_outputs.append(gen)
            labels.append(normalize_label_text(gen))
            if idx % 100 == 0:
                print(f"[fingpt] Predicted {idx}/{len(texts)} rows")
    return labels, raw_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinBERT and/or FinGPT sentiment prediction on a CSV file.")
    parser.add_argument("--input-csv", default="src/malaysia_news_last_1_months_test.csv")
    parser.add_argument("--output-csv", default="results/predictions/sentiment_predictions.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick testing.")
    parser.add_argument("--text-column", default=None, help="Optional explicit text column to use.")
    parser.add_argument("--use-body", action="store_true", help="Append body text to the text column.")
    parser.add_argument("--skip-finbert", action="store_true")
    parser.add_argument("--skip-fingpt", action="store_true")
    parser.add_argument("--finbert-model", default="anasamri12/finbert-malaysia-sentiment-run2")
    parser.add_argument("--finbert-zeroshot-model", default="ProsusAI/finbert")
    parser.add_argument("--fingpt-base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--fingpt-adapter", default="anasamri12/fingpt-malaysia-lora-run2")
    parser.add_argument("--fingpt-zeroshot-base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--run-finbert-zeroshot", action="store_true")
    parser.add_argument("--run-fingpt-zeroshot", action="store_true")
    parser.add_argument("--finbert-batch-size", type=int, default=32)
    args = parser.parse_args()

    run_finetuned = not (args.skip_finbert and args.skip_fingpt)
    run_any = run_finetuned or args.run_finbert_zeroshot or args.run_fingpt_zeroshot
    if not run_any:
        raise ValueError(
            "No model run selected. Use fine-tuned defaults or set --run-finbert-zeroshot/--run-fingpt-zeroshot."
        )

    df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    text_col = args.text_column or infer_column(df, TEXT_CANDIDATES, "text")
    body_col = infer_column(df, BODY_CANDIDATES, "body") if args.use_body else None
    print(f"Using text column: {text_col}")
    if body_col:
        print(f"Appending body column: {body_col}")

    text_values = df[text_col].fillna("").astype(str).map(normalize_whitespace)
    if body_col:
        body_values = df[body_col].fillna("").astype(str).map(normalize_whitespace)
        text_values = (text_values + " " + body_values).map(normalize_whitespace)
    texts = text_values.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    result_df = df.copy()

    if not args.skip_finbert:
        print(f"Loading FinBERT model: {args.finbert_model}")
        fb_label, p_neg, p_neu, p_pos = predict_finbert(
            texts=texts,
            model_id=args.finbert_model,
            device=device,
            batch_size=args.finbert_batch_size,
        )
        result_df["finbert_label"] = fb_label
        result_df["finbert_prob_negative"] = p_neg
        result_df["finbert_prob_neutral"] = p_neu
        result_df["finbert_prob_positive"] = p_pos

    if args.run_finbert_zeroshot:
        print(f"Loading FinBERT zero-shot model: {args.finbert_zeroshot_model}")
        z_label, z_p_neg, z_p_neu, z_p_pos = predict_finbert(
            texts=texts,
            model_id=args.finbert_zeroshot_model,
            device=device,
            batch_size=args.finbert_batch_size,
        )
        result_df["finbert_zeroshot_label"] = z_label
        result_df["finbert_zeroshot_prob_negative"] = z_p_neg
        result_df["finbert_zeroshot_prob_neutral"] = z_p_neu
        result_df["finbert_zeroshot_prob_positive"] = z_p_pos

    if not args.skip_fingpt:
        print(f"Loading FinGPT base model: {args.fingpt_base_model}")
        print(f"Loading FinGPT adapter: {args.fingpt_adapter}")
        fg_label, fg_raw = predict_fingpt(
            texts=texts,
            base_model_id=args.fingpt_base_model,
            adapter_id=args.fingpt_adapter,
            device=device,
        )
        result_df["fingpt_label"] = fg_label
        result_df["fingpt_raw_output"] = fg_raw

    if args.run_fingpt_zeroshot:
        print(f"Loading FinGPT zero-shot base model: {args.fingpt_zeroshot_base_model}")
        zg_label, zg_raw = predict_fingpt(
            texts=texts,
            base_model_id=args.fingpt_zeroshot_base_model,
            adapter_id=None,
            device=device,
        )
        result_df["fingpt_zeroshot_label"] = zg_label
        result_df["fingpt_zeroshot_raw_output"] = zg_raw

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved predictions to: {output_path}")
    print(f"Rows processed: {len(result_df)}")


if __name__ == "__main__":
    main()
