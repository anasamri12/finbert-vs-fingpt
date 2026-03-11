from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_CANDIDATES = ("date", "Date", "published_date", "published_at", "datetime")
CLOSE_CANDIDATES = ("Adj Close", "adj_close", "Close", "close", "price")


def _normalize_col_name(col: Any) -> str:
    if isinstance(col, tuple):
        parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
        return " ".join(parts).strip()
    return str(col).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_names = [_normalize_col_name(c) for c in out.columns]

    # Ensure uniqueness after flattening.
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in flat_names:
        base = name if name else "col"
        count = seen.get(base, 0)
        if count == 0:
            unique = base
        else:
            unique = f"{base}_{count+1}"
        seen[base] = count + 1
        unique_names.append(unique)

    out.columns = unique_names
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: compare daily sentiment signals against stock movement."
    )
    parser.add_argument(
        "--pred-glob",
        default="results/predictions/*preds.csv",
        help="Glob for prediction CSV files.",
    )
    parser.add_argument(
        "--stock-csv",
        default=None,
        help="CSV file with daily stock/index prices (must include date and close columns).",
    )
    parser.add_argument(
        "--yahoo-ticker",
        default=None,
        help="Yahoo Finance ticker (example: ^KLSE, AAPL, 5296.KL).",
    )
    parser.add_argument(
        "--yahoo-start",
        default=None,
        help="Yahoo Finance start date YYYY-MM-DD (default: --min-date).",
    )
    parser.add_argument(
        "--yahoo-end",
        default=None,
        help="Yahoo Finance end date YYYY-MM-DD (default: today + 1 day).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/experiment2",
        help="Output folder.",
    )
    parser.add_argument(
        "--min-date",
        default="2023-01-01",
        help="Filter both sentiment and stock data to this date or later.",
    )
    args = parser.parse_args()
    if not args.stock_csv and not args.yahoo_ticker:
        parser.error("Provide either --stock-csv or --yahoo-ticker.")
    return args


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    normalized = {_normalize_col_name(c).lower(): c for c in df.columns}

    # Exact match first.
    for cand in candidates:
        col = normalized.get(cand.lower())
        if col is not None:
            return col

    # Fallback for labels like "Adj Close ^KLSE" from yfinance multi-index columns.
    for cand in candidates:
        cand_low = cand.lower()
        for norm_name, raw_col in normalized.items():
            if norm_name.startswith(cand_low + " ") or norm_name.startswith(cand_low + "_"):
                return raw_col
            if cand_low in norm_name and cand_low in {"adj close", "close", "date"}:
                return raw_col

    raise ValueError(f"Could not infer {label} column. Available columns: {list(df.columns)}")


def detect_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.endswith("_label"):
            return col
    for col in ("label", "sentiment"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer sentiment label column. Available columns: {list(df.columns)}")


def detect_prob_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pos = None
    neg = None
    for col in df.columns:
        low = col.lower()
        if "prob_positive" in low:
            pos = col
        elif "prob_negative" in low:
            neg = col
    return pos, neg


def label_to_score(label_series: pd.Series) -> pd.Series:
    raw = label_series.astype(str).str.strip().str.lower()
    mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    score = raw.map(mapping)
    return score.fillna(0.0).astype(float)


def build_daily_sentiment(pred_csv: str, min_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(pred_csv)
    date_col = infer_column(df, DATE_CANDIDATES, "prediction date")
    label_col = detect_label_column(df)
    pos_prob_col, neg_prob_col = detect_prob_columns(df)

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
    work = work.dropna(subset=["date"]).copy()
    work = work.loc[work["date"] >= min_date.date()].copy()

    if pos_prob_col and neg_prob_col:
        work["sentiment_score"] = (
            pd.to_numeric(work[pos_prob_col], errors="coerce").fillna(0.0)
            - pd.to_numeric(work[neg_prob_col], errors="coerce").fillna(0.0)
        )
        score_mode = "prob_positive_minus_negative"
    else:
        work["sentiment_score"] = label_to_score(work[label_col])
        score_mode = "label_mapped_to_-1_0_1"

    lbl = work[label_col].astype(str).str.lower()
    work["is_positive"] = (lbl == "positive").astype(int)
    work["is_negative"] = (lbl == "negative").astype(int)
    work["is_neutral"] = (lbl == "neutral").astype(int)

    daily = (
        work.groupby("date", as_index=False)
        .agg(
            article_count=("sentiment_score", "size"),
            sentiment_score_mean=("sentiment_score", "mean"),
            sentiment_score_std=("sentiment_score", "std"),
            positive_ratio=("is_positive", "mean"),
            negative_ratio=("is_negative", "mean"),
            neutral_ratio=("is_neutral", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["sentiment_score_std"] = daily["sentiment_score_std"].fillna(0.0)

    meta = {
        "prediction_file": pred_csv,
        "rows_input": int(len(df)),
        "rows_after_date_filter": int(len(work)),
        "daily_rows": int(len(daily)),
        "date_col": date_col,
        "label_col": label_col,
        "score_mode": score_mode,
        "pos_prob_col": pos_prob_col,
        "neg_prob_col": neg_prob_col,
    }
    return daily, meta


def load_stock_prices(stock_csv: str, min_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(stock_csv)
    raw = _flatten_columns(raw)
    date_col = infer_column(raw, DATE_CANDIDATES, "stock date")
    close_col = infer_column(raw, CLOSE_CANDIDATES, "stock close price")

    stock = raw.copy()
    stock["date"] = pd.to_datetime(stock[date_col], errors="coerce")
    stock["close"] = pd.to_numeric(stock[close_col], errors="coerce")
    stock = stock.dropna(subset=["date", "close"]).copy()
    stock = stock.loc[stock["date"] >= min_date].copy()
    stock = stock.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    stock["return_1d"] = stock["close"].pct_change()
    stock["next_return_1d"] = stock["return_1d"].shift(-1)

    meta = {
        "stock_file": stock_csv,
        "rows_input": int(len(raw)),
        "rows_after_clean": int(len(stock)),
        "date_col": date_col,
        "close_col": close_col,
        "date_min": str(stock["date"].min().date()) if len(stock) else None,
        "date_max": str(stock["date"].max().date()) if len(stock) else None,
    }
    return stock, meta


def load_stock_prices_yahoo(
    ticker: str,
    min_date: pd.Timestamp,
    yahoo_start: str | None = None,
    yahoo_end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for --yahoo-ticker. Install with: pip install yfinance") from exc

    start = pd.to_datetime(yahoo_start if yahoo_start else min_date).date().isoformat()
    if yahoo_end:
        end = pd.to_datetime(yahoo_end).date().isoformat()
    else:
        # yfinance treats `end` as exclusive, so include tomorrow.
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"No Yahoo Finance data returned for ticker={ticker}, start={start}, end={end}.")

    raw = raw.reset_index()
    raw = _flatten_columns(raw)
    date_col = infer_column(raw, DATE_CANDIDATES, "stock date")
    close_col = infer_column(raw, CLOSE_CANDIDATES, "stock close price")

    stock = raw.copy()
    stock["date"] = pd.to_datetime(stock[date_col], errors="coerce")
    stock["close"] = pd.to_numeric(stock[close_col], errors="coerce")
    stock = stock.dropna(subset=["date", "close"]).copy()
    stock = stock.loc[stock["date"] >= min_date].copy()
    stock = stock.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    stock["return_1d"] = stock["close"].pct_change()
    stock["next_return_1d"] = stock["return_1d"].shift(-1)

    meta = {
        "stock_source": "yahoo_finance",
        "ticker": ticker,
        "query_start": start,
        "query_end_exclusive": end,
        "rows_after_clean": int(len(stock)),
        "date_col": date_col,
        "close_col": close_col,
        "date_min": str(stock["date"].min().date()) if len(stock) else None,
        "date_max": str(stock["date"].max().date()) if len(stock) else None,
    }
    return stock, meta


def evaluate_signal(score: pd.Series, ret: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"score": score, "ret": ret}).dropna()
    if len(df) < 5:
        return {
            "n_rows": int(len(df)),
            "pearson_corr": np.nan,
            "spearman_corr": np.nan,
            "directional_accuracy": np.nan,
            "directional_coverage": np.nan,
        }

    pearson = float(df["score"].corr(df["ret"], method="pearson"))
    spearman = float(df["score"].corr(df["ret"], method="spearman"))

    pred_dir = np.sign(df["score"].to_numpy())
    true_dir = np.sign(df["ret"].to_numpy())
    valid = (pred_dir != 0) & (true_dir != 0)
    if valid.any():
        dir_acc = float(np.mean(pred_dir[valid] == true_dir[valid]))
        coverage = float(np.mean(valid))
    else:
        dir_acc = np.nan
        coverage = 0.0

    return {
        "n_rows": int(len(df)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "directional_accuracy": dir_acc,
        "directional_coverage": coverage,
    }


def merge_exact(daily_sentiment: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    merged = daily_sentiment.merge(stock[["date", "close", "return_1d", "next_return_1d"]], on="date", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


def merge_next_trading_day(daily_sentiment: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    left = daily_sentiment.sort_values("date").reset_index(drop=True)
    right = stock[["date", "close", "return_1d", "next_return_1d"]].sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right,
        on="date",
        direction="forward",
        allow_exact_matches=False,
    )
    return merged.dropna(subset=["close"]).reset_index(drop=True)


def safe_name(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_date = pd.to_datetime(args.min_date)

    pred_files = sorted(glob.glob(args.pred_glob))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files matched: {args.pred_glob}")

    if args.stock_csv:
        stock, stock_meta = load_stock_prices(args.stock_csv, min_date=min_date)
        stock_source_label = f"csv:{args.stock_csv}"
    else:
        stock, stock_meta = load_stock_prices_yahoo(
            ticker=args.yahoo_ticker,
            min_date=min_date,
            yahoo_start=args.yahoo_start,
            yahoo_end=args.yahoo_end,
        )
        stock_source_label = f"yahoo:{args.yahoo_ticker}"
        yahoo_dump = out_dir / f"yahoo_{safe_name(args.yahoo_ticker)}_prices.csv"
        stock.to_csv(yahoo_dump, index=False)
        stock_meta["downloaded_stock_csv"] = str(yahoo_dump)

    all_metrics: list[dict[str, Any]] = []
    per_file_meta: list[dict[str, Any]] = []

    for pred_csv in pred_files:
        daily, meta = build_daily_sentiment(pred_csv, min_date=min_date)
        file_tag = safe_name(pred_csv)

        daily_out = out_dir / f"{file_tag}_daily_sentiment.csv"
        daily.to_csv(daily_out, index=False)

        merged_exact = merge_exact(daily, stock)
        merged_exact_out = out_dir / f"{file_tag}_merged_exact.csv"
        merged_exact.to_csv(merged_exact_out, index=False)

        merged_next = merge_next_trading_day(daily, stock)
        merged_next_out = out_dir / f"{file_tag}_merged_next_trading.csv"
        merged_next.to_csv(merged_next_out, index=False)

        exact_same_day = evaluate_signal(merged_exact["sentiment_score_mean"], merged_exact["return_1d"])
        exact_next_day = evaluate_signal(merged_exact["sentiment_score_mean"], merged_exact["next_return_1d"])
        next_trade = evaluate_signal(merged_next["sentiment_score_mean"], merged_next["return_1d"])

        all_metrics.append(
            {
                "prediction_file": pred_csv,
                "mode": "exact_same_day",
                **exact_same_day,
            }
        )
        all_metrics.append(
            {
                "prediction_file": pred_csv,
                "mode": "exact_next_day",
                **exact_next_day,
            }
        )
        all_metrics.append(
            {
                "prediction_file": pred_csv,
                "mode": "next_trading_day",
                **next_trade,
            }
        )

        meta.update(
            {
                "daily_output_csv": str(daily_out),
                "merged_exact_output_csv": str(merged_exact_out),
                "merged_next_output_csv": str(merged_next_out),
                "rows_merged_exact": int(len(merged_exact)),
                "rows_merged_next": int(len(merged_next)),
            }
        )
        per_file_meta.append(meta)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = out_dir / "experiment2_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "stock_meta": stock_meta,
        "prediction_files": pred_files,
        "per_file_meta": per_file_meta,
        "metrics_csv": str(metrics_csv),
    }
    summary_json = out_dir / "experiment2_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_txt = out_dir / "experiment2_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 2: Sentiment vs Stock Movement\n")
        f.write(f"Stock source: {stock_source_label}\n")
        f.write(f"Prediction files: {len(pred_files)}\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved text summary: {summary_txt}")


if __name__ == "__main__":
    main()
