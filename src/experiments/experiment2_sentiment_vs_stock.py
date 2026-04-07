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
DEFAULT_LAGS = (0, 1, 2)


def _normalize_col_name(col: Any) -> str:
    if isinstance(col, tuple):
        parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
        return " ".join(parts).strip()
    return str(col).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_names = [_normalize_col_name(c) for c in out.columns]

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


def _split_repeated_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                out.append(item)
    return out


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def safe_name(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_").replace("^", "").replace("/", "_").replace("\\", "_")


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    normalized = {_normalize_col_name(c).lower(): c for c in df.columns}

    for cand in candidates:
        col = normalized.get(cand.lower())
        if col is not None:
            return col

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
        if str(col).endswith("_label"):
            return str(col)
    for col in ("label", "sentiment"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer sentiment label column. Available columns: {list(df.columns)}")


def detect_prob_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pos = None
    neg = None
    for col in df.columns:
        low = str(col).lower()
        if "prob_positive" in low:
            pos = str(col)
        elif "prob_negative" in low:
            neg = str(col)
    return pos, neg


def label_to_score(label_series: pd.Series) -> pd.Series:
    raw = label_series.astype(str).str.strip().str.lower()
    mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    score = raw.map(mapping)
    return score.fillna(0.0).astype(float)


def parse_lags(raw: str) -> list[int]:
    if not raw.strip():
        return list(DEFAULT_LAGS)
    lags = sorted({int(part.strip()) for part in raw.split(",") if part.strip() != ""})
    if any(lag < 0 for lag in lags):
        raise ValueError("Lags must be non-negative.")
    return lags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: compare aggregated daily sentiment signals against stock movement."
    )
    parser.add_argument(
        "--pred-glob",
        default="results/predictions/*preds.csv",
        help="Glob for prediction CSV files. Ignored if --pred-csv is provided.",
    )
    parser.add_argument(
        "--pred-csv",
        action="append",
        default=[],
        help="Specific prediction CSV to evaluate. Repeat or pass comma-separated paths.",
    )
    parser.add_argument(
        "--stock-csv",
        action="append",
        default=[],
        help="CSV file with daily stock/index prices. Repeat or pass comma-separated paths.",
    )
    parser.add_argument(
        "--yahoo-ticker",
        action="append",
        default=[],
        help="Yahoo Finance ticker such as ^KLSE, 1155.KL, 5347.KL. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--yahoo-start",
        default=None,
        help="Yahoo Finance start date YYYY-MM-DD. Defaults to --min-date.",
    )
    parser.add_argument(
        "--yahoo-end",
        default=None,
        help="Yahoo Finance end date YYYY-MM-DD (exclusive). Defaults to one day after --max-date, or tomorrow.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/experiment2",
        help="Output folder.",
    )
    parser.add_argument(
        "--min-date",
        default="2023-01-01",
        help="Filter sentiment and stock data to this date or later.",
    )
    parser.add_argument(
        "--max-date",
        default=None,
        help="Optional inclusive end date filter YYYY-MM-DD.",
    )
    parser.add_argument(
        "--lags",
        default="0,1,2",
        help="Comma-separated return lags to evaluate. Example: 0,1,2,3",
    )
    parser.add_argument(
        "--alignment",
        choices=("same_or_next_trading", "same_day", "next_trading_day"),
        default="same_or_next_trading",
        help="How article dates are aligned to trading dates.",
    )
    parser.add_argument(
        "--min-articles-per-market-day",
        type=int,
        default=1,
        help="Drop aligned market days with fewer than this many articles.",
    )
    parser.add_argument(
        "--granger-maxlag",
        type=int,
        default=3,
        help="Maximum lag to use for Granger causality tests.",
    )
    parser.add_argument(
        "--skip-granger",
        action="store_true",
        help="Skip Granger causality tests even if statsmodels is installed.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    args.pred_csv = _unique_keep_order(_split_repeated_args(args.pred_csv))
    args.stock_csv = _unique_keep_order(_split_repeated_args(args.stock_csv))
    args.yahoo_ticker = _unique_keep_order(_split_repeated_args(args.yahoo_ticker))
    args.lags = parse_lags(args.lags)

    if not args.stock_csv and not args.yahoo_ticker:
        parser.error("Provide at least one --stock-csv or --yahoo-ticker.")
    return args


def resolve_prediction_files(args: argparse.Namespace) -> list[str]:
    if args.pred_csv:
        pred_files = [str(Path(p)) for p in args.pred_csv]
    else:
        pred_files = sorted(glob.glob(args.pred_glob))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files matched. pred_csv={args.pred_csv}, pred_glob={args.pred_glob}")
    return pred_files


def load_prediction_rows(
    pred_csv: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(pred_csv)
    date_col = infer_column(df, DATE_CANDIDATES, "prediction date")
    label_col = detect_label_column(df)
    pos_prob_col, neg_prob_col = detect_prob_columns(df)

    rows = df.copy()
    rows["calendar_date"] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["calendar_date"]).copy()
    rows = rows.loc[rows["calendar_date"] >= min_date].copy()
    if max_date is not None:
        rows = rows.loc[rows["calendar_date"] <= max_date].copy()

    if pos_prob_col and neg_prob_col:
        rows["sentiment_score"] = (
            pd.to_numeric(rows[pos_prob_col], errors="coerce").fillna(0.0)
            - pd.to_numeric(rows[neg_prob_col], errors="coerce").fillna(0.0)
        )
        score_mode = "prob_positive_minus_negative"
    else:
        rows["sentiment_score"] = label_to_score(rows[label_col])
        score_mode = "label_mapped_to_-1_0_1"

    rows["label_norm"] = rows[label_col].astype(str).str.strip().str.lower()
    rows["is_positive"] = (rows["label_norm"] == "positive").astype(int)
    rows["is_negative"] = (rows["label_norm"] == "negative").astype(int)
    rows["is_neutral"] = (rows["label_norm"] == "neutral").astype(int)

    kept = rows[
        ["calendar_date", "sentiment_score", "label_norm", "is_positive", "is_negative", "is_neutral"]
    ].copy()

    meta = {
        "prediction_file": pred_csv,
        "rows_input": int(len(df)),
        "rows_after_date_filter": int(len(kept)),
        "date_col": date_col,
        "label_col": label_col,
        "score_mode": score_mode,
        "pos_prob_col": pos_prob_col,
        "neg_prob_col": neg_prob_col,
        "date_min": str(kept["calendar_date"].min().date()) if len(kept) else None,
        "date_max": str(kept["calendar_date"].max().date()) if len(kept) else None,
    }
    return kept, meta


def aggregate_sentiment(rows: pd.DataFrame, date_col: str, out_date_col: str) -> pd.DataFrame:
    daily = (
        rows.groupby(date_col, as_index=False)
        .agg(
            article_count=("sentiment_score", "size"),
            sentiment_score_mean=("sentiment_score", "mean"),
            sentiment_score_std=("sentiment_score", "std"),
            positive_ratio=("is_positive", "mean"),
            negative_ratio=("is_negative", "mean"),
            neutral_ratio=("is_neutral", "mean"),
            source_day_count=("calendar_date", "nunique"),
            first_calendar_date=("calendar_date", "min"),
            last_calendar_date=("calendar_date", "max"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    daily = daily.rename(columns={date_col: out_date_col})
    daily["sentiment_score_std"] = daily["sentiment_score_std"].fillna(0.0)
    return daily


def load_stock_prices(
    stock_csv: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(stock_csv)
    raw = _flatten_columns(raw)
    date_col = infer_column(raw, DATE_CANDIDATES, "stock date")
    close_col = infer_column(raw, CLOSE_CANDIDATES, "stock close price")

    stock = raw.copy()
    stock["date"] = pd.to_datetime(stock[date_col], errors="coerce").dt.normalize()
    stock["close"] = pd.to_numeric(stock[close_col], errors="coerce")
    stock = stock.dropna(subset=["date", "close"]).copy()
    stock = stock.loc[stock["date"] >= min_date].copy()
    if max_date is not None:
        stock = stock.loc[stock["date"] <= max_date].copy()
    stock = stock.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    meta = {
        "stock_source": "csv",
        "stock_file": stock_csv,
        "source_label": Path(stock_csv).stem,
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
    max_date: pd.Timestamp | None,
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
    elif max_date is not None:
        end = (max_date + pd.Timedelta(days=1)).date().isoformat()
    else:
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"No Yahoo Finance data returned for ticker={ticker}, start={start}, end={end}.")

    raw = raw.reset_index()
    raw = _flatten_columns(raw)
    date_col = infer_column(raw, DATE_CANDIDATES, "stock date")
    close_col = infer_column(raw, CLOSE_CANDIDATES, "stock close price")

    stock = raw.copy()
    stock["date"] = pd.to_datetime(stock[date_col], errors="coerce").dt.normalize()
    stock["close"] = pd.to_numeric(stock[close_col], errors="coerce")
    stock = stock.dropna(subset=["date", "close"]).copy()
    stock = stock.loc[stock["date"] >= min_date].copy()
    if max_date is not None:
        stock = stock.loc[stock["date"] <= max_date].copy()
    stock = stock.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    meta = {
        "stock_source": "yahoo_finance",
        "ticker": ticker,
        "source_label": ticker,
        "query_start": start,
        "query_end_exclusive": end,
        "rows_after_clean": int(len(stock)),
        "date_col": date_col,
        "close_col": close_col,
        "date_min": str(stock["date"].min().date()) if len(stock) else None,
        "date_max": str(stock["date"].max().date()) if len(stock) else None,
    }
    return stock, meta


def add_return_columns(stock: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = stock.copy()
    out["return_tplus0_1d"] = out["close"].pct_change()
    for lag in lags:
        if lag == 0:
            continue
        out[f"return_tplus{lag}_1d"] = out["return_tplus0_1d"].shift(-lag)
    return out


def align_rows_to_market_dates(rows: pd.DataFrame, stock: pd.DataFrame, alignment: str) -> pd.DataFrame:
    market_calendar = stock[["date"]].drop_duplicates().sort_values("date").rename(columns={"date": "market_date"})
    left = rows.sort_values("calendar_date").reset_index(drop=True)

    if alignment == "same_day":
        merged = left.merge(market_calendar, left_on="calendar_date", right_on="market_date", how="inner")
        return merged.reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        market_calendar,
        left_on="calendar_date",
        right_on="market_date",
        direction="forward",
        allow_exact_matches=(alignment == "same_or_next_trading"),
    )
    return merged.dropna(subset=["market_date"]).reset_index(drop=True)


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


def evaluate_lags(merged: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag in lags:
        ret_col = f"return_tplus{lag}_1d"
        if ret_col not in merged.columns:
            continue
        metrics = evaluate_signal(merged["sentiment_score_mean"], merged[ret_col])
        rows.append({"lag": lag, "return_col": ret_col, **metrics})
    return pd.DataFrame(rows)


def run_granger_tests(
    merged: pd.DataFrame,
    maxlag: int,
    sentiment_col: str = "sentiment_score_mean",
    return_col: str = "return_tplus0_1d",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return (
            pd.DataFrame(),
            {
                "status": "skipped",
                "reason": "statsmodels_not_installed",
                "hint": "Install statsmodels to enable Granger causality tests.",
            },
        )

    df = merged[[sentiment_col, return_col]].dropna().copy()
    df = df.rename(columns={sentiment_col: "score", return_col: "ret"})
    if len(df) < max(30, maxlag * 8):
        return (
            pd.DataFrame(),
            {
                "status": "skipped",
                "reason": "insufficient_rows",
                "n_rows": int(len(df)),
                "required_min_rows": int(max(30, maxlag * 8)),
            },
        )

    rows: list[dict[str, Any]] = []
    try:
        forward = grangercausalitytests(df[["ret", "score"]], maxlag=maxlag, verbose=False)
        reverse = grangercausalitytests(df[["score", "ret"]], maxlag=maxlag, verbose=False)
    except Exception as exc:
        return pd.DataFrame(), {"status": "skipped", "reason": "granger_failed", "error": str(exc)}

    for lag, result in forward.items():
        test_stats = result[0]["ssr_ftest"]
        rows.append(
            {
                "direction": "score_causes_return",
                "lag": int(lag),
                "f_stat": float(test_stats[0]),
                "p_value": float(test_stats[1]),
                "df_denom": float(test_stats[2]),
                "df_num": float(test_stats[3]),
            }
        )
    for lag, result in reverse.items():
        test_stats = result[0]["ssr_ftest"]
        rows.append(
            {
                "direction": "return_causes_score",
                "lag": int(lag),
                "f_stat": float(test_stats[0]),
                "p_value": float(test_stats[1]),
                "df_denom": float(test_stats[2]),
                "df_num": float(test_stats[3]),
            }
        )

    return pd.DataFrame(rows), {"status": "ok", "n_rows": int(len(df)), "maxlag": int(maxlag)}


def make_plots(
    merged: pd.DataFrame,
    lag_df: pd.DataFrame,
    out_prefix: Path,
) -> dict[str, str] | dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "skipped", "reason": "matplotlib_not_installed"}

    if merged.empty:
        return {"status": "skipped", "reason": "empty_merged_dataframe"}

    plot_paths: dict[str, str] = {}
    plot_df = merged.sort_values("market_date").copy()

    overview_png = out_prefix.with_name(out_prefix.name + "_sentiment_vs_returns.png")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)

    axes[0].plot(plot_df["market_date"], plot_df["sentiment_score_mean"], color="tab:blue", label="Sentiment score")
    axes[0].set_title("Daily Market-Aligned Sentiment")
    axes[0].set_ylabel("Sentiment")
    ax0b = axes[0].twinx()
    ax0b.bar(
        plot_df["market_date"],
        plot_df["article_count"],
        width=1.0,
        alpha=0.25,
        color="tab:gray",
        label="Article count",
    )
    ax0b.set_ylabel("Articles")

    axes[1].bar(plot_df["market_date"], plot_df["return_tplus0_1d"], color="tab:green", alpha=0.7)
    axes[1].set_title("Aligned Same-Day Returns")
    axes[1].set_ylabel("Return")

    normalized_close = plot_df["close"] / plot_df["close"].iloc[0]
    axes[2].plot(plot_df["market_date"], normalized_close, color="tab:orange")
    axes[2].set_title("Normalized Close Price")
    axes[2].set_ylabel("Close / first close")
    axes[2].set_xlabel("Market date")

    fig.savefig(overview_png, dpi=160)
    plt.close(fig)
    plot_paths["sentiment_vs_returns_png"] = str(overview_png)

    if not lag_df.empty:
        heatmap_png = out_prefix.with_name(out_prefix.name + "_lag_heatmap.png")
        heatmap_cols = ["pearson_corr", "spearman_corr", "directional_accuracy"]
        heatmap_df = lag_df.set_index("lag")[heatmap_cols].astype(float)

        fig, ax = plt.subplots(figsize=(8, max(3, 1.25 * len(heatmap_df))))
        im = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Lagged Sentiment / Return Metrics")
        ax.set_xticks(range(len(heatmap_cols)))
        ax.set_xticklabels(heatmap_cols, rotation=20, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels([f"T+{int(idx)}" for idx in heatmap_df.index])

        for row_idx in range(heatmap_df.shape[0]):
            for col_idx in range(heatmap_df.shape[1]):
                value = heatmap_df.iat[row_idx, col_idx]
                label = "nan" if pd.isna(value) else f"{value:.3f}"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9, color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(heatmap_png, dpi=160)
        plt.close(fig)
        plot_paths["lag_heatmap_png"] = str(heatmap_png)

    plot_paths["status"] = "ok"
    return plot_paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_date = pd.to_datetime(args.min_date).normalize()
    max_date = pd.to_datetime(args.max_date).normalize() if args.max_date else None
    pred_files = resolve_prediction_files(args)

    stock_inputs: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    for stock_csv in args.stock_csv:
        stock_df, stock_meta = load_stock_prices(stock_csv, min_date=min_date, max_date=max_date)
        stock_inputs.append((stock_df, stock_meta))
    for ticker in args.yahoo_ticker:
        stock_df, stock_meta = load_stock_prices_yahoo(
            ticker=ticker,
            min_date=min_date,
            max_date=max_date,
            yahoo_start=args.yahoo_start,
            yahoo_end=args.yahoo_end,
        )
        dump_path = out_dir / f"yahoo_{safe_name(ticker)}_prices.csv"
        stock_df.to_csv(dump_path, index=False)
        stock_meta["downloaded_stock_csv"] = str(dump_path)
        stock_inputs.append((stock_df, stock_meta))

    all_metrics: list[dict[str, Any]] = []
    combo_meta: list[dict[str, Any]] = []

    for pred_csv in pred_files:
        pred_rows, pred_meta = load_prediction_rows(pred_csv, min_date=min_date, max_date=max_date)
        pred_tag = safe_name(pred_csv)

        calendar_daily = aggregate_sentiment(pred_rows, "calendar_date", "calendar_date")
        calendar_daily_csv = out_dir / f"{pred_tag}_calendar_daily_sentiment.csv"
        calendar_daily.to_csv(calendar_daily_csv, index=False)

        for stock_df, stock_meta in stock_inputs:
            stock_label = safe_name(stock_meta["source_label"])
            combo_tag = f"{pred_tag}__{stock_label}"

            enriched_stock = add_return_columns(stock_df, args.lags)
            aligned_rows = align_rows_to_market_dates(pred_rows, enriched_stock, alignment=args.alignment)
            market_daily = aggregate_sentiment(aligned_rows, "market_date", "market_date")
            market_daily = market_daily.loc[
                market_daily["article_count"] >= args.min_articles_per_market_day
            ].reset_index(drop=True)

            merged = market_daily.merge(enriched_stock, left_on="market_date", right_on="date", how="inner")
            merged = merged.sort_values("market_date").reset_index(drop=True)

            market_daily_csv = out_dir / f"{combo_tag}_market_daily_sentiment.csv"
            merged_csv = out_dir / f"{combo_tag}_merged_market.csv"
            market_daily.to_csv(market_daily_csv, index=False)
            merged.to_csv(merged_csv, index=False)

            lag_df = evaluate_lags(merged, args.lags)
            lag_csv = out_dir / f"{combo_tag}_lag_metrics.csv"
            lag_df.to_csv(lag_csv, index=False)

            granger_df: pd.DataFrame
            if args.skip_granger:
                granger_df = pd.DataFrame()
                granger_meta = {"status": "skipped", "reason": "skip_granger_flag"}
            else:
                granger_df, granger_meta = run_granger_tests(merged, maxlag=args.granger_maxlag)
            granger_csv = out_dir / f"{combo_tag}_granger_metrics.csv"
            granger_df.to_csv(granger_csv, index=False)

            if args.skip_plots:
                plot_meta: dict[str, Any] = {"status": "skipped", "reason": "skip_plots_flag"}
            else:
                plot_meta = make_plots(merged, lag_df, out_dir / combo_tag)

            for row in lag_df.to_dict(orient="records"):
                all_metrics.append(
                    {
                        "prediction_file": pred_csv,
                        "stock_source": stock_meta["source_label"],
                        "alignment": args.alignment,
                        "min_date": str(min_date.date()),
                        "max_date": str(max_date.date()) if max_date is not None else None,
                        "min_articles_per_market_day": args.min_articles_per_market_day,
                        **row,
                    }
                )

            combo_meta.append(
                {
                    "prediction_file": pred_csv,
                    "prediction_meta": pred_meta,
                    "stock_meta": stock_meta,
                    "alignment": args.alignment,
                    "lags": args.lags,
                    "calendar_daily_csv": str(calendar_daily_csv),
                    "market_daily_csv": str(market_daily_csv),
                    "merged_market_csv": str(merged_csv),
                    "lag_metrics_csv": str(lag_csv),
                    "granger_metrics_csv": str(granger_csv),
                    "plot_meta": plot_meta,
                    "granger_meta": granger_meta,
                    "rows_aligned_articles": int(len(aligned_rows)),
                    "rows_market_daily": int(len(market_daily)),
                    "rows_merged_market": int(len(merged)),
                }
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = out_dir / "experiment2_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "pred_files": pred_files,
        "stock_inputs": [meta for _, meta in stock_inputs],
        "alignment": args.alignment,
        "lags": args.lags,
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()) if max_date is not None else None,
        "min_articles_per_market_day": args.min_articles_per_market_day,
        "combo_meta": combo_meta,
        "metrics_csv": str(metrics_csv),
    }
    summary_json = out_dir / "experiment2_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_txt = out_dir / "experiment2_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 2: Sentiment vs Stock Movement\n")
        f.write(f"Prediction files: {len(pred_files)}\n")
        f.write(f"Stock inputs: {len(stock_inputs)}\n")
        f.write(f"Alignment: {args.alignment}\n")
        f.write(f"Lags: {args.lags}\n\n")
        if metrics_df.empty:
            f.write("No lag metrics were produced.\n")
        else:
            f.write(metrics_df.to_string(index=False))
            f.write("\n")

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved text summary: {summary_txt}")


if __name__ == "__main__":
    main()
