from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}

MOJIBAKE_REPLACEMENTS = {
    "Ã¢â‚¬â„¢": "'",
    "Ã¢â‚¬Ëœ": "'",
    "Ã¢â‚¬Å“": '"',
    "Ã¢â‚¬\x9d": '"',
    "Ã¢â‚¬â€œ": "-",
    "Ã¢â‚¬â€\x9d": "-",
    "Ã¢â‚¬Â¦": "...",
    "Ã‚ ": " ",
    "\xa0": " ",
}

BOILERPLATE_INLINE_PATTERNS = (
    re.compile(r"stay current\s*-\s*follow fmt[^.]*\.?", re.I),
    re.compile(r"follow us on[^.]*\.?", re.I),
    re.compile(r"subscribe to our newsletter[^.]*\.?", re.I),
    re.compile(r"registration on or use of this site[^.]*\.?", re.I),
    re.compile(r"terms of service[^.]*\.?", re.I),
    re.compile(r"privacy policy[^.]*\.?", re.I),
    re.compile(r"cookies policy[^.]*\.?", re.I),
    re.compile(r"copyright\s*[^\n.]*\.?", re.I),
    re.compile(r"all rights reserved\.?", re.I),
    re.compile(r"\bread also\b[^.]*\.?", re.I),
    re.compile(r"\brelated\b[^.]*\.?", re.I),
)

DATELINE_PREFIX_PATTERNS = (
    # Examples: TOKYO:  |  TOKYO :
    re.compile(r"^\s*[A-Z][A-Z .,&'/-]{2,}(?:\s+[A-Z][A-Z .,&'/-]{2,}){0,5}\s*:\s+"),
    # Example: KUALA LUMPUR (31 MARCH):
    re.compile(r"^\s*[A-Z][A-Z .,&'/-]{2,}(?:\s+[A-Z][A-Z .,&'/-]{2,}){0,5}\s*\(\s*\d{1,2}\s+[A-Z]{3,12}\s*\)\s*:\s+"),
    # Example: KUALA LUMPUR (Jan 31): / PUTRAJAYA (March 2, 2026):
    re.compile(
        r"^\s*[A-Z][A-Z .,&'/-]{2,}(?:\s+[A-Z][A-Z .,&'/-]{2,}){0,5}\s*"
        r"\(\s*[A-Za-z]{3,12}\s+\d{1,2}(?:,\s*\d{4})?\s*\)\s*:\s+"
    ),
    # Reuters wires: TOKYO (Reuters) -  | LONDON, March 31 (Reuters) -
    re.compile(
        r"^\s*[A-Z][A-Z .,&'/-]{2,}(?:\s+[A-Z][A-Z .,&'/-]{2,}){0,5}"
        r"(?:,\s*[A-Za-z]{3,12}\s+\d{1,2})?\s*\(Reuters\)\s*[-:]\s+",
        re.I,
    ),
)

COLUMN_CANDIDATES = {
    "source": ("source", "publisher", "site", "news_source"),
    "category": ("category", "section", "topic"),
    "url": ("url", "link", "article_url"),
    "title": ("title", "headline"),
    "date": ("date", "published_date", "published_at", "datetime"),
    "body": ("body", "content", "article", "text"),
}

SENTIMENT_COLUMN_CANDIDATES = (
    "sentiment",
    "sentiment_label",
    "label",
    "finbert_label",
    "fingpt_label",
    "finbert_zeroshot_label",
    "fingpt_zeroshot_label",
)

SENTIMENT_BUCKETS = ("negative", "neutral", "positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine and preprocess multiple news CSV files into one cleaned dataset."
    )
    parser.add_argument(
        "--input-glob",
        default="results/data/*since_2023*.csv",
        help="Glob pattern for input CSV files (default includes news_scraper + The Star since_2023 files).",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="",
        help="Optional Kaggle dataset slug (for example: anasamri/news-from-different-sources).",
    )
    parser.add_argument(
        "--kaggle-file-glob",
        default="*.csv",
        help="Glob used inside downloaded Kaggle dataset folder (default: *.csv, searched recursively).",
    )
    parser.add_argument(
        "--kaggle-local-dir",
        default="",
        help="Use an already-downloaded Kaggle dataset directory instead of downloading again.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/data/news_combined_preprocessed.csv",
        help="Output CSV path for the full cleaned dataset.",
    )
    parser.add_argument(
        "--output-csv-2023-2025",
        default="results/data/news_combined_preprocessed_2023_2025.csv",
        help="Output CSV path for the cleaned 2023-2025 subset.",
    )
    parser.add_argument(
        "--output-csv-2026",
        default="results/data/news_combined_preprocessed_2026.csv",
        help="Output CSV path for the cleaned 2026 subset.",
    )
    parser.add_argument(
        "--sample-output-csv",
        default="results/data/news_human_label_sample_300.csv",
        help="Output CSV path for the separate human-label sample file.",
    )
    parser.add_argument(
        "--sample-output-csv-2025",
        default="results/data/news_human_label_sample_2025_100.csv",
        help="Output CSV path for the 2025-only human-label sample file.",
    )
    parser.add_argument(
        "--sample-output-csv-2026",
        default="results/data/news_human_label_sample_2026_200.csv",
        help="Output CSV path for the 2026-only human-label sample file.",
    )
    parser.add_argument(
        "--sample-2025",
        type=int,
        default=100,
        help="Number of sample rows to pick from 2025.",
    )
    parser.add_argument(
        "--sample-2026",
        type=int,
        default=200,
        help="Number of sample rows to pick from 2026.",
    )
    parser.add_argument(
        "--sample-random-seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sample selection.",
    )
    parser.add_argument(
        "--sample-sentiment-column",
        default="",
        help=(
            "Optional sentiment label column to use for sentiment-balanced sampling "
            "(for example: finbert_label). If omitted, auto-detect is attempted."
        ),
    )
    parser.add_argument(
        "--report-json",
        default="results/data/news_combined_preprocess_report.json",
        help="Output JSON summary/report path.",
    )
    parser.add_argument(
        "--min-date",
        default="2023-01-01",
        help="Keep rows with date >= this value (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=80,
        help="Drop rows where cleaned body is shorter than this many characters.",
    )
    parser.add_argument(
        "--no-text-dedupe",
        action="store_true",
        help="Disable dedupe by normalized title+body fingerprint.",
    )
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> tuple[list[Path], dict[str, str]]:
    if args.kaggle_local_dir:
        dataset_dir = Path(args.kaggle_local_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"--kaggle-local-dir not found: {dataset_dir}")
        files = sorted(dataset_dir.rglob(args.kaggle_file_glob))
        if not files:
            raise FileNotFoundError(
                f"No files matched '{args.kaggle_file_glob}' under --kaggle-local-dir: {dataset_dir}"
            )
        return files, {"input_mode": "kaggle_local_dir", "dataset_dir": str(dataset_dir)}

    if args.kaggle_dataset:
        try:
            import kagglehub  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for --kaggle-dataset. Install with: pip install kagglehub"
            ) from exc
        dataset_dir = Path(kagglehub.dataset_download(args.kaggle_dataset))
        files = sorted(dataset_dir.rglob(args.kaggle_file_glob))
        if not files:
            raise FileNotFoundError(
                f"No files matched '{args.kaggle_file_glob}' in Kaggle dataset folder: {dataset_dir}"
            )
        return files, {
            "input_mode": "kaggle_dataset",
            "kaggle_dataset": args.kaggle_dataset,
            "dataset_dir": str(dataset_dir),
        }

    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No input files matched pattern: {args.input_glob}")
    return files, {"input_mode": "local_glob", "input_glob": args.input_glob}


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_to_real = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in lower_to_real:
            return lower_to_real[cand]
    return None


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_leading_dateline(text: str) -> str:
    cleaned = text.lstrip()
    for _ in range(3):
        prefix = cleaned[:180]
        previous = cleaned
        for pattern in DATELINE_PREFIX_PATTERNS:
            match = pattern.match(prefix)
            if match:
                cleaned = cleaned[match.end() :].lstrip()
                break
        if cleaned == previous:
            break
    return cleaned


def clean_article_body_text(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = strip_leading_dateline(text)
    if not text:
        return ""
    cleaned = text
    for pattern in BOILERPLATE_INLINE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def canonicalize_url(value: object) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    raw = raw.split("#", 1)[0]
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw

    if not parts.scheme or not parts.netloc:
        return raw

    query_items = [
        (k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in TRACKING_QUERY_KEYS
    ]
    clean_query = urlencode(query_items, doseq=True)
    clean_path = re.sub(r"/+", "/", parts.path).rstrip("/")
    rebuilt = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), clean_path, clean_query, ""))
    return rebuilt


def load_and_standardize(csv_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, low_memory=False)
    original_rows = len(df)
    if original_rows == 0:
        return pd.DataFrame(columns=["source", "category", "url", "title", "date", "body", "source_file"]), {
            "input_rows": 0,
            "output_rows": 0,
        }

    selected: dict[str, str | None] = {
        key: pick_column(df, candidates) for key, candidates in COLUMN_CANDIDATES.items()
    }

    out = pd.DataFrame()
    out["source"] = (
        df[selected["source"]].astype(str).str.strip() if selected["source"] else csv_path.stem.replace(" ", "_")
    )
    out["category"] = df[selected["category"]].astype(str).str.strip() if selected["category"] else ""
    out["url"] = df[selected["url"]].astype(str).str.strip() if selected["url"] else ""
    out["title"] = df[selected["title"]].astype(str).str.strip() if selected["title"] else ""
    out["date"] = df[selected["date"]].astype(str).str.strip() if selected["date"] else ""
    out["body"] = df[selected["body"]].astype(str) if selected["body"] else ""
    out["source_file"] = csv_path.name

    # Preserve sentiment-like columns (if present) for downstream balanced sampling.
    lower_to_real = {col.lower(): col for col in df.columns}
    selected_source_cols = {col for col in selected.values() if col}
    sentiment_source_cols: list[str] = []
    for candidate in SENTIMENT_COLUMN_CANDIDATES:
        real = lower_to_real.get(candidate.lower())
        if real and real not in selected_source_cols and real not in sentiment_source_cols:
            sentiment_source_cols.append(real)
    for col in df.columns:
        low = col.lower()
        is_sentiment_like = ("sentiment" in low) or ("label" in low and ("finbert" in low or "fingpt" in low))
        if is_sentiment_like and col not in selected_source_cols and col not in sentiment_source_cols:
            sentiment_source_cols.append(col)
    for col in sentiment_source_cols:
        out[col] = df[col].astype(str).str.strip()

    return out, {"input_rows": original_rows, "output_rows": len(out)}


def preprocess_all(df: pd.DataFrame, min_date: date, min_body_chars: int, dedupe_text: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    report: dict[str, int] = {"rows_loaded": len(df)}

    # Base cleaning
    for col in ("source", "category", "title"):
        df[col] = df[col].map(clean_text)
    df["body"] = df["body"].map(clean_article_body_text)
    df["url"] = df["url"].map(canonicalize_url)
    df["source"] = df["source"].replace("", "unknown_source")

    # Date parsing/filtering
    dt = pd.to_datetime(df["date"], errors="coerce")
    report["rows_with_valid_date"] = int(dt.notna().sum())
    df = df.loc[dt.notna()].copy()
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["_date_dt"] = dt.dt.date
    df = df.loc[df["_date_dt"] >= min_date].copy()
    df["date"] = pd.to_datetime(df["_date_dt"]).dt.strftime("%Y-%m-%d")
    report["rows_after_min_date"] = len(df)

    # Required text fields
    df = df.loc[df["title"].str.len() > 0].copy()
    df = df.loc[df["body"].str.len() >= min_body_chars].copy()
    report["rows_after_text_filters"] = len(df)

    # Deduplication: url first
    before_url_dedupe = len(df)
    df_nonempty_url = df["url"].str.len() > 0
    deduped_url = pd.concat(
        [
            df.loc[df_nonempty_url].drop_duplicates(subset=["url"], keep="first"),
            df.loc[~df_nonempty_url],
        ],
        ignore_index=True,
    )
    df = deduped_url
    report["rows_dropped_duplicate_url"] = before_url_dedupe - len(df)

    # Deduplication: same title+body text fingerprint
    if dedupe_text:
        before_text_dedupe = len(df)
        combo = (df["title"].str.lower() + "\n" + df["body"].str.lower()).map(
            lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()
        )
        df["_text_hash"] = combo
        df = df.drop_duplicates(subset=["_text_hash"], keep="first").copy()
        report["rows_dropped_duplicate_text"] = before_text_dedupe - len(df)
    else:
        report["rows_dropped_duplicate_text"] = 0

    df["text"] = (df["title"] + ". " + df["body"]).str.strip()
    df["body_chars"] = df["body"].str.len()

    df = df.sort_values(["date", "source"], ascending=[True, True]).reset_index(drop=True)
    df = df.drop(columns=["_date_dt", "_text_hash"], errors="ignore")
    base_cols = ["source", "category", "url", "title", "date", "body", "text", "body_chars", "source_file"]
    extra_cols = [col for col in df.columns if col not in base_cols]
    df = df[base_cols + extra_cols]
    report["rows_final"] = len(df)
    return df, report


def normalize_sentiment_label(value: object) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    if re.search(r"\bnegative\b|\bneg\b", text):
        return "negative"
    if re.search(r"\bneutral\b|\bneu\b", text):
        return "neutral"
    if re.search(r"\bpositive\b|\bpos\b", text):
        return "positive"
    return ""


def detect_sentiment_column(df: pd.DataFrame, preferred_column: str = "") -> str | None:
    if preferred_column:
        if preferred_column not in df.columns:
            raise ValueError(f"--sample-sentiment-column '{preferred_column}' was not found in combined data")
        return preferred_column

    lower_to_real = {col.lower(): col for col in df.columns}
    for candidate in SENTIMENT_COLUMN_CANDIDATES:
        real = lower_to_real.get(candidate.lower())
        if real is not None:
            return real
    return None


def allocate_even_quota(total: int, groups: list[str], capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    quotas = {group: 0 for group in groups}
    if total == 0:
        return quotas

    active = [group for group in groups if capacities.get(group, 0) > 0]
    remaining = total
    while remaining > 0 and active:
        exhausted: list[str] = []
        for group in active:
            if remaining == 0:
                break
            if quotas[group] < capacities[group]:
                quotas[group] += 1
                remaining -= 1
            if quotas[group] >= capacities[group]:
                exhausted.append(group)
        active = [group for group in active if group not in exhausted]

    if remaining > 0:
        raise ValueError(
            f"Unable to allocate quota={total}; only {total - remaining} available across requested strata"
        )
    return quotas


def sample_month_balanced(df_pool: pd.DataFrame, target_n: int, random_seed: int) -> pd.DataFrame:
    if target_n <= 0:
        return df_pool.iloc[0:0].copy()
    if len(df_pool) < target_n:
        raise ValueError(f"Requested {target_n} rows but only {len(df_pool)} rows are available")

    work = df_pool.copy()
    work["_sample_month"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").astype(str)
    work = work.loc[work["_sample_month"] != "NaT"].copy()
    if len(work) < target_n:
        raise ValueError(f"Requested {target_n} rows but only {len(work)} rows have valid month information")

    months = sorted(work["_sample_month"].unique().tolist())
    month_caps = {month: int((work["_sample_month"] == month).sum()) for month in months}
    month_quota = allocate_even_quota(target_n, months, month_caps)

    sampled_parts: list[pd.DataFrame] = []
    for idx, month in enumerate(months):
        n_pick = month_quota.get(month, 0)
        if n_pick <= 0:
            continue
        part = work.loc[work["_sample_month"] == month].sample(n=n_pick, random_state=random_seed + idx)
        sampled_parts.append(part)

    out = pd.concat(sampled_parts, ignore_index=False)
    return out


def sample_year_rows(
    df_year: pd.DataFrame,
    target_n: int,
    random_seed: int,
    sentiment_column: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(df_year) < target_n:
        raise ValueError(f"Not enough rows for year sample: requested {target_n}, available {len(df_year)}")

    strategy_details: dict[str, object] = {"target_rows": target_n, "available_rows": len(df_year)}
    if not sentiment_column:
        chosen = sample_month_balanced(df_year, target_n, random_seed)
        strategy_details["strategy"] = "month_balanced_only"
        return chosen, strategy_details

    work = df_year.copy()
    work["_sample_sentiment"] = work[sentiment_column].map(normalize_sentiment_label)
    labeled = work.loc[work["_sample_sentiment"].isin(SENTIMENT_BUCKETS)].copy()

    if labeled.empty:
        chosen = sample_month_balanced(df_year, target_n, random_seed)
        strategy_details["strategy"] = "month_balanced_only_no_valid_sentiment"
        strategy_details["sentiment_column"] = sentiment_column
        strategy_details["sentiment_labeled_rows"] = 0
        return chosen, strategy_details

    sentiment_caps = {
        bucket: int((labeled["_sample_sentiment"] == bucket).sum()) for bucket in SENTIMENT_BUCKETS
    }
    sentiment_quota_total = min(target_n, len(labeled))
    sentiment_quota = allocate_even_quota(sentiment_quota_total, list(SENTIMENT_BUCKETS), sentiment_caps)

    picked_parts: list[pd.DataFrame] = []
    for idx, bucket in enumerate(SENTIMENT_BUCKETS):
        n_pick = sentiment_quota.get(bucket, 0)
        if n_pick <= 0:
            continue
        bucket_pool = labeled.loc[labeled["_sample_sentiment"] == bucket]
        part = sample_month_balanced(bucket_pool, n_pick, random_seed + (100 * (idx + 1)))
        picked_parts.append(part)
    chosen = pd.concat(picked_parts, ignore_index=False) if picked_parts else df_year.iloc[0:0].copy()

    remaining = target_n - len(chosen)
    if remaining > 0:
        remainder_pool = work.drop(index=chosen.index, errors="ignore")
        top_up = sample_month_balanced(remainder_pool, remaining, random_seed + 777)
        chosen = pd.concat([chosen, top_up], ignore_index=False)

    strategy_details["strategy"] = "sentiment_and_month_balanced"
    strategy_details["sentiment_column"] = sentiment_column
    strategy_details["sentiment_labeled_rows"] = len(labeled)
    strategy_details["sentiment_quota"] = sentiment_quota
    return chosen, strategy_details


def build_human_label_sample(
    cleaned_df: pd.DataFrame,
    sample_2025: int,
    sample_2026: int,
    random_seed: int,
    preferred_sentiment_column: str = "",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if sample_2025 < 0 or sample_2026 < 0:
        raise ValueError("sample sizes must be >= 0")

    with_year = cleaned_df.copy()
    with_year["_date_dt"] = pd.to_datetime(with_year["date"], errors="coerce")
    with_year = with_year.loc[with_year["_date_dt"].notna()].copy()
    with_year["_year"] = with_year["_date_dt"].dt.year

    sentiment_column = detect_sentiment_column(with_year, preferred_column=preferred_sentiment_column)
    if sentiment_column is not None:
        non_empty_count = int(with_year[sentiment_column].astype(str).str.strip().ne("").sum())
        if non_empty_count == 0:
            sentiment_column = None

    year_targets = {2025: sample_2025, 2026: sample_2026}
    year_parts: list[pd.DataFrame] = []
    year_debug: dict[str, object] = {}

    for year, target_n in year_targets.items():
        if target_n == 0:
            continue
        pool = with_year.loc[with_year["_year"] == year].copy()
        chosen, details = sample_year_rows(
            pool,
            target_n=target_n,
            random_seed=random_seed + year,
            sentiment_column=sentiment_column,
        )
        chosen["_sample_target_year"] = year
        year_parts.append(chosen)
        year_debug[str(year)] = details

    sampled = pd.concat(year_parts, ignore_index=False) if year_parts else with_year.iloc[0:0].copy()
    sampled = sampled.sample(frac=1.0, random_state=random_seed + 2026).reset_index(drop=True)
    sampled["_sample_month"] = sampled["_date_dt"].dt.to_period("M").astype(str)
    sampled = sampled.rename(
        columns={
            "_sample_target_year": "sample_target_year",
            "_sample_month": "sample_month",
            "_sample_sentiment": "sample_sentiment_bucket",
        }
    )

    drop_cols = ["_date_dt", "_year"]
    keep_cols = [col for col in sampled.columns if col not in drop_cols]
    sampled = sampled[keep_cols]

    sample_report = {
        "sample_rows": len(sampled),
        "sample_2025": sample_2025,
        "sample_2026": sample_2026,
        "sentiment_column_used": sentiment_column or "",
        "year_details": year_debug,
    }
    return sampled, sample_report


def to_date_text_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text" not in out.columns:
        title = out["title"] if "title" in out.columns else ""
        body = out["body"] if "body" in out.columns else ""
        out["text"] = (title.astype(str) + ". " + body.astype(str)).str.strip() if isinstance(title, pd.Series) else ""
    if "date" not in out.columns:
        out["date"] = ""
    out = out[["date", "text"]].copy()
    out["date"] = out["date"].astype(str).str.strip()
    out["text"] = out["text"].astype(str).str.strip()
    out = out.loc[out["date"].str.len() > 0].copy()
    out = out.loc[out["text"].str.len() > 0].copy()
    out = out.sort_values(["date", "text"], ascending=[True, True]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    if args.sample_2025 + args.sample_2026 <= 0:
        raise ValueError("sample size must be > 0 (sample_2025 + sample_2026)")

    min_date = date.fromisoformat(args.min_date)
    input_paths, input_info = resolve_input_paths(args)

    frames: list[pd.DataFrame] = []
    file_stats: dict[str, dict[str, int]] = {}
    generated_output_names = {
        Path(args.output_csv).name,
        Path(args.output_csv_2023_2025).name,
        Path(args.output_csv_2026).name,
        Path(args.sample_output_csv).name,
        Path(args.sample_output_csv_2025).name,
        Path(args.sample_output_csv_2026).name,
        Path(args.report_json).name,
    }
    for path in input_paths:
        if input_info.get("input_mode") == "local_glob":
            # Avoid recursively re-ingesting generated outputs only when reading local outputs dir.
            if path.name in generated_output_names:
                continue
            if path.stem.startswith("news_combined_preprocessed") or path.stem.startswith("news_human_label_sample"):
                continue
        frame, stats = load_and_standardize(path)
        frames.append(frame)
        file_stats[str(path)] = stats

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cleaned, report = preprocess_all(
        merged,
        min_date=min_date,
        min_body_chars=args.min_body_chars,
        dedupe_text=not args.no_text_dedupe,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_export = to_date_text_only(cleaned)
    cleaned_export.to_csv(output_csv, index=False, encoding="utf-8")

    clean_dates = pd.to_datetime(cleaned["date"], errors="coerce")
    clean_year = clean_dates.dt.year
    cleaned_2023_2025 = cleaned.loc[(clean_year >= 2023) & (clean_year <= 2025)].copy()
    cleaned_2026 = cleaned.loc[clean_year == 2026].copy()

    output_csv_2023_2025 = Path(args.output_csv_2023_2025)
    output_csv_2023_2025.parent.mkdir(parents=True, exist_ok=True)
    cleaned_2023_2025_export = to_date_text_only(cleaned_2023_2025)
    cleaned_2023_2025_export.to_csv(output_csv_2023_2025, index=False, encoding="utf-8")

    output_csv_2026 = Path(args.output_csv_2026)
    output_csv_2026.parent.mkdir(parents=True, exist_ok=True)
    cleaned_2026_export = to_date_text_only(cleaned_2026)
    cleaned_2026_export.to_csv(output_csv_2026, index=False, encoding="utf-8")

    sample_df, sample_report = build_human_label_sample(
        cleaned_df=cleaned,
        sample_2025=args.sample_2025,
        sample_2026=args.sample_2026,
        random_seed=args.sample_random_seed,
        preferred_sentiment_column=args.sample_sentiment_column,
    )
    sample_output_csv = Path(args.sample_output_csv)
    sample_output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_export = to_date_text_only(sample_df)
    sample_export.to_csv(sample_output_csv, index=False, encoding="utf-8")

    sample_years = (
        pd.to_datetime(sample_export["date"], errors="coerce").dt.year
        if not sample_export.empty
        else pd.Series(dtype="float64")
    )
    sample_df_2025 = sample_export.loc[sample_years == 2025].copy()
    sample_df_2026 = sample_export.loc[sample_years == 2026].copy()

    sample_output_csv_2025 = Path(args.sample_output_csv_2025)
    sample_output_csv_2025.parent.mkdir(parents=True, exist_ok=True)
    sample_df_2025.to_csv(sample_output_csv_2025, index=False, encoding="utf-8")

    sample_output_csv_2026 = Path(args.sample_output_csv_2026)
    sample_output_csv_2026.parent.mkdir(parents=True, exist_ok=True)
    sample_df_2026.to_csv(sample_output_csv_2026, index=False, encoding="utf-8")

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "inputs": [str(p) for p in input_paths],
        "input_info": input_info,
        "file_stats": file_stats,
        "preprocess_report": report,
        "output_csv_full": str(output_csv),
        "output_rows_full": len(cleaned_export),
        "output_csv_2023_2025": str(output_csv_2023_2025),
        "output_rows_2023_2025": len(cleaned_2023_2025_export),
        "output_csv_2026": str(output_csv_2026),
        "output_rows_2026": len(cleaned_2026_export),
        "sample_output_csv": str(sample_output_csv),
        "sample_output_csv_2025": str(sample_output_csv_2025),
        "sample_output_rows_2025": len(sample_df_2025),
        "sample_output_csv_2026": str(sample_output_csv_2026),
        "sample_output_rows_2026": len(sample_df_2026),
        "sample_report": sample_report,
        "unique_sources": int(cleaned["source"].nunique()) if not cleaned.empty else 0,
        "unique_categories": int(cleaned["category"].nunique()) if not cleaned.empty else 0,
    }
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved combined CSV (full): {output_csv} ({len(cleaned_export)} rows)")
    print(f"Saved combined CSV (2023-2025): {output_csv_2023_2025} ({len(cleaned_2023_2025_export)} rows)")
    print(f"Saved combined CSV (2026): {output_csv_2026} ({len(cleaned_2026_export)} rows)")
    print(f"Saved human-label sample CSV: {sample_output_csv} ({len(sample_export)} rows)")
    print(f"Saved human-label sample CSV (2025): {sample_output_csv_2025} ({len(sample_df_2025)} rows)")
    print(f"Saved human-label sample CSV (2026): {sample_output_csv_2026} ({len(sample_df_2026)} rows)")
    print(f"Saved preprocess report: {report_json}")


if __name__ == "__main__":
    main()
