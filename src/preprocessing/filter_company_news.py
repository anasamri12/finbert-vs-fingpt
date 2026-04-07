from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from .combine_preprocess_news import load_and_standardize, preprocess_all
    from .exclude_human_labelled_from_corpora import exclude_rows, load_date_text_csv
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.preprocessing.combine_preprocess_news import load_and_standardize, preprocess_all
    from src.preprocessing.exclude_human_labelled_from_corpora import exclude_rows, load_date_text_csv


DEFAULT_INPUT_GLOB = "data/raw/news_sources/*.csv"
DEFAULT_CONFIG_JSON = "data/company_aliases/top10_malaysia_large_caps.json"
DEFAULT_OUTPUT_DIR = "data/company_news"
DEFAULT_HUMAN_2025 = "data/news_labels/news_human_label_sample_2025_100_cleaned.csv"
DEFAULT_HUMAN_2026 = "data/news_labels/news_human_label_sample_2026_200_cleaned.csv"


@dataclass(frozen=True)
class CompanySpec:
    ticker: str
    name: str
    aliases: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter cleaned news into company-specific corpora for Experiment 2."
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for raw source CSVs.",
    )
    parser.add_argument(
        "--config-json",
        default=DEFAULT_CONFIG_JSON,
        help="JSON file containing company tickers, names, and aliases.",
    )
    parser.add_argument(
        "--company",
        action="append",
        default=[],
        help="Optional ticker or company name to keep. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-company outputs.",
    )
    parser.add_argument(
        "--min-date",
        default="2023-01-01",
        help="Keep rows with date >= this value (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-date",
        default="2025-12-31",
        help="Optional inclusive end date filter YYYY-MM-DD.",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=80,
        help="Minimum body length used during cleaning.",
    )
    parser.add_argument(
        "--no-text-dedupe",
        action="store_true",
        help="Disable title+body dedupe during cleaning.",
    )
    parser.add_argument(
        "--skip-exclude-human",
        action="store_true",
        help="Keep human-labelled benchmark rows in the output.",
    )
    parser.add_argument(
        "--human-2025",
        default=DEFAULT_HUMAN_2025,
        help="2025 human-labelled CSV used for exclusion.",
    )
    parser.add_argument(
        "--human-2026",
        default=DEFAULT_HUMAN_2026,
        help="2026 human-labelled CSV used for exclusion.",
    )
    return parser.parse_args()


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


def normalize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_company_specs(config_json: str) -> list[CompanySpec]:
    path = Path(config_json)
    if not path.exists():
        raise FileNotFoundError(f"Company config not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_companies = payload["companies"] if isinstance(payload, dict) else payload
    if not isinstance(raw_companies, list) or not raw_companies:
        raise ValueError(f"No companies found in config: {path}")

    specs: list[CompanySpec] = []
    for item in raw_companies:
        ticker = str(item.get("ticker", "")).strip()
        name = str(item.get("name", "")).strip()
        aliases_raw = item.get("aliases", [])
        if not ticker or not name or not aliases_raw:
            raise ValueError(f"Invalid company entry in {path}: {item}")
        aliases = tuple(_unique_keep_order([str(alias).strip() for alias in aliases_raw if str(alias).strip()]))
        if not aliases:
            raise ValueError(f"Company entry has no usable aliases in {path}: {item}")
        specs.append(CompanySpec(ticker=ticker, name=name, aliases=aliases))
    return specs


def select_company_specs(specs: list[CompanySpec], selectors: list[str]) -> list[CompanySpec]:
    if not selectors:
        return specs

    wanted = {selector.lower() for selector in _unique_keep_order(_split_repeated_args(selectors))}
    selected: list[CompanySpec] = []
    for spec in specs:
        spec_keys = {spec.ticker.lower(), spec.name.lower(), normalize_slug(spec.name)}
        if wanted.intersection(spec_keys):
            selected.append(spec)

    if not selected:
        available = ", ".join(f"{spec.ticker} ({spec.name})" for spec in specs)
        raise ValueError(f"No companies matched --company selectors. Available: {available}")
    return selected


def resolve_input_paths(input_glob: str) -> list[Path]:
    paths = sorted(Path().glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No input files matched pattern: {input_glob}")
    return paths


def load_text_only_corpus(input_paths: list[Path], min_date: date) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    file_stats: dict[str, int] = {}
    for path in input_paths:
        frame = load_date_text_csv(path).copy()
        frame["source"] = path.stem
        frame["category"] = ""
        frame["url"] = ""
        frame["title"] = frame["text"].astype(str).str.split(".", n=1).str[0].str.strip()
        frame["body"] = frame["text"].astype(str).str.strip()
        frame["source_file"] = path.name
        frames.append(frame[["source", "category", "url", "title", "date", "body", "text", "source_file"]])
        file_stats[str(path)] = int(len(frame))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise ValueError("No rows were loaded from the text-only corpus inputs.")

    dt = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.loc[dt.notna()].copy()
    dt = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.loc[dt.dt.date >= min_date].copy()
    combined["date"] = dt.loc[combined.index].dt.strftime("%Y-%m-%d")
    combined["body_chars"] = combined["body"].astype(str).str.len()
    combined = combined.sort_values(["date", "source"], ascending=[True, True]).reset_index(drop=True)

    report: dict[str, object] = {
        "input_mode": "text_only_corpus",
        "input_files": len(input_paths),
        "rows_per_file_before_clean": file_stats,
        "rows_final": int(len(combined)),
    }
    return combined, report


def load_and_clean_sources(
    input_paths: list[Path],
    min_date: date,
    min_body_chars: int,
    dedupe_text: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    file_stats: dict[str, int] = {}
    for path in input_paths:
        frame, stats = load_and_standardize(path)
        frames.append(frame)
        file_stats[str(path)] = stats["output_rows"]

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cleaned, report = preprocess_all(
        merged,
        min_date=min_date,
        min_body_chars=min_body_chars,
        dedupe_text=dedupe_text,
    )
    report["input_files"] = len(input_paths)
    report["rows_per_file_before_clean"] = file_stats
    return cleaned, report


def load_and_prepare_inputs(
    input_paths: list[Path],
    min_date: date,
    min_body_chars: int,
    dedupe_text: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = pd.read_csv(input_paths[0], nrows=2, dtype=str, keep_default_na=False, low_memory=False, encoding="utf-8-sig")
    sample_cols = {str(col).strip().lower() for col in sample.columns}
    if {"date", "text"}.issubset(sample_cols) and not {"title", "body"}.intersection(sample_cols):
        return load_text_only_corpus(input_paths, min_date=min_date)
    return load_and_clean_sources(
        input_paths=input_paths,
        min_date=min_date,
        min_body_chars=min_body_chars,
        dedupe_text=dedupe_text,
    )


def exclude_human_rows(
    cleaned: pd.DataFrame,
    human_2025_path: str,
    human_2026_path: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    human_frames: list[pd.DataFrame] = []
    for raw_path in (human_2025_path, human_2026_path):
        path = Path(raw_path)
        if path.exists():
            human_frames.append(load_date_text_csv(path))
    if not human_frames:
        return cleaned, {"excluded_rows": 0}

    human = pd.concat(human_frames, ignore_index=True).drop_duplicates(subset=["date", "text"]).reset_index(drop=True)
    filtered, excluded = exclude_rows(cleaned, human)
    return filtered, {"excluded_rows": excluded, "human_rows_loaded": int(len(human))}


def compile_alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    if re.fullmatch(r"[A-Za-z0-9.&'/-]+", alias):
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = escaped
    return re.compile(pattern, re.IGNORECASE)


def find_alias_matches(text: str, alias_patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    haystack = text if isinstance(text, str) else ""
    matches: list[str] = []
    for alias, pattern in alias_patterns:
        if pattern.search(haystack):
            matches.append(alias)
    return matches


def build_company_frame(cleaned: pd.DataFrame, spec: CompanySpec) -> pd.DataFrame:
    patterns = [(alias, compile_alias_pattern(alias)) for alias in spec.aliases]
    work = cleaned.copy()
    work["title_matches"] = work["title"].map(lambda value: find_alias_matches(str(value), patterns))
    work["body_matches"] = work["body"].map(lambda value: find_alias_matches(str(value), patterns))
    work["text_matches"] = work["text"].map(lambda value: find_alias_matches(str(value), patterns))
    work["matched_aliases"] = work.apply(
        lambda row: _unique_keep_order(row["title_matches"] + row["body_matches"] + row["text_matches"]),
        axis=1,
    )
    work["match_in_title"] = work["title_matches"].map(bool)
    work["match_in_body"] = work["body_matches"].map(bool)
    work["match_in_text"] = work["text_matches"].map(bool)
    work = work.loc[work["matched_aliases"].map(bool)].copy()
    work["ticker"] = spec.ticker
    work["company_name"] = spec.name
    work["matched_aliases"] = work["matched_aliases"].map(lambda values: "; ".join(values))
    work = work.drop(columns=["title_matches", "body_matches", "text_matches"])
    work = work.sort_values(["date", "source", "title"], ascending=[True, True, True]).reset_index(drop=True)
    return work


def export_company_outputs(company_df: pd.DataFrame, spec: CompanySpec, output_dir: Path) -> dict[str, str]:
    stem = f"{spec.ticker.replace('.', '_')}_{normalize_slug(spec.name)}"
    rich_csv = output_dir / f"{stem}_rich.csv"
    sentiment_csv = output_dir / f"{stem}_for_sentiment.csv"

    company_df.to_csv(rich_csv, index=False, encoding="utf-8-sig")
    company_df[["date", "text"]].to_csv(sentiment_csv, index=False, encoding="utf-8-sig")

    return {
        "rich_csv": str(rich_csv),
        "sentiment_csv": str(sentiment_csv),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_date = date.fromisoformat(args.min_date)
    max_date = date.fromisoformat(args.max_date) if args.max_date else None

    company_specs = select_company_specs(load_company_specs(args.config_json), args.company)
    input_paths = resolve_input_paths(args.input_glob)
    cleaned, clean_report = load_and_prepare_inputs(
        input_paths=input_paths,
        min_date=min_date,
        min_body_chars=args.min_body_chars,
        dedupe_text=not args.no_text_dedupe,
    )

    if max_date is not None:
        cleaned = cleaned.loc[pd.to_datetime(cleaned["date"], errors="coerce").dt.date <= max_date].copy()

    exclusion_report: dict[str, int] = {"excluded_rows": 0}
    if not args.skip_exclude_human:
        cleaned, exclusion_report = exclude_human_rows(
            cleaned=cleaned,
            human_2025_path=args.human_2025,
            human_2026_path=args.human_2026,
        )

    summary_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "input_glob": args.input_glob,
        "config_json": args.config_json,
        "min_date": args.min_date,
        "max_date": args.max_date,
        "skip_exclude_human": args.skip_exclude_human,
        "clean_report": clean_report,
        "exclusion_report": exclusion_report,
        "companies": [],
    }

    for spec in company_specs:
        company_df = build_company_frame(cleaned, spec)
        exported = export_company_outputs(company_df, spec, output_dir)
        summary_rows.append(
            {
                "ticker": spec.ticker,
                "company_name": spec.name,
                "rows": int(len(company_df)),
                "date_min": company_df["date"].min() if len(company_df) else None,
                "date_max": company_df["date"].max() if len(company_df) else None,
                "sources": int(company_df["source"].nunique()) if len(company_df) else 0,
                "rich_csv": exported["rich_csv"],
                "sentiment_csv": exported["sentiment_csv"],
            }
        )
        manifest["companies"].append(
            {
                "ticker": spec.ticker,
                "name": spec.name,
                "aliases": list(spec.aliases),
                **exported,
                "rows": int(len(company_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["rows", "ticker"], ascending=[False, True])
    summary_csv = output_dir / "company_filter_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    manifest["summary_csv"] = str(summary_csv)
    manifest_json = output_dir / "company_filter_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved company summary: {summary_csv}")
    print(f"Saved manifest: {manifest_json}")
    for row in summary_rows:
        print(f"{row['ticker']}: {row['rows']} rows -> {row['sentiment_csv']}")


if __name__ == "__main__":
    main()
