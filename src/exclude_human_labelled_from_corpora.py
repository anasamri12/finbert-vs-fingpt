from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path("results/kaggle_news_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exclude cleaned human-labelled rows from cleaned corpus CSVs."
    )
    parser.add_argument(
        "--corpus-2023-2025",
        default=str(DEFAULT_BASE_DIR / "news_combined_preprocessed_2023_2025_cleaned.csv"),
        help="Cleaned 2023-2025 corpus CSV.",
    )
    parser.add_argument(
        "--corpus-2026",
        default=str(DEFAULT_BASE_DIR / "news_combined_preprocessed_2026_cleaned.csv"),
        help="Cleaned 2026 corpus CSV.",
    )
    parser.add_argument(
        "--human-2025",
        default=str(DEFAULT_BASE_DIR / "news_human_label_sample_2025_100_cleaned.csv"),
        help="Cleaned 2025 human-labelled CSV.",
    )
    parser.add_argument(
        "--human-2026",
        default=str(DEFAULT_BASE_DIR / "news_human_label_sample_2026_200_cleaned.csv"),
        help="Cleaned 2026 human-labelled CSV.",
    )
    parser.add_argument(
        "--output-2023-2025",
        default=str(DEFAULT_BASE_DIR / "news_combined_preprocessed_2023_2025_excl_human_cleaned.csv"),
        help="Output CSV for 2023-2025 corpus after excluding human-labelled rows.",
    )
    parser.add_argument(
        "--output-2026",
        default=str(DEFAULT_BASE_DIR / "news_combined_preprocessed_2026_excl_human_cleaned.csv"),
        help="Output CSV for 2026 corpus after excluding human-labelled rows.",
    )
    return parser.parse_args()


def load_date_text_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, encoding="utf-8-sig")
    required = {"date", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = out["date"].astype(str).map(normalize_date_text)
    out["text"] = out["text"].astype(str)
    out["text"] = out["text"].map(normalize_match_text)
    out = out.loc[out["date"].notna()].copy()
    return out


def normalize_date_text(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def normalize_match_text(text: str) -> str:
    cleaned = text
    replacements = {
        "\x91": "'",
        "\x92": "'",
        "\x93": '"',
        "\x94": '"',
        "\x96": "-",
        "\x97": "-",
        "\xa0": " ",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = cleaned.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def build_key_series(df: pd.DataFrame) -> pd.Series:
    return df["date"] + "\n" + df["text"]


def texts_match_loosely(corpus_text: str, human_text: str) -> bool:
    if corpus_text == human_text:
        return True
    shorter, longer = sorted((corpus_text, human_text), key=len)
    if len(shorter) >= 120 and longer.startswith(shorter):
        return True
    if len(shorter) >= 120 and shorter.startswith(longer):
        return True
    common_prefix = 0
    for a, b in zip(corpus_text, human_text):
        if a != b:
            break
        common_prefix += 1
    return common_prefix >= 160


def exclude_rows(corpus: pd.DataFrame, human_labelled: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    exact_keys = set(build_key_series(human_labelled))
    exact_mask = build_key_series(corpus).isin(exact_keys)

    remaining_human = human_labelled.loc[~build_key_series(human_labelled).isin(set(build_key_series(corpus)))].copy()
    filtered = corpus.loc[~exact_mask].copy()

    if not remaining_human.empty:
        loose_drop_indices: set[int] = set()
        corpus_by_date = {
            date_value: group[["text"]]
            for date_value, group in filtered.groupby("date", sort=False)
        }
        for row_idx, row in remaining_human.iterrows():
            candidates = corpus_by_date.get(row["date"])
            if candidates is None:
                continue
            for corpus_idx, corpus_row in candidates.itertuples():
                if texts_match_loosely(corpus_row, row["text"]):
                    loose_drop_indices.add(corpus_idx)
                    break
        if loose_drop_indices:
            filtered = filtered.drop(index=list(loose_drop_indices))

    excluded = len(corpus) - len(filtered)
    filtered = filtered.reset_index(drop=True)
    return filtered, excluded


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    corpus_2023_2025_path = Path(args.corpus_2023_2025)
    corpus_2026_path = Path(args.corpus_2026)
    human_2025_path = Path(args.human_2025)
    human_2026_path = Path(args.human_2026)
    output_2023_2025_path = Path(args.output_2023_2025)
    output_2026_path = Path(args.output_2026)

    corpus_2023_2025 = load_date_text_csv(corpus_2023_2025_path)
    corpus_2026 = load_date_text_csv(corpus_2026_path)
    human_2025 = load_date_text_csv(human_2025_path)
    human_2026 = load_date_text_csv(human_2026_path)

    filtered_2023_2025, excluded_2025 = exclude_rows(corpus_2023_2025, human_2025)
    filtered_2026, excluded_2026 = exclude_rows(corpus_2026, human_2026)

    save_csv(filtered_2023_2025, output_2023_2025_path)
    save_csv(filtered_2026, output_2026_path)

    print(f"Input 2023-2025 corpus: {corpus_2023_2025_path} ({len(corpus_2023_2025)} rows)")
    print(f"Human-labelled 2025: {human_2025_path} ({len(human_2025)} rows)")
    print(f"Excluded from 2023-2025 corpus: {excluded_2025} rows")
    print(f"Output 2023-2025 corpus: {output_2023_2025_path} ({len(filtered_2023_2025)} rows)")
    print(f"Input 2026 corpus: {corpus_2026_path} ({len(corpus_2026)} rows)")
    print(f"Human-labelled 2026: {human_2026_path} ({len(human_2026)} rows)")
    print(f"Excluded from 2026 corpus: {excluded_2026} rows")
    print(f"Output 2026 corpus: {output_2026_path} ({len(filtered_2026)} rows)")


if __name__ == "__main__":
    main()
