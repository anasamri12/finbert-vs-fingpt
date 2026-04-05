from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import kagglehub
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle news dataset and export The Star corporate/business rows."
    )
    parser.add_argument(
        "--dataset",
        default="azraimohamad/news-article-weekly-updated",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Keep rows on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/raw/news_sources/the_star_corporate_business_since_2023.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--categories",
        default="corporate,business",
        help="Comma-separated category keywords to keep.",
    )
    return parser.parse_args()


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in df.columns}
    for name in candidates:
        col = lower_to_original.get(name.lower())
        if col:
            return col
    return None


def load_all_tabular_files(dataset_dir: Path) -> pd.DataFrame:
    files = sorted(dataset_dir.rglob("*.csv")) + sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found under {dataset_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".csv":
                frame = pd.read_csv(file_path)
            else:
                frame = pd.read_parquet(file_path)
            if not frame.empty:
                frame["_source_file"] = str(file_path.relative_to(dataset_dir))
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipping unreadable file {file_path}: {exc}")

    if not frames:
        raise RuntimeError("All dataset files were empty or unreadable.")
    return pd.concat(frames, ignore_index=True)


def infer_source_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] downloading dataset: {args.dataset}")
    dataset_dir = Path(kagglehub.dataset_download(args.dataset))
    print(f"[info] dataset folder: {dataset_dir}")

    df = load_all_tabular_files(dataset_dir)
    print(f"[info] loaded rows: {len(df)}")
    print(f"[info] columns: {list(df.columns)}")

    source_col = find_first_column(df, ["source", "publisher", "site", "news_source"])
    category_col = find_first_column(df, ["category", "topic"])
    section_col = find_first_column(df, ["section", "desk", "vertical", "channel"])
    date_col = find_first_column(df, ["date", "published_date", "published_at", "datetime"])
    title_col = find_first_column(df, ["title", "headline"])
    body_col = find_first_column(df, ["body", "content", "article", "text"])
    url_col = find_first_column(df, ["url", "link", "article_url"])

    if date_col is None:
        raise RuntimeError("Could not detect date-like column.")
    if category_col is None and section_col is None:
        raise RuntimeError("Could not detect category/section-like columns.")
    if source_col is None and url_col is None:
        raise RuntimeError(
            "Could not detect source-like data. Need source column or URL column."
        )

    out = df.copy()
    if source_col is None:
        out["__source"] = out[url_col].astype(str).map(infer_source_from_url)
        source_col = "__source"
    else:
        out[source_col] = out[source_col].astype(str)

    if category_col is not None:
        out[category_col] = out[category_col].astype(str)
    if section_col is not None:
        out[section_col] = out[section_col].astype(str)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date

    category_keywords = [x.strip().lower() for x in args.categories.split(",") if x.strip()]

    source_mask = out[source_col].str.contains(r"thestar|the\s*star|thestar\.com\.my", case=False, na=False)

    category_text = pd.Series("", index=out.index)
    if category_col is not None:
        category_text = category_text + " " + out[category_col].fillna("")
    if section_col is not None:
        category_text = category_text + " " + out[section_col].fillna("")

    category_mask = pd.Series(False, index=out.index)
    for keyword in category_keywords:
        category_mask = category_mask | category_text.str.contains(keyword, case=False, na=False)
    date_mask = out[date_col] >= pd.to_datetime(args.start_date).date()

    filtered = out[source_mask & category_mask & date_mask].copy()
    filtered = filtered.sort_values(by=date_col, ascending=True)
    if url_col is not None and url_col in filtered.columns:
        filtered = filtered.drop_duplicates(subset=[url_col], keep="first")

    output_category_col = category_col if category_col is not None else section_col
    keep_cols = [col for col in [source_col, output_category_col, date_col, title_col, body_col, url_col] if col]
    filtered = filtered[keep_cols].rename(
        columns={
            source_col: "source",
            output_category_col: "category",
            date_col: "date",
            title_col: "title" if title_col else "",
            body_col: "body" if body_col else "",
            url_col: "url" if url_col else "",
        }
    )

    # Drop empty rename artifacts when title/body/url columns do not exist.
    filtered = filtered.loc[:, [c for c in filtered.columns if c]]

    filtered.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[done] saved rows: {len(filtered)}")
    print(f"[done] output: {output_path}")


if __name__ == "__main__":
    main()
