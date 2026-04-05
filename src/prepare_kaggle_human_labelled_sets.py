from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT_2025 = "results/kaggle_news_label/news_human_label_sample_2025_100_cleaned.csv"
DEFAULT_INPUT_2026 = "results/kaggle_news_label/news_human_label_sample_2026_200_cleaned.csv"
DEFAULT_OUT_DIR = "results/kaggle_dataset_human_labelled"
DEFAULT_KAGGLE_ID = "anasamri/fyp-human-labelled-news-benchmarks"
DEFAULT_TITLE = "FYP Human-Labelled News Benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean Kaggle dataset bundle for the 2025 and 2026 human-labelled benchmark sets."
    )
    parser.add_argument("--input-2025", default=DEFAULT_INPUT_2025, help="Input CSV for the 2025 human-labelled set.")
    parser.add_argument("--input-2026", default=DEFAULT_INPUT_2026, help="Input CSV for the 2026 human-labelled set.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for the Kaggle bundle.")
    parser.add_argument("--kaggle-id", default=DEFAULT_KAGGLE_ID, help="Kaggle dataset id, e.g. user/slug.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Kaggle dataset title.")
    parser.add_argument(
        "--message",
        default="Human-labelled 2025 and 2026 benchmark sets for sentiment evaluation.",
        help="Suggested Kaggle version message to print at the end.",
    )
    return parser.parse_args()


def clean_cell(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def sanitize_three_column_csv(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["date", "text", "sentiment"])

        header_consumed = False
        for row in reader:
            if not row:
                continue

            first = clean_cell(row[0]) if len(row) >= 1 else ""
            second = clean_cell(row[1]) if len(row) >= 2 else ""
            third = clean_cell(row[2]) if len(row) >= 3 else ""

            if not header_consumed:
                header_consumed = True
                if first.lower() == "date" and second.lower() == "text" and third.lower() == "sentiment":
                    continue

            if not first and not second and not third:
                continue

            writer.writerow([first, second, third])
            rows_written += 1

    return rows_written


def write_metadata(out_dir: Path, kaggle_id: str, title: str) -> Path:
    metadata_path = out_dir / "dataset-metadata.json"
    payload = {
        "title": title,
        "id": kaggle_id,
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def write_readme(out_dir: Path, rows_2025: int, rows_2026: int) -> Path:
    readme_path = out_dir / "README.md"
    readme = f"""# Human-Labelled News Benchmarks

This dataset bundle contains the human-labelled benchmark sets used for sentiment evaluation.

## Files

- `news_human_label_sample_2025_100_cleaned.csv`
  Rows: {rows_2025}
  Columns: `date`, `text`, `sentiment`
- `news_human_label_sample_2026_200_cleaned.csv`
  Rows: {rows_2026}
  Columns: `date`, `text`, `sentiment`

## Notes

- The 2025 set is intended for development/validation and comparison across modelling stages.
- The 2026 set is intended for the final held-out test.
- Labels are retained exactly as provided in the cleaned benchmark files.
"""
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_2025 = out_dir / "news_human_label_sample_2025_100_cleaned.csv"
    out_2026 = out_dir / "news_human_label_sample_2026_200_cleaned.csv"

    rows_2025 = sanitize_three_column_csv(Path(args.input_2025), out_2025)
    rows_2026 = sanitize_three_column_csv(Path(args.input_2026), out_2026)

    metadata_path = write_metadata(out_dir, args.kaggle_id, args.title)
    readme_path = write_readme(out_dir, rows_2025, rows_2026)

    print(f"Prepared Kaggle bundle: {out_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"README: {readme_path}")
    print(f"2025 rows: {rows_2025}")
    print(f"2026 rows: {rows_2026}")
    print()
    print("Next steps:")
    print(f"  kaggle datasets create -p {out_dir}")
    print(f"  kaggle datasets version -p {out_dir} -m \"{args.message}\"")


if __name__ == "__main__":
    main()
