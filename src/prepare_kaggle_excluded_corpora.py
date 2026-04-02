from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT_2023_2025 = "results/kaggle_news_label/news_combined_preprocessed_2023_2025_excl_human_cleaned.csv"
DEFAULT_INPUT_2026 = "results/kaggle_news_label/news_combined_preprocessed_2026_excl_human_cleaned.csv"
DEFAULT_OUT_DIR = "results/kaggle_dataset_excl_human"
DEFAULT_KAGGLE_ID = "anasamri12/fyp-excl-human-corpora"
DEFAULT_TITLE = "FYP Excluded Human-Labelled Corpora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean Kaggle dataset bundle for the exclude-human-labelled corpora."
    )
    parser.add_argument("--input-2023-2025", default=DEFAULT_INPUT_2023_2025, help="Input CSV for 2023-2025 corpus.")
    parser.add_argument("--input-2026", default=DEFAULT_INPUT_2026, help="Input CSV for 2026 corpus.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for the Kaggle bundle.")
    parser.add_argument("--kaggle-id", default=DEFAULT_KAGGLE_ID, help="Kaggle dataset id, e.g. user/slug.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Kaggle dataset title.")
    parser.add_argument(
        "--message",
        default="Excluded human-labelled corpora for adaptation and evaluation workflows.",
        help="Suggested Kaggle version message to print at the end.",
    )
    return parser.parse_args()


def clean_cell(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def sanitize_two_column_csv(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["date", "text"])

        header_consumed = False
        for row in reader:
            if not row:
                continue

            first = clean_cell(row[0]) if len(row) >= 1 else ""
            second = clean_cell(row[1]) if len(row) >= 2 else ""

            if not header_consumed:
                header_consumed = True
                if first.lower() == "date" and second.lower() == "text":
                    continue

            if not first and not second:
                continue

            writer.writerow([first, second])
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


def write_readme(out_dir: Path, rows_2023_2025: int, rows_2026: int) -> Path:
    readme_path = out_dir / "README.md"
    readme = f"""# Excluded Human-Labelled Corpora

This dataset bundle contains the cleaned corpora used for downstream adaptation and evaluation workflows after removing rows that appear in the human-labelled benchmark sets.

## Files

- `news_combined_preprocessed_2023_2025_excl_human_cleaned.csv`
  Rows: {rows_2023_2025}
  Columns: `date`, `text`
- `news_combined_preprocessed_2026_excl_human_cleaned.csv`
  Rows: {rows_2026}
  Columns: `date`, `text`

## Notes

- The 2023-2025 corpus is intended for pseudo-labelling and adaptation.
- The 2026 corpus is intended for downstream holdout/forecasting workflows.
- Human-labelled rows were excluded upstream before this Kaggle bundle was created.
"""
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_2023_2025 = out_dir / "news_combined_preprocessed_2023_2025_excl_human_cleaned.csv"
    out_2026 = out_dir / "news_combined_preprocessed_2026_excl_human_cleaned.csv"

    rows_2023_2025 = sanitize_two_column_csv(Path(args.input_2023_2025), out_2023_2025)
    rows_2026 = sanitize_two_column_csv(Path(args.input_2026), out_2026)

    metadata_path = write_metadata(out_dir, args.kaggle_id, args.title)
    readme_path = write_readme(out_dir, rows_2023_2025, rows_2026)

    print(f"Prepared Kaggle bundle: {out_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"README: {readme_path}")
    print(f"2023-2025 rows: {rows_2023_2025}")
    print(f"2026 rows: {rows_2026}")
    print()
    print("Next steps:")
    print(f"  kaggle datasets create -p {out_dir}")
    print(f"  kaggle datasets version -p {out_dir} -m \"{args.message}\"")


if __name__ == "__main__":
    main()
