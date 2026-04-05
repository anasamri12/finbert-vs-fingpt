from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MOJIBAKE_REPLACEMENTS = {
    "Â": "",
    "Â": "",
    "â€•": "-",
    "â€‹": "",
    "â€Œ": "",
    "â€": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    "Ã¢Â€Â™": "'",
    "Ã¢Â€Â˜": "'",
    "Ã¢Â€Âœ": '"',
    "Ã¢Â€Â": '"',
    "Ã¢Â€Â": "-",
    "Ã¢Â€Â”": "-",
    "Ã¢Â€Â¦": "...",
    "â\x80\x98": "'",
    "â\x80\x99": "'",
    "â\x80\x9c": '"',
    "â\x80\x9d": '"',
    "â\x80\x93": "-",
    "â\x80\x94": "-",
    "â\x80\xa6": "...",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€\x98": "'",
    "â€\x99": "'",
    "â€\x9c": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "\xa0": " ",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢": "'",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“": "'",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": '"',
    "ÃƒÂ¢Ã¢â€šÂ¬\x9d": '"',
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“": "-",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬\x9d": "-",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦": "...",
    "Ãƒâ€š ": " ",
}

DEFAULT_COLUMNS = ("text", "title", "body")
READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean mojibake symbols in pulled news CSV files."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. Defaults to <input>_cleaned.csv.",
    )
    parser.add_argument(
        "--columns",
        default="text,title,body",
        help="Comma-separated columns to clean if they exist.",
    )
    return parser.parse_args()


def _mojibake_score(text: str) -> int:
    markers = ("Ã", "Â", "â‚¬", "â€œ", "â€", "â€™", "â€¦")
    return sum(text.count(marker) for marker in markers)


def repair_mojibake(text: str) -> str:
    repaired = text
    for _ in range(2):
        best = repaired
        best_score = _mojibake_score(repaired)
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = repaired.encode(encoding, errors="strict").decode("utf-8", errors="strict")
            except UnicodeError:
                continue
            candidate_score = _mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == repaired:
            break
        repaired = best
    return repaired


def clean_mojibake(text: object) -> str:
    if text is None:
        return ""
    cleaned = repair_mojibake(str(text))
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return " ".join(cleaned.split())


def read_csv_with_fallback(input_csv: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            return pd.read_csv(
                input_csv,
                dtype=str,
                keep_default_na=False,
                low_memory=False,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding if isinstance(last_error, UnicodeDecodeError) else "unknown",
        last_error.object if isinstance(last_error, UnicodeDecodeError) else b"",
        last_error.start if isinstance(last_error, UnicodeDecodeError) else 0,
        last_error.end if isinstance(last_error, UnicodeDecodeError) else 0,
        f"Unable to decode {input_csv} with encodings {READ_ENCODINGS}",
    )


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = Path(args.output_csv) if args.output_csv else input_csv.with_name(f"{input_csv.stem}_cleaned.csv")
    requested_columns = tuple(col.strip() for col in args.columns.split(",") if col.strip()) or DEFAULT_COLUMNS

    df = read_csv_with_fallback(input_csv)
    cleaned_columns: list[str] = []
    for column in requested_columns:
        if column in df.columns:
            df[column] = df[column].map(clean_mojibake)
            cleaned_columns.append(column)

    if not cleaned_columns:
        raise ValueError(
            f"None of the requested columns were found. Requested={requested_columns}, available={tuple(df.columns)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Cleaned columns: {', '.join(cleaned_columns)}")


if __name__ == "__main__":
    main()
