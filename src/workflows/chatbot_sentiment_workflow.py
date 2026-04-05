from __future__ import annotations

import argparse
import difflib
import json
from io import StringIO
from pathlib import Path
import re

import pandas as pd


READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
DEFAULT_PROVIDERS = ("gemini", "deepseek", "grok", "claude", "chatgpt")
TEXT_CANDIDATES = (
    "text",
    "cleaned_text",
    "news_text",
    "news",
    "headline",
    "title",
    "body",
    "content",
    "article",
    "sentence",
)
BODY_CANDIDATES = ("body", "content", "article", "article_body")
VALID_LABELS = {"pos", "neg", "neut"}
RESPONSE_SUFFIXES = {".txt", ".md", ".csv", ".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare copy-paste sentiment batches for chatbot UIs and merge their pasted results "
            "back into a single CSV."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create prompt batches for chatbot websites/apps.")
    prepare.add_argument("--input-csv", required=True, help="Input CSV path.")
    prepare.add_argument(
        "--work-dir",
        default="",
        help="Output work directory. Defaults to results/chatbot_sentiment/<input_stem>.",
    )
    prepare.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help="Comma-separated providers: gemini,deepseek,grok,claude,chatgpt",
    )
    prepare.add_argument("--text-column", default="", help="Optional explicit text column.")
    prepare.add_argument("--body-column", default="", help="Optional explicit body column.")
    prepare.add_argument("--use-body", action="store_true", help="Append the body column to the main text.")
    prepare.add_argument("--batch-size", type=int, default=20, help="Rows per prompt batch.")
    prepare.add_argument("--start-row", type=int, default=0, help="Zero-based starting row.")
    prepare.add_argument("--max-rows", type=int, default=None, help="Optional cap for testing.")

    merge = subparsers.add_parser("merge", help="Merge pasted chatbot outputs into one CSV.")
    merge.add_argument("--work-dir", required=True, help="Work directory created by the prepare step.")
    merge.add_argument(
        "--input-csv",
        default="",
        help="Optional input CSV override. Defaults to <work-dir>/prepared_input.csv.",
    )
    merge.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. Defaults to <work-dir>/merged_sentiment.csv.",
    )
    merge.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help="Comma-separated providers: gemini,deepseek,grok,claude,chatgpt",
    )

    return parser.parse_args()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            return pd.read_csv(
                path,
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
        f"Unable to decode {path} with encodings {READ_ENCODINGS}",
    )


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], explicit: str, kind: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{kind} column '{explicit}' was not found. Available columns: {list(df.columns)}")
        return explicit
    lower_to_real = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    raise ValueError(f"Could not infer {kind} column. Available columns: {list(df.columns)}")


def normalize_whitespace(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: object) -> str:
    text = normalize_whitespace(value).lower()
    if not text:
        return ""
    if text in VALID_LABELS:
        return text
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        candidate = normalize_label(payload.get("label", ""))
        if candidate:
            return candidate
    if re.search(r"\bpos\b", text):
        return "pos"
    if re.search(r"\bneg\b", text):
        return "neg"
    if re.search(r"\bneut\b", text):
        return "neut"
    if "positive" in text or "bullish" in text:
        return "pos"
    if "negative" in text or "bearish" in text:
        return "neg"
    if "neutral" in text or "mixed" in text:
        return "neut"
    return ""


def parse_providers(raw: str) -> list[str]:
    providers = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [provider for provider in providers if provider not in DEFAULT_PROVIDERS]
    if invalid:
        raise ValueError(f"Unknown providers: {invalid}. Valid options: {list(DEFAULT_PROVIDERS)}")
    if not providers:
        raise ValueError("At least one provider is required.")
    return providers


def build_text_series(df: pd.DataFrame, text_column: str, body_column: str | None) -> pd.Series:
    text_values = df[text_column].fillna("").astype(str).map(normalize_whitespace)
    if not body_column:
        return text_values
    body_values = df[body_column].fillna("").astype(str).map(normalize_whitespace)
    return (text_values + " " + body_values).map(normalize_whitespace)


def make_work_dir(input_csv: Path, raw_work_dir: str) -> Path:
    if raw_work_dir:
        return Path(raw_work_dir)
    return Path("results") / "chatbot_sentiment" / input_csv.stem


def suggest_similar_csv_paths(target_path: Path, search_root: Path, limit: int = 5) -> list[Path]:
    if not search_root.exists():
        return []
    csv_files = [path for path in search_root.rglob("*.csv") if path.is_file()]
    if not csv_files:
        return []

    target_name = target_path.name.lower()
    candidate_names = {path.name.lower(): path for path in csv_files}
    close_name_matches = difflib.get_close_matches(target_name, list(candidate_names), n=limit, cutoff=0.45)
    suggestions: list[Path] = [candidate_names[name] for name in close_name_matches]

    if suggestions:
        return suggestions[:limit]

    target_tokens = [token for token in re.split(r"[_\W]+", target_path.stem.lower()) if token]
    scored: list[tuple[int, Path]] = []
    for path in csv_files:
        name = path.name.lower()
        score = sum(token in name for token in target_tokens)
        if score:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored[:limit]]


def row_id_for(position: int) -> str:
    return f"row_{position + 1:05d}"


def provider_ui_hint(provider: str) -> str:
    hints = {
        "chatgpt": "Use GPT-5.4 with the highest Thinking setting available in the UI.",
        "claude": "Use Claude Sonnet 4.6 with Extended Thinking enabled.",
        "gemini": "Use Gemini 3.1 Pro with Thinking enabled/high.",
        "grok": "Use the strongest Grok model available in the UI, ideally your Grok 4.x Expert/Thinking option.",
        "deepseek": "Use DeepSeek Reasoner / thinking mode.",
    }
    return hints[provider]


def render_prompt(provider: str, batch_name: str, rows_df: pd.DataFrame) -> str:
    rows_csv = rows_df.to_csv(index=False, lineterminator="\n")
    return (
        f"Batch: {batch_name}\n"
        f"Provider target: {provider}\n"
        f"UI setting: {provider_ui_hint(provider)}\n\n"
        "You are a financial news sentiment classifier.\n"
        "Classify the overall sentiment of each news item for market or company impact.\n\n"
        "Label definitions:\n"
        "- pos = clearly positive\n"
        "- neg = clearly negative\n"
        "- neut = neutral, mixed, or unclear\n\n"
        "Output rules:\n"
        "1. Return exactly one CSV code block.\n"
        "2. Use exactly these columns: row_id,label\n"
        "3. Keep every row_id exactly as given.\n"
        "4. Use only pos, neg, or neut in the label column.\n"
        "5. Do not add explanations, comments, notes, confidence scores, or extra columns.\n\n"
        "Rows to classify:\n\n"
        "```csv\n"
        f"{rows_csv}"
        "```\n"
    )


def write_workflow_notes(work_dir: Path, providers: list[str]) -> None:
    notes = [
        "1. Open the prompt files under prompts/<provider>/ and paste each batch into the matching chatbot UI.",
        "2. Make sure the chatbot is set to the requested model/thinking mode before you paste.",
        "3. Save each chatbot reply into responses/<provider>/ using any .txt, .md, .csv, or .json filename.",
        "4. Then run the merge command to combine provider labels and compute llm_consensus_label.",
        "",
        "Provider UI targets:",
    ]
    for provider in providers:
        notes.append(f"- {provider}: {provider_ui_hint(provider)}")
    (work_dir / "WORKFLOW.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def run_prepare(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        suggestions = suggest_similar_csv_paths(input_csv, Path.cwd())
        message = f"Input CSV not found: {input_csv}"
        if suggestions:
            suggestion_lines = "\n".join(f"- {path}" for path in suggestions)
            message = f"{message}\nDid you mean one of these?\n{suggestion_lines}"
        raise FileNotFoundError(message)

    providers = parse_providers(args.providers)
    work_dir = make_work_dir(input_csv, args.work_dir)
    df = read_csv_with_fallback(input_csv)
    df["source_row_index"] = [str(i) for i in range(len(df))]
    df = df.iloc[max(args.start_row, 0) :].copy()
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    df = df.reset_index(drop=True)

    text_column = infer_column(df, TEXT_CANDIDATES, args.text_column, "text")
    body_column: str | None = None
    if args.use_body:
        body_column = infer_column(df, BODY_CANDIDATES, args.body_column, "body")
    text_series = build_text_series(df, text_column, body_column)

    prepared = df.copy()
    prepared.insert(0, "chatbot_row_id", [row_id_for(i) for i in range(len(prepared))])
    prepared["chatbot_text"] = text_series

    if prepared["chatbot_text"].eq("").all():
        raise ValueError("All selected rows have empty text after preprocessing.")

    prompts_dir = work_dir / "prompts"
    batches_dir = work_dir / "batches"
    responses_dir = work_dir / "responses"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    prepared.to_csv(work_dir / "prepared_input.csv", index=False, encoding="utf-8-sig")

    manifest_cols = ["chatbot_row_id", "source_row_index", "chatbot_text"]
    for optional_col in ("title", "headline", "date"):
        if optional_col in prepared.columns and optional_col not in manifest_cols:
            manifest_cols.append(optional_col)
    prepared[manifest_cols].to_csv(work_dir / "manifest.csv", index=False, encoding="utf-8-sig")

    for provider in providers:
        (prompts_dir / provider).mkdir(parents=True, exist_ok=True)
        (responses_dir / provider).mkdir(parents=True, exist_ok=True)

    batch_size = max(args.batch_size, 1)
    batch_count = 0
    for start in range(0, len(prepared), batch_size):
        batch_count += 1
        stop = min(start + batch_size, len(prepared))
        batch_name = f"batch_{batch_count:03d}"
        batch_df = prepared.iloc[start:stop][["chatbot_row_id", "chatbot_text"]].copy()
        batch_df.columns = ["row_id", "text"]
        batch_df.to_csv(batches_dir / f"{batch_name}_rows.csv", index=False, encoding="utf-8-sig")

        for provider in providers:
            prompt_text = render_prompt(provider, batch_name, batch_df)
            (prompts_dir / provider / f"{batch_name}.txt").write_text(prompt_text, encoding="utf-8")

    write_workflow_notes(work_dir, providers)

    print(f"Prepared rows: {len(prepared)}")
    print(f"Batches: {batch_count}")
    print(f"Work dir: {work_dir}")
    print(f"Text column: {text_column}")
    if body_column:
        print(f"Body column appended: {body_column}")


def extract_code_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text)]


def parse_json_payload(text: str) -> dict[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}

    records: dict[str, str] = {}
    if isinstance(payload, dict):
        if "rows" in payload and isinstance(payload["rows"], list):
            payload = payload["rows"]
        else:
            for row_id, label in payload.items():
                normalized = normalize_label(label)
                if normalized:
                    records[str(row_id).strip()] = normalized
            return records

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                row_id = str(item.get("row_id", "")).strip()
                label = normalize_label(item.get("label", ""))
                if row_id and label:
                    records[row_id] = label
            elif isinstance(item, list) and len(item) >= 2:
                row_id = str(item[0]).strip()
                label = normalize_label(item[1])
                if row_id and label:
                    records[row_id] = label
    return records


def parse_csv_payload(text: str) -> dict[str, str]:
    candidates = [text.strip()]
    if "row_id" in text.lower() and "label" in text.lower():
        candidates.append(text[text.lower().find("row_id") :].strip())

    for candidate in candidates:
        try:
            frame = pd.read_csv(StringIO(candidate), dtype=str, keep_default_na=False)
        except Exception:
            continue
        if frame.empty:
            continue
        lower_to_real = {col.lower(): col for col in frame.columns}
        row_col = lower_to_real.get("row_id")
        label_col = lower_to_real.get("label")
        if not row_col or not label_col:
            if len(frame.columns) >= 2:
                row_col, label_col = frame.columns[:2]
            else:
                continue
        records: dict[str, str] = {}
        for _, row in frame.iterrows():
            row_id = normalize_whitespace(row[row_col])
            label = normalize_label(row[label_col])
            if row_id and label:
                records[row_id] = label
        if records:
            return records
    return {}


def parse_line_payload(text: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith("row_id"):
            continue
        pair_match = re.match(r"^(row_[A-Za-z0-9_-]+)\s*[:,-]\s*([A-Za-z]+)\s*$", line)
        if pair_match:
            row_id = pair_match.group(1).strip()
            label = normalize_label(pair_match.group(2))
            if row_id and label:
                records[row_id] = label
            continue

        delimiter = "," if "," in line else ("\t" if "\t" in line else ("|" if "|" in line else ""))
        if not delimiter:
            continue
        parts = [part.strip().strip('"').strip("'") for part in line.split(delimiter)]
        if delimiter == "|":
            parts = [part for part in parts if part]
            if parts and set(parts[0]) <= {"-"}:
                continue
        if len(parts) < 2:
            continue
        row_id = normalize_whitespace(parts[0])
        label = normalize_label(parts[1])
        if row_id and label:
            records[row_id] = label
    return records


def parse_response_text(text: str) -> dict[str, str]:
    blocks = extract_code_blocks(text)
    payload_candidates = blocks + [text]
    for payload in payload_candidates:
        for parser in (parse_json_payload, parse_csv_payload, parse_line_payload):
            records = parser(payload)
            if records:
                return records
    return {}


def load_provider_records(provider_dir: Path) -> tuple[dict[str, str], list[str]]:
    if not provider_dir.exists():
        return {}, [f"Missing responses directory: {provider_dir}"]

    files = sorted(path for path in provider_dir.rglob("*") if path.is_file() and path.suffix.lower() in RESPONSE_SUFFIXES)
    records: dict[str, str] = {}
    warnings: list[str] = []

    if not files:
        warnings.append(f"No response files found in {provider_dir}")
        return records, warnings

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_response_text(text)
        if not parsed:
            warnings.append(f"Could not parse {file_path}")
            continue
        for row_id, label in parsed.items():
            if row_id in records and records[row_id] != label:
                warnings.append(f"Conflicting labels for {row_id} in {file_path}: keeping latest value {label}")
            records[row_id] = label
    return records, warnings


def update_consensus_column(df: pd.DataFrame, providers: list[str]) -> None:
    values: list[str] = []
    for _, row in df.iterrows():
        labels = [normalize_label(row.get(f"{provider}_label", "")) for provider in providers]
        valid = [label for label in labels if label in VALID_LABELS]
        if len(valid) != len(providers):
            values.append("")
        elif len(set(valid)) == 1:
            values.append(valid[0])
        else:
            values.append("mixed")
    df["llm_consensus_label"] = values


def run_merge(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    providers = parse_providers(args.providers)
    input_csv = Path(args.input_csv) if args.input_csv else work_dir / "prepared_input.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Prepared input CSV not found: {input_csv}")

    output_csv = Path(args.output_csv) if args.output_csv else work_dir / "merged_sentiment.csv"
    df = read_csv_with_fallback(input_csv)
    if "chatbot_row_id" not in df.columns:
        raise ValueError(f"{input_csv} does not contain chatbot_row_id")
    df = df.reset_index(drop=True)

    all_warnings: list[str] = []
    for provider in providers:
        records, warnings = load_provider_records(work_dir / "responses" / provider)
        all_warnings.extend(warnings)
        label_col = f"{provider}_label"
        df[label_col] = df["chatbot_row_id"].map(lambda row_id: records.get(str(row_id).strip(), ""))

    update_consensus_column(df, providers)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Merged output: {output_csv}")
    for provider in providers:
        non_empty = int(df[f"{provider}_label"].astype(str).str.strip().ne("").sum())
        print(f"{provider}: {non_empty} labeled rows")
    consensus_non_empty = int(df["llm_consensus_label"].astype(str).str.strip().ne("").sum())
    print(f"llm_consensus_label: {consensus_non_empty} rows")
    if all_warnings:
        print("Warnings:")
        for warning in all_warnings:
            print(f"- {warning}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        run_prepare(args)
        return
    if args.command == "merge":
        run_merge(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
