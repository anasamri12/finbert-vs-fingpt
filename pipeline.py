"""
Convenience runner for the most common project steps.

`all` runs the core prepared-data workflow:
    python pipeline.py all

Available individual steps:
    python pipeline.py finetune-finbert
    python pipeline.py finetune-fingpt
    python pipeline.py predict
    python pipeline.py evaluate
    python pipeline.py experiment2
    python pipeline.py experiment3

Pass extra flags after '--' to forward them to the underlying script:
    python pipeline.py experiment2 -- --yahoo-ticker "^KLSE,1155.KL" --lags 0,1,2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def check_file(path: str, label: str) -> bool:
    target = ROOT / path
    if not target.exists():
        print(f"  [MISSING] {label}: {path}")
        return False
    print(f"  [OK]      {label}: {path}")
    return True


def check_glob(pattern: str, label: str) -> bool:
    import glob as _glob

    matches = _glob.glob(str(ROOT / pattern))
    if not matches:
        print(f"  [MISSING] {label}: {pattern}")
        return False
    print(f"  [OK]      {label}: {len(matches)} file(s) matched {pattern}")
    return True


STEPS = {
    "finetune-finbert": {
        "description": "Fine-tune FinBERT on the local fixed-split sentiment dataset",
        "script": ["src/finetune_finbert_kaggle.py"],
        "default_args": [
            "--train-csv",
            "data/splits/train.csv",
            "--val-csv",
            "data/splits/validation.csv",
            "--test-csv",
            "data/splits/test.csv",
            "--model",
            "ProsusAI/finbert",
            "--output-dir",
            "models/finbert_kaggle",
            "--epochs",
            "3",
        ],
        "prereqs": [
            ("data/splits/train.csv", "Finetuning split (train)"),
            ("data/splits/validation.csv", "Finetuning split (validation)"),
            ("data/splits/test.csv", "Finetuning split (test)"),
        ],
    },
    "finetune-fingpt": {
        "description": "Fine-tune FinGPT (TinyLlama + LoRA) on the local fixed-split sentiment dataset",
        "script": ["src/finetune_fingpt_kaggle.py"],
        "default_args": [
            "--train-csv",
            "data/splits/train.csv",
            "--val-csv",
            "data/splits/validation.csv",
            "--test-csv",
            "data/splits/test.csv",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--output-dir",
            "models/fingpt_kaggle_lora",
            "--epochs",
            "2",
        ],
        "prereqs": [
            ("data/splits/train.csv", "Finetuning split (train)"),
            ("data/splits/validation.csv", "Finetuning split (validation)"),
            ("data/splits/test.csv", "Finetuning split (test)"),
        ],
    },
    "predict": {
        "description": "Generate a sentiment prediction CSV from a local news CSV",
        "script": ["src/test_sentiment_models.py"],
        "default_args": [
            "--input-csv",
            "data/samples/malaysia_news_last_1_months_test.csv",
            "--output-csv",
            "results/predictions/finbert_preds.csv",
            "--skip-fingpt",
        ],
        "prereqs": [
            ("data/samples/malaysia_news_last_1_months_test.csv", "Sample prediction input CSV"),
        ],
    },
    "evaluate": {
        "description": "Experiment 1 - evaluate sentiment models on human-labelled Malaysian news",
        "script": ["src/evaluate_sentiment_models.py"],
        "default_args": [
            "--input-csv",
            "data/news_labels/news_human_label_sample_2025_100_cleaned.csv",
            "--output-dir",
            "results/evaluations/sentiment_eval",
        ],
        "prereqs": [
            ("data/news_labels/news_human_label_sample_2025_100_cleaned.csv", "Human-labelled evaluation CSV"),
        ],
    },
    "experiment2": {
        "description": "Experiment 2 - correlate daily sentiment scores with Bursa Malaysia returns",
        "script": ["src/experiment2_sentiment_vs_stock.py"],
        "default_args": [
            "--pred-glob",
            "results/predictions/*preds.csv",
            "--yahoo-ticker",
            "^KLSE,1155.KL,5347.KL,1023.KL,1066.KL",
            "--out-dir",
            "results/experiment2",
        ],
        "prereq_globs": [
            ("results/predictions/*preds.csv", "Sentiment prediction CSVs"),
        ],
    },
    "experiment3": {
        "description": "Experiment 3 - train LSTM/GRU hybrid forecasting models",
        "script": ["src/experiment3_hybrid_forecasting.py"],
        "default_args": [
            "--input-glob",
            "results/experiment2/*_merged_market.csv",
            "--out-dir",
            "results/experiment3",
            "--model-types",
            "lstm,gru",
            "--epochs",
            "100",
        ],
        "prereq_globs": [
            (
                "results/experiment2/*_merged_market.csv",
                "Merged market CSVs from Experiment 2",
            ),
        ],
    },
}


def run_step(name: str, extra_args: list[str]) -> None:
    step = STEPS[name]
    print(f"\n>>> {name}: {step['description']}")

    ok = True
    for path, label in step.get("prereqs", []):
        ok &= check_file(path, label)
    for pattern, label in step.get("prereq_globs", []):
        ok &= check_glob(pattern, label)
    if not ok:
        print("\nPrerequisite check failed. See above for missing files.")
        sys.exit(1)

    args = [PYTHON] + step["script"]
    if extra_args:
        args += extra_args
    else:
        args += step["default_args"]
    run(args)


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "step",
        choices=list(STEPS.keys()) + ["all"],
        help="Which step to run. Use 'all' to run the prepared-data workflow.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the underlying script (after '--').",
    )
    ns = parser.parse_args()
    extra = [arg for arg in ns.extra if arg != "--"]
    return ns.step, extra


def main() -> None:
    step, extra = parse_args()

    if step == "all":
        if extra:
            print("Extra args are not supported with 'all'. Run individual steps instead.")
            sys.exit(1)
        ordered = ["predict", "evaluate", "experiment2", "experiment3"]
        for name in ordered:
            run_step(name, [])
        print("\n\nCore prepared-data workflow completed successfully.")
        return

    run_step(step, extra)


if __name__ == "__main__":
    main()
