# FinBERT vs FinGPT: Enhancing Malaysian Stock Predictions with News-Driven Insights

This repository contains the code, local data-prep utilities, and thesis artifacts for an FYP on sentiment-driven stock forecasting for Bursa Malaysia.

## Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Before You Run Anything

- Most CSV datasets under `data/` are local and git-ignored. A fresh clone will not include them.
- Hugging Face model downloads are required for prediction, evaluation, and fine-tuning.
- Kaggle credentials are required only for the Kaggle bundle/download helpers.
- Experiment 2 downloads market data from Yahoo Finance.
- FinGPT fine-tuning is GPU-oriented; FinBERT fine-tuning is much lighter.

## Quick Start

The convenience runner now treats `all` as the prepared-data workflow:

```bash
python pipeline.py all
```

That runs:

1. `predict`
2. `evaluate`
3. `experiment2`
4. `experiment3`

Fine-tuning is available separately because it is slower and depends on local split files:

```bash
python pipeline.py finetune-finbert
python pipeline.py finetune-fingpt
```

Forward custom flags with `--`:

```bash
python pipeline.py predict -- --input-csv data/company_news/my_company_for_sentiment.csv --output-csv results/predictions/my_company_preds.csv
python pipeline.py experiment2 -- --yahoo-ticker "^KLSE,1155.KL" --lags 0,1,2
python pipeline.py experiment3 -- --epochs 50 --model-types lstm
```

## Main Commands

### Fine-tune FinBERT

Uses the fixed split files in `data/splits/`.

```bash
python src/finetune_finbert_kaggle.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/validation.csv \
  --test-csv data/splits/test.csv \
  --model ProsusAI/finbert \
  --output-dir models/finbert_kaggle
```

### Fine-tune FinGPT

```bash
python src/finetune_fingpt_kaggle.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/validation.csv \
  --test-csv data/splits/test.csv \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir models/fingpt_kaggle_lora
```

### Generate Sentiment Predictions

Despite the filename, `src/test_sentiment_models.py` is the prediction entry point.

```bash
python src/test_sentiment_models.py \
  --input-csv data/samples/malaysia_news_last_1_months_test.csv \
  --output-csv results/predictions/finbert_preds.csv \
  --skip-fingpt
```

### Experiment 1: Sentiment Evaluation

```bash
python src/evaluate_sentiment_models.py \
  --input-csv data/news_labels/news_human_label_sample_2025_100_cleaned.csv \
  --output-dir results/evaluations/sentiment_eval
```

### Experiment 2: Sentiment vs. Stock Movement

```bash
python src/experiment2_sentiment_vs_stock.py \
  --pred-glob "results/predictions/*preds.csv" \
  --yahoo-ticker "^KLSE,1155.KL,5347.KL,1023.KL,1066.KL" \
  --out-dir results/experiment2
```

### Experiment 3: Hybrid LSTM/GRU Forecasting

```bash
python src/experiment3_hybrid_forecasting.py \
  --input-glob "results/experiment2/*_merged_market.csv" \
  --out-dir results/experiment3 \
  --model-types lstm,gru \
  --epochs 100
```

## Canonical Data Layout

`data/` is the canonical home for local datasets and generated corpora:

- `data/raw/news_sources/`: raw pulled news CSVs
- `data/news_labels/`: cleaned benchmark/evaluation sets
- `data/samples/`: small local sample inputs
- `data/splits/`: train/validation/test splits for fine-tuning
- `data/company_news/`: company-filtered corpora for downstream prediction
- `data/kaggle_bundles/`: Kaggle upload/download bundles

Because these folders are mostly git-ignored, keep local copies outside the repo if you need a durable backup.

## Project Structure

```text
FYP/
|-- pipeline.py
|-- requirements.txt
|-- data/
|-- docs/
|   |-- latex/        # thesis source
|   |-- references/   # papers and reference PDFs
|   `-- examples/     # example thesis materials
`-- src/
    |-- collection/   # scraping and source ingestion
    |-- preprocessing/
    |-- datasets/
    |-- models/
    |-- evaluation/
    |-- experiments/
    |-- workflows/
    `-- bin/          # exploratory notebooks
```

## Notes

- The top-level scripts in `src/*.py` are compatibility entry points; the full implementations live in the package subfolders.
- Run commands from the project root so `src` imports resolve consistently.
- Notebooks under `src/bin/` are exploratory helpers, not the authoritative workflow.
- Archived runs already present under `results/experiment2_finbert_run2_*` and `results/experiment3_main` are reference outputs, not the default pipeline targets.

## Validation

There is currently no automated pytest suite in the repository. For now, validation is done with script-level smoke runs and generated output inspection.

## License

This repository's original code is released under the MIT License. See [LICENSE](LICENSE).

Third-party datasets, papers, checkpoints, and other externally sourced materials remain subject to their own original licenses and terms.
