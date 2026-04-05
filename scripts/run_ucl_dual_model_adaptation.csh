#!/bin/tcsh

if ($#argv < 1) then
  echo "Usage: tcsh scripts/run_ucl_dual_model_adaptation.csh <kaggle_dataset_slug>"
  echo "Example: tcsh scripts/run_ucl_dual_model_adaptation.csh your_kaggle_username/fyp-excl-human-corpora"
  exit 1
endif

set KAGGLE_DATASET = $1
set DATA_DIR = data/kaggle_bundles/excl_human
set CORPUS_2023_2025 = $DATA_DIR/news_combined_preprocessed_2023_2025_excl_human_cleaned.csv
set HUMAN_2025 = data/news_labels/news_human_label_sample_2025_100_cleaned.csv

set FINBERT_MODEL = anasamri12/finbert-malaysia-sentiment-run2
set FINGPT_ADAPTER = anasamri12/fingpt-malaysia-lora-run2
set FINGPT_BASE = TinyLlama/TinyLlama-1.1B-Chat-v1.0

mkdir -p $DATA_DIR results/predictions results/adaptation results/evaluations

if (! -f $CORPUS_2023_2025) then
  kaggle datasets download -d $KAGGLE_DATASET -p $DATA_DIR --unzip
  if ($status != 0) exit 1
endif

if (! -f $CORPUS_2023_2025) then
  echo "Missing corpus after Kaggle download: $CORPUS_2023_2025"
  exit 1
endif

echo "[1/6] FinBERT pseudo-labeling"
python src/test_sentiment_models.py \
  --input-csv $CORPUS_2023_2025 \
  --output-csv results/predictions/finbert_pseudolabels_2023_2025.csv \
  --finbert-model $FINBERT_MODEL \
  --skip-fingpt
if ($status != 0) exit 1

echo "[2/6] FinGPT pseudo-labeling"
python src/test_sentiment_models.py \
  --input-csv $CORPUS_2023_2025 \
  --output-csv results/predictions/fingpt_pseudolabels_2023_2025.csv \
  --skip-finbert \
  --fingpt-base-model $FINGPT_BASE \
  --fingpt-adapter $FINGPT_ADAPTER
if ($status != 0) exit 1

echo "[3/6] FinBERT adaptation"
python src/finetune_finbert_kaggle.py \
  --input-csv results/predictions/finbert_pseudolabels_2023_2025.csv \
  --text-column text \
  --label-column finbert_label \
  --model $FINBERT_MODEL \
  --output-dir results/adaptation/finbert_adapted_2023_2025 \
  --epochs 2 \
  --batch-size 16
if ($status != 0) exit 1

echo "[4/6] FinGPT adaptation"
python src/finetune_fingpt_kaggle.py \
  --input-csv results/predictions/fingpt_pseudolabels_2023_2025.csv \
  --text-column text \
  --label-column fingpt_label \
  --model $FINGPT_BASE \
  --output-dir results/adaptation/fingpt_adapted_2023_2025 \
  --epochs 1 \
  --batch-size 2 \
  --eval-max-rows 500
if ($status != 0) exit 1

if (-f $HUMAN_2025) then
  echo "[5/6] FinBERT 2025 evaluation"
  python src/evaluate_sentiment_models.py \
    --input-csv $HUMAN_2025 \
    --output-dir results/evaluations/adapted_2025_finbert \
    --finbert-model results/adaptation/finbert_adapted_2023_2025 \
    --skip-fingpt
  if ($status != 0) exit 1

  echo "[6/6] FinGPT 2025 evaluation"
  python src/evaluate_sentiment_models.py \
    --input-csv $HUMAN_2025 \
    --output-dir results/evaluations/adapted_2025_fingpt \
    --skip-finbert \
    --fingpt-base-model $FINGPT_BASE \
    --fingpt-adapter results/adaptation/fingpt_adapted_2023_2025
  if ($status != 0) exit 1
else
  echo "[note] Skipping 2025 evaluation because $HUMAN_2025 was not found on this server."
endif

echo "[done] Completed both model paths."
echo "FinBERT base model: $FINBERT_MODEL"
echo "FinGPT adapter: $FINGPT_ADAPTER"

