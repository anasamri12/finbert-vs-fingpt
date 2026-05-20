# Data Layout

This project treats `data/` as the canonical home for local datasets, corpora, and reusable bundles.

Most contents of `data/` are git-ignored, so a fresh clone will not include the CSV files listed below.

## Folders

- `raw/news_sources/`
  - Source CSVs collected from news sites and Kaggle pulls.
- `news_labels/`
  - Cleaned human-labelled sets and excluded-human corpora used by evaluation and adaptation scripts.
- `news_labels/pre-cleaned/`
  - Intermediate combined datasets before final cleaning.
- `news_labels/pre-excluded/`
  - Intermediate cleaned datasets before human-labelled rows are excluded.
- `samples/`
  - Small local sample inputs for quick testing.
- `splits/`
  - Fixed train/validation/test CSV splits for supervised tuning runs.
- `company_news/`
  - Company-filtered corpora used for downstream sentiment prediction.
- `kaggle_bundles/`
  - Folders prepared for Kaggle dataset upload or download.

## Compatibility Notes

- Older outputs may still exist under `results/`.
- New default script paths point to `data/`.
- The notebook paths should now point to `data/samples/` and `data/splits/`, not `src/testing/`.
- Old command entry points under `src/*.py` still work through compatibility wrappers.
