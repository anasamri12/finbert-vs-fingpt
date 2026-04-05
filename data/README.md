# Data Layout

This project now treats `data/` as the canonical home for datasets and reusable bundles.

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
- `kaggle_bundles/`
  - Folders prepared for Kaggle dataset upload or download.

## Compatibility Notes

- Older copies of some datasets may still exist under `results/` or `src/testing/`.
- New default script paths point to `data/`.
- Old command entry points under `src/*.py` still work through compatibility wrappers.
