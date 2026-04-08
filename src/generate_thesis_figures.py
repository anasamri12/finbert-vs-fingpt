"""
Generate all thesis figures for docs/latex/.

Usage (from repo root):
    python src/generate_thesis_figures.py --out-dir docs/latex/figures

Figures produced
----------------
Chapter 3
  ch3_model_comparison.png          -- macro-F1 bar chart across all models
  ch3_per_class_recall.png          -- per-class recall heatmap (FinBERT run2)
  ch3_class_distribution.png        -- benchmark class distribution
  ch3_confusion_matrix.png          -- FinBERT run2 confusion matrix

Chapter 4 (uses existing PNGs from experiment2 output)
  ch4_sentiment_vs_returns_<ticker>.png  -- scatter copy/symlink
  ch4_lag_heatmap_<ticker>.png           -- lag heatmap copy

Chapter 5
  ch5_directional_accuracy.png      -- grouped bar chart across stocks & models
  ch5_r2_comparison.png             -- R² grouped bar chart
  ch5_training_history_maybank.png  -- train/val loss curves for Maybank
  ch5_predictions_maybank.png       -- predicted vs actual return on test set
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}
plt.rcParams.update(STYLE)

PALETTE = {
    "finbert": "#2166ac",
    "llama": "#4dac26",
    "qwen": "#d6604d",
    "fingpt": "#762a83",
    "persistence": "#888888",
    "price_only": "#2166ac",
    "price_plus_sentiment": "#d6604d",
    "lstm": "#4dac26",
    "gru": "#f4a582",
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVALUATIONS = RESULTS / "evaluations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def load_classification_report(eval_dir: Path) -> dict | None:
    """Load classification_report.json from an evaluation directory."""
    for sub in eval_dir.iterdir():
        candidate = sub / "classification_report.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def load_confusion_matrix(eval_dir: Path) -> pd.DataFrame | None:
    for sub in eval_dir.iterdir():
        candidate = sub / "confusion_matrix.csv"
        if candidate.exists():
            return pd.read_csv(candidate, index_col=0)
    return None


# ---------------------------------------------------------------------------
# Chapter 3 figures
# ---------------------------------------------------------------------------

def ch3_model_comparison(out_dir: Path) -> None:
    """Grouped bar chart: macro-F1 for every evaluated model on the 2025 benchmark."""

    # Hand-curated from results/evaluations/ — column indices differ between
    # finbert (no adapter_source) and fingpt (has adapter_source).
    models = [
        # label, family, macro_f1
        ("FinBERT\nZero-shot",          "finbert",  0.250),
        ("FinBERT\nFine-tuned",         "finbert",  0.620),
        ("FinBERT\nAdapted",            "finbert",  0.546),
        ("TinyLlama\nZero-shot",        "fingpt",   0.133),
        ("FinGPT\n(TinyLlama FT)",      "fingpt",   0.320),
        ("HF FinGPT\n(LLaMA2-13B)",     "fingpt",   0.133),
        ("Llama-3.1-8B\nZero-shot",     "llama",    0.458),
        ("Llama-3.1-8B\nFT ep1",        "llama",    0.431),
        ("Llama-3.1-8B\nFT ep3 low-lr", "llama",    0.418),
        ("Qwen-2.5-7B\nZero-shot",      "qwen",     0.410),
        ("Qwen-2.5-7B\nFT",             "qwen",     0.343),
    ]

    labels = [m[0] for m in models]
    families = [m[1] for m in models]
    f1s = [m[2] for m in models]

    colors = [PALETTE[f] for f in families]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), f1s, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Sentiment Model Comparison — 2025 Malaysian Benchmark (n=100)")
    ax.set_ylim(0, 0.75)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, label="0.5 reference")

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE["finbert"], label="FinBERT family"),
        mpatches.Patch(color=PALETTE["fingpt"],  label="FinGPT / TinyLlama"),
        mpatches.Patch(color=PALETTE["llama"],   label="Llama-3.1-8B"),
        mpatches.Patch(color=PALETTE["qwen"],    label="Qwen-2.5-7B"),
    ]
    ax.legend(handles=patches, loc="upper left")

    savefig(fig, out_dir / "ch3_model_comparison.png")


def ch3_confusion_matrix(out_dir: Path) -> None:
    """Normalised confusion matrix for FinBERT run2."""
    cm_df = load_confusion_matrix(EVALUATIONS / "base_2025_finbert_run2")
    if cm_df is None:
        print("  [skip] ch3_confusion_matrix: no file found")
        return

    cm = cm_df.values.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-normalise

    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("FinBERT Run2 — Normalised Confusion Matrix\n(2025 benchmark, n=100)")

    for i in range(3):
        for j in range(3):
            count = int(cm[i, j])
            frac = cm_norm[i, j]
            color = "white" if frac > 0.6 else "black"
            ax.text(j, i, f"{count}\n({frac:.2f})", ha="center", va="center",
                    fontsize=10, color=color)

    savefig(fig, out_dir / "ch3_confusion_matrix.png")


def ch3_class_distribution(out_dir: Path) -> None:
    """Bar chart of class distribution in the 2025 benchmark."""
    # From confusion matrix: support = 22 negative, 25 neutral, 53 positive
    labels = ["Negative", "Neutral", "Positive"]
    counts = [22, 25, 53]
    colors = ["#d6604d", "#888888", "#4dac26"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of samples")
    ax.set_title("2025 Benchmark Class Distribution (n=100)")
    ax.set_ylim(0, 65)
    savefig(fig, out_dir / "ch3_class_distribution.png")


def ch3_per_class_recall(out_dir: Path) -> None:
    """Per-class recall heatmap across key models."""
    report_finbert_zs = load_classification_report(EVALUATIONS / "zeroshot_2025_finbert")
    report_finbert_ft = load_classification_report(EVALUATIONS / "base_2025_finbert_run2")
    report_finbert_ad = load_classification_report(EVALUATIONS / "adapted_2025_finbert")
    report_llama_zs   = load_classification_report(EVALUATIONS / "zeroshot_2025_llama31_8b")
    report_llama_ft   = load_classification_report(EVALUATIONS / "llama31_8b_ft_2025")

    reports = {
        "FinBERT\nZero-shot":    report_finbert_zs,
        "FinBERT\nFine-tuned":   report_finbert_ft,
        "FinBERT\nAdapted":      report_finbert_ad,
        "Llama-3.1-8B\nZero-shot": report_llama_zs,
        "Llama-3.1-8B\nFT ep1": report_llama_ft,
    }
    reports = {k: v for k, v in reports.items() if v is not None}
    if not reports:
        print("  [skip] ch3_per_class_recall: no classification reports found")
        return

    classes = ["negative", "neutral", "positive"]
    model_names = list(reports.keys())
    data = np.array([[reports[m][c]["recall"] for c in classes] for m in model_names])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("Per-class Recall Comparison Across Key Models (2025 Benchmark)")

    for i, m in enumerate(model_names):
        for j, c in enumerate(classes):
            val = data[i, j]
            color = "black" if 0.3 < val < 0.7 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color)

    savefig(fig, out_dir / "ch3_per_class_recall.png")


# ---------------------------------------------------------------------------
# Chapter 4 figures — copy existing PNGs
# ---------------------------------------------------------------------------

EXP2_DIRS = {
    "1155": RESULTS / "experiment2_finbert_run2_1155_filtered",
    "1023": RESULTS / "experiment2_finbert_run2_1023_filtered",
    "5347": RESULTS / "experiment2_finbert_run2_5347_filtered",
    "1066": RESULTS / "experiment2_finbert_run2_1066_filtered",
}

TICKER_NAMES = {
    "1155": "Maybank",
    "1023": "CIMB",
    "5347": "Tenaga",
    "1066": "RHB",
}


def ch4_copy_existing(out_dir: Path) -> None:
    """Copy the scatter and lag-heatmap PNGs generated by experiment 2."""
    for ticker, src_dir in EXP2_DIRS.items():
        name = TICKER_NAMES[ticker]
        for pattern, dest_stem in [
            (f"{ticker}_KL_finbert_run2_preds__{ticker}_sentiment_vs_returns.png",
             f"ch4_sentiment_vs_returns_{ticker}_{name}"),
            (f"{ticker}_KL_finbert_run2_preds__{ticker}_lag_heatmap.png",
             f"ch4_lag_heatmap_{ticker}_{name}"),
        ]:
            src = src_dir / pattern
            if src.exists():
                dest = out_dir / f"{dest_stem}.png"
                shutil.copy2(src, dest)
                print(f"  copied: {dest.name}")
            else:
                print(f"  [skip] {pattern} not found")


# ---------------------------------------------------------------------------
# Chapter 5 figures
# ---------------------------------------------------------------------------

EXP3_DIR = RESULTS / "experiment3_main"

# Map of dataset folder suffix → display name
STOCKS = {
    "1155_KL_finbert_run2_preds__1155_merged_market": ("Maybank\n(1155.KL)", "1155"),
    "1023_KL_finbert_run2_preds__1023_merged_market": ("CIMB\n(1023.KL)",   "1023"),
    "5347_KL_finbert_run2_preds__5347_merged_market": ("Tenaga\n(5347.KL)", "5347"),
    "1066_KL_finbert_run2_preds__1066_merged_market": ("RHB\n(1066.KL)",    "1066"),
}


def _load_exp3_metrics() -> dict[str, pd.DataFrame]:
    """Return {dataset_key: metrics_df} for all 4 stocks."""
    out = {}
    for key in STOCKS:
        path = EXP3_DIR / key / "experiment3_metrics.csv"
        if path.exists():
            out[key] = pd.read_csv(path)
    return out


def ch5_directional_accuracy(out_dir: Path) -> None:
    """Grouped bar chart: directional accuracy for all models × all stocks."""
    all_metrics = _load_exp3_metrics()
    if not all_metrics:
        print("  [skip] ch5_directional_accuracy: no metrics found")
        return

    model_configs = [
        ("persistence", "price_only",         "Persistence",          "#aaaaaa", "//"),
        ("lstm",        "price_only",          "LSTM Price-only",      "#2166ac", ""),
        ("gru",         "price_only",          "GRU Price-only",       "#4393c3", ""),
        ("lstm",        "price_plus_sentiment","LSTM+Sentiment",       "#d6604d", ""),
        ("gru",         "price_plus_sentiment","GRU+Sentiment",        "#f4a582", ""),
    ]

    stock_keys = list(STOCKS.keys())
    stock_labels = [STOCKS[k][0] for k in stock_keys]
    n_stocks = len(stock_keys)
    n_models = len(model_configs)
    width = 0.14
    x = np.arange(n_stocks)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (mf, fs, label, color, hatch) in enumerate(model_configs):
        vals = []
        for key in stock_keys:
            df = all_metrics.get(key)
            if df is None:
                vals.append(0)
                continue
            row = df[(df["model_family"] == mf) & (df["feature_set"] == fs)]
            if row.empty or mf == "persistence":
                vals.append(np.nan)
            else:
                vals.append(float(row["directional_accuracy"].values[0]))
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=label, color=color,
                      hatch=hatch, edgecolor="white", linewidth=0.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(stock_labels)
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Experiment 3 — Directional Accuracy by Stock and Model")
    ax.set_ylim(0.40, 0.72)
    ax.legend(loc="upper right", ncol=2)
    savefig(fig, out_dir / "ch5_directional_accuracy.png")


def ch5_r2_comparison(out_dir: Path) -> None:
    """Grouped bar chart: R² for learned models vs persistence across stocks."""
    all_metrics = _load_exp3_metrics()
    if not all_metrics:
        print("  [skip] ch5_r2_comparison: no metrics found")
        return

    model_configs = [
        ("persistence", "price_only",         "Persistence",          "#aaaaaa"),
        ("lstm",        "price_only",          "LSTM Price-only",      "#2166ac"),
        ("gru",         "price_only",          "GRU Price-only",       "#4393c3"),
        ("lstm",        "price_plus_sentiment","LSTM+Sentiment",       "#d6604d"),
        ("gru",         "price_plus_sentiment","GRU+Sentiment",        "#f4a582"),
    ]

    stock_keys = list(STOCKS.keys())
    stock_labels = [STOCKS[k][0] for k in stock_keys]
    n_stocks = len(stock_keys)
    n_models = len(model_configs)
    width = 0.14
    x = np.arange(n_stocks)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (mf, fs, label, color) in enumerate(model_configs):
        vals = []
        for key in stock_keys:
            df = all_metrics.get(key)
            if df is None:
                vals.append(0)
                continue
            row = df[(df["model_family"] == mf) & (df["feature_set"] == fs)]
            vals.append(float(row["r2"].values[0]) if not row.empty else np.nan)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, label="Persistence R²≈0 reference")
    ax.set_xticks(x)
    ax.set_xticklabels(stock_labels)
    ax.set_ylabel("$R^2$")
    ax.set_title("Experiment 3 — $R^2$ Score by Stock and Model")
    ax.legend(loc="upper right", ncol=2)
    savefig(fig, out_dir / "ch5_r2_comparison.png")


def ch5_training_history_maybank(out_dir: Path) -> None:
    """Train / val loss curves for Maybank LSTM and GRU (both feature sets)."""
    base = EXP3_DIR / "1155_KL_finbert_run2_preds__1155_merged_market"
    configs = [
        ("training_history_lstm_price_only.csv",          "LSTM Price-only",    "#2166ac", "-"),
        ("training_history_gru_price_only.csv",           "GRU Price-only",     "#4393c3", "-"),
        ("training_history_lstm_price_plus_sentiment.csv","LSTM+Sentiment",     "#d6604d", "--"),
        ("training_history_gru_price_plus_sentiment.csv", "GRU+Sentiment",      "#f4a582", "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    titles = ["Training Loss", "Validation Loss"]

    for fname, label, color, ls in configs:
        path = base / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        axes[0].plot(df["epoch"], df["train_loss"], color=color, linestyle=ls, label=label)
        axes[1].plot(df["epoch"], df["val_loss"],   color=color, linestyle=ls, label=label)

    for ax, title in zip(axes, titles):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (scaled)")
        ax.set_title(f"Maybank — {title}")
        ax.legend(fontsize=9)

    fig.suptitle("Training History — Maybank (1155.KL)", fontsize=13)
    savefig(fig, out_dir / "ch5_training_history_maybank.png")


def ch5_predictions_maybank(out_dir: Path) -> None:
    """Predicted vs actual return on Maybank test set for best model (LSTM price-only)."""
    base = EXP3_DIR / "1155_KL_finbert_run2_preds__1155_merged_market"

    configs = [
        ("predictions_lstm_price_only.csv",          "LSTM Price-only",     "#2166ac"),
        ("predictions_gru_price_plus_sentiment.csv", "GRU+Sentiment",       "#d6604d"),
    ]

    actual = None
    dates = None
    fig, ax = plt.subplots(figsize=(12, 4))

    for fname, label, color in configs:
        path = base / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["market_date"])
        if actual is None:
            actual = df["actual_target"].values * 100  # to %
            dates = df["market_date"].values
            ax.plot(dates, actual, color="black", linewidth=1.0, label="Actual return", zorder=3)
        pred = df["predicted_target"].values * 100
        ax.plot(dates, pred, color=color, linewidth=0.8, alpha=0.8, label=label, zorder=2)

    ax.axhline(0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("1-day Return (%)")
    ax.set_title("Maybank (1155.KL) — Predicted vs Actual Return on Test Set")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    savefig(fig, out_dir / "ch5_predictions_maybank.png")


def ch5_rmse_comparison(out_dir: Path) -> None:
    """RMSE bar chart across all stocks and models."""
    all_metrics = _load_exp3_metrics()
    if not all_metrics:
        return

    model_configs = [
        ("persistence", "price_only",         "Persistence",          "#aaaaaa"),
        ("lstm",        "price_only",          "LSTM Price-only",      "#2166ac"),
        ("gru",         "price_only",          "GRU Price-only",       "#4393c3"),
        ("lstm",        "price_plus_sentiment","LSTM+Sentiment",       "#d6604d"),
        ("gru",         "price_plus_sentiment","GRU+Sentiment",        "#f4a582"),
    ]

    stock_keys = list(STOCKS.keys())
    stock_labels = [STOCKS[k][0] for k in stock_keys]
    n_stocks = len(stock_keys)
    n_models = len(model_configs)
    width = 0.14
    x = np.arange(n_stocks)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (mf, fs, label, color) in enumerate(model_configs):
        vals = []
        for key in stock_keys:
            df = all_metrics.get(key)
            if df is None:
                vals.append(np.nan)
                continue
            row = df[(df["model_family"] == mf) & (df["feature_set"] == fs)]
            vals.append(float(row["rmse"].values[0]) if not row.empty else np.nan)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(stock_labels)
    ax.set_ylabel("RMSE (daily return)")
    ax.set_title("Experiment 3 — RMSE by Stock and Model")
    ax.legend(loc="upper right", ncol=2)
    savefig(fig, out_dir / "ch5_rmse_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis figures.")
    parser.add_argument("--out-dir", default="docs/latex/figures",
                        help="Output directory for figures.")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    print("=== Chapter 3 ===")
    ch3_model_comparison(out_dir)
    ch3_confusion_matrix(out_dir)
    ch3_class_distribution(out_dir)
    ch3_per_class_recall(out_dir)

    print("\n=== Chapter 4 ===")
    ch4_copy_existing(out_dir)

    print("\n=== Chapter 5 ===")
    ch5_directional_accuracy(out_dir)
    ch5_r2_comparison(out_dir)
    ch5_rmse_comparison(out_dir)
    ch5_training_history_maybank(out_dir)
    ch5_predictions_maybank(out_dir)

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
